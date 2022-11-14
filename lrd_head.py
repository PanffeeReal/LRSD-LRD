# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, images_to_levels, multi_apply,
                        unmap)
from ..builder import HEADS
from ..losses.accuracy import accuracy
from ..losses.utils import weight_reduce_loss
from .retina_head import RetinaHead


@HEADS.register_module()
class LRDHead_simple(RetinaHead):
    #
    #
    """

    Args:
        *args: Same as its base class in :class:`RetinaHead`
        score_threshold (float, optional): The score_threshold to calculate
            positive recall. If given, prediction scores lower than this value
            is counted as incorrect prediction. Default to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
        **kwargs: Same as its base class in :class:`RetinaHead`

    Example:
        >>> import torch
        >>> self = LRDHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == self.num_classes
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 *args,
                 alpha=None, # loss_reweight_func used
                 level_select=False,  ##use level select
                 #range_para, #scale_range cal used with base_edge_list
                 score_threshold=None,
                 init_cfg=None,
                 #base_edge_list=(16, 32, 64, 128, 256),  #default value is ratio of downsampling
                 #scale_ranges=None,
                 **kwargs):
        # The positive bias in self.retina_reg conv is to prevent predicted \
        #  bbox with 0 area
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(
                        type='Normal',
                        name='retina_cls',
                        std=0.01,
                        bias_prob=0.01),
                    dict(
                        type='Normal', name='retina_reg', std=0.01, bias=0.25)
                ])
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.score_threshold = score_threshold
        # self.scale_ranges=scale_ranges
        self.alpha=alpha
        self.level_select=level_select
        # self.range_para=range_para
        # self.base_edge_list=base_edge_list

    def forward_single(self, x):
        """Forward feature map of a single scale level.

        Args:
            x (Tensor): Feature map of a single scale level.

        Returns:
            tuple (Tensor):
                cls_score (Tensor): Box scores for each scale level
                    Has shape (N, num_points * num_classes, H, W).
                bbox_pred (Tensor): Box energies / deltas for each scale
                    level with shape (N, num_points * 4, H, W).
        """
        cls_score, bbox_pred = super().forward_single(x)
        # relu: TBLR encoder only accepts positive bbox_pred
        return cls_score, self.relu(bbox_pred)

    ##
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            num_level_anchors=None,#lpf
                            # # base_edge_list=(16, 32, 64, 128, 256),
                            # scale_ranges=None,##(r/x,r*x) x==4
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Most of the codes are the same with the base class
          :obj: `AnchorHead`, except that it also collects and returns
          the matched gt index in the image (from 0 to num_gt-1). If the
          anchor bbox is not matched to any gt, the corresponding value in
          pos_gt_inds is -1.
        """

        #scale_ranges=self.scale_ranges
        # range_para=self.range_para
        # base_edge_list=self.base_edge_list
#         scale_ranges=[(ba/range_para,ba*range_para) for ba in base_edge_list]
#         scale_ranges=self.scale_ranges
#         print('==============================')
#         print(scale_ranges)

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # Assign gt and sample anchors
        anchors = flat_anchors[inside_flags.type(torch.bool), :]

        #lpf
        # print(inside_flags.shape,inside_flags[0])
        # print(type(num_level_anchors),num_level_anchors)
        # print('======================')
        # num_level_anchors_inside = self.get_num_level_anchors_inside(
        #     num_level_anchors, inside_flags)
        # print(type(num_level_anchors_inside),num_level_anchors_inside)
        # print('======================')


        #end lpf
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors, label_channels),
                                          dtype=torch.float)
        pos_gt_inds = anchors.new_full((num_valid_anchors, ),
                                       -1,
                                       dtype=torch.long)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # The assigned gt_index for each anchor. (0-based)
            pos_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # shadowed_labels is a tensor composed of tuples
        #  (anchor_inds, class_label) that indicate those anchors lying in the
        #  outer region of a gt or overlapped by another gt with a smaller
        #  area.
        #
        # Therefore, only the shadowed labels are ignored for loss calculation.
        # the key `shadowed_labels` is defined in :obj:`CenterRegionAssigner`
        shadowed_labels = assign_result.get_extra_property('shadowed_labels')
        if shadowed_labels is not None and shadowed_labels.numel():
            if len(shadowed_labels.shape) == 2:
                idx_, label_ = shadowed_labels[:, 0], shadowed_labels[:, 1]
                assert (labels[idx_] != label_).all(), \
                    'One label cannot be both positive and ignored'
                label_weights[idx_, label_] = 0
            else:
                label_weights[shadowed_labels] = 0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pos_gt_inds = unmap(
                pos_gt_inds, num_total_anchors, inside_flags, fill=-1)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, pos_gt_inds)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        for i in range(len(bbox_preds)):  # loop over fpn level
            # avoid 0 area of the predicted bbox
            bbox_preds[i] = bbox_preds[i].clamp(min=1e-4)
        # TODO: It may directly use the base-class loss function.
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        batch_size = len(gt_bboxes)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg,
         pos_assigned_gt_inds_list) = cls_reg_targets

        num_gts = np.array(list(map(len, gt_labels)))
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        # `pos_assigned_gt_inds_list` (length: fpn_levels) stores the assigned
        # gt index of each anchor bbox in each fpn level.
        cum_num_gts = list(np.cumsum(num_gts))  # length of batch_size
        for i, assign in enumerate(pos_assigned_gt_inds_list):
            # loop over fpn levels
            for j in range(1, batch_size):
                # loop over batch size
                # Convert gt indices in each img to those in the batch
                assign[j][assign[j] >= 0] += int(cum_num_gts[j - 1])
            pos_assigned_gt_inds_list[i] = assign.flatten()
            labels_list[i] = labels_list[i].flatten()
        num_gts = sum(map(len, gt_labels))  # total number of gt in the batch
        # The unique label index of each gt in the batch
        label_sequence = torch.arange(num_gts, device=device)
        # Collect the average loss of each gt in each level
        with torch.no_grad():
            loss_levels, = multi_apply(
                self.collect_loss_level_single,
                losses_cls,
                losses_bbox,
                pos_assigned_gt_inds_list,
                labels_seq=label_sequence)
            # Shape: (fpn_levels, num_gts). Loss of each gt at each fpn level
            loss_levels = torch.stack(loss_levels, dim=0)
            # Locate the best fpn level for loss back-propagation
            if loss_levels.numel() == 0:  # zero gt
                argmin = loss_levels.new_empty((num_gts, ), dtype=torch.long)
            else:
                # _, argmin = loss_levels.min(dim=0)
                #lpf find the three levels sorted result
                loss_levels, argmin = loss_levels.sort(dim=0)
                argmin_lpf=argmin[0:3,:]
                loss_levels_lpf=loss_levels[0:3,:]
                # print(argmin_lpf)
                # print('=====================')

        # Reweight the loss of each (anchor, label) pair, so that only those
        #  at the best gt level are back-propagated.
        losses_cls, losses_bbox, pos_inds ,second_wei_sum= multi_apply(
            self.reweight_loss_single,
            losses_cls,
            losses_bbox,
            pos_assigned_gt_inds_list,
            labels_list,
            list(range(len(losses_cls))),
            loss_levels_lpf=loss_levels_lpf,
            min_levels=argmin_lpf)  ##lpf :  argmin  -->  argmin_lpf
        num_pos = torch.cat(pos_inds, 0).sum().float()
        pos_recall = self.calculate_pos_recall(cls_scores, labels_list,
                                               pos_inds)
        #cal weight_second sum
        sum_reweight=0
        for abc in second_wei_sum:
            sum_reweight += abc
        # print('yyyyy',sum_reweight)
        #

        if num_pos == 0:  # No gt
            avg_factor = num_pos + float(num_total_neg)
        else:
            # avg_factor = num_pos
            avg_factor = num_pos+sum_reweight
        for i in range(len(losses_cls)):
            losses_cls[i] /= avg_factor
            losses_bbox[i] /= avg_factor
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            num_pos=num_pos / batch_size,
            pos_recall=pos_recall)
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        # count=0
        # for iii in range(len(labels)):
        #     if labels[iii] == 0:
        #         count+=1
        #         print('uuuuuuuuuuuuuuuu',labels[iii])
        # print(count)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        ##lpf
#         centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        # print(labels.size(),bbox_targets.size())
        # for i ,j in zip(labels,bbox_targets):
        #     print ('label',i ,'targetsss',j)

        pos_inds = ((labels >= 0)
                    & (labels < self.num_classes)).nonzero().squeeze(1)
        pos_inds_222=bbox_targets.any(dim=1).nonzero().squeeze(1)
        # pos_inds_333=bbox_weights.any(dim=1).nonzero().squeeze(1)
        # pos_inds_444=label_weights.nonzero().squeeze(1)
        # print('ccccccccccccc')
        # print(pos_inds_222)
        # print(len(pos_inds),len(pos_inds_222),len(pos_inds_333),len(pos_inds_444))
        # for i, j, z in zip(labels[pos_inds],  bbox_targets[pos_inds], bbox_weights[pos_inds]):
        #     print('fun',i,j,z)

        # print(self.cls_out_channels, self.num_classes)
        # print('==============',labels.shape,pos_inds.shape,labels.min())
        # print(pos_inds)
        # print('------------------------------------------',len(labels[pos_inds]),labels[pos_inds])
####### pos_centerness = centerness[pos_inds_222] #used for loss calculation
        pos_bbox_targets = bbox_targets[pos_inds_222]
        # print('****************************************', len(pos_bbox_targets), pos_bbox_targets)
        anchors = anchors.reshape(-1, 4)
        pos_anchors = anchors[pos_inds_222]
        # print('before centerness',anchors.shape,pos_inds_222.shape,pos_anchors.shape,pos_bbox_targets.shape)
        centerness_targets = self.centerness_target(
            pos_anchors, pos_bbox_targets)
        # print(bbox_weights.shape,label_weights.shape,centerness_targets.shape)
        if self.level_select==True:
            print("TTTTTTTTTTTTTTTTTTTTTTTTTT")
            bbox_weights[pos_inds_222]=centerness_targets.unsqueeze(1)

        # label_weights[pos_inds_222]=centerness_targets
        # for i, j, z in zip(labels[pos_inds],  bbox_targets[pos_inds], bbox_weights[pos_inds]):
        #     print('fun',i,j,z)
        ##lpf

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            weight=bbox_weights,
            avg_factor=num_total_samples)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # # centerness loss
        # loss_centerness = self.loss_centerness(
        #     pos_centerness,
        #     centerness_targets,
        #     avg_factor=num_total_samples)
        return loss_cls, loss_bbox  ##,loss_centerness
    def calculate_pos_recall(self, cls_scores, labels_list, pos_inds):
        """Calculate positive recall with score threshold.

        Args:
            cls_scores (list[Tensor]): Classification scores at all fpn levels.
                Each tensor is in shape (N, num_classes * num_anchors, H, W)
            labels_list (list[Tensor]): The label that each anchor is assigned
                to. Shape (N * H * W * num_anchors, )
            pos_inds (list[Tensor]): List of bool tensors indicating whether
                the anchor is assigned to a positive label.
                Shape (N * H * W * num_anchors, )

        Returns:
            Tensor: A single float number indicating the positive recall.
        """
        with torch.no_grad():
            num_class = self.num_classes
            scores = [
                cls.permute(0, 2, 3, 1).reshape(-1, num_class)[pos]
                for cls, pos in zip(cls_scores, pos_inds)
            ]
            labels = [
                label.reshape(-1)[pos]
                for label, pos in zip(labels_list, pos_inds)
            ]
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(dim=1)

            return accuracy(scores, labels, thresh=self.score_threshold)

    def collect_loss_level_single(self, cls_loss, reg_loss, assigned_gt_inds,
                                  labels_seq):
        """Get the average loss in each FPN level w.r.t. each gt label.

        Args:
            cls_loss (Tensor): Classification loss of each feature map pixel,
              shape (num_anchor, num_class)
            reg_loss (Tensor): Regression loss of each feature map pixel,
              shape (num_anchor, 4)
            assigned_gt_inds (Tensor): It indicates which gt the prior is
              assigned to (0-based, -1: no assignment). shape (num_anchor),
            labels_seq: The rank of labels. shape (num_gt)

        Returns:
            shape: (num_gt), average loss of each gt in this level
        """
        if len(reg_loss.shape) == 2:  # iou loss has shape (num_prior, 4)
            reg_loss = reg_loss.sum(dim=-1)  # sum loss in tblr dims
        if len(cls_loss.shape) == 2:
            cls_loss = cls_loss.sum(dim=-1)  # sum loss in class dims
        loss = cls_loss + reg_loss
        assert loss.size(0) == assigned_gt_inds.size(0)
        # Default loss value is 1e6 for a layer where no anchor is positive
        #  to ensure it will not be chosen to back-propagate gradient
        losses_ = loss.new_full(labels_seq.shape, 1e6)
        for i, l in enumerate(labels_seq):
            match = assigned_gt_inds == l
            if match.any():
                losses_[i] = loss[match].mean()
        return losses_,
    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        # print('=========',centerness)
        assert not torch.isnan(centerness).any()
        return centerness
    def reweight_loss_single(self, cls_loss, reg_loss, assigned_gt_inds,
                             labels, level, loss_levels_lpf,min_levels):
        """Reweight loss values at each level.

        Reassign loss values at each level by masking those where the
        pre-calculated loss is too large. Then return the reduced losses.

        Args:
            cls_loss (Tensor): Element-wise classification loss.
              Shape: (num_anchors, num_classes)
            reg_loss (Tensor): Element-wise regression loss.
              Shape: (num_anchors, 4)
            assigned_gt_inds (Tensor): The gt indices that each anchor bbox
              is assigned to. -1 denotes a negative anchor, otherwise it is the
              gt index (0-based). Shape: (num_anchors, ),
            labels (Tensor): Label assigned to anchors. Shape: (num_anchors, ).
            level (int): The current level index in the pyramid

            #lpf
            #   (0-4 for RetinaNet)
            # min_levels (Tensor): The best-matching level for each gt.
            #   Shape: (num_gts, ),
            argmin_lpf (Tensor):(3, num_gts)

        Returns:
            tuple:
                - cls_loss: Reduced corrected classification loss. Scalar.
                - reg_loss: Reduced corrected regression loss. Scalar.
                - pos_flags (Tensor): Corrected bool tensor indicating the
                  final positive anchors. Shape: (num_anchors, ).
        """
        loc_weight = torch.ones_like(reg_loss)
        cls_weight = torch.ones_like(cls_loss)
        pos_flags = assigned_gt_inds >= 0  # positive pixel flag   shape:(num_points,)   bool
        pos_indices = torch.nonzero(pos_flags, as_tuple=False).flatten()  #pos_pixels' indexes in assigned_gt_inds ;shape: (num_pos,)
        weight_second=torch.zeros(0)
        weight_third=torch.zeros(0)
#=======================================================
        # if pos_flags.any():  # pos pixels exist
        #     pos_assigned_gt_inds = assigned_gt_inds[pos_flags]   #shape (num_pos,) --> gt_indes of every_pos_points
        #     zeroing_indices = (min_levels[pos_assigned_gt_inds] != level)   #shape:(num_pos,) --> bool --> pos_pixels'best level not correspond to current level
        #     neg_indices = pos_indices[zeroing_indices]      #SHAPE(num_negs,)   -->  indexs of assigned_gt_inds need to be negtive
        #
        #     if neg_indices.numel():
        #         pos_flags[neg_indices] = 0
        #         loc_weight[neg_indices] = 0
        #         # Only the weight corresponding to the label is
        #         #  zeroed out if not selected
        #         zeroing_labels = labels[neg_indices]
        #         assert (zeroing_labels >= 0).all()
        #         cls_weight[neg_indices, zeroing_labels] = 0

        #lpf :

        if pos_flags.any():  # pos pixels exist
            pos_assigned_gt_inds = assigned_gt_inds[pos_flags]   #shape (num_pos,) --> gt_inds of every_pos_point   type : number
            zeroing_indices = (min_levels[0,pos_assigned_gt_inds] !=level)  #shape:(num_pos,) --> pos_pixels' need to be change   type : bool
            second_indices  = (min_levels[1,pos_assigned_gt_inds] ==level)
            third_indices   = (min_levels[2,pos_assigned_gt_inds] ==level)

#find zeroing_indices and neg_indices
            zeroing_indices = zeroing_indices    #  pos samples need to be weight by 0   shape:(num_neg,)    type : bool
            neg_indices = pos_indices[zeroing_indices]      #SHAPE(num_negs,)   -->  indexs of assigned_gt_inds need to be negtive   type : number

            # find second and third points_indexs and reweight by a and b
            #shape : (num_second,), type : number --> index
            second_index= pos_indices[second_indices]   ## samples need to reweight by 0.5    (eg)  their index(position) of points_tensor
            third_index = pos_indices[third_indices]     ## samples need to reweight by 0.2    (eg)

            ##calculate weight by each_gt's loss
            #(3,num_gts)-->each gt's weight to 3 selected levels
            # alpha=third_index.new_full((1,),0.5)

            alpha=self.alpha
            # print('****************************************')
            # print(alpha)

            para=2.71828182846**(alpha)
            # weight_tensor=torch.exp(alpha*(loss_levels_lpf[0,:]/(loss_levels_lpf+1e-6)) - 1)
            temp=(loss_levels_lpf[0, :] / (loss_levels_lpf + 1e-6))
            weight_tensor = (torch.exp(alpha*temp)-1)/(para-1)
            #(num_second,)  (num_third,)
            # weight_second=second_index.new_full(second_index.shape,0)
            # weight_third=third_index.new_full(third_index.shape,0)

            weight_second=(weight_tensor[1,assigned_gt_inds[second_index]])
            # print('===========================')
            # # print(loss_levels_lpf.shape,weight_tensor.shape)
            # # # print(weight_tensor)
            # print(weight_second)
            weight_third=(weight_tensor[2,assigned_gt_inds[third_index]])

            # print(weight_second,weight_third)
            # weight_second=0
            # weight_third=4.0

            if neg_indices.numel():
                pos_flags[neg_indices] = 0
                loc_weight[neg_indices] = 0
#lpf
                # print('==================')
                # print(second_indices)
                loc_weight[second_index] = weight_second
                loc_weight[third_index] = weight_third

                # Only the weight corresponding to the label is
                #  zeroed out if not selected
                zeroing_labels = labels[neg_indices]
                second_labels  = labels[second_index]
                third_labels   = labels[third_index]
                assert (zeroing_labels >= 0).all()
                assert (second_labels >= 0).all()
                assert (third_labels >= 0).all()
                cls_weight[neg_indices, zeroing_labels] = 0
                cls_weight[second_index, second_labels] = weight_second
                cls_weight[third_index, third_labels] = weight_third

        # Weighted loss for both cls and reg loss
        cls_loss = weight_reduce_loss(cls_loss, cls_weight, reduction='sum')
        reg_loss = weight_reduce_loss(reg_loss, loc_weight, reduction='sum')
        weight_second_sum=weight_second.sum()
        weight_third_sum=weight_third.sum()
        # print('xxxxxxxxxxxxxxxx',weight_second.shape,weight_second_sum)
        return cls_loss, reg_loss, pos_flags ,weight_second_sum+weight_third_sum
    #lpf
    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):

        #lpf
        # print(type(num_level_anchors),num_level_anchors)
        # print(type(inside_flags),inside_flags.shape )

        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
