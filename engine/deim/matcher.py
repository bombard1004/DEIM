"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict, List

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.
    It incorporates query specialization, where queries are pre-assigned to object category groups.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, 
                 weight_dict: Dict[str, float],
                 num_queries: int = 300,
                 num_classes: int = 80,
                 specialization_groups: List[List[int]] = [
                    list(range(0, 20)),
                    list(range(20, 40)),
                    list(range(40, 60)),
                    list(range(60, 80)),
                 ],
                 use_focal_loss: bool = False, 
                 alpha: float = 0.25, 
                 gamma: float = 2.0):
        """Creates the matcher

        Params:
            weight_dict: Dict with keys 'cost_class', 'cost_bbox', 'cost_giou'
            num_queries: Total number of queries used by the model.
            num_classes: Number of object classes in the dataset.
            specialization_groups: A list of lists, where each inner list contains class IDs 
                                   belonging to one specialization group.
            use_focal_loss: Whether to use Focal Loss for classification cost.
            alpha: Alpha_param for Focal Loss.
            gamma: Gamma_param for Focal Loss.
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

        self.num_queries = num_queries
        self.num_classes = num_classes

        if not specialization_groups:
            raise ValueError("specialization_groups cannot be empty.") 
        self.num_specialization_groups = len(specialization_groups)

        self.register_buffer('coco_class_to_group_id', torch.full((num_classes,), -1, dtype=torch.long))
        for group_idx, class_ids_in_group in enumerate(specialization_groups):
            if not class_ids_in_group: 
                raise ValueError(f"Specialization group {group_idx} is empty.") 
            for class_id in class_ids_in_group:
                if not (0 <= class_id < num_classes): 
                    raise ValueError(
                        f"Class ID {class_id} in specialization_groups is out of bounds "
                        f"for num_classes {num_classes}. Expected 0 to {num_classes - 1}."
                    ) 
                if self.coco_class_to_group_id[class_id] != -1: 
                    raise ValueError(
                        f"Class ID {class_id} is assigned to multiple specialization groups. "
                        f"Previous group: {self.coco_class_to_group_id[class_id]}, new group: {group_idx}."
                    ) 
                self.coco_class_to_group_id[class_id] = group_idx
        
        self.register_buffer('query_to_group_id', torch.zeros(num_queries, dtype=torch.long))
        queries_per_group_base = num_queries // self.num_specialization_groups
        remainder_queries = num_queries % self.num_specialization_groups
        
        current_query_start_idx = 0
        for group_idx in range(self.num_specialization_groups):
            num_queries_for_this_group = queries_per_group_base + (1 if group_idx < remainder_queries else 0)
            self.query_to_group_id[current_query_start_idx : current_query_start_idx + num_queries_for_this_group] = group_idx
            current_query_start_idx += num_queries_for_this_group

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert not return_topk

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # --- 쿼리 전문화: 입력 쿼리 수 검증 ---
        if num_queries != self.num_queries:
            raise ValueError(
                f"Number of queries in model output ({num_queries}) does not match "
                f"self.num_queries ({self.num_queries}) defined in matcher init."
            )

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # FIXME，RT-DETR, different way to set NaN
        C = torch.nan_to_num(C, nan=1.0)

        C_eachs = [c[i] for i, c in enumerate(C.split(sizes, -1))]

        indices_pre = []

        for i in range(len(C_eachs)):
            target_group_ids_item = self.coco_class_to_group_id[targets[i]["labels"]]
            specialization_mask = self.query_to_group_id.unsqueeze(1) != target_group_ids_item.unsqueeze(0)

            C_eachs[i][specialization_mask] = np.inf

            try:
                ids = linear_sum_assignment(C_eachs[i])
                indices_pre.append(ids)
            except:
                import sys
                torch.set_printoptions(threshold=sys.maxsize)
                print(f"sizes: {sizes}")
                print(f"i: {i}")
                print(f"c_each.shape: {C_eachs[i].shape}")
                print(f"target_group_ids_item: {target_group_ids_item}")
                print(f"specialization_mask: {specialization_mask}")
                print(f"c_each: {C_eachs[i]}")
                raise

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
