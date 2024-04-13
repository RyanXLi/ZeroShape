# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from datasets.symm_proj_map import QUERY_NORMALS
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss



class Matcher(nn.Module):

    def __init__(self, cost_class, cost_normal): #cost_objectness, 
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.query_normals = None


    @torch.no_grad()
    def forward(self, outputs, targets):

        batch_size = outputs["cls_prob"].shape[0]
        nprop = outputs["cls_prob"].shape[1]
        ngt = targets["gt_box_present"].shape[1]
        num_actual_gt = targets["num_actual_gt"]
        device = outputs["cls_prob"].device

        if self.query_normals is None or self.query_normals.shape[0] != batch_size:
            self.query_normals = torch.tensor(QUERY_NORMALS).to(device).unsqueeze(0).repeat(batch_size, 1, 1)

        normal_query_gt_dist = 1 - torch.matmul(self.query_normals, targets["gt_normal_normalized"].transpose(1, 2))
        final_cost = normal_query_gt_dist.detach().cpu().numpy()

        assignments = []

        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            assign = []
            if num_actual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : num_actual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }
    

class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(2)
        semcls_percls_weights[0] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_cls": self.loss_cls,
            "loss_normal": self.loss_normal,
            "loss_offset": self.loss_offset, 
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # how many planes are present in prediction

        pred_logits = outputs["cls_logits"]
        is_object = pred_logits.argmax(-1) != 0
        pred_objects = is_object.sum(1)
        card_err = F.l1_loss(pred_objects.float(), torch.zeros_like(targets["num_actual_gt"]))
        return {"loss_cardinality": card_err}

    def loss_cls(self, outputs, targets, assignments):

        # pred_logits: B x Q x 2
        # batch_size = pred_logits.shape[0]
        # nprop = pred_logits.shape[1]
        # device = pred_logits.device
        pred_logits = outputs["cls_logits"]
        # gt_box_label = torch.zeros((batch_size, nprop), dtype=torch.int64, device=device)
        gt_box_label = assignments["proposal_matched_mask"].type(torch.int64)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, 2),
            gt_box_label.reshape(-1),
            self.semcls_percls_weights,
            reduction="mean",
        )
        # loss = sigmoid_focal_loss(
        #     pred_logits,
        #     gt_box_label,
        #     # self.semcls_percls_weights,
        #     reduction="mean",
        # )


        return {"loss_cls": loss}



    def loss_normal(self, outputs, targets, assignments):
        normal_dist = outputs["normal_dist"]
        if targets["num_boxes_replica"] > 0:

            # select appropriate distances by using proposal to gt matching
            # this leaves only the valid ones, and filters out all gt that is not matching
            normal_loss = torch.gather(
                normal_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            normal_loss = normal_loss * assignments["proposal_matched_mask"]
            normal_loss = normal_loss.sum()

            if targets["num_boxes"] > 0:
                normal_loss /= targets["num_boxes"]
        else:
            normal_loss = torch.zeros(1, device=normal_dist.device).squeeze()

        return {"loss_normal": normal_loss}    
    

    def loss_offset(self, outputs, targets, assignments):
        
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:
            # zero-out non-matched proposals
            offset_loss = center_dist * assignments["proposal_matched_mask"]
            offset_loss = offset_loss.sum()

            if targets["num_boxes"] > 0:
                offset_loss /= targets["num_boxes"]
        else:
            offset_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_offset": offset_loss}    

    def single_output_forward(self, outputs, targets):

        outputs["normal_dist"] = 1 - torch.matmul(outputs["normal_normalized"], targets["gt_normal_normalized"].transpose(1, 2))

        centers = targets["center_coords"]
        center_dist = torch.matmul(outputs["normal_normalized"], centers.unsqueeze(-1)) + outputs["plane_offset"]
        center_dist = center_dist.squeeze(-1)
        outputs["center_dist"] = center_dist ** 2

        assignments = self.matcher(outputs, targets)

        outputs["assignments"] = assignments["assignments"]
        outputs["proposal_matched_mask"] = assignments["proposal_matched_mask"]
        outputs["per_prop_gt_inds"] = assignments["per_prop_gt_inds"]
        

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)


        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        num_actual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(num_actual_gt.sum()), min=1).item()
        targets["num_actual_gt"] = num_actual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = num_actual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )  

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_normal=args.matcher_normal_cost,
        # cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_cls_weight": args.loss_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_normal_weight": args.loss_normal_weight,
        "loss_offset_weight": args.loss_offset_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion
