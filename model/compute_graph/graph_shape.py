import torch
import torch.nn as nn

from utils.util import EasyDict as edict
from utils.loss import Loss
from model.shape.implicit import Implicit
from model.shape.seen_coord_enc import CoordEncAtt, CoordEncRes
from model.shape.seen_coord_enc_threedetr import CoordEncThreeDetr
from model.shape.seen_coord_enc_pointtr import CoordEncPointTr
from model.shape.rgb_enc import RGBEncAtt, RGBEncRes
from model.depth.dpt_depth import DPTDepthModel
from model.shape.symm_decoder import SymmDecoder
from utils.util import toggle_grad, interpolate_coordmap, get_child_state_dict
from utils.camera import unproj_depth, valid_norm_fac
from utils.layers import Bottleneck_Conv

import numpy as np
import json
from scipy.optimize import linear_sum_assignment


class Graph(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # define the intrinsics head
        self.intr_feat_channels = 768
        self.intr_head = nn.Sequential(
            Bottleneck_Conv(self.intr_feat_channels, kernel_size=3),
            Bottleneck_Conv(self.intr_feat_channels, kernel_size=3),
        )
        self.intr_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.intr_proj = nn.Linear(self.intr_feat_channels, 3)
        # init the last linear layer so it outputs zeros
        nn.init.zeros_(self.intr_proj.weight)
        nn.init.zeros_(self.intr_proj.bias)
            
        # define the depth pred model based on omnidata
        self.dpt_depth = DPTDepthModel(backbone='vitb_rn50_384')
        # load the pretrained depth model
        # when intrinsics need to be predicted we need to load that part as well
        self.load_pretrained_depth(opt)
        if opt.optim.fix_dpt:
            toggle_grad(self.dpt_depth, False)
            toggle_grad(self.intr_head, False)
            toggle_grad(self.intr_proj, False)

        # encoder that encode seen surface to impl conditioning vec
        if opt.arch.depth.encoder == 'resnet':
            opt.arch.depth.dsp = 1
            self.coord_encoder = CoordEncRes(opt)
        elif opt.arch.depth.encoder == 'pointtr':
            opt.arch.depth.dsp = 1
            self.coord_encoder = CoordEncPointTr(opt)
        elif opt.arch.depth.encoder == 'threedetr':
            opt.arch.depth.dsp = 1
            self.coord_encoder = CoordEncThreeDetr(opt)
        else:
            self.coord_encoder = CoordEncAtt(embed_dim=opt.arch.latent_dim, n_blocks=opt.arch.depth.n_blocks, 
                                        num_heads=opt.arch.num_heads, win_size=opt.arch.win_size//opt.arch.depth.dsp)
            
        # rgb branch (not used in final model, keep here for extension)
        if opt.arch.rgb.encoder == 'resnet':
            self.rgb_encoder = RGBEncRes(opt)
        elif opt.arch.depth.encoder == 'pointtr':
            self.rgb_encoder = None
        elif opt.arch.depth.encoder == 'threedetr':
            self.rgb_encoder = None
        elif opt.arch.rgb.encoder == 'transformer':
            self.rgb_encoder = RGBEncAtt(img_size=opt.H, embed_dim=opt.arch.latent_dim, n_blocks=opt.arch.rgb.n_blocks, 
                                        num_heads=opt.arch.num_heads, win_size=opt.arch.win_size)
        else:
            self.rgb_encoder = None
        
        # implicit function
        feat_res = opt.H // opt.arch.win_size
        self.impl_network = Implicit(feat_res**2, latent_dim=opt.arch.latent_dim*2 if self.rgb_encoder else opt.arch.latent_dim, 
                                     semantic=self.rgb_encoder is not None, n_channels=opt.arch.impl.n_channels, 
                                     n_blocks_attn=opt.arch.impl.att_blocks, n_layers_mlp=opt.arch.impl.mlp_layers, 
                                     num_heads=opt.arch.num_heads, posenc_3D=opt.arch.impl.posenc_3D, 
                                     mlp_ratio=opt.arch.impl.mlp_ratio, skip_in=opt.arch.impl.skip_in, 
                                     pos_perlayer=opt.arch.impl.posenc_perlayer)
        
        # loss functions
        self.loss_fns = Loss(opt)

        # decoder for symmetry
        with open("data/symm/candidate_planes.json") as json_file:
            candidate_planes = json.load(json_file)
        self.query_normals = torch.from_numpy(np.array(candidate_planes)).float()
        self.symm_decoder = SymmDecoder(opt, self.query_normals)
            
    def load_pretrained_depth(self, opt):
        if opt.pretrain.depth:
            # loading from our pretrained depth and intr model
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.pretrain.depth))
            checkpoint = torch.load(opt.pretrain.depth, map_location="cuda:{}".format(opt.device))
            self.dpt_depth.load_state_dict(get_child_state_dict(checkpoint["graph"], "dpt_depth"))
            # load the intr head
            if opt.device == 0:
                print("loading pretrained intr from {}...".format(opt.pretrain.depth))
            self.intr_head.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_head"))
            self.intr_proj.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_proj"))
        elif opt.arch.depth.pretrained:
            # loading from omnidata weights
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.arch.depth.pretrained))
            checkpoint = torch.load(opt.arch.depth.pretrained, map_location="cuda:{}".format(opt.device))
            state_dict = checkpoint['model_state_dict']
            self.dpt_depth.load_state_dict(state_dict)

    def intr_param2mtx(self, opt, intr_params):
        '''
        Parameters:
            opt: config
            intr_params: [B, 3], [scale_f, delta_cx, delta_cy]
        Return:
            intr: [B, 3, 3]
        '''
        batch_size = len(intr_params)
        f = 1.3875
        intr = torch.zeros(3, 3).float().to(intr_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
        intr[:, 2, 2] += 1
        # scale the focal length
        # range: [-1, 1], symmetric
        scale_f = torch.tanh(intr_params[:, 0])
        # range: [1/4, 4], symmetric
        scale_f = torch.pow(4. , scale_f)
        intr[:, 0, 0] += f * opt.W * scale_f
        intr[:, 1, 1] += f * opt.H * scale_f
        # shift the optic center, (at most to the image border)
        shift_cx = torch.tanh(intr_params[:, 1]) * opt.W / 2
        shift_cy = torch.tanh(intr_params[:, 2]) * opt.H / 2
        intr[:, 0, 2] += opt.W / 2 + shift_cx
        intr[:, 1, 2] += opt.H / 2 + shift_cy
        return intr
    
    def prepare_symm_targets(self, opt, var):
        return {
            "gt_normal_normalized": var.gt_normal_cam,
            "center_coords": var.gt_center_cam,
            "num_actual_gt": var.num_actual_gt,
            "num_total_actual_gt": var.num_actual_gt.sum(),
        }
    
    def prepare_symm_output(self, opt, var, outputs):
        targets = var.symm_targets
        normal_dist = 1 - torch.matmul(outputs["normal_normalized"], targets["gt_normal_normalized"].transpose(1, 2))

        centers = targets["center_coords"]
        centers = centers.squeeze(1)
        # max_num_planes = outputs["normal_normalized"].shape[1]
        # centers = centers.unsqueeze(1).repeat(1, max_num_planes, 1)

        # squared distance to center
        # Ax+By+Cz+D=0, center: (x,y,z), normal: (A,B,C), D from outputs
        # Input: outputs["normal_normalized"]: (B, Q, 3), centers: (B, 3), outputs["plane_offset"]: (B, Q, 1)
        # Output: outputs["center_dist"]: (B, Q)
        center_dist = torch.matmul(outputs["normal_normalized"], centers.unsqueeze(-1)) + outputs["plane_offset"]
        center_dist = center_dist.squeeze(-1)
        center_dist = center_dist ** 2
    
        return {
            "cls_prob": outputs["cls_prob"],
            "cls_logits": outputs["cls_logits"], 
            "normal": outputs["normal_normalized"],
            "offset": outputs["plane_offset"],
            "center_dist": center_dist,
            "normal_dist": normal_dist
        }
    
    def prepare_symm_assignments(self, opt, var, query_normals):
        outputs = var.symm_outputs
        batch_size = outputs["cls_prob"].shape[0]
        nprop = outputs["cls_prob"].shape[1]
        device = outputs["cls_prob"].device

        targets = var.symm_targets
        num_actual_gt = targets["num_actual_gt"]
        gt_normal_normalized = targets["gt_normal_normalized"]

        query_normals = query_normals.to(device).unsqueeze(0).repeat(batch_size, 1, 1)

        normal_query_gt_dist = 1 - torch.matmul(query_normals, gt_normal_normalized.transpose(1, 2))
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

    def forward(self, opt, var, training=False, get_loss=True):
        # def log_tensor_strides(tensor, message):
        #     print(f"{message}: sizes = {tensor.size()}, strides = {tensor.stride()}")

        batch_size = len(var.idx)
        
        # encode the rgb, [B, 3, H, W] -> [B, 1+H/(ws)*W/(ws), C], not used in our final model
        var.latent_semantic = self.rgb_encoder(var.rgb_input_map) if self.rgb_encoder else None

        # predict the depth map and intrinsics
        var.depth_pred, intr_feat = self.dpt_depth(var.rgb_input_map, get_feat=True)
        depth_map = var.depth_pred
        # predict the intrinsics
        intr_feat = self.intr_head(intr_feat)
        intr_feat = self.intr_pool(intr_feat).squeeze(-1).squeeze(-1)
        intr_params = self.intr_proj(intr_feat)
        # [B, 3, 3]
        var.intr_pred = self.intr_param2mtx(opt, intr_params)
        intr_forward = var.intr_pred
        # record the validity mask, [B, H*W]
        var.validity_mask = (var.mask_input_map>0.5).float().view(batch_size, -1)

        # project the depth to 3D points in view-centric frame
        # [B, H*W, 3], in camera coordinates
        seen_points_3D_pred = unproj_depth(opt, depth_map, intr_forward)
        # [B, H*W, 3], [B, 1, H, W] (boolean) -> [B, 3], [B]
        seen_points_mean_pred, seen_points_scale_pred = valid_norm_fac(seen_points_3D_pred, var.mask_input_map > 0.5)
        # normalize the seen surface, [B, H*W, 3]
        var.seen_points = (seen_points_3D_pred - seen_points_mean_pred.unsqueeze(1)) / seen_points_scale_pred.unsqueeze(-1).unsqueeze(-1)
        var.seen_points[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
        # [B, 3, H, W]
        seen_3D_map = var.seen_points.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
        seen_3D_dsp, mask_dsp = interpolate_coordmap(seen_3D_map, var.mask_input_map, (opt.H//opt.arch.depth.dsp, opt.W//opt.arch.depth.dsp))
        
        # encode the depth, [B, 1, H/k, W/k] -> [B, 1+H/(ws)*W/(ws), C]
        if opt.arch.depth.encoder == 'resnet':
            var.latent_depth = self.coord_encoder(seen_3D_dsp, mask_dsp)
        elif opt.arch.depth.encoder == 'pointtr':
            var.enc_xyz, var.enc_pos, var.latent_depth = self.coord_encoder(seen_3D_dsp, mask_dsp>0.5)
        elif opt.arch.depth.encoder == 'threedetr':
            # log_tensor_strides(seen_3D_dsp, "seen_3D_dsp Before operation XYZ")
            var.enc_pos, var.latent_depth = self.coord_encoder(seen_3D_dsp.permute(0, 2, 3, 1).contiguous(), mask_dsp.squeeze(1)>0.5)
            # log_tensor_strides(seen_3D_dsp, "seen_3D_dsp After operation XYZ")
        else:
            var.latent_depth = self.coord_encoder(seen_3D_dsp.permute(0, 2, 3, 1).contiguous(), mask_dsp.squeeze(1)>0.5)
        

        var.pose = var.pose_gt
        # forward for loss calculation (only during training)
        if 'gt_sample_points' in var and 'gt_sample_sdf' in var:
            with torch.no_grad():
                # get the normalizing fac based on the GT seen surface
                # project the GT depth to 3D points in view-centric frame
                # [B, H*W, 3], in camera coordinates
                seen_points_3D_gt = unproj_depth(opt, var.depth_input_map, var.intr)
                # [B, H*W, 3], [B, 1, H, W] (boolean) -> [B, 3], [B]
                seen_points_mean_gt, seen_points_scale_gt = valid_norm_fac(seen_points_3D_gt, var.mask_input_map > 0.5)
                var.seen_points_gt = (seen_points_3D_gt - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                var.seen_points_gt[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
                
                # transform the GT points accordingly
                # [B, 3, 3]
                R_gt = var.pose_gt[:, :, :3]
                # [B, 3, 1]
                T_gt = var.pose_gt[:, :, 3:]
                # [B, 3, N]
                gt_sample_points_transposed = var.gt_sample_points.permute(0, 2, 1).contiguous()
                # camera coordinates, [B, N, 3]
                gt_sample_points_cam = (R_gt @ gt_sample_points_transposed + T_gt).permute(0, 2, 1).contiguous()
                # normalize with seen std and mean, [B, N, 3]
                var.gt_points_cam = (gt_sample_points_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                
                # TODO: transform center and normal accordingly
                # [B, 3, 1]
                gt_center_transposed = var.center_coords.permute(0, 2, 1).contiguous()
                # gt_normal_valid = var.gt_normal_normalized[:, :var.num_actual_gt, :] # TODO: check indexing correctness
                gt_normal_transposed = var.gt_normal_normalized.permute(0, 2, 1).contiguous()
                # camera coordinates, [B, N, 3]
                gt_center_cam = (R_gt @ gt_center_transposed + T_gt).permute(0, 2, 1).contiguous()
                gt_normal_cam_dirty = (R_gt @ gt_normal_transposed + T_gt).permute(0, 2, 1).contiguous()
                # normalize with seen std and mean, [B, N, 3]
                var.gt_center_cam = (gt_center_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)

                gt_normal_cam_dirty = (gt_normal_cam_dirty - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                gt_normal_cam_dirty -= var.gt_center_cam
                gt_normal_cam_dirty = gt_normal_cam_dirty / torch.norm(gt_normal_cam_dirty, dim=-1, keepdim=True)
                bs = gt_normal_cam_dirty.size(0)
                for i in range(bs):
                    gt_normal_cam_dirty[i, var.num_actual_gt[i]:, :] = 0
                var.gt_normal_cam  = gt_normal_cam_dirty
                

                # get near-surface points for visualization
                # [B, 100, 3]
                close_surf_idx = torch.topk(var.gt_sample_sdf.abs(), k=100, dim=1, largest=False)[1].unsqueeze(-1).repeat(1, 1, 3)
                # [B, 100, 3]
                var.gt_surf_points = torch.gather(var.gt_points_cam, dim=1, index=close_surf_idx)
        
            # [B, N], [B, N, 1+feat_res**2], inference the impl_network for 3D loss
            # def log_tensor_strides(tensor, message):
            #     print(f"{message}: sizes = {tensor.size()}, strides = {tensor.stride()}")
            # log_tensor_strides(var.latent_depth, "Before operation XYZ")
            # print(f"vram before running of impl_network: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
            # print(f"vram before running of impl_network: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
            var.pred_sample_occ, attn = self.impl_network(var.latent_depth, var.latent_semantic, var.gt_points_cam, pos=var.enc_pos)
            # print(f"vram after running of impl_network: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
            # print(f"vram after running of impl_network: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
            # log_tensor_strides(var.latent_depth, "After operation XYZ")
            query_normals = self.query_normals.to(var.latent_depth.device)
            symm_outputs = self.symm_decoder(var.latent_depth, enc_xyz=var.enc_xyz)
            symm_outputs = symm_outputs["outputs"] # ommitted the "aux_outputs" 
            var.symm_targets = self.prepare_symm_targets(opt, var)
            var.symm_outputs = self.prepare_symm_output(opt, var, symm_outputs)
            var.symm_assignments = self.prepare_symm_assignments(opt, var, query_normals)
        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.depth is not None:
            loss.depth = self.loss_fns.depth_loss(var.depth_pred, var.depth_input_map, var.mask_input_map)
        if opt.loss_weight.intr is not None and training:
            loss.intr = self.loss_fns.intr_loss(var.seen_points, var.seen_points_gt, var.validity_mask)
        if opt.loss_weight.shape is not None and training:
            loss.shape = self.loss_fns.shape_loss(var.pred_sample_occ, var.gt_sample_sdf)
        if opt.loss_weight.symm_cls is not None and training:
            loss.symm_cls = self.loss_fns.symm_cls_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)
        if opt.loss_weight.symm_normal is not None and training:
            loss.symm_normal = self.loss_fns.symm_normal_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)
        if opt.loss_weight.symm_offset is not None and training:
            loss.symm_offset = self.loss_fns.symm_offset_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)
        return loss
