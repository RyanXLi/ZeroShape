# This source code is written based on https://github.com/facebookresearch/MCC 
# The original code base is licensed under the license found in the LICENSE file in the root directory.

import torch
import torch.nn as nn
import torchvision

from functools import partial
from timm.models.vision_transformer import Block
from utils.pos_embed import get_2d_sincos_pos_embed
from utils.layers import Bottleneck_Conv


from external.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from external.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points

from model.threedetr.helpers import GenericMLP
from model.threedetr.position_embedding import PositionEmbeddingCoordsSine
from model.threedetr.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


class CoordEncThreeDetr(nn.Module):
    """ 
    Seen surface encoder based on threedetr encoder, includes a pointnet++ layer and a transformer encoder.
    """
    def __init__(self, opt):
        super().__init__()

        # self.encoder = torchvision.models.resnet50(pretrained=True)
        # self.encoder.fc = nn.Sequential(
        #     Bottleneck_Conv(2048),
        #     Bottleneck_Conv(2048),
        #     nn.Linear(2048, opt.arch.latent_dim)
        # )

        # ------------ begin threedetr --------------
        encoder_dim=256
        decoder_dim=256
        position_embedding="fourier"
        mlp_dropout=0.3
        
        self.pre_encoder = build_preencoder(opt.threedetr)
        self.encoder = build_encoder(opt.threedetr)
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=False
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        
        # print(f"vram after init of threedetr: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
        # print(f"vram after init of threedetr: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
        # self.decoder = build_decoder(opt.threedetr)
        # self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        # ------------ end threedetr ---------------

        # # define hooks
        # self.seen_feature = None
        # def feature_hook(model, input, output):
        #     self.seen_feature = output
        
        # # attach hooks
        # assert opt.arch.depth.dsp == 1
        # if (opt.arch.win_size) == 16:
        #     self.encoder.layer3.register_forward_hook(feature_hook)
        #     self.depth_feat_proj = nn.Sequential(
        #         Bottleneck_Conv(1024),
        #         Bottleneck_Conv(1024),
        #         nn.Conv2d(1024, opt.arch.latent_dim, 1)
        #     )
        # elif (opt.arch.win_size) == 32:
        #     self.encoder.layer4.register_forward_hook(feature_hook)
        #     self.depth_feat_proj = nn.Sequential(
        #         Bottleneck_Conv(2048),
        #         Bottleneck_Conv(2048),
        #         nn.Conv2d(2048, opt.arch.latent_dim, 1)
        #     )
        # else:
        #     print('Make sure win_size is 16 or 32 when using resnet backbone!')
        #     raise NotImplementedError
   
    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        return enc_xyz, enc_features, enc_inds
    

    def forward(self, coord_obj, mask_obj):
        # print(f"vram before running of threedetr: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
        # print(f"vram before running of threedetr: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

        # image = image[~bg_mask, :]
        # image /= 255.0
        # image -= 0.5
        # image *= 2.0
        # # print(image.shape)
        # point_cloud = image.reshape(-1, 3) 
        # point_cloud, choices = pc_util.random_sampling(
        #     point_cloud, self.num_points, return_choices=True
        # )

        batch_size = coord_obj.shape[0]
        
        # print(f"shape of mask_obj: {mask_obj.shape}")
        # print(f"shape of coord_obj: {coord_obj.shape}")
        # shape of mask_obj: torch.Size([1, 224, 224])
        # shape of coord_obj: torch.Size([1, 224, 224, 3])

        
        # print(f"shape of mask_obj: {mask_obj.shape}")
        # print(f"shape of coord_obj: {coord_obj.shape}")
        # coord_obj = coord_obj[mask_obj]
        coord_obj = coord_obj * mask_obj.unsqueeze(-1)
        # print(f"shape of coord_obj after: {coord_obj.shape}")
        # print(f"max of coord_obj: {torch.max(coord_obj)}")
        # print(f"min of coord_obj: {torch.min(coord_obj)}")
        # print(f"mean of coord_obj: {torch.mean(coord_obj)}")
        # print(f"shape of coord_obj: {coord_obj.shape}")
        # print()

        coord_obj = coord_obj.view(batch_size, -1, 3).contiguous()
        # coord_obj = coord_obj[:, :25088, :]

        # ----------------- prev -------------------
        # # [B, 1, C]
        # global_feat = self.encoder(coord_obj).unsqueeze(1)
        # # [B, C, H/ws*W/ws]
        # local_feat = self.depth_feat_proj(self.seen_feature).view(batch_size, global_feat.shape[-1], -1)
        # # [B, H/ws*W/ws, C]
        # local_feat = local_feat.permute(0, 2, 1).contiguous()
        # # [B, 1+H/ws*W/ws, C]
        # seen_embedding = torch.cat([global_feat, local_feat], dim=1)

        point_clouds = coord_obj

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0).contiguous()
        ).permute(2, 0, 1).contiguous()
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        point_cloud_dims = [torch.tensor(-1.).to(point_clouds.device), torch.tensor(1.).to(point_clouds.device)]
        # query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(0, 2, 1).contiguous()
        # query_embed = query_embed.permute(2, 0, 1)
        # tgt = torch.zeros_like(query_embed)
        # box_features = self.decoder(
        #     tgt, enc_features, query_pos=query_embed, pos=enc_pos
        # )[0]

        # print(f"vram after running of threedetr: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
        # print(f"vram after running of threedetr: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")



    # if encoder_only:
        # return: batch x npoints x channels
        # print(f"enc_features.shape: {enc_features.shape}")
        # print(f"enc_pos.shape: {enc_pos.shape}")
        return enc_pos, enc_features.transpose(0, 1).contiguous()

        # # point_cloud_dims = [
        # #     inputs["point_cloud_dims_min"],
        # #     inputs["point_cloud_dims_max"],
        # # ]
        # point_cloud_dims = [torch.tensor(-1.).to(point_clouds.device), torch.tensor(1.).to(point_clouds.device)]
        # query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # # query_embed: batch x channel x npoint
        # enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # # decoder expects: npoints x batch x channel
        # enc_pos = enc_pos.permute(2, 0, 1)
        # query_embed = query_embed.permute(2, 0, 1)
        # tgt = torch.zeros_like(query_embed)
        # box_features = self.decoder(
        #     tgt, enc_features, query_pos=query_embed, pos=enc_pos
        # )[0]

        # box_predictions = self.get_plane_predictions(
        #     query_xyz, point_cloud_dims, box_features
        # )
        # return box_predictions
