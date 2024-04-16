
from model.threedetr.helpers import GenericMLP
from model.threedetr.position_embedding import PositionEmbeddingCoordsSine
from model.threedetr.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class SymmDecoder(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        opt,
        queries,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()
        # hidden_dims = 376
        # self.encoder_to_decoder_projection = GenericMLP(
        #     input_dim=encoder_dim,
        #     hidden_dims=hidden_dims,
        #     output_dim=decoder_dim,
        #     norm_fn_name="bn1d",
        #     activation="relu",
        #     use_conv=True,
        #     output_use_activation=True,
        #     output_use_norm=True,
        #     output_use_bias=False,
        # )
        decoder_dim = opt.symm_decoder.dim

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
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=opt.symm_decoder.num_head,
            dim_feedforward=opt.symm_decoder.ffn_dim,
            dropout=opt.symm_decoder.dropout,
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers=opt.symm_decoder.num_layers, return_intermediate=False
        )
        self.build_mlp_heads(decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.queries = queries
        latent_dim = 384
        n_channels = 256
        self.latent_proj = nn.Linear(latent_dim, n_channels, bias=True)

    def build_mlp_heads(self, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # 2 classes (yes or no)
        cls_head = mlp_func(output_dim=2)

        # # continuous rotation representation ortho6d
        # regression_head = mlp_func(output_dim=6)
        #  rotation representation quaternion
        # regression_head = mlp_func(output_dim=4)
        regression_head = mlp_func(output_dim=3)

        # offset to actual plane
        plane_offset_head = mlp_func(output_dim=1)

        mlp_heads = [
            ("cls_head", cls_head),
            ("regression_head", regression_head),
            ("plane_offset_head", plane_offset_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def assemble_quaternion(self, normal_offset_bq_4):
        batch = normal_offset_bq_4.shape[0]
        
        angle = normal_offset_bq_4[...,0].contiguous().view(batch, 1)
        axis = normal_offset_bq_4[...,1:].contiguous().view(batch, 3)

        angle = torch.deg2rad(F.sigmoid(angle) * 30) # limit rotation to 0-30 degrees
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        w = torch.cos(angle / 2)
        xyz = axis * torch.sin(angle / 2)
        quat = torch.cat((w, xyz), 1)
        norm = torch.norm(quat, dim=1, keepdim=True)
        # print(norm)
        return quat

    # rotation version:
    # def compute_predicted_normal(self, normal_offset, query_xyz):

    #     # normal_unnormalized = query_xyz + normal_offset
    #     # normal_normalized = normal_unnormalized / normal_unnormalized.norm(dim=-1, keepdim=True)
    #     # normal_offset: B x Q x 6, query_xyz: B x Q x 3
    #     batch_size, query_size, _ = normal_offset.shape
    #     # normal_offset_bq_6 = normal_offset.reshape(-1, 6)
    #     # rot: B*Q x 3 x 3
    #     # rot = self.compute_rotation_matrix_from_ortho6d(normal_offset_bq_6)

    #     normal_offset_bq_4 = normal_offset.reshape(-1, 4)
    #     quat = self.assemble_quaternion(normal_offset_bq_4)
    #     # rot: B*Q x 3 x 3
    #     rot = self.compute_rotation_matrix_from_quaternion(quat)

    #     query_xyz_bq_3= query_xyz.view(-1, 3)
    #     normal_normalized = torch.matmul(rot, query_xyz_bq_3.unsqueeze(-1)).squeeze(-1)

    #     # # Use below to disable regression
    #     # normal_normalized = query_xyz_bq_3

    #     normal_normalized = normal_normalized.view(batch_size, query_size, 3)

    #     return normal_normalized, normal_normalized


    def compute_predicted_normal(self, normal_offset, query_xyz):
        normal_offset = normal_offset / torch.norm(normal_offset, dim=-1, keepdim=True)
        normal_offset *= 0.95
        normal_unnormalized = query_xyz + normal_offset
        normal_normalized = normal_unnormalized / torch.norm(normal_unnormalized, dim=-1, keepdim=True)


        return normal_normalized, normal_normalized

    def get_query_embeddings(self, device, bs, point_cloud_dims):
        # query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        # query_inds = query_inds.long()
        # query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        # query_xyz = torch.stack(query_xyz)
        # query_xyz = query_xyz.permute(1, 2, 0) # 8 x 128 x 3 in training

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)

        query_xyz = self.queries.clone().detach().to(device).unsqueeze(0).repeat(bs, 1, 1)

        pos_embed = self.pos_embedding(query_xyz)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    # def run_encoder(self, point_clouds):
    #     xyz, features = self._break_up_pc(point_clouds)
    #     pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
    #     # xyz: batch x npoints x 3
    #     # features: batch x channel x npoints
    #     # inds: batch x npoints

    #     # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
    #     pre_enc_features = pre_enc_features.permute(2, 0, 1)

    #     # xyz points are in batch x npointx channel order
    #     enc_xyz, enc_features, enc_inds = self.encoder(
    #         pre_enc_features, xyz=pre_enc_xyz
    #     )
    #     if enc_inds is None:
    #         # encoder does not perform any downsampling
    #         enc_inds = pre_enc_inds
    #     else:
    #         # use gather here to ensure that it works for both FPS and random sampling
    #         enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
    #     return enc_xyz, enc_features, enc_inds
    

    # batch*n
    def normalize_vector(self, v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v

    # u, v batch*n
    def cross_product(self, u, v):
        batch = u.shape[0]
        #print (u.shape)
        #print (v.shape)
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
            
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
            
        return out

    # def compute_rotation_matrix_from_ortho6d(self, ortho6d):
    #     x_raw = ortho6d[:,0:3]#batch*3
    #     y_raw = ortho6d[:,3:6]#batch*3
            
    #     x = self.normalize_vector(x_raw) #batch*3
    #     z = self.cross_product(x,y_raw) #batch*3
    #     z = self.normalize_vector(z)#batch*3
    #     y = self.cross_product(z,x)#batch*3
            
    #     x = x.view(-1,3,1)
    #     y = y.view(-1,3,1)
    #     z = z.view(-1,3,1)
    #     matrix = torch.cat((x,y,z), 2) #batch*3*3
    #     return matrix

    # batch*n
    def normalize_vector(self, v, return_mag=False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v

    #quaternion batch*4
    def compute_rotation_matrix_from_quaternion(self, quaternion):
        batch=quaternion.shape[0]
        
        
        quat = self.normalize_vector(quaternion).contiguous()
        
        qw = quat[...,0].contiguous().view(batch, 1)
        qx = quat[...,1].contiguous().view(batch, 1)
        qy = quat[...,2].contiguous().view(batch, 1)
        qz = quat[...,3].contiguous().view(batch, 1)

        # Unit quaternion rotation matrices computatation  
        xx = qx*qx
        yy = qy*qy
        zz = qz*qz
        xy = qx*qy
        xz = qx*qz
        yz = qy*qz
        xw = qx*qw
        yw = qy*qw
        zw = qz*qw
        
        row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
        row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
        row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
        
        matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
        
        return matrix

    def get_plane_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.unsqueeze(0)
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["cls_head"](box_features).transpose(1, 2)
        normal_offset = (
            self.mlp_heads["regression_head"](box_features).transpose(1, 2)
        )
        plane_offset = (
            self.mlp_heads["plane_offset_head"](box_features).transpose(1, 2)
        )

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        normal_offset = normal_offset.reshape(num_layers, batch, num_queries, -1)
        plane_offset = plane_offset.reshape(num_layers, batch, num_queries, -1)

        outputs = []
        for l in range(num_layers):
            (
                normal_normalized,
                normal_unnormalized,
            ) = self.compute_predicted_normal(
                normal_offset[l], query_xyz
            )
            

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                cls_prob = torch.nn.functional.softmax(cls_logits[l], dim=-1)
                # cls_prob = cls_prob[..., :-1]

            plane_prediction = {
                "cls_logits": cls_logits[l],
                "normal_normalized": normal_normalized.contiguous(),
                "normal_unnormalized": normal_unnormalized.contiguous(),
                "plane_offset": plane_offset[l],
                "cls_prob": cls_prob,
            }
            outputs.append(plane_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }


    def forward(self, features, enc_xyz):

        features = self.latent_proj(features) # features: B, 128 (ntoken), 384 (dim)
        features = features.permute(1, 0, 2) # features: 128, B, 384
        # enc_xyz = enc_xyz.permute(1, 0, 2) # enc_xyz: 128, B, 3
        # enc_features = self.encoder_to_decoder_projection(
        #     enc_features.permute(1, 2, 0)
        # ).permute(2, 0, 1)
        # # encoder features: npoints x batch x channel
        # # encoder xyz: npoints x batch x 3

        # point_cloud_dims = [
        #     inputs["point_cloud_dims_min"],
        #     inputs["point_cloud_dims_max"],
        # ]
        point_cloud_dims = [torch.tensor(-1.).to(features.device), torch.tensor(1.).to(features.device)] # TODO:fix
        query_xyz, query_embed = self.get_query_embeddings(device=features.device, bs=features.shape[1], point_cloud_dims=point_cloud_dims)
        # query_embed: batch x channel x npoint

        # point_cloud_dims = [
        #     inputs["point_cloud_dims_min"],
        #     inputs["point_cloud_dims_max"],
        # ]
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel 
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_plane_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions

        


# decoder_layer = TransformerDecoderLayer(
#     d_model=args.dec_dim,
#     nhead=args.dec_nhead,
#     dim_feedforward=args.dec_ffn_dim,
#     dropout=args.dec_dropout,
# )
# decoder = TransformerDecoder(
#     decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
# )