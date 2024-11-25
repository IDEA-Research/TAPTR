
import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn
from models.dino.deformable_transformer import DeformableTransformer


class PointDeformableTransformer(DeformableTransformer):
    """A transformer that can handle multiple tasks, including detection, segmentation, and point tracking.

    Args:
        activate_det_seg (bool): whether to activate the detection / segmentation task. If it is true, then it will conduct the operations for detection / segmentation.
        activate_point_tracking (bool): whether to activate the point tracking task. If it is true, then it will conduct the extra operations for point tracking.
        query_feature_initialize_mode (List[str]): which features to be used to initialize the feature of each points
            first_frame_feature_bilinear: conduct bilinear interpolation at the in the feature maps of the first frame.
            first_frame_pe_bilinear     : conduct bilinear interpolation at the in the positional encoding of feature maps of the first frame.
            initial_visib               : the visibility of the points at the first frame.
    """
    def __init__(self, 
            d_model=256, nhead=8, num_queries=300, num_encoder_layers=6, num_unicoder_layers=0, num_decoder_layers=6, dim_feedforward=2048, dropout=0, activation="relu", normalize_before=False, return_intermediate_dec=False, query_dim=4, num_patterns=0, modulate_hw_attn=False, deformable_encoder=False, deformable_decoder=False, num_feature_levels=1, enc_n_points=4, dec_n_points=4, box_attn_type='roi_align', learnable_tgt_init=False, decoder_query_perturber=None, add_channel_attention=False, add_pos_value=False, random_refpoints_xy=False, two_stage_type='no', two_stage_pat_embed=0, two_stage_add_query_num=0, two_stage_learn_wh=False, two_stage_keep_all_tokens=False, dec_layer_number=None, rm_enc_query_scale=True, rm_dec_query_scale=True, rm_self_attn_layers=None, key_aware_type=None, layer_share_type=None, rm_detach=None, decoder_sa_type='ca', module_seq=['sa', 'ca', 'ffn'], embed_init_tgt=False, use_detached_boxes_dec_out=False,
            activate_det_seg=False,
            activate_point_tracking=False,
            query_feature_initialize_mode=["first_frame_ms_feature_bilinear", "first_frame_ms_pe_bilinear", "initial_visib"],
            query_feature_initializer="Linear",
            transformer_pe_temperature=10000,
            ignore_wh_inca=False,
            use_checkpoint=False,
            use_keyaware_ca=False,
            update_pos_in_ca=False,
            ca_update_aware=False,
        ):
        super().__init__(d_model, nhead, num_queries, num_encoder_layers, num_unicoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec, query_dim, num_patterns, modulate_hw_attn, deformable_encoder, deformable_decoder, num_feature_levels, enc_n_points, dec_n_points, box_attn_type, learnable_tgt_init, decoder_query_perturber, add_channel_attention, add_pos_value, random_refpoints_xy, two_stage_type, two_stage_pat_embed, two_stage_add_query_num, two_stage_learn_wh, two_stage_keep_all_tokens, dec_layer_number, rm_enc_query_scale, rm_dec_query_scale, rm_self_attn_layers, key_aware_type, layer_share_type, rm_detach, decoder_sa_type, module_seq, embed_init_tgt, use_detached_boxes_dec_out, ignore_wh_inca, use_checkpoint, use_keyaware_ca, update_pos_in_ca, ca_update_aware=ca_update_aware)
        self.activate_det_seg = activate_det_seg
        self.activate_point_tracking = activate_point_tracking
        self.query_feature_initialize_mode = query_feature_initialize_mode
        self.encoder_classifier = None
        self.encoder_regressor  = None
        self.encoder_attn_projector = None
        self.transformer_pe_temperature = transformer_pe_temperature
        self.decoder.transformer_pe_temperature = transformer_pe_temperature
        for l_id in range(len(self.decoder.layers)):
                self.decoder.layers[l_id].transformer_pe_temperature = transformer_pe_temperature
        if not activate_det_seg:
            self.tgt_embed = None
            self.refpoint_embed = None
        if self.activate_point_tracking:
            dim_point_initial_feature = 0
            if "first_emerge_frame_ms_feature_bilinear" in self.query_feature_initialize_mode:
                dim_point_initial_feature += self.num_feature_levels*d_model
            
            if query_feature_initializer == "Linear":
                self.query_feature_initializer = nn.Linear(dim_point_initial_feature, d_model)
            elif query_feature_initializer == "MLP":
                self.query_feature_initializer = nn.Sequential(*[
                        nn.Linear(dim_point_initial_feature, d_model * 2),
                        nn.ReLU(),
                        nn.Linear(d_model * 2, d_model * 2),
                        nn.ReLU(),
                        nn.Linear(d_model * 2, d_model)
                    ]
                )
            
            if "attn_first_emerge_global_visual" in self.query_feature_initialize_mode:
                dim_point_initial_feature += d_model
                self.query_feature_initializer_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.0, batch_first=True)
                self.query_feature_initializer_ffn = nn.Sequential(*[
                        nn.Linear(d_model * 2, d_model * 2),
                        nn.ReLU(),
                        nn.Linear(d_model * 2, d_model * 2),
                        nn.ReLU(),
                        nn.Linear(d_model * 2, d_model)
                    ]
                )
        self.use_checkpoint = use_checkpoint
    
    def forward_encoder(self, 
            srcs, masks, pos_embeds
        ):
        """
        Args:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            # Det/Seg DN
            - refpoint_embed: [bs, num_dn, 4]. The DN part query's PE for Det/Seg, None in infer
            - tgt: [bs, num_dn, d_model]. The DN part query's label encoding, None in infer
            - attn_mask: [bs ??]. The attention mask for self-attention when DN is activated.
            # point tracking
            - pt_refpoint: [bs len_temporal num_point 2]. The initial position of points.
            - pt_tgt: [bs len_temporal num_point d_model]. The initial encoding of the points' visibility.
        Return:
            - hs: (n_dec, bs, num_point, d_model)
            - references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
            - hs_enc: (n_enc+1, bs, num_point, d_model) or (1, bs, num_point, d_model) or None
            - ref_enc: sigmoid coordinates. \
                    (n_enc+1, bs, num_point, query_dim) or (1, bs, num_point, query_dim) or None
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        if self.encoder is None:
            return src_flatten, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed, # bs, nq, 4
                )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # -------------------------------------------------------
        # Begin preparing tgt for detection or segmentation task, or preparing tgt for point tracking task.
        #########################################################
        return memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios

    def forward_pt_tracking_decoder(self, 
            memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, 
            mask_flatten, valid_ratios, 
            multi_task_tgt, multi_task_pos_embed, multi_task_corr_tgt,
            len_temporal, temp_tgt_mask, tgt_mask, update_mask,
            history_tgt_list,
        ):
        #########################################################
        # Begin Decoder
        #########################################################
        outputs = self.decoder(
                tgt=multi_task_tgt.transpose(0, 1), 
                tgt_corr=multi_task_corr_tgt.transpose(0, 1),
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                pos=lvl_pos_embed_flatten.transpose(0, 1),
                refpoints_unsigmoid=multi_task_pos_embed.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios, temp_tgt_mask=temp_tgt_mask, tgt_mask=tgt_mask,
                len_temporal=len_temporal, freeze_first_frame=True, update_mask=update_mask,
                history_tgt_list=history_tgt_list)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################     
        if self.activate_det_seg and self.two_stage_type == 'standard':
            raise not NotImplementedError
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################     
        if self.decoder.layers[0].update_pos_in_ca:
            hs, references, references_ca = outputs
            return hs, references, hs_enc, ref_enc, references_ca
        else:
            hs, references = outputs   
            return hs, references, hs_enc, ref_enc, None
    
    def prepare_point_query(self, flattened_multi_scale_features, flattened_multi_scale_pos_enc, flattened_multi_scale_mask, level_start_index, multi_scale_spatial_shapes, pt_refpoint, pt_tgt, first_emerge_frame=None):
        """prepare the initial feature of each points according to the "query_feature_initialize_mode".
            These features are supported: first_frame_feature_bilinear, first_frame_pe_bilinear, initial_pe, initial_visib.
            Please refer to the docstring of the class for more detail.

        Args:
            flattened_multi_scale_features (torch.tensor): [bs*len_temp, \sum(hw), C], the flattened multi_scale feature maps
            flattened_multi_scale_pos_enc (torch.tensor) : [bs*len_temp, \sum(hw), C], the flattened multi_scale feature maps' positional encoding.
            level_start_index (torch.tensor): [num_scales], the start index of multi_scale feature maps in the flattend version.
            multi_scale_spatial_shapes (torch.tensor): [num_scales, 2], the resolution of multi_scale feature maps.
            pt_refpoint (torch.tensor): [bs, len_temp, num_point, len_temp, num_points, 2], the points' initial positions.
            pt_tgt (torch.tensor): [bs, len_temp, num_point, len_temp, num_points], the points' initial visibility.
        Return:
            point_feature (torch.tensor): [bs*len_temp, num_point, d_model]
            point_embed   (torch.tensor): [bs*len_temp, num_point, 4]
        """
        def get_first_frame_multi_scale_features(features, bs, len_temp):
            """get the multi_scale feature maps of the first frame.

            Returns:
                features_first_frame_multi_scales (List[torch.Tensor]): 
            """
            features_first_frame = features.reshape(bs, len_temp, *features.shape[1:])[:, 0]
            # features_first_frame_multi_scales = []
            # for start_index, spatial_shape in zip(level_start_index, multi_scale_spatial_shapes):
            #     features_first_frame_multi_scales.append(features_first_frame[:, start_index: start_index + spatial_shape[0] * spatial_shape[1]].reshape(bs, *spatial_shape, -1))
            return features_first_frame
        
        def get_first_emerge_frame_multi_scale_features(features, bs, len_temp, first_emerge_frame):
            """get the multi_scale feature maps of the first frame.

            Returns:
                features_first_emerge_frame (List[torch.Tensor]): bs num_point sum(wh) C
            """
            features_ = features.reshape(bs, len_temp, *features.shape[1:])
            features_first_emerge_frame = torch.gather(features_, 1, first_emerge_frame[:, :, None, None].repeat(1, 1, *features_.shape[2:]))
            return features_first_emerge_frame
        
        bs, len_temp = pt_refpoint.shape[:2]
        pt_features = []
        for init_feature_mode in self.query_feature_initialize_mode:
            if init_feature_mode == "first_emerge_frame_ms_feature_bilinear":
                for spatial_shape, start_index in zip(multi_scale_spatial_shapes, level_start_index):
                    feature_maps = flattened_multi_scale_features[:, start_index: start_index+spatial_shape.prod()].view(bs, len_temp, *spatial_shape, -1).permute(0,1,4,2,3)  # bs len_temp h w c
                    sampling_loc_3d = torch.cat([first_emerge_frame[..., None], pt_refpoint[:,0] * (spatial_shape-1)], dim=-1)[:,None]
                    pt_local_feature_lvl = sample_features5d(feature_maps, sampling_loc_3d)  # bs 1 num_point C
                    pt_features.append(pt_local_feature_lvl.squeeze(1))
            else:
                raise NotImplementedError
        # summarize the point features.
        point_feature = self.query_feature_initializer(torch.cat(pt_features, dim=2))
        point_embed   = torch.cat([
            inverse_sigmoid(pt_refpoint[:, 0]),
            inverse_sigmoid(torch.ones_like(pt_refpoint[:, 0]) * 0.5)
        ], dim=2)
        point_traj_feature = point_feature[:, None].repeat(1, len_temp, 1, 1)
        point_traj_embed = point_embed[:, None].repeat(1, len_temp, 1, 1)
        point_traj_visib = F.one_hot(pt_tgt, num_classes=3).float()  # bs, len_temporal, num_points 3
        return point_traj_feature, point_traj_embed, point_traj_visib


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    coords -= 1

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features5d(input, coords):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def build_point_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber=RandomBoxPerturber(
                x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise, 
                w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = getattr(args, "use_detached_boxes_dec_out", False)
    rm_detach = getattr(args, "rm_detach", None)
    rm_self_attn_layers = getattr(args, "rm_self_attn_layers", None)
    transformer_pe_temperature = getattr(args, "transformer_pe_temperature", 10000)

    ignore_wh_inca = getattr(args, "ignore_wh_inca", False)
    use_checkpoint = getattr(args, "use_checkpoint", False)
    use_keyaware_ca = getattr(args, "use_keyaware_ca", False)
    update_pos_in_ca = getattr(args, "update_pos_in_ca", False)
    ca_update_aware = getattr(args, "ca_update_aware", False)

    return PointDeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=rm_self_attn_layers,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=rm_detach,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,

        activate_det_seg=args.activate_det_seg,
        activate_point_tracking=args.activate_point_tracking,
        query_feature_initialize_mode=args.query_feature_initialize_mode,
        query_feature_initializer=args.query_feature_initializer,
        transformer_pe_temperature=transformer_pe_temperature,
        ignore_wh_inca=ignore_wh_inca,
        use_checkpoint=use_checkpoint,
        use_keyaware_ca=use_keyaware_ca,
        update_pos_in_ca=update_pos_in_ca,
        ca_update_aware=ca_update_aware,
    )
