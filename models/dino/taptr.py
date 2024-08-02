import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from copy import deepcopy

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, nested_temporal_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
# from .deformable_transformer import build_deformable_transformer
from .point_deformable_transformer import build_point_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
import time
from models.dino.dino import DINO, SetCriterion, PostProcess


class TAPTR(DINO):
    def __init__(self, 
            backbone, transformer, num_classes, num_queries, aux_loss=False, iter_update=False, query_dim=2, random_refpoints_xy=False, fix_refpoints_hw=-1, num_feature_levels=1, nheads=8, two_stage_type='no', two_stage_add_query_num=0, dec_pred_class_embed_share=True, dec_pred_bbox_embed_share=True, two_stage_class_embed_share=True, two_stage_bbox_embed_share=True, decoder_sa_type='sa', num_patterns=0, dn_number=100, dn_box_noise_scale=0.4, dn_label_noise_ratio=0.5, dn_labelbook_size=100,
            activate_det_seg=False,
            activate_point_tracking=False,
            query_initialize_mode="repeat_first_frame",
            sliding_window_size=-1,
            sliding_window_stride=-1,
            add_temporal_embedding=False,
            pointquery_position_initialize_mode="padding_newest",
            point_size_initialize_value=0.5,
            multi_layer_full_seq=False,
            add_corr_block=False,
            corr_once=False,
            update_query_mode = "normal",
            update_query_between_window_mode = "normal",
            dropout_update_query_between_window = 0.0,
            mode_class_feat = "default",
        ):
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss, iter_update, query_dim, random_refpoints_xy, fix_refpoints_hw, num_feature_levels, nheads, two_stage_type, two_stage_add_query_num, dec_pred_class_embed_share, dec_pred_bbox_embed_share, two_stage_class_embed_share, two_stage_bbox_embed_share, decoder_sa_type, num_patterns, dn_number, dn_box_noise_scale, dn_label_noise_ratio, dn_labelbook_size)
        self.activate_det_seg = activate_det_seg
        self.activate_point_tracking = activate_point_tracking
        self.query_initialize_mode = query_initialize_mode
        self.sliding_window_size = sliding_window_size  # -1 means no sliding window
        self.sliding_window_stride = sliding_window_stride
        self.add_temporal_embedding = add_temporal_embedding
        if self.add_temporal_embedding:
            self.temporal_embedding = nn.Embedding(self.sliding_window_size, self.transformer.d_model)
        self.max_length = 24 # max=48 in dgx
        self.label_enc = None
        self.pointquery_position_initialize_mode = pointquery_position_initialize_mode  # [padding_newest]
        self.point_size_initialize_value = point_size_initialize_value
        self.transformer.point_size_initialize_value = point_size_initialize_value
        self.multi_layer_full_seq = multi_layer_full_seq
        # cost-volume (correlation) - related.
        self.add_corr_block = add_corr_block
        self.corr_once = corr_once
        if self.add_corr_block:
            from .deformable_transformer import CorrBlock
            self.transformer.decoder.add_corr_block = add_corr_block
            self.transformer.decoder.corr_once = corr_once
            if corr_once:
                self.transformer.decoder.corr_query_proj = nn.Sequential(*[
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                    nn.ReLU(),
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                ])
            single2multi = True if num_feature_levels == 1 else False
            self.transformer.decoder.corr_block = CorrBlock(num_feature_levels, 3, self.sliding_window_size, self.transformer.d_model, padding_mode="border", single2multi=single2multi)
            for l_id in range(len(self.transformer.decoder.layers)):
                self.transformer.decoder.layers[l_id].add_corr_block = add_corr_block
                self.transformer.decoder.layers[l_id].corr_once = corr_once
                self.transformer.decoder.layers[l_id].corr_fuser = nn.Sequential(*[
                    nn.Linear(self.transformer.d_model + (3*2+1)**2*4, self.transformer.d_model),
                    nn.ReLU(),
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                ])
                if self.corr_once:
                    continue
                self.transformer.decoder.layers[l_id].corr_query_proj = nn.Sequential(*[
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                    nn.ReLU(),
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                ])
        # query updating.
        self.update_query_mode = update_query_mode
        self.transformer.decoder.update_query_mode = update_query_mode
        if self.update_query_mode == "mlp_delta":
            self.transformer.decoder.query_updater = nn.Sequential(*[
                    nn.Linear(self.transformer.d_model * 2, self.transformer.d_model),
                    nn.ReLU(),
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                ]
            )
        self.update_query_between_window_mode = update_query_between_window_mode
        self.dropout_update_query_between_window = dropout_update_query_between_window
        if self.update_query_between_window_mode in ["mlp_delta_with_init"]:
            self.query_between_window_updater = nn.Sequential(*[
                    nn.Linear(self.transformer.d_model * 3, self.transformer.d_model),
                    nn.ReLU(),
                    nn.Linear(self.transformer.d_model, self.transformer.d_model),
                ]
            )
        if self.dropout_update_query_between_window > 0:
            self.dropout_window_update = nn.Dropout(self.dropout_update_query_between_window)
        # class feature.
        self.mode_class_feat = mode_class_feat
        self.memory_efficient_mode = False

    def prepare_video_features(self, video_data):
        """Extract the feature of the video using convolutional backbone.

        Args:
            video_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Conv Part
        features, poss = self.backbone(video_data)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = video_data.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        # Transformer encoder part
        memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios = self.transformer.forward_encoder(
            srcs, masks, poss
        )
        return memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios

    def prepare_point_initial_states(self, targets, batch_size, len_temporal):
        """get the initial state of the points across the whole temporal sequence.

        Args:
            targets (dict): the gt state of the video.
            batch_size (int): _description_
            len_temporal (int): _description_

        Returns:
            pt_tracking_masks: [bs, num_points, len_temporal], the mask of the points, whether it should be tracked or not across the whole temporal sequence.
            pt_start_tracking_frames: [bs, num_points], the frame id of the points when it starts to be tracked.
            pt_initial_feature: [bs, num_points, len_temporal, hidden_dim], the initial feature of the points across the whole temporal sequence. Initialized with zeros.
            pt_initial_location: [bs, num_points, len_temporal, 4], the initial normalized location (0~1) of the points across the whole temporal sequence. Initialized with the location where it is tracked.
            pt_initial_visibility: [bs, num_points, len_temporal], the initial visibility of the points across the whole temporal sequence. Initialized with the visibility where it is tracked.
        """
        pt_tracking_masks = torch.stack([target["pt_tracking_mask"] for target in targets], dim=0)  # batchsize, num_points, len_temporal
        pt_start_tracking_frames = pt_tracking_masks.float().argmax(dim=2)  # batchsize, num_points
        # pt_updating_masks = torch.ones(pt_tracking_masks.shape, device=pt_start_tracking_frames.device, dtype=torch.float32).scatter_(2, pt_start_tracking_frames[..., None], torch.zeros(pt_tracking_masks.shape, device=pt_start_tracking_frames.device, dtype=torch.float32))
        pt_updating_masks = pt_tracking_masks.clone()
        pt_updating_masks = pt_updating_masks.scatter_(2, pt_start_tracking_frames[..., None], torch.zeros(pt_tracking_masks.shape, device=pt_start_tracking_frames.device, dtype=torch.bool))
        pt_initial_location = torch.stack([targets[i]["pt_boxes"] for i in range(batch_size)], dim=0)  # bs, num_points, len_temporal, 4
        pt_initial_visibility = torch.stack([targets[i]["pt_labels"] for i in range(batch_size)], dim=0)
        pt_initial_location = torch.gather(pt_initial_location, 2, pt_start_tracking_frames[..., None, None].repeat(1,1,1,4)).repeat(1, 1, len_temporal, 1).transpose(2,1)  # bs, len_temporal, num_points, 4
        pt_initial_visibility = torch.gather(pt_initial_visibility, 2, pt_start_tracking_frames[..., None]).repeat(1, 1, len_temporal).transpose(2,1)  # bs, len_temporal, num_points
        if "pt_group_mask" in targets[0].keys():
            pt_group_mask = torch.stack([target["pt_group_mask"] for target in targets], dim=0)  # batchsize, num_points, num_points
        else:
            pt_group_mask = None
        return pt_tracking_masks, pt_start_tracking_frames, pt_updating_masks, pt_initial_location, pt_initial_visibility, pt_group_mask
    
    def get_window_range(self, window_id, len_temporal):
        start_frame = window_id * self.sliding_window_stride
        stop_frame = start_frame + self.sliding_window_size - 1
        if stop_frame >= len_temporal:
            stop_frame = len_temporal - 1
            start_frame = stop_frame - self.sliding_window_size + 1
        return start_frame, stop_frame
    
    def prepare_window_forward(self, 
                              batchsize, len_temporal,
                              window_head, window_tail,
                              pt_start_tracking_frames, pt_tracking_masks, pt_updating_masks,
                              memory, lvl_pos_embed_flatten, mask_flatten, valid_ratios,
                              pt_current_traj_features, pt_current_traj_embeds,
                              pt_initial_traj_features=None,
                              pt_group_mask=None,
                        ):
        """prepare the required data for the forward pass for each window.

        Args:
            batchsize (int): 
            len_temporal (int): the length of the whole temporal.
            window_head (_type_): the head pointer for the window.
            window_tail (int): the tail pointer for the window.
            pt_start_tracking_frames (torch.IntTensor): [bs, num_point], from which frame the points start to be tracked.
            pt_tracking_masks (torch.BoolTensor): [bs, num_point, len_temporal], the mask of the points, whether it should be tracked or not across the whole temporal sequence.
            memory (torch.FloatTensor): [bs*len_temporal, sum(wh), C], the multi-scale features of the video.
            lvl_pos_embed_flatten (torch.FloatTensor): [bs*len_temporal, sum(wh), C], the multi-scale positional encoding of the video features.
            mask_flatten (torch.BoolTensor): [bs*len_temporal, sum(wh)], whether the feature is padded or not.
            valid_ratios (torch.FloatTensor): [bs*len_temporal, num_scale, 2], ...
            pt_current_traj_features (torch.FloatTensor): [bs, len_temporal, num_points, C], the feature for each point across the whole temporal sequence.
            pt_current_traj_embeds (torch.FloatTensor): [bs, len_temporal, num_points, 4], the location for each point across the whole temporal sequence.

        Returns:
            window_memory (torch.FloatTensor): [bs*window_size, sum(wh), C], the multi-scale features of the video in the window.
            window_lvl_pos_embed_flatten (torch.FloatTensor): [bs*window_size, sum(wh), C], the multi-scale positional encoding of the video features in the window.
            window_mask_flatten (torch.BoolTensor): [bs*window_size, sum(wh)], whether the feature is padded or not in the window.
            window_valid_ratios (torch.FloatTensor): [bs*window_size, num_scale, 2], ...
            window_temp_attn_mask (torch.BoolTensor): [bs*window_size, num_point, num_head, window_size, window_size], the temporal attention mask in the window.
            window_update_mask (torch.BoolTensor): [bs, num_point, window_size], whether the locations of each points should be updated in the window.
        """
        # prepare for initializing the points' features
        # window_update_mask = torch.logical_and(pt_start_tracking_frames >= window_head, pt_start_tracking_frames <= window_tail)
        window_track_mask = pt_tracking_masks[:,:,window_head:window_tail+1]
        window_update_mask = pt_updating_masks[:,:,window_head:window_tail+1]
        if window_update_mask.sum() == 0:  # none of the points should be updated in the window.
            return [None, ] * 11
        # construct temporal attention mask in the window.
        window_memory = memory.view(batchsize, len_temporal, *memory.shape[1:])[:, window_head:window_tail+1].flatten(0,1)
        window_lvl_pos_embed_flatten = lvl_pos_embed_flatten.view(batchsize, len_temporal, *lvl_pos_embed_flatten.shape[1:])[:, window_head:window_tail+1].flatten(0,1)
        window_mask_flatten = mask_flatten.view(batchsize, len_temporal, *mask_flatten.shape[1:])[:, window_head:window_tail+1].flatten(0,1)
        window_valid_ratios = valid_ratios.view(batchsize, len_temporal, *valid_ratios.shape[1:])[:, window_head:window_tail+1].flatten(0,1)

        # n_head_temp_self_attn = self.transformer.decoder.layers[0].temp_self_attn.num_heads if not (self.transformer.decoder.layers[0].temp_self_attn is None) else 8
        if not (self.transformer.decoder.layers[0].temp_self_attn is None):
            n_head_temp_self_attn = self.transformer.decoder.layers[0].temp_self_attn.num_heads
        else:
            n_head_temp_self_attn = 8
        window_temp_attn_mask = (window_track_mask[..., None].int() + window_track_mask[..., None, :].int()) == 1  # bs num_points, window_size, window_size
        window_temp_attn_mask = window_temp_attn_mask[:, :, None].repeat(1, 1, n_head_temp_self_attn, 1, 1).flatten(0,2)  # bs, num_point, num_head, window_size, window_size --> bs*num_point*num_head, window_size, window_size

        n_head_self_attn = self.transformer.decoder.layers[0].self_attn.num_heads if not (self.transformer.decoder.layers[0].self_attn is None) else 8
        window_self_attn_mask = (window_track_mask.transpose(1,2)[..., None].int() + window_track_mask.transpose(1,2)[..., None, :].int()) == 1  # bs window_size, num_points, num_points
        window_self_attn_mask = window_self_attn_mask[:, :, None].repeat(1, 1, n_head_self_attn, 1, 1).flatten(0,2)  # bs, window_size, num_head, num_points, num_points --> bs*window_size*num_head, num_points, num_points

        pt_window_traj_feature = pt_current_traj_features[:, window_head:window_tail+1].flatten(0,1)  # bs*window_size, num_points, C
        pt_window_traj_embeds = pt_current_traj_embeds[:, window_head:window_tail+1].flatten(0,1)  # bs*window_size, num_points, 4
        if self.add_temporal_embedding:
            pt_window_traj_feature = pt_window_traj_feature + self.temporal_embedding(torch.arange(self.sliding_window_size, device=pt_window_traj_feature.device))[None, :, None].repeat(batchsize, 1, 1, 1).flatten(0,1)
        
        pt_window_traj_corr_feature = pt_window_traj_feature
        if not pt_group_mask is None:
            window_self_attn_mask = torch.logical_or(window_self_attn_mask, pt_group_mask)  # "True" indicates that there should be no interaction.

        return window_memory, window_lvl_pos_embed_flatten, window_mask_flatten, window_valid_ratios, \
            window_temp_attn_mask, window_self_attn_mask, window_track_mask, window_update_mask, \
            pt_window_traj_feature, pt_window_traj_embeds, pt_window_traj_corr_feature

    def update_query_states(self, window_head, window_tail, window_update_mask, pt_current_traj_embeds, pt_current_traj_visibility, pt_current_traj_features, pt_initial_traj_features, outputs_coord_list, outputs_class, outputs_hs, padding_to_future=True):
        """Update the trajectory states of the points.

        Args:
            window_head (int): the pointer of the head of the window.
            window_tail (int): the pointer of the tail of the window.
            window_update_mask (torch.tensor): [bs, num_point, window_size], whether the locations of each points should be updated in the window.
            pt_current_traj_embeds (torch.FloatTensor): [bs, len_temporal, num_points, 4], the location for each point across the whole temporal sequence.
            pt_current_traj_visibility (torch.FloatTensor): [bs, len_temporal, num_points, 3], the visibility for each point across the whole temporal sequence.
            outputs_coords (torch.FloatTensor): [num_dec, bs*window_size, num_points, 4]
            outputs_class (torch.FloatTensor): [num_dec, bs*window_size, num_points, 3]
        """
        window_update_mask_ = window_update_mask.to(outputs_coord_list[-1].dtype).transpose(1,2).unsqueeze(-1)
        # 1. updating of trajectory, including location and visibility
        if not self.multi_layer_full_seq:
            pt_current_traj_embeds[:, window_head:window_tail+1]     = pt_current_traj_embeds.clone()[:, window_head:window_tail+1]     * (1 - window_update_mask_) + outputs_coord_list[-1].view(-1, self.sliding_window_size, *outputs_coord_list.shape[2:]) * window_update_mask_
            pt_current_traj_visibility[:, window_head:window_tail+1] = pt_current_traj_visibility.clone()[:, window_head:window_tail+1] * (1 - window_update_mask_) + outputs_class[-1].view(-1, self.sliding_window_size, *outputs_class.shape[2:])           * window_update_mask_
            if padding_to_future and self.pointquery_position_initialize_mode == "padding_newest":
                pt_current_traj_embeds[:, window_tail+1:] = pt_current_traj_embeds[:, window_tail].unsqueeze(1)
                pt_current_traj_visibility[:, window_tail+1:] = pt_current_traj_visibility[:, window_tail].unsqueeze(1)
        else:
            for l_id in range(len(self.transformer.decoder.layers) + len(self.transformer.decoder.global_layers)):
                pt_current_traj_embeds[l_id][:, window_head:window_tail+1]     = pt_current_traj_embeds[l_id].clone()[:, window_head:window_tail+1]     * (1 - window_update_mask_) + outputs_coord_list[l_id].view(-1, self.sliding_window_size, *outputs_coord_list.shape[2:]) * window_update_mask_
                pt_current_traj_visibility[l_id][:, window_head:window_tail+1] = pt_current_traj_visibility[l_id].clone()[:, window_head:window_tail+1] * (1 - window_update_mask_) + outputs_class[l_id].view(-1, self.sliding_window_size, *outputs_class.shape[2:])           * window_update_mask_
                if padding_to_future and self.pointquery_position_initialize_mode == "padding_newest":
                    pt_current_traj_embeds[l_id][:, window_tail+1:] = pt_current_traj_embeds[l_id][:, window_tail].unsqueeze(1)
                    pt_current_traj_visibility[l_id][:, window_tail+1:] = pt_current_traj_visibility[l_id][:, window_tail].unsqueeze(1)
        
        # 2. updating of content feature.
        if self.update_query_between_window_mode == "normal":
            pt_current_traj_features = pt_current_traj_features
        elif self.update_query_between_window_mode == "mlp_delta_with_init":
            window_pt_features = outputs_hs[-1]  # bs*window_size, num_points, C
            pt_window_feature_delta = self.query_between_window_updater(
                                            torch.cat([
                                                pt_initial_traj_features[:, window_head:window_tail+1],
                                                pt_current_traj_features[:, window_head:window_tail+1],
                                                window_pt_features.view(-1, self.sliding_window_size, *window_pt_features.shape[1:]),
                                            ], dim=-1)
                                        ) * window_update_mask_
            if self.dropout_update_query_between_window > 0:
                # TODO: the dropout should consider the window_update_mask_.
                pt_window_feature_delta = self.dropout_window_update(pt_window_feature_delta)
            if not self.training and self.long_video_mode and self.dropout_update_query_between_window > 0:
                if not self.window_id % self.padding_gap == 0:
                    print(f"Do not padding the delta, between {self.padding_gap} windows, keep 1 window.")
                    padding_to_future = False
                else:
                    padding_to_future = True
            pt_current_traj_features[:, window_head:window_tail+1] = pt_current_traj_features.clone()[:, window_head:window_tail+1] + pt_window_feature_delta
            if padding_to_future and self.pointquery_position_initialize_mode == "padding_newest":
                pt_current_traj_features[:, window_tail+1:] = pt_current_traj_features[:, window_tail].unsqueeze(1)
        else:
            raise NotImplementedError
        return pt_current_traj_embeds, pt_current_traj_visibility, pt_current_traj_features

    def prepare_window_loss(self, outputs_class, outputs_coord_list, targets, window_head, window_tail, window_update_mask):
        """post process the outputs and targets for the calculation of loss.

        Args:
            outputs_class (torch.tensor): [num_dec, bs*window_size, num_points, 3] (3 classes), the classification logits (including no-object) for all points in the window.
            outputs_coord_list (torch.tensor): [num_dec, bs*window_size, num_points, 4], the normalized boxes coordinates for all points in the window.
            targets (dict): the learning targets.
            window_head (int): the pointer of the head of the window.
            window_tail (int): the pointer of the tail of the window.
            window_update_mask (torch.tensor): [bs num_points, window_size], whether the locations of each points should be updated in the window.
        """
        # outputs_class = outputs_class * window_update_mask.transpose(1,2).flatten(0,1)[None, ..., None]
        # outputs_coord_list = outputs_coord_list * window_update_mask.transpose(1,2).flatten(0,1)[None, ..., None]
        window_output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], "dn_meta": None}
        if self.aux_loss:
            window_output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
            if len(window_output['aux_outputs']) == 0:
                window_output.pop('aux_outputs')
        window_targets = []
        for bs_id, (target, update_mask) in enumerate(zip(targets, window_update_mask)):
            window_target = {}
            for tgt_name, tgt_value in target.items():
                if tgt_name in ['num_real_pt', 'query_frames']:
                    window_target[tgt_name] = tgt_value
                else:
                    window_target[tgt_name] = tgt_value[:, window_head:window_tail+1]
            window_target["ptq_update_mask"] = update_mask  # use update mask to instruct the calculation of loss.
            window_targets.append(window_target)
        return window_output, window_targets

    def forward(self, samples: NestedTensor, targets:List=None):
        #! 1. Prepare video feature
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_temporal_tensor_from_tensor_list(samples)
        samples, batch_size, len_temporal = self.flatten_temporal_dimension(samples)
        # for memory efficient inference:
        shortest_length_for_memory_efficient = 100
        num_of_windows_per_group = 2
        if self.memory_efficient_mode and (len_temporal > shortest_length_for_memory_efficient):
            self.num_frames_processed = 0
            self.num_frames_per_group = self.sliding_window_stride * num_of_windows_per_group
            # TODO: make sure the first group should meets the requirement of self.transformer.prepare_point_query.
        else:
            self.num_frames_per_group = None
        if not self.training:  # eval path
            # len_temporal = samples.tensors.shape[0]
            # Split the video into clips to lower down the usage of memory. Larger self.max_length provide faster speed.
            num_sub_samples = len_temporal // self.max_length
            if len_temporal % self.max_length != 0:
                num_sub_samples = num_sub_samples + 1
            memory = lvl_pos_embed_flatten = level_start_index = spatial_shapes = mask_flatten = valid_ratios = None
            if self.memory_efficient_mode:  # split the video into multiple groups and process the video in streaming mode to save memory.
                sub_samples = samples.tensors[self.num_frames_processed: self.num_frames_processed+self.num_frames_per_group]
                self.num_frames_per_group = min(self.num_frames_per_group, sub_samples.shape[0])
                sub_samples = nested_temporal_tensor_from_tensor_list([sub_samples])
                sub_samples, _, _ = self.flatten_temporal_dimension(sub_samples)
                memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios = self.prepare_video_features(sub_samples)
            else:
                for sub_sample_id in range(num_sub_samples):
                    sub_samples = samples.tensors[sub_sample_id*self.max_length:(sub_sample_id+1)*self.max_length]
                    sub_samples = nested_temporal_tensor_from_tensor_list([sub_samples])
                    sub_samples, _, _ = self.flatten_temporal_dimension(sub_samples)
                    sub_memory, sub_lvl_pos_embed_flatten, sub_level_start_index, sub_spatial_shapes, sub_mask_flatten, sub_valid_ratios = self.prepare_video_features(sub_samples)
                    if memory is None:
                        memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios = \
                            sub_memory, sub_lvl_pos_embed_flatten, sub_level_start_index, sub_spatial_shapes, sub_mask_flatten, sub_valid_ratios
                    else:
                        memory = torch.cat([memory, sub_memory], dim=0)
                        lvl_pos_embed_flatten = torch.cat([lvl_pos_embed_flatten, sub_lvl_pos_embed_flatten], dim=0)
                        mask_flatten = torch.cat([mask_flatten, sub_mask_flatten], dim=0)
                        valid_ratios = torch.cat([valid_ratios, sub_valid_ratios], dim=0)
            if len_temporal > 24:  # For the updating of content features between windows.
                self.long_video_mode = True
                self.padding_gap = len_temporal // 24
            else:
                self.long_video_mode = False
        else:  # training path
            memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios = self.prepare_video_features(samples)
        #! 2. Prepare the initial states of the point queries (ptq). Use the states when their belonging points first emerge or start to be tracked to initialize their states.
        ptq_tracking_masks, ptq_start_tracking_frames, ptq_updating_masks, \
        ptq_initial_traj_location, ptq_initial_traj_visibility, ptq_group_mask = self.prepare_point_initial_states(targets, batch_size, len_temporal)
        # Get the initial states of each point's whole trajectory. Sampling image features as point queries' initial content features and locations (inverse sigmoid). 
        ptq_current_traj_features, ptq_current_traj_embeds, ptq_current_traj_visibility = self.transformer.prepare_point_query(
            memory, lvl_pos_embed_flatten, mask_flatten, level_start_index, spatial_shapes, 
            ptq_initial_traj_location[..., :2], ptq_initial_traj_visibility, ptq_start_tracking_frames,
            len_group = self.num_frames_per_group,
        )
        #! 3. Other preparations, including the feature updating and loss.
        if (self.update_query_between_window_mode in ["mlp_delta_with_init"]):
            ptq_initial_traj_features = ptq_current_traj_features.clone()
        else:
            ptq_initial_traj_features = None
        if self.multi_layer_full_seq:
            ptq_current_traj_embeds = [ptq_current_traj_embeds.clone() for _ in range(len(self.transformer.decoder.layers) + len(self.transformer.decoder.global_layers))]
            ptq_current_traj_visibility = [ptq_current_traj_visibility.clone() for _ in range(len(self.transformer.decoder.layers) + len(self.transformer.decoder.global_layers))]
        #! 4.Sliding window with Decoder.
        num_windows = (len_temporal - self.sliding_window_size) // self.sliding_window_stride + 1
        if (self.sliding_window_size + (num_windows-1) * self.sliding_window_stride) < len_temporal:
            num_windows += 1
        window_output_list = []
        window_target_list = []

        window_ids = list(range(num_windows))
        for window_id in window_ids:
            # Window preparation.
            self.window_id = window_id
            window_head, window_tail = self.get_window_range(window_id, len_temporal)
            if self.memory_efficient_mode and window_tail > self.num_frames_processed + self.num_frames_per_group:
                self.num_frames_processed = self.num_frames_processed + (self.num_frames_per_group - self.sliding_window_stride)
                sub_samples = samples.tensors[self.num_frames_processed: self.num_frames_processed+self.num_frames_per_group]
                self.num_frames_per_group = min(self.num_frames_per_group, sub_samples.shape[0])
                sub_samples = nested_temporal_tensor_from_tensor_list([sub_samples])
                sub_samples, _, _ = self.flatten_temporal_dimension(sub_samples)
                memory, lvl_pos_embed_flatten, level_start_index, spatial_shapes, mask_flatten, valid_ratios = self.prepare_video_features(sub_samples)
            window_memory, window_lvl_pos_embed_flatten, window_mask_flatten, window_valid_ratio, \
            window_temp_attn_mask, window_self_attn_mask, window_track_mask, window_update_mask, \
            ptq_window_feature, ptq_window_embed, ptq_window_traj_corr_feature = self.prepare_window_forward(
                batch_size, len_temporal if not self.memory_efficient_mode else self.num_frames_per_group,
                window_head if not self.memory_efficient_mode else window_head - self.num_frames_processed, 
                window_tail if not self.memory_efficient_mode else window_tail - self.num_frames_processed, 
                ptq_start_tracking_frames, ptq_tracking_masks, ptq_updating_masks, 
                memory, lvl_pos_embed_flatten, mask_flatten, valid_ratios,
                ptq_current_traj_features, 
                ptq_current_traj_embeds if not self.multi_layer_full_seq else ptq_current_traj_embeds[-1],  # use the latest point states as the initial states of the window.
                ptq_initial_traj_features,
                ptq_group_mask,
            )
            if window_memory is None:
                continue  # no points should be updated in the window.
            # Forward each window.
            hs, reference, _, _ = self.transformer.forward_pt_tracking_decoder(
                window_memory, window_lvl_pos_embed_flatten, level_start_index, spatial_shapes, 
                window_mask_flatten, window_valid_ratio,
                ptq_window_feature, ptq_window_embed, ptq_window_traj_corr_feature,
                self.sliding_window_size, window_temp_attn_mask, window_self_attn_mask, window_update_mask,
            )
            # Get decoder output.
            outputs_coord_list = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_delta_unsig = layer_delta_unsig * window_update_mask.permute(0,2,1).flatten(0,1).unsqueeze(-1).to(layer_delta_unsig.dtype)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                outputs_coord_list.append(layer_outputs_unsig)
            outputs_coord_list = torch.stack(outputs_coord_list)  # num_dec, bs*window_size, num_queries, 4
            outputs_class = []
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs):
                if self.mode_class_feat == "default":
                    outputs_class.append(layer_cls_embed(layer_hs))
                elif self.mode_class_feat == "diff_init":
                    outputs_class.append(layer_cls_embed(layer_hs - ptq_window_feature))
            outputs_class = torch.stack(outputs_class)
            # Update the point-query states.
            ptq_current_traj_embeds, ptq_current_traj_visibility, ptq_current_traj_features = self.update_query_states(
                window_head, window_tail, 
                window_update_mask, ptq_current_traj_embeds, ptq_current_traj_visibility, ptq_current_traj_features, 
                ptq_initial_traj_features, 
                outputs_coord_list, outputs_class, hs, 
                padding_to_future=window_id<(num_windows-1)
            )
            outputs_coord_list = outputs_coord_list.sigmoid()  # transform query states into the readable format.
            # Prepare for the window loss.
            if self.training:
                window_output, window_target = self.prepare_window_loss(outputs_class, outputs_coord_list, targets, window_head, window_tail, window_update_mask)
                window_output_list.append(window_output)
                window_target_list.append(window_target)
        assert window_tail == len_temporal - 1, "the last window should end at the last frame."
        #! 5.Prepare the loss calculation.
        if self.multi_layer_full_seq:
            if self.training:
                ptq_current_traj_embeds = torch.stack(ptq_current_traj_embeds, dim=0).flatten(1,2)  # num_dec, bs, len_temporal, num_queries, 4 --> num_dec, bs*len_temporal, num_queries, 4
                ptq_current_traj_visibility = torch.stack(ptq_current_traj_visibility, dim=0).flatten(1,2)
                ptq_current_traj_embeds = ptq_current_traj_embeds.sigmoid()
                full_seq_output, full_seq_targets = self.prepare_window_loss(ptq_current_traj_visibility, ptq_current_traj_embeds, targets, 0, len_temporal-1, ptq_updating_masks)
            else:
                ptq_current_traj_embeds = ptq_current_traj_embeds[-1]  # num_dec, bs, len_temporal, num_queries, 4 --> bs, len_temporal, num_queries, 4
                ptq_current_traj_visibility = ptq_current_traj_visibility[-1]
                ptq_current_traj_embeds = ptq_current_traj_embeds.sigmoid()
                full_seq_output = {'pred_logits': ptq_current_traj_visibility, 'pred_boxes': ptq_current_traj_embeds, "dn_meta": None}
                full_seq_targets = []
                for target, ptq_update_mask in zip(targets, ptq_updating_masks):
                    target["ptq_update_mask"] = ptq_update_mask
                    full_seq_targets.append(target)
        else:
            ptq_current_traj_embeds = ptq_current_traj_embeds.sigmoid()
            if self.training:
                full_seq_output = {'pred_logits': ptq_current_traj_visibility.flatten(0,1), 'pred_boxes': ptq_current_traj_embeds.flatten(0,1), "dn_meta": None}
            else:
                full_seq_output = {'pred_logits': ptq_current_traj_visibility, 'pred_boxes': ptq_current_traj_embeds, "dn_meta": None}
            full_seq_targets = []
            for target, ptq_update_mask in zip(targets, ptq_updating_masks):
                target["ptq_update_mask"] = ptq_update_mask
                full_seq_targets.append(target)
        
        self.num_frames_processed = 0
        outputs = {
            "full_seq_output": full_seq_output,
            "window_output_list": window_output_list,
        }
        targets = {
            "full_seq_target": full_seq_targets,
            "window_target_list": window_target_list,
        }
        return outputs, targets

    def flatten_temporal_dimension(self, data):
        """fuse the temporal dimension (dim1) into the batch dimension (dim0).
        bs, len_temporal, ... --> bs*len_temporal, ...
        """
        if isinstance(data, torch.Tensor):
            batch_size, len_temporal = data.shape[:2]
            data = data.flatten(0,1)
            return data, batch_size, len_temporal
        elif isinstance(data, NestedTensor):
            batch_size, len_temporal = data.tensors.shape[:2]
            data.tensors = data.tensors.flatten(0, 1)
            data.mask = data.mask.flatten(0,1)
            return data, batch_size, len_temporal
    
    def unflatten_temporal_dimension(self, data, batch_size, len_temporal=-1):
        """seperate the temporal dimension (dim1) and the batch dimension (dim0).
        bs*len_temporal, ... --> bs, len_temporal, ...
        """
        if isinstance(data, torch.Tensor):
            data = data.view(batch_size, len_temporal, *data.shape[1:])
            return data
        elif isinstance(data, NestedTensor):
            data.tensors = data.tensors.view(batch_size, len_temporal, *data.tensors.shape[1:])
            data.mask = data.mask.view(batch_size, len_temporal, *data.mask.shape[1:])
            return data


class PointTrackingCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, avg_window_loss=True, ignore_outsight=False):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.avg_window_loss = avg_window_loss
        self.ignore_outsight = ignore_outsight
    
    def align_gt_point2box(self, targets):
        """Align the format of gts in point tracking into the format of detection for easy loss calculation.
        """
        targets_aligned = []
        for target in targets:
            gt_point = target.pop("pt_boxes").unbind(dim=1)
            gt_visib = target.pop("pt_labels").unbind(dim=1)
            update_mask = target.pop("ptq_update_mask").unbind(dim=1)
            for gt_p, gt_v, mask in zip(gt_point, gt_visib, update_mask):
                tgt = {}
                tgt["labels"] = gt_v
                tgt["boxes"] = gt_p
                tgt["mask"] = mask
                targets_aligned.append(tgt)
        return targets_aligned

    def forward(self, outputs, targets, return_indices=False):
        losses = {}
        # Full sequence loss
        outputs_full_seq = outputs["full_seq_output"]
        targets_full_seq = self.align_gt_point2box(targets["full_seq_target"])
        matching_indices = [(torch.arange(target["labels"].shape[0]).long(), torch.arange(target["labels"].shape[0]).long()) for bs_id, target in enumerate(targets_full_seq)]
        losses_full_seq_ = super().forward(outputs_full_seq, targets_full_seq, return_indices, matching_indices)
        losses_full_seq  = {}
        for loss_name, loss_value in losses_full_seq_.items():
            losses_full_seq['pt_full_' + loss_name] = loss_value
        losses.update(losses_full_seq)
        # Window loss
        num_windows = len(outputs["window_output_list"]) if self.avg_window_loss else 1
        outputs_window_list = outputs["window_output_list"]
        targets_window_list = [self.align_gt_point2box(window_tgt) for window_tgt in targets["window_target_list"]]
        losses_windows  = {}
        for output_window, target_window in zip(outputs_window_list, targets_window_list):
            matching_indices = [(torch.arange(target["labels"].shape[0]).long(), torch.arange(target["labels"].shape[0]).long()) for bs_id, target in enumerate(target_window)]
            losses_window_ = super().forward(output_window, target_window, return_indices, matching_indices)
            for loss_name, loss_value in losses_window_.items():
                if not loss_name in losses_windows.keys():
                    losses_windows['pt_window_' + loss_name] = loss_value / num_windows
                else:
                    losses_windows['pt_window_' + loss_name] += loss_value / num_windows
        losses.update(losses_windows)
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'pt_visibs': self.loss_labels,
            'pt_boxes': self.loss_boxes,
            'pt_location': self.loss_location,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        if "mask" in targets[0].keys():
            src_masks = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.bool, device=src_logits.device)
            src_masks[idx] = torch.cat([t["mask"][J] for t, (_, J) in zip(targets, indices)])
            src_logits = src_logits[src_masks]
            target_classes_onehot = target_classes_onehot[src_masks]
            num_boxes = src_masks.sum()
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_location(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the points (which is the center of boxes), the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if "mask" in targets[0].keys():
            target_masks = torch.cat([t['mask'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if self.ignore_outsight:
                target_masks = target_masks & torch.logical_and(
                    (target_boxes[..., 0] >= 0) & (target_boxes[..., 0] <= 1),
                    (target_boxes[..., 1] >= 0) & (target_boxes[..., 1] <= 1),
                )
            src_boxes = src_boxes[target_masks]
            target_boxes = target_boxes[target_masks]
        loss_boxes = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_location = loss_boxes * loss_boxes.new_tensor([1,1,0,0])[None, :]

        losses = {}
        num_boxes = target_masks.sum()
        losses['loss_bbox'] = loss_location.sum() / num_boxes

        with torch.no_grad():
            loss_size = loss_boxes * loss_boxes.new_tensor([0,0,1,1])[None, :]
            losses['loss_hw'] = loss_size.sum() / num_boxes
            losses['loss_xy'] = losses['loss_bbox'].detach()
        return losses


def get_weight_dict(args):
    # prepare weight dict
    def get_det_seg_weight_dict(args, layer_decay_ratio=1.0):
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        # for DN training
        if args.use_dn:
            weight_dict['loss_ce_dn'] = args.cls_loss_coef
            weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
            weight_dict['loss_giou_dn'] = args.giou_loss_coef

        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
        clean_weight_dict = copy.deepcopy(weight_dict)

        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                # aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
                for k, v in clean_weight_dict.items():
                    if "bbox" in k:
                        aux_weight_dict[k + f'_{i}'] = v*(layer_decay_ratio**(args.dec_layers - 1 - i))
                    else:
                        aux_weight_dict[k + f'_{i}'] = v
            weight_dict.update(aux_weight_dict)

        if args.two_stage_type != 'no':
            interm_weight_dict = {}
            try:
                no_interm_box_loss = args.no_interm_box_loss
            except:
                no_interm_box_loss = False
            _coeff_weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
                'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            }
            try:
                interm_loss_coef = args.interm_loss_coef
            except:
                interm_loss_coef = 1.0
            interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)
        return weight_dict
    def get_point_tracking_weight_dict(args):
        layer_decay_ratio = getattr(args, "layer_decay_ratio", 1.0)
        weight_dict_ = get_det_seg_weight_dict(args, layer_decay_ratio)
        weight_dict = {}
        ignore_full_seq_loss = getattr(args, "ignore_full_seq_loss", False)
        ignore_window_loss = getattr(args, "ignore_window_loss", False)
        ignore_middle_cls_loss = getattr(args, "ignore_middle_cls_loss", False)
        if ignore_middle_cls_loss:
            for key, value in weight_dict_.items():
                if "loss_ce_" in key:
                    weight_dict_[key] = 0.0
                else:
                    weight_dict_[key] = value
        for key, value in weight_dict_.items():
            if not ignore_full_seq_loss:
                weight_dict['pt_full_' + key] = value
            else:
                weight_dict['pt_full_' + key] = 0.0
        for key, value in weight_dict_.items():
            if not ignore_window_loss:
                weight_dict['pt_window_' + key] = value
            else:
                weight_dict['pt_window_' + key] = 0.0
        return weight_dict
    weight_dict_all = {}
    if args.activate_det_seg:
        weight_dict = get_det_seg_weight_dict(args)
        weight_dict_all.update(weight_dict)
    if args.activate_point_tracking:
        weight_dict = get_point_tracking_weight_dict(args)
        weight_dict_all.update(weight_dict)
    return weight_dict_all


def get_criterions(args, device, weight_dict):
    losses = []
    if args.activate_point_tracking:
        losses += args.pt_losses # ['pt_boxes', 'pt_visibs']  # 'pt_label', 
        pt_criterion = PointTrackingCriterion(num_classes=args.num_classes, weight_dict=weight_dict, 
                                                     matcher=None, focal_alpha=args.focal_alpha, losses=losses, avg_window_loss=args.avg_window_loss, ignore_outsight=args.ignore_outsight)
        pt_criterion.to(device)
    else:
        pt_criterion = None
    return None, pt_criterion


@MODULE_BUILD_FUNCS.registe_with_name(module_name='taptr')
def build_taptr(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_point_deformable_transformer(args)

    dec_pred_class_embed_share = getattr(args, "dec_pred_class_embed_share", True)
    dec_pred_bbox_embed_share = getattr(args, "dec_pred_bbox_embed_share", True)
    dn_labelbook_size = getattr(args, "dn_labelbook_size", num_classes)
    pointquery_position_initialize_mode = getattr(args, "pointquery_position_initialize_mode", "padding_newest")
    point_size_initialize_value = getattr(args, "point_size_initialize_value", 0.5)
    multi_layer_full_seq = getattr(args, "multi_layer_full_seq", False)
    add_corr_block = getattr(args, "add_corr_block", False)
    corr_once = getattr(args, "corr_once", False)
    update_query_mode = getattr(args, "update_query_mode", "normal")
    update_query_between_window_mode = getattr(args, "update_query_between_window_mode", "normal")
    dropout_update_query_between_window = getattr(args, "dropout_update_query_between_window", 0.0)
    mode_class_feat = getattr(args, "mode_class_feat", "default")

    model = TAPTR(
        backbone,
        transformer,
        num_classes=num_classes+1,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        activate_det_seg = args.activate_det_seg,
        activate_point_tracking = args.activate_point_tracking,

        sliding_window_size=args.sliding_window_size,
        sliding_window_stride=args.sliding_window_stride,
        add_temporal_embedding=args.add_temporal_embedding,
        pointquery_position_initialize_mode=pointquery_position_initialize_mode,
        point_size_initialize_value=point_size_initialize_value,
        multi_layer_full_seq=multi_layer_full_seq,
        add_corr_block=add_corr_block,
        corr_once=corr_once,
        update_query_mode=update_query_mode,
        update_query_between_window_mode=update_query_between_window_mode,
        dropout_update_query_between_window=dropout_update_query_between_window,
        mode_class_feat=mode_class_feat,
    )
    
    weight_dict = get_weight_dict(args)
    criterion, pt_criterion = get_criterions(args, device, weight_dict)
    criterions = {
        "det_seg_criterion": criterion,
        "pt_criterion": pt_criterion,
    }
    
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    
    return model, criterions, postprocessors