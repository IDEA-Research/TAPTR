# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch.utils.checkpoint as checkpoint

from ..functions import MSDeformAttnFunction  # , KeyAwareMSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ignore_wh_inca=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 1024

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        self.ignore_wh_inca = ignore_wh_inca

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            if not self.ignore_wh_inca:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :] + reference_points[:, :, None, :, None, 2:] * 0.0  # + wh*0.0 to avoid gradient of wh.
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # for amp
        if value.dtype == torch.float16:
            # for mixed precision
            output = MSDeformAttnFunction.apply(
            value.to(torch.float32), input_spatial_shapes, input_level_start_index, sampling_locations.to(torch.float32), attention_weights, self.im2col_step)
            output = output.to(torch.float16)
            output = self.output_proj(output)
            return output


        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class MSDeformAttn_KeyAware(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ignore_wh_inca=False, use_pytorch_version=False, value_proj_after=False, key_aware=True, add=True, proj_key=True, deformable_use_checkpoint=True, same_loc=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
        self.use_pytorch_version = use_pytorch_version
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.same_loc = same_loc

        if not same_loc:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        if not key_aware:
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.proj_key = proj_key
        if proj_key:
            self.key_proj = nn.Linear(d_model, d_model)
        else:
            self.key_proj = None
        # self.key_proj = None
        self.query_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.key_aware = key_aware
        self.add = add
        self.deformable_use_checkpoint = deformable_use_checkpoint

        print("use_pytorch_version key_aware, add same_loc", use_pytorch_version, key_aware, add, same_loc)
        self.value_proj_after = value_proj_after

        self._reset_parameters()
        self.ignore_wh_inca = ignore_wh_inca

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.same_loc:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
            for i in range(self.n_points):
                grid_init[:, i, :] *= i + 1
        else:
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                                  self.n_levels,
                                                                                                                  self.n_points,
                                                                                                                  1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if not self.key_aware:
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        if self.proj_key:
            xavier_uniform_(self.key_proj.weight.data)
            constant_(self.key_proj.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, get_attn_weights=False, get_sampling_offsets=False, input_pe_flatten=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        if not self.value_proj_after:
            value = self.value_proj(input_flatten)
        # if key is None:
        key = input_flatten
        if self.proj_key:
            key = self.key_proj(key)
        else:
            key = value
        # value = input_flatten
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            key = key.masked_fill(input_padding_mask[..., None], float(0))
        key = key.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if not self.same_loc:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_points, 2)
            sampling_offsets = sampling_offsets[:, :, :, None].repeat(1, 1, 1, self.n_levels, 1, 1)
        attention_weights = None
        if not self.key_aware:
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # sampling_locations = reference_points[:, :, None, :, None, :2] \
            #                      + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            if not self.ignore_wh_inca:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :] + reference_points[:, :, None, :, None, 2:] * 0.0  # + wh*0.0 to avoid gradient of wh.
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if self.key_aware:
            if not self.deformable_use_checkpoint or not (self.training):
                output, attention_weights_ = ms_deform_attn_core_pytorch_key_aware(
                    self.query_proj(query), value, key, input_padding_mask, 
                    input_spatial_shapes, sampling_locations, self.key_proj, self.value_proj, 
                    attention_weights, self.add
                )
                # output, attention_weights_ = KeyAwareMSDeformAttnFunction.apply(
                #     self.query_proj(query).float(), value.float(), key.float(), 
                #     input_spatial_shapes, input_level_start_index, sampling_locations.float(), 128
                # )
            else:
                output, attention_weights_ = checkpoint.checkpoint(ms_deform_attn_core_pytorch_key_aware, 
                    self.query_proj(query), value, key, input_padding_mask,
                    input_spatial_shapes, sampling_locations, self.key_proj, self.value_proj,
                    attention_weights, self.add
                )
        elif self.use_pytorch_version:
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights,
            )
        else:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        if self.value_proj_after:
            output = self.value_proj(output)
        output = self.output_proj(output)
        returns = [output]
        if get_attn_weights:
            returns.append(attention_weights_)
            # return output, attention_weights_
        if get_sampling_offsets:
            returns.append(sampling_offsets)
        if len(returns) == 1:
            return returns[0]
        return returns


def ms_deform_attn_core_pytorch_key_aware(query, value, key, input_padding_mask, value_spatial_shapes, sampling_locations, key_proj, value_proj, attention_weights_linear, add):
    # for debug and test only,
    # need to use cuda version instead
    # N: batch szie; S_: total value num;   M_: head num 8; mD: 256/M (32)
    # Lq_: len q;  L_: num levels (4); P_: sample point per-level (4)
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    sampling_key_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        # N_*M_, D_, Lq_, P_
        sampling_key_l__ = F.grid_sample(key_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_key_list.append(sampling_key_l__)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # import ipdb; ipdb.set_trace()
    key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
    value = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # N_*M_, D_, Lq_, P_ -> N_*M_, D_, Lq_, L_, P_ -> N_*M_, D_, Lq_, L_*P_

    # key = key.view(N_, M_, D_, Lq_, L_*P_).flatten(1, 2).permute(0, 2, 3, 1)
    # # N_, M_, D_, Lq_, L_*P_ -> N, Lq_, L_*P_, M_*D_
    #
    # key = key_proj(key)
    #
    # key = key.permute(0, 3, 1, 2).view(N_, M_, D_, Lq_, L_*P_).flatten(0, 1)
    # # N, Lq_, L_*P_, M_*D_ -> N_*M_, D_, Lq_, L_*P_
    #
    # key = key.permute(0, 2, 3, 1)  # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D

    key = key.permute(0, 2, 3, 1).flatten(0, 1)   # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D -> N*M*Lq, L*P, D

    N_, Lq, DD_ = query.shape
    # query = query_proj(query)
    query = query.view(N_, Lq, M_, DD_ // M_)
    query = query.permute(0, 2, 1, 3).flatten(0, 2)  # N, Lq, M, D -> N, M, Lq, D -> N*M*Lq, D

    query = query.unsqueeze(-2)  # N*M*Lq, D-> N*M*Lq, 1, D
    dk = query.size()[-1]
    # import ipdb; ipdb.set_trace()
    # attention_weights = torch.matmul(key, query[..., None]).squeeze(-1)/ math.sqrt(dk)
    # attention_weights_linear = attention_weights_linear.transpose(1, 2).reshape(N_ * M_*Lq_, 1, L_ * P_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_)
    attention_weights_ = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)  # bs*n_head*n_query, 1, n_lvl*n_point, 
    # N*M*Lq, 1, D, x N*M*Lq, L*P, D   ==  N*M, Lq, 1,  L*P
    #
    # if add:
    #     attention_weights = attention_weights + attention_weights_linear
    # else:
    #     attention_weights = attention_weights * attention_weights_linear
    attention_weights = F.softmax(attention_weights_, -1)

    # attention_weights = attention_weights.reshape(N_*M_, 1, Lq_, L_*P_)
    # attention_weights = attention_weights.transpose(1, 2)

    # value = value.view(N_, M_, D_, Lq_, L_ * P_).flatten(1, 2).permute(0, 2, 3, 1)
    # value = value_proj(value)
    # value = value.permute(0, 3, 1, 2).view(N_, M_, D_, Lq_, L_ * P_).flatten(0, 1)
    value = value.permute(0, 2, 3, 1).flatten(0, 1)  # N*M*Lq, L*P, D

    output = attention_weights.matmul(value)  # N*M, Lq, 1,  L*P x N*M*Lq, L*P, D -> N*M, Lq, 1,  D

    output = output.squeeze(-2).view(N_, M_, Lq_, D_).permute(0, 2, 1, 3)  # N*M, Lq, 1,  D -> N, Lq, M,  D

    output = output.flatten(2)

    # output = (value * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    attention_weights_ = attention_weights_.view(N_, M_, Lq_, -1).permute(2, 0, 1, 3).flatten(2)  # [bs, n_head, n_query, n_lvl*n_point] --> [n_query, bs, n_head, n_lvl*n_point] --> [n_query, bs, n_head*n_lvl*n_point]
    return output.contiguous(), attention_weights_