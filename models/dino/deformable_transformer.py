# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn, MSDeformAttn_KeyAware
import math


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, 
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no', # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca', 
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 ignore_wh_inca=False,
                 use_checkpoint=False,
                 use_keyaware_ca=False,
                 update_pos_in_ca=False,
                 ca_update_aware=False,

                 # for cca
                 use_cca_in_apu=None,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, add_channel_attention=add_channel_attention, box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, 
            encoder_norm, d_model=d_model, 
            num_queries=num_queries,
            deformable_encoder=deformable_encoder, 
            enc_layer_share=enc_layer_share, 
            two_stage_type=two_stage_type,
            use_checkpoint=use_checkpoint,
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, box_attn_type=box_attn_type,
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq,
                                                          ignore_wh_inca=ignore_wh_inca,
                                                          use_checkpoint=use_checkpoint,
                                                          use_keyaware_ca=use_keyaware_ca,
                                                          update_pos_in_ca=update_pos_in_ca,
                                                          use_cca_in_apu=use_cca_in_apu,
                                                        )

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model, query_dim=query_dim, 
                                        modulate_hw_attn=modulate_hw_attn,
                                        num_feature_levels=num_feature_levels,
                                        deformable_decoder=deformable_decoder,
                                        decoder_query_perturber=decoder_query_perturber, 
                                        dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                        dec_layer_share=dec_layer_share,
                                        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
                                        use_checkpoint=use_checkpoint,
                                        ca_update_aware=ca_update_aware,
                                        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None
        
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None
            
        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type =='standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)      
            
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:

                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries) # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
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
        #########################################################

        if self.two_stage_type =='standard':
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            
            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0) 
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)
                tgt=torch.cat([tgt,tgt_],dim=1)
            else:
                refpoint_embed,tgt=refpoint_embed_,tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)
                tgt=torch.cat([tgt,tgt_],dim=1)
            else:
                refpoint_embed,tgt=refpoint_embed_,tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1) # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 
        ######################################################### 

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
                tgt=tgt.transpose(0, 1), 
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                pos=lvl_pos_embed_flatten.transpose(0, 1),
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,tgt_mask=attn_mask)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################        

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):

    def __init__(self, 
        encoder_layer, num_layers, norm=None, d_model=256, 
        num_queries=300,
        deformable_encoder=False, 
        enc_layer_share=False, enc_layer_dropout_prob=None,                  
        two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        use_checkpoint=False,
    ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1) ])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1) ]) 
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            print("\n WARNING, Using checkpoint in encoder to save memory."*5)
            from fairscale.nn.checkpoint import checkpoint_wrapper
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, 
            src: Tensor, 
            pos: Tensor, 
            spatial_shapes: Tensor, 
            level_start_index: Tensor, 
            valid_ratios: Tensor, 
            key_padding_mask: Tensor,
            ref_token_index: Optional[Tensor]=None,
            ref_token_coord: Optional[Tensor]=None 
            ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        # preparation and reshape
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True
            
            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=key_padding_mask)  
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1), key_padding_mask=key_padding_mask).transpose(0, 1)        

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))
                
                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1] # bs, nq
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output) # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, 
                    return_intermediate=False, 
                    d_model=256, query_dim=4, 
                    modulate_hw_attn=False,
                    num_feature_levels=1,
                    deformable_decoder=False,
                    decoder_query_perturber=None,
                    dec_layer_number=None, # number of queries each layer in decoder
                    rm_dec_query_scale=False,
                    dec_layer_share=False,
                    dec_layer_dropout_prob=None,
                    use_detached_boxes_dec_out=False,
                    use_checkpoint=False,
                    ca_update_aware=False,
                    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        
        self.ref_point_head = MLP(query_dim // 2 * 256, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None
        self.transformer_pe_temperature = 10000
        # update query feature between layers.
        self.query_updater = None
        self.norm = None
        self.ca_update_aware = ca_update_aware
        if self.ca_update_aware:
            self.ca_update_fuser = nn.Sequential(*[
                nn.Linear(d_model + 256, d_model), nn.ReLU(), 
                nn.Linear(d_model, d_model), nn.ReLU(), 
                nn.Linear(d_model, d_model), 
            ])

        if use_checkpoint:
            from fairscale.nn.checkpoint import checkpoint_wrapper
            print("\n WARNING, Checkpoint in decoder does not work."*5)
            # for layer in self.layers:
            #     # ffn part
            #     layer.linear1 = checkpoint_wrapper(layer.linear1)
            #     layer.linear2 = checkpoint_wrapper(layer.linear2)
            #     # self attn
            #     # layer.self_attn = checkpoint_wrapper(layer.self_attn)
            #     # temporal self attn
            #     # layer.temp_self_attn = checkpoint_wrapper(layer.temp_self_attn)

    def forward(self, tgt, tgt_corr, memory,
                tgt_mask: Optional[Tensor] = None,
                temp_tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                len_temporal = None,
                freeze_first_frame=False,
                update_mask=None,
                tgt_long_temporal_feature = None,
                long_temp_tgt_mask = None,
                ptq_initial_context_features = None,
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
            - len_temporal: int, the length of temporal. It is required when doing temporal-self-attention.
        """
        output = tgt
        output_init = output.clone()

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  
        if self.layers[0].update_pos_in_ca:
            assert valid_ratios.min() == 1  # if valid ratios in different scales not the same, then the update of reference points in cross attention should be considered more in detail.
            ref_points_ca = []
        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.transformer_pe_temperature) # nq, bs, 256*2 
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points, self.transformer_pe_temperature) # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                input_to_layer = None
                if layer_id == 0:
                    input_to_layer = output_init
                else:
                    delta = self.query_updater(torch.cat([output, output_init], dim=-1))
                    input_to_layer = output + delta
                output = layer(
                    tgt = input_to_layer,
                    tgt_query_pos = query_pos,
                    tgt_query_sine_embed = query_sine_embed,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    tgt_reference_points = reference_points_input,

                    memory = memory,
                    memory_key_padding_mask = memory_key_padding_mask,
                    memory_level_start_index = level_start_index,
                    memory_spatial_shapes = spatial_shapes,
                    memory_pos = pos,

                    self_attn_mask = tgt_mask,
                    cross_attn_mask = memory_mask,
                    temp_self_attn_mask = temp_tgt_mask,
                    len_temporal = len_temporal,
                    tgt_long_temporal_feature = tgt_long_temporal_feature,
                    ltemp_self_attn_mask = long_temp_tgt_mask,
                    layer_id=layer_id,
                    update_mask=update_mask,

                    ptq_initial_context_features=ptq_initial_context_features,
                )
            if layer.update_pos_in_ca:
                output, reference_points_input, reference_points_caup = output
                reference_points = reference_points_input[..., 0, :]
                ref_points_ca.append(inverse_sigmoid(reference_points).sigmoid())
            # iter update
            if self.ca_update_aware:
                output_caup_aware = gen_sineembed_for_position(reference_points_caup[..., 0, :], self.transformer_pe_temperature)
                output_caup_aware = self.ca_update_fuser(torch.cat([output, output_caup_aware], dim=-1))
                output_for_regress = output_caup_aware
            else:
                output_for_regress = output
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output_for_regress)
                # if freeze_first_frame:
                #     delta_unsig = self.ignore_first_frame_update(delta_unsig, len_temporal)
                if not update_mask is None:
                    delta_unsig = delta_unsig * update_mask.permute(1,0,2).flatten(1,2).unsqueeze(-1).to(delta_unsig.dtype)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # # select # ref points
                # if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                #     nq_now = new_reference_points.shape[0]
                #     select_number = self.dec_layer_number[layer_id + 1]
                #     if nq_now != select_number:
                #         class_unselected = self.class_embed[layer_id](output) # nq, bs, 91
                #         topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1] # new_nq, bs
                #         new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)
            intermediate.append(output_for_regress)
        if self.layers[0].update_pos_in_ca:
            return [
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points_ca]
            ]
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]
    
    def ignore_first_frame_update(self, update, len_temporal):
        scale_update = torch.ones_like(update)
        scale_update = scale_update.view(update.shape[0], scale_update.shape[1]//len_temporal, len_temporal, scale_update.shape[-1])
        scale_update[:, :, 0] *= 0.0
        scale_update = scale_update.flatten(1,2)
        update = update * scale_update
        return update


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class WeightedMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, positional_embedding=None):
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = weighted_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = weighted_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights, positional_embedding=positional_embedding)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn.functional import _mha_shape_check, _in_projection_packed, _in_projection, pad, softmax, linear, dropout
def weighted_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    positional_embedding=None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            weighted_multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask.is_floating_point():
        float_attn_mask = True
    else:
        float_attn_mask = False
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            # warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    #! add positional embedding
    if not (positional_embedding is None):
        q = positional_embedding.rotate_queries_or_keys(q.flip(dims=[1]), seq_dim=1).flip(dims=[1])
        k = positional_embedding.rotate_queries_or_keys(k.flip(dims=[1]), seq_dim=1).flip(dims=[1])
    #!
    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None:
        if not float_attn_mask:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
        else:
            # v1: - max value, push all values into negative ones. / the visibility score, the larger the visibility, the larger the value (since they are negative).
            # attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            # attn_output_weights = attn_output_weights - attn_output_weights.max(dim=-1, keepdim=True)[0]
            # mask_inf = attn_mask == -torch.inf
            # attn_mask_float = torch.masked_fill(attn_mask, mask_inf, 1.0)   
            # attn_mask_inf = torch.masked_fill(attn_mask, ~mask_inf, 0.0) 
            # attn_output_weights = attn_output_weights / (attn_mask_float + 1e-7)  # since attn_output_weights < 0.0, if we want a larger value, we should divide it.
            # attn_output_weights = attn_output_weights + attn_mask_inf
            # attn_output_weights = softmax(attn_output_weights, dim=-1)

            # v2: reweight the attention score after we have obtained it through softmax. and renormalize it linearly.
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            mask_inf = attn_mask == -torch.inf
            attn_mask_float = torch.masked_fill(attn_mask, mask_inf, 0.0)   
            attn_output_weights = attn_output_weights * attn_mask_float
            attn_output_weights = attn_output_weights / (attn_output_weights.sum(dim=-1, keepdim=True) + 1e-7)
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ignore_wh_inca = False,
                 use_checkpoint=False,
                 use_keyaware_ca=False,
                 update_pos_in_ca=False,
                 use_cca_in_apu=None,
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert (sorted(module_seq) == ['ca', 'ffn', 'sa', 'tsa']) or (sorted(module_seq) == ['ca', 'ffn', 'sa']) or (sorted(module_seq) == ['ffn', 'sa', 'tsa']) \
        or (sorted(module_seq) == ['ca', 'ffn', 'lta', 'sa']) or (sorted(module_seq) == ['ca', 'ffn', 'lta', 'sa', 'tsa'])
        # cross attention
        if "ca" in module_seq:
            if use_keyaware_ca:
                self.cross_attn = MSDeformAttn_KeyAware(d_model, n_levels, n_heads, n_points, ignore_wh_inca=ignore_wh_inca, deformable_use_checkpoint=use_checkpoint, use_cca_in_apu=use_cca_in_apu)
            else:
                self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, ignore_wh_inca=ignore_wh_inca)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
        
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if "tsa" in module_seq:
            # self.temp_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.temp_self_attn = WeightedMultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout_temp = nn.Dropout(dropout)
            self.norm_temp = nn.LayerNorm(d_model)
        else:
            self.temp_self_attn = None
        if 'lta' in module_seq:
            self.ltemp_self_attn = WeightedMultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout_ltemp = nn.Dropout(dropout)
            self.norm_ltemp = nn.LayerNorm(d_model)
            from .rotary_position_embedding import RotaryEmbedding
            self.lta_temporal_embedding = RotaryEmbedding(d_model)
        else:
            self.ltemp_self_attn = None
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # add corr
        self.update_pos_in_ca = update_pos_in_ca
        if self.update_pos_in_ca:
            self.update_pos_embed_head = MLP(512, d_model, d_model, 2)
            self.transformer_pe_temperature = 10

            self.clamp_update = True
            self.caupdate_weight_normalizer = nn.Sequential(*[
                nn.Linear(n_heads*n_levels*n_points, d_model), nn.ReLU(), 
                nn.Linear(d_model, d_model), nn.ReLU(), 
            ])
            self.caupdate_weight_regressor = nn.Sequential(*[
                nn.Linear(d_model, n_heads*n_levels*n_points)
            ])
    
    def zero_init_regressor(self):
        from torch.nn.init import xavier_uniform_, constant_
        constant_(self.caupdate_regressor[-1].weight.data, 0.)
        constant_(self.caupdate_regressor[-1].bias.data, 0.)
    
    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention

                # use context features in APU
                ptq_initial_context_features: Optional[Tensor] = None,
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        if self.update_pos_in_ca:
            tgt2, attention_weights_before_sfmx, sampling_offsets = self.cross_attn(
                               self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask, get_attn_weights=True, get_sampling_offsets=True,
                               input_pe_flatten=memory_pos.transpose(0, 1), ptq_initial_context_features=ptq_initial_context_features)
            tgt2 = tgt2.transpose(0, 1)
            sampling_offsets = (sampling_offsets / memory_spatial_shapes[None, None, None, :, None, :]).flatten(2, -2).permute(1,0,3,2)  # [bs, n_query, n_head, n_level, n_points, 2] --> [n_query, bs, 2, n_head*n_level*n_points]
            attention_weights_before_sfmx = self.caupdate_weight_normalizer(attention_weights_before_sfmx)
            attention_weights_before_sfmx = self.caupdate_weight_regressor(attention_weights_before_sfmx)
            pos_delta = (sampling_offsets @ attention_weights_before_sfmx.softmax(dim=-1)[..., None]).squeeze(-1)  # [n_query, bs, 2]
        else:
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                tgt_reference_points.transpose(0, 1).contiguous(),
                                memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.update_pos_in_ca:
            tgt_reference_points_updated = tgt_reference_points.clone()
            tgt_reference_points_updated[..., :2] = tgt_reference_points_updated[..., :2] + pos_delta[..., None, :]
            if self.clamp_update:
                tgt_reference_points_updated[..., :2] = inverse_sigmoid(tgt_reference_points_updated[..., :2].clone()).sigmoid()
            return tgt, tgt_reference_points_updated

        return tgt
    
    def forward_tsa(self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

        # for memory
        memory: Optional[Tensor] = None, # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None, # num_levels
        memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None, # pos for memory

        # sa
        self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
        len_temporal = None
    ):
        # self attention
        if self.temp_self_attn is not None:
            if self.decoder_sa_type == 'sa':
                batch_size, num_queries = tgt.shape[1]//len_temporal, tgt.shape[0]
                tgt_tmp = tgt.view(num_queries, batch_size, len_temporal, -1).permute(2, 1, 0, 3).flatten(1,2)  # num_point, bs*len_temporal, C --> len_temporal, bs*num_point, C
                tgt_query_pos_tmp = tgt_query_pos.view(num_queries, batch_size, len_temporal, -1).permute(2, 1, 0, 3).flatten(1,2)
                q = k = self.with_pos_embed(tgt_tmp, tgt_query_pos_tmp)
                tgt_tmp2 = self.temp_self_attn(q, k, tgt_tmp, attn_mask=self_attn_mask)[0]
                tgt_tmp = tgt_tmp + self.dropout_temp(tgt_tmp2)
                tgt = tgt_tmp.view(len_temporal, batch_size, num_queries, -1).permute(2, 1, 0, 3).flatten(1, 2)  # len_temporal, bs*num_point, C --> num_point, bs*len_temporal, C
                tgt = self.norm_temp(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt
    
    def forward_lta(self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model

        # sa
        self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
        len_temporal = None,
        tgt_long_temporal_feature = None,  # num_point, bs*len_long_temporal, C
    ):
        # update the features in current window into the long-temporal feature.
        batch_size, num_queries = tgt.shape[1]//len_temporal, tgt.shape[0]
        len_long_temporal = tgt_long_temporal_feature.shape[1] // batch_size
        tgt_long_temporal_feature = tgt_long_temporal_feature.view(num_queries, batch_size, len_long_temporal, -1)  # num_point, bs, len_long_temporal, C

        # self attention
        if self.ltemp_self_attn is not None:
            if self.decoder_sa_type == 'sa':
                # batch_size, num_queries = tgt.shape[1]//len_temporal, tgt.shape[0]
                tgt_tmp = tgt.view(num_queries, batch_size, len_temporal, -1).permute(2, 1, 0, 3).flatten(1,2)  # num_point, bs*len_temporal, C --> len_temporal, bs*num_point, C
                tgt_long_temporal_feature_tmp = tgt_long_temporal_feature.view(num_queries, batch_size, len_long_temporal, -1).permute(2, 1, 0, 3).flatten(1,2)  # num_point, bs*len_long_temporal, C --> len_long_temporal, bs*num_point, C
                # taking the 1'th window as an example:
                # query: 4, 5, 6, 7
                # k/v  : 0, 1, 2, 3, 4, 5, 6, 7
                # if we get rotary embedding directly, 4 in query and 0 in k/v will get the most close distance in the embedding space.
                # so we need to reverse the temporal dimension
                # query: 7, 6, 5, 4
                # k/v  : 7, 6, 5, 4, 3, 2, 1, 0
                q = self.lta_temporal_embedding.rotate_queries_or_keys(tgt_tmp.flip(dims=[0]), seq_dim=-3).flip(dims=[0])
                k = self.lta_temporal_embedding.rotate_queries_or_keys(tgt_long_temporal_feature_tmp.flip(dims=[0]), seq_dim=-3).flip(dims=[0])
                positional_embedding_inside_lta = None
                tgt_tmp2, _ = self.ltemp_self_attn(
                    q, k, tgt_long_temporal_feature_tmp, 
                    attn_mask=self_attn_mask.detach(), 
                    positional_embedding = positional_embedding_inside_lta
                )
                tgt_tmp = tgt_tmp + self.dropout_ltemp(tgt_tmp2)
                tgt = tgt_tmp.view(len_temporal, batch_size, num_queries, -1).permute(2, 1, 0, 3).flatten(1, 2)  # len_temporal, bs*num_point, C --> num_point, bs*len_temporal, C
                tgt = self.norm_ltemp(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt
    
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention

                # tsa
                len_temporal = None,
                temp_self_attn_mask = None,
                temp_cross_attn_mask = None,

                # history attention
                tgt_long_temporal_feature = None,
                ltemp_self_attn_mask = None,
                layer_id=None,
                update_mask=None,

                # use context features in APU
                ptq_initial_context_features: Optional[Tensor] = None,
            ):
        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                if self.update_pos_in_ca:
                    tgt, tgt_reference_points_ = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                        tgt_key_padding_mask, tgt_reference_points, \
                            memory, memory_key_padding_mask, memory_level_start_index, \
                                memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, ptq_initial_context_features)
                    # update positional encoding.
                    tgt_delta_pos = update_mask.permute(1,2,0).unsqueeze(-1).to(tgt_reference_points_.dtype) * (tgt_reference_points_ - tgt_reference_points)[..., :2]
                    tgt_reference_points = tgt_reference_points_ * update_mask.permute(1,2,0).unsqueeze(-1).to(tgt_reference_points_.dtype) + \
                        tgt_reference_points * (1 - update_mask.permute(1,2,0).unsqueeze(-1).to(tgt_reference_points_.dtype))
                    tgt_query_sine_embed = gen_sineembed_for_position(tgt_reference_points[:, :, 0, :], self.transformer_pe_temperature) # nq, bs, 256*2 
                    tgt_query_pos = self.update_pos_embed_head(tgt_query_sine_embed) # nq, bs, 256

                else:
                    tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                        tgt_key_padding_mask, tgt_reference_points, \
                            memory, memory_key_padding_mask, memory_level_start_index, \
                                memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'tsa':
                tgt = self.forward_tsa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, temp_self_attn_mask, temp_cross_attn_mask, len_temporal)
            elif funcname == "lta":
                if tgt_long_temporal_feature is not None:
                    tgt = self.forward_lta(tgt, ltemp_self_attn_mask, len_temporal, tgt_long_temporal_feature)
                else:
                    pass
            else:
                raise ValueError('unknown funcname {}'.format(funcname))
        if self.update_pos_in_ca:
            return tgt, tgt_reference_points, tgt_delta_pos
        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber=RandomBoxPerturber(
                x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise, 
                w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out =False

    return DeformableTransformer(
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
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out
    )


