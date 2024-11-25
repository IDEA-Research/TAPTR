# for gradient accumulating
grad_acc_steps = 4
# for basic training setting.
num_classes=2
lr = 0.0001 * 2
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 1
weight_decay = 0.0001
epochs = 150
lr_drop = 100
save_checkpoint_interval = 5
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = True
lr_drop_list = [100, 130]
freq_evaluate = 10000  # never evaluate
# for model
modelname = 'taptr'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 10
pe_temperatureW = 10
transformer_pe_temperature = 10
return_interm_indices = [0, 1, 2, 3]
backbone_freeze_keywords = None
freeze_before_layer2=False
enc_layers = 2
dec_layers = 5
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'no'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
avg_window_loss = True
ignore_middle_cls_loss = True

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['ca', 'tsa', 'sa', 'ffn']
nms_iou_threshold = -1
use_keyaware_ca = True
update_pos_in_ca = True
supervise_ca_update = True
ca_update_aware = True

dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = False
ignore_wh_inca = True
use_detached_boxes_dec_out = False

# for dn (has been abandoned)
use_dn = False
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 91

match_unstable_error = True

# for ema
use_ema = True
ema_decay = 0.9997
ema_epoch = 0


# for Point Tracking
mini_box_size = 0.01  # 0.01 * 512 = 5.12 pixels --> 5 pixels to 
num_samples_per_video = -1
num_queries_per_video = 800
num_queries_per_video_eval = 800 
sample_continuous_clip = True
sample_visible_at_first_middle_frame = True
activate_det_seg = False
activate_point_tracking = True
query_initialize_mode = "repeat_first_frame"
query_feature_initialize_mode = ["first_emerge_frame_ms_feature_bilinear"]
query_feature_initializer = "Linear"
ignore_outsight = True
# for unrolled temporal 
sliding_window_size = 8
sliding_window_stride = 4
add_temporal_embedding = False
# for losses
pt_losses = ['pt_location', 'pt_visibs']
ignore_full_seq_loss = False
ignore_window_loss = True
compensate_loss = True

# For data augmentation
# 0. Photometric part.
add_random_photometric_aug = True
blur_aug_prob = 0.25
color_aug_prob = 0.25
eraser_aug_prob = 0.5
eraser_bounds = [2, 100]
eraser_max = 10
erase = True
replace = True
# 1. Panning part. (Only Training)
add_random_temporal_camera_panning = True
pad_bounds = [0, 25]
resize_lim = [0.75, 1.25]
resize_delta = 0.05
crop_size = (512, 512)
max_crop_offset = 15
# 2. Flip part. (Only Training)
add_random_temporal_flip = True
h_flip_prob = 0.5
v_flip_prob = 0.5
# 3. Scale part.
add_704_temporal_consistent_resize = False
add_random_temporal_consistent_resize = True
data_aug_scales = [512]  # the resolution of original image is 512x512
data_aug_max_size = 512
data_aug_scale_overlap = None
compensate_HW_ratio = True

random_restart_trajectory = True
ratio_restart_trajectory = 1/4