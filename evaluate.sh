CONFIG="TAPTRv2.py"
EXP_NAME="TAPTRv2"
CHECKPOINT="logs/$EXP_NAME/taptrv2.pth"

DATASET="tapvid_davis_first"
# DATASET="tapvid_davis_strided"
# DATASET="tapvid_rgb_stacking_first"
# DATASET="tapvid_kinetics_first"
DATAROOT="datas"
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
    --data_path $DATAROOT \
    --dataset_file $DATASET \
    --eval_checkpoint $CHECKPOINT \
	--output_dir logs/$EXP_NAME \
	-c config/$CONFIG \
    --num_workers 0