CONFIG="TAPTR.py"
EXP_NAME="TAPTR"
CHECKPOINT="logs/$EXP_NAME/taptr.pth"

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