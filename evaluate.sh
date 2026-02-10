DATAROOT="datas"
EXP_NAME="TAPTRv3"

#! Available configs (choose one):
#  - TAPTRv3_resnet18_384x512, with ResNet-18 backbone and 384x512 input resolution (default)
#  - TAPTRv3_resnet50_512x512, with ResNet-50 backbone and 512x512 input resolution
CONFIG="TAPTRv3_resnet18_384x512"

CHECKPOINT="logs/$EXP_NAME/$CONFIG.pth"

#! Available datasets (choose one):
#  - tapvid_davis_first        (DAVIS)
#  - tapvid_kinetics_first     (Kinetics)
#  - tapvid_rgb_stacking_first (RGB-Stacking)
#  - tapvid_robotap_first      (RoboTAP)
#
# Set DATASET by editing the line below or by exporting DATASET in the
# environment before running this script, e.g.:
#   DATASET=tapvid_kinetics_first ./evaluate_tapvid.sh
DEFAULT_DATASET="tapvid_davis_first"

# Validate dataset choice to avoid silent typos
DATASET="${DATASET:-$DEFAULT_DATASET}"
case "$DATASET" in
    tapvid_davis_first|tapvid_kinetics_first|tapvid_rgb_stacking_first|tapvid_robotap_first)
        ;; # valid
    *)
        echo "Error: unknown DATASET '$DATASET'" >&2
        echo "Valid options: tapvid_davis_first, tapvid_kinetics_first, tapvid_rgb_stacking_first, tapvid_robotap_first" >&2
        exit 2
        ;;
esac

# Set support point mode according to dataset. RoboTAP uses random support points.
case "$DATASET" in
    tapvid_robotap_first)
        support_points_mode="random"
        ;;
    *)
        support_points_mode="local_and_grid"
        ;;
esac

# Streaming mode evaluation (default)
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
    --data_path $DATAROOT \
    --dataset_file $DATASET \
    --eval_checkpoint $CHECKPOINT \
	--output_dir logs/$EXP_NAME \
	-c config/$CONFIG.py \
    --num_workers 0 \
    --support_points_mode $support_points_mode \
    --streaming