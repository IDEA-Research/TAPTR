export TORCH_DISTRIBUTED_DEBUG=DETAIL  # for debug the "unused_parameter"

#! Available configs (choose one)
#  - TAPTRv3_resnet18_384x512, with ResNet-18 backbone and 384x512 input resolution (default)
#  - TAPTRv3_resnet50_512x512, with ResNet-50 backbone and 512x512 input resolution
cfg_name="TAPTRv3_resnet18_384x512"

python -m torch.distributed.launch --nproc_per_node=8 main.py \
	--output_dir logs/$cfg_name \
	-c config/$cfg_name.py
