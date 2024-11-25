export TORCH_DISTRIBUTED_DEBUG=DETAIL  # for debug the "unused_parameter"
exp_name="TAPTRv2"
cfg_name="TAPTRv2"

python -m torch.distributed.launch --nproc_per_node=8 main.py \
	--output_dir logs/$exp_name \
	-c config/$cfg_name.py
