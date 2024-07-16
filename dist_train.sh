export TORCH_DISTRIBUTED_DEBUG=DETAIL  # for debug the "unused_parameter"
exp_name="TAPTR"
cfg_name="TAPTR"

data_path="datas/kubric_movif"
python -m torch.distributed.launch --nproc_per_node=8 main.py \
	--dataset_file kubric \
	--data_path $data_path \
	--output_dir logs/$exp_name \
	-c config/$cfg_name.py