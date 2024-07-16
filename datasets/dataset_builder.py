import torch.utils.data
import torchvision


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        from .coco import build as build_coco
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'kubric':
        from .kubric import build_kubric
        return build_kubric(image_set, args)
    if args.dataset_file in ['tapvid_davis_first', 'tapvid_davis_strided', 'tapvid_kinetics_first', "tapvid_rgb_stacking_first"]:
        from .tapvid import build_tapvid
        return build_tapvid(image_set, args)
    
    raise ValueError(f'dataset {args.dataset_file} not supported')
