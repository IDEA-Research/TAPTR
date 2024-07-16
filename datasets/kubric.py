"""
Point tracking dataset.
"""

import os
import random
from pathlib import Path

import cv2
import json
import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Callable, Any, Dict

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from datasets.coco import get_aux_target_hacks_list as get_aux_target_hacks_list_coco
from datasets.coco import RandomSelectBoxes, RandomSelectBoxlabels, label2compat, label_compat2onehot
import datasets.transforms as T


def get_aux_target_hacks_list(image_set: str, args: Dict=None):
    """Preparing targets into a specific format.

    Args:
        image_set (str): the split of dataset, e.g. train, val, test.
        args (Dict): some more arguments.

    Returns:
        _type_: _description_
    """
    # from query to mini box, which is used to represent a point.
    if args.modelname in ['taptr']:
        aux_target_hacks_list = [
            T.Point2Minibox(mini_box_size=args.mini_box_size), 
            T.Visibility2Label(), 
            # FlattenTemporal(),
            # label2compat(), 
            # label_compat2onehot(), 
            # RandomSelectBoxlabels(args.num_classes),
            # UnflattenTemporal()
        ]
    elif args.modelname in ['q2bs_mask', 'q2bs', 'q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = get_aux_target_hacks_list_coco(image_set, args)
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


def make_temporal_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    compensate_HW_ratio = getattr(args, 'compensate_HW_ratio', False)
    normalize = T.Compose([
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], compensate=compensate_HW_ratio)
    ])
    if image_set in ['mini_train', 'train', 'train_reg']:
        # config the params for data aug
        tranforms = []
        add_random_temporal_consistent_resize = getattr(args, 'add_random_temporal_consistent_resize', False)
        add_704_temporal_consistent_resize = getattr(args, 'add_704_temporal_consistent_resize', True)
        add_random_temporal_camera_panning = getattr(args, 'add_random_temporal_camera_panning', False)
        add_random_temporal_flip = getattr(args, 'add_random_temporal_flip', False)
        add_random_photometric_aug = getattr(args, 'add_random_photometric_aug', False)
        random_restart_trajectory = getattr(args, 'random_restart_trajectory', False)
        if add_random_photometric_aug:
            blur_aug_prob = getattr(args, 'blur_aug_prob', 0.25)
            color_aug_prob = getattr(args, 'color_aug_prob', 0.25)
            eraser_aug_prob = getattr(args, 'eraser_aug_prob', 0.5)
            eraser_bounds = getattr(args, 'eraser_bounds', [2, 100])
            eraser_max = getattr(args, 'eraser_max', 10)
            replace_aug_prob = getattr(args, 'replace_aug_prob', 0.5)
            replace_bounds = getattr(args, 'replace_bounds', [2, 100])
            replace_max = getattr(args, 'replace_max', 10)
            erase = getattr(args, 'erase', True)
            replace = getattr(args, 'replace', True)
            contrastive_erase = getattr(args, 'contrastive_erase', False)
            tranforms.append(T.RandomPhotometricAug(blur_aug_prob, color_aug_prob, eraser_aug_prob, replace_aug_prob, replace_bounds, replace_max, eraser_bounds, eraser_max, erase, replace, contrastive_erase=contrastive_erase))
        if add_random_temporal_camera_panning:
            pad_bounds = getattr(args, 'pad_bounds', [0, 100])
            resize_lim = getattr(args, 'resize_lim', [0.25, 2.0])
            resize_delta = getattr(args, 'resize_delta', 0.2)
            crop_size = getattr(args, 'crop_size', (256, 256))
            max_crop_offset = getattr(args, 'max_crop_offset', 50)
            tranforms.append(T.RandomTemporalCameraPanning(pad_bounds, resize_lim, resize_delta, crop_size, max_crop_offset))
        if add_random_temporal_flip:
            h_flip_prob = getattr(args, 'h_flip_prob', 0.5)
            v_flip_prob = getattr(args, 'v_flip_prob', 0.5)
            tranforms.append(T.RandomTemporalFlipping(h_flip_prob, v_flip_prob))
        if add_random_temporal_consistent_resize:
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            max_size = 1333
        
            scales = getattr(args, 'data_aug_scales', scales)
            max_size = getattr(args, 'data_aug_max_size', max_size)
            tranforms.append(T.RandomTemporalConsistentResize(scales, max_size=max_size, compensate=compensate_HW_ratio))
        if add_704_temporal_consistent_resize:
            tranforms.append(T.RandomTemporalConsistentResize([704], max_size=800))
        if random_restart_trajectory:
            ratio_restart_trajectory = getattr(args, 'ratio_restart_trajectory', False)
            tranforms.append(T.RandomRestartTrajectories(ratio_restart_trajectory))
        tranforms.append(normalize)
        return T.Compose(tranforms)
    elif image_set in ['test', 'val']:
        tranforms = []
        add_random_temporal_consistent_resize = getattr(args, 'add_random_temporal_consistent_resize', False)
        add_704_temporal_consistent_resize = getattr(args, 'add_704_temporal_consistent_resize', True)
        if add_random_temporal_consistent_resize:
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            max_size = 1333
        
            scales = getattr(args, 'data_aug_scales', scales)
            max_size = getattr(args, 'data_aug_max_size', max_size)
            compensate_HW_ratio = getattr(args, 'compensate_HW_ratio', False)
            eval_scale = getattr(args, 'eval_scale', None)
            if eval_scale is None:
                tranforms.append(T.RandomTemporalConsistentResize([max(scales)], max_size=max_size, compensate=compensate_HW_ratio))
            else:
                tranforms.append(T.RandomTemporalConsistentResize([eval_scale], max_size=max_size, compensate=compensate_HW_ratio))
        if add_704_temporal_consistent_resize:
            tranforms.append(T.RandomTemporalConsistentResize([704], max_size=800))
        tranforms.append(normalize)
        return T.Compose(tranforms)
    raise ValueError(f'unknown {image_set}')


class PointTrackingDataset(torch.utils.data.Dataset):
    """Point Tracking Dataset.

    Args:
        data_root (str): the root of a specific image_set for this dataset. Defaults to datas/kubric_movif_cotracker/r256/train.  
        ann_file (str) : the path of annotation file. The annotation should be formatted as:
            video_id0: 
                points    : List[[float, float], ], [num_points, 2]; the coordinates of points in the image coordinate system.
                occluded  : List[int], [num_points]; 0: not occluded, 1: occluded.
                num_frames: int, 
            video_id1:
                ...
            Please refer to datasets/preprocess_kubric.py for more detail. Defaults to datas/kubric_movif_cotracker/r256/annotations/train.json.
        transforms (List(Callable), optional): . The transformations applied to the input images. Defaults to None.
        aux_target_hacks (List(Callable), optional): The transformations applied to the targets. Defaults to None.
        num_samples_per_video (int): number of frames sampled from one video, -1 means all frames. Defaults to -1.
        num_queries_per_video (int): number of points to be tracked in one video, -1 means all points. Defaults to -1.
        sample_continuous_clip (bool): whether or not the sampled frames is a continuous clip.
    """
    def __init__(self, 
                data_root: str="datas/kubric_movif_cotracker/r256/train", 
                ann_file: str="datas/kubric_movif_cotracker/r256/annotations/train.json", 
                transforms: List[Callable]=None, 
                aux_target_hacks: List[Callable]=None,
                load_prepared_annotation=False,
                num_samples_per_video = -1,
                num_queries_per_video = -1,
                num_groups_per_video = 1,
                sample_continuous_clip = False,
                sample_visible_at_first_middle_frame = False,
                input_width = 512,
                input_height = 512,
        ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.transforms = transforms
        self.aux_target_hacks = aux_target_hacks
        self.num_samples_per_video = num_samples_per_video
        self.sample_continuous_clip = sample_continuous_clip
        self.num_queries_per_video = num_queries_per_video
        self.sample_visible_at_first_middle_frame = sample_visible_at_first_middle_frame
        self.input_width = input_width
        self.input_height = input_height

        # load annotations
        print("----- Constructing Point Tracking Dataset... -----")
        self.video_ids = self.load_video_ids()
        print("1. loading annotations into memory...")
        self.load_prepared_annotation = load_prepared_annotation
        self.annotations = self.load_annotations()
        print("2. loading image paths into memory...")
        self.image_paths = self.load_image_paths()
        if data_root.name == "mini_train":
            self.video_ids = self.video_ids * 2000
        self.num_groups_per_video = num_groups_per_video
    
    def load_video_ids(self):
        def sort_video_ids(video_ids):
            """sort the video ids in ascending order. where the video ids are numbers in strings format.

            Args:
                video_ids (List[str]): the video ids

            Returns:
                sorted_video_ids (List[str]): the sorted video ids in string format.
            """
            video_ids_int = [int(video_id) for video_id in video_ids]
            sorted_video_ids = [video_id for _, video_id in sorted(zip(video_ids_int, video_ids))]
            return sorted_video_ids

        video_ids = []
        video_ids = os.listdir(self.data_root)
        video_ids = sort_video_ids(video_ids)
        return video_ids
    
    def load_annotations(self):
        """Load annotaions into memory or just load the paths of annotations.

        Returns:
            annotations: 
                - if load_prepared_annotation is True, annotations is a dict of annotations, where the keys are video ids, and the values are the annotations of the corresponding video.
                - if load_prepared_annotation is False, annotations is a dict of path of corresponding annotations, where the keys are video ids, and the values are the paths of the corresponding annotations.
        """
        if self.load_prepared_annotation:
            annotations = json.load(open(self.ann_file, 'r'))
        else:
            annotations = {}
            for video_id in tqdm(self.video_ids):
                path_anno = os.path.join(self.data_root, video_id, f'{video_id}.npy')
                annotations[video_id] = path_anno
        return annotations
    
    def load_image_paths(self):
        """Prepare the image paths for this dataset. 
        
        Return:
            image_paths (dict): 
                - keys: video ids
                - values: a list of image paths of the corresponding video.
        """
        image_paths = {}
        for video_id in tqdm(self.video_ids):
            ann = self.get_one_anno(video_id)
            if ann is None:
                print(f"{video_id} has no valid points to be tracked.")
                continue
            image_paths_per_video = []
            for frame_id in range(ann['num_frames']):
                image_paths_per_video.append(os.path.join(self.data_root, video_id, "frames", f'{frame_id:03d}.png'))
            image_paths[video_id] = image_paths_per_video
        return image_paths
    
    def downsample_temporal(self, annots):
        """downsample the annotations along the temporal dimension.

        Args:
            annots (dict): the annotations

        Returns:
            annots: the downsampled annotations. Adding 'sampled_frame_ids' to indicates which frames are sampled.
        """
        if self.num_samples_per_video > 0 and self.num_samples_per_video < annots["num_frames"]:
            if self.sample_continuous_clip:
                clip_start_idx = random.sample(range(annots["num_frames"] - self.num_samples_per_video), 1)[0]
                sampled_frame_ids = list(range(clip_start_idx, clip_start_idx + self.num_samples_per_video))
            else:
                sampled_frame_ids = sorted(random.sample(range(annots["num_frames"]), self.num_samples_per_video))
            annots["points"] = annots["points"][:, sampled_frame_ids]
            annots["occluded"] = annots["occluded"][:, sampled_frame_ids]
        else:
            sampled_frame_ids = list(range(annots["num_frames"]))
        annots["sampled_frame_ids"] = sampled_frame_ids
        return annots
    
    def downsample_points(self, annots):
        """Downsample the points to be tracked according to the visibility of points.

        Args:
            annots (dict): the annotations to be downsampled.
            self.sample_visible_at_first_middle_frame: only sample the points that are visible at the first frame or the middle frame.

        Returns:
            _type_: _description_
        """
        # annots["pt_padding_mask"] = np.ones(self.num_queries_per_video) < 0
        valid_mask = np.ones(annots["points"].shape[0]) > 0
        if self.sample_visible_at_first_middle_frame:
            middle_frame = annots["sampled_frame_ids"][len(annots["sampled_frame_ids"])//2]
            valid_mask = annots["occluded"][:, 0] == 0
            sampled_ids = np.where(valid_mask)[0]
            valid_mask = annots["occluded"][:, middle_frame] == 0
            sampled_ids = np.concatenate([sampled_ids, np.where(valid_mask)[0]])
        if len(sampled_ids) >= self.num_queries_per_video:
            sampled_ids = sorted(random.sample(sampled_ids.tolist(), self.num_queries_per_video))
            start_tracking_frame_ids = np.argmax(annots["occluded"][sampled_ids]==0, axis=1)
            all_occluded = np.all(annots["occluded"][sampled_ids], axis=1)  # to prevent all occluded ones.
            start_tracking_frame_ids[all_occluded] = annots["occluded"].shape[-1] - 1
            start_tracking_frame_ids, sampled_ids = zip(*sorted(zip(start_tracking_frame_ids, sampled_ids)))
            sampled_ids = np.array(sampled_ids)
            annots["tracking_mask"] = np.zeros([self.num_queries_per_video, len(annots["sampled_frame_ids"])], dtype=np.bool_)
            for point_id, start_tracking_frame_id in enumerate(start_tracking_frame_ids):
                annots["tracking_mask"][point_id, start_tracking_frame_id:] = True
            annots["occluded"] = annots["occluded"][sampled_ids]
            annots["sampled_point_ids"] = sampled_ids
            annots["points"] = annots["points"][sampled_ids]
        elif len(sampled_ids) == 0:
            return None
        else:
            return None
        return annots

    def get_one_anno(self, video_id):
        """load the annotations for each frame of one video according to the video_id, and also num_samples_per_video and num_queries_per_video. 

        Args:
            video_id (_type_): the video id of the annotations to be loaded.

        Returns:
            annots (dict): 
                points: np.array, [num_queries_per_video, num_samples_per_video, 2]; the coordinates of points in the image coordinate system across the video.
                occluded: np.array, [num_queries_per_video, num_samples_per_video]; the occlusion of points across the video. 0: not occluded, 1: occluded.
                sampled_frame_ids: the sampled frame ids.
                sampled_point_ids: the sampled point ids.
        """

        if self.load_prepared_annotation:
            annots = self.annotations[video_id]
        else:
            annots = np.load(self.annotations[video_id], allow_pickle=True).item()
            num_frame = len(os.listdir(os.path.join(self.data_root, video_id, 'frames')))
            annots["visibility"] = annots["visibility"] | (annots["coords"][..., 0]<0) | (annots["coords"][..., 1]<0) | (annots["coords"][..., 0]>(self.input_width-1)) | (annots["coords"][..., 1]>(self.input_height-1))
            annots = {
                "points": annots["coords"],
                "occluded": annots["visibility"].astype(np.int64),
                "num_frames": num_frame
            }
        annots = self.downsample_temporal(annots)
        annots = self.downsample_points(annots)
        if annots is None:
            return None
        for key in annots.keys():
            annots[key] = torch.tensor(annots[key])
        return annots
    
    def get_initial_anno(self, annotations):
        """get the initial position / visibility of the points to be tracked.
        """
        initial_anno = {
            "pt_initial_pos": annotations["boxes"][:, 0:1, :2],
            "pt_initial_visib": annotations["labels"][:, 0:1],
            # "pt_padding_mask": annotations.pop("pt_padding_mask")
        }
        return initial_anno

    def get_one_video(self, video_id, sampled_frame_ids):
        """load the images of one video according to the video_id, and also "num_samples_per_video"

        Args:
            video_id (str): the video id of the video to be loaded.

        Returns:
            image (np.array): [num_samples_per_video, 3, H, W]; the images of one donwsampled video.
        """
        image_paths = self.image_paths[video_id]
        image_paths = [image_paths[rd_id] for rd_id in sampled_frame_ids]
        image = [cv2.imread(image_path) for image_path in image_paths]  # [(H W 3), ...]
        return torch.FloatTensor(np.array(image)).permute(0,3,1,2) / 255.0  # [num_samples_per_video 3 H W]
    
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        Output:
            - images: num_frames 3 H W
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        video_id = self.video_ids[idx]
        targets  = self.get_one_anno(video_id)
        if targets is None:
            print(f"\nWARNING!!!!{video_id} has no valid points to be tracked.\n")
            return self.__getitem__(self.video_ids[random.randint(0, len(self.video_ids)-1)])
        #! debug
        # from ipdb import set_trace; set_trace()
        # print("Num Occluded points At First: ", targets["occluded"].sum())
        # targets["points"] = targets["points"] * (1-targets["occluded"][...,None]) + targets["occluded"][...,None] * 255
        #!
        images   = self.get_one_video(video_id, targets["sampled_frame_ids"])
        if self.transforms is not None:
            def debug():
                from ipdb import set_trace; set_trace()
                start_tracking_frame_ids = torch.tensor(targets["tracking_mask"]).float().argmax(dim=1)
                points = targets["points"]
                sampled_points = torch.gather(torch.FloatTensor(points),1, torch.LongTensor(start_tracking_frame_ids)[:,None,None].repeat(1,1,2))
                print(sampled_points.min(), sampled_points.max())
                for trans in self.transforms.transforms:
                    images, targets = trans(images, targets)
                    start_tracking_frame_ids = torch.tensor(targets["tracking_mask"]).float().argmax(dim=1)
                    not_all_occluded = torch.any(targets["tracking_mask"], dim=1)
                    points = targets["points"]
                    sampled_points = torch.gather(torch.FloatTensor(points),1, torch.LongTensor(start_tracking_frame_ids)[:,None,None].repeat(1,1,2))
                    sampled_points = sampled_points[not_all_occluded]
                    print(trans, sampled_points.min(), sampled_points.max())
                    print(f"{(~not_all_occluded).sum()} points are occluded all the time.")
            images, targets = self.transforms(images, targets)
            if (targets["tracking_mask"].sum(dim=-1) != 1).sum() == 0:  # none of the points are visible.
                return self.__getitem__(self.video_ids[random.randint(0, len(self.video_ids)-1)])

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                targets, images = hack_runner(targets, img=images)
        # initial_points = self.get_initial_anno(targets)
        targets = {
            "pt_boxes": targets.pop("boxes"),
            "pt_labels": targets.pop("labels"),
            "pt_tracking_mask": targets.pop("tracking_mask"),
        }
        # self.debug_vis_data(images, targets)
        # self.debug_check_data(images, targets)
        if self.num_groups_per_video > 0:
            def get_group_mask(num_points, num_groups):
                group_size = num_points // num_groups
                assert group_size * num_groups == num_points, f"num_points: {num_points} % num_groups: {num_groups} should be zero."
                group_mask = torch.ones(num_points, num_points)
                for g_id in range(num_groups):
                    group_mask[g_id*group_size: (g_id+1)*group_size, g_id*group_size: (g_id+1)*group_size] = 0
                shuffle_index = torch.randperm(num_points)
                group_mask = group_mask[shuffle_index]
                group_mask = group_mask[:, shuffle_index]  # points are randomly grouped
                group_mask = group_mask > 0
                return group_mask
            group_mask = get_group_mask(self.num_queries_per_video, self.num_groups_per_video)
            targets["pt_group_mask"] = group_mask  # [num_points, num_points]
        return images, targets
    
    def debug_vis_data(self, images, targets):
        """visualize the data.

        Args:
            data (dict): the data to be visualized.
        """
        def images2video(images, path_video, fps=15):
            '''Transform a series of images into a video.

            Args:
                images (np.array): [num_frame, height, width, 3].
                path_video (string): path to save the resulting video.
            '''
            num_frame, height, width, _ = images.shape
            print(f"Integrating {num_frame} frames into a video with shape H{height}-W{width}...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            videowrite = cv2.VideoWriter(path_video, fourcc, fps=fps, frameSize=(width, height)) 
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                videowrite.write(img)
            videowrite.release()
            print(f"Video saved to {path_video}")
        
        def unnormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            # image: 3 H W
            image = image * image.new_tensor(std)[:, None, None] + image.new_tensor(mean)[:, None, None]
            image = image * 255
            image = np.ascontiguousarray(image.permute(1,2,0).long().numpy())
            return image.astype(np.uint8)
        
        import cv2
        pt_boxes = targets["pt_boxes"]
        pt_labels = targets["pt_labels"]
        pt_tracking_mask = targets["pt_tracking_mask"]
        images_ploted = []
        for frame_id in range(pt_boxes.shape[1]):
            image = unnormalize_image(images[frame_id])
            image = (image * 255).astype(np.uint8)
            for pt_id in range(pt_boxes.shape[0]):
                if pt_tracking_mask[pt_id, frame_id]:
                    box = pt_boxes[pt_id, frame_id]
                    label = pt_labels[pt_id, frame_id]
                    box = box.numpy()
                    box = (box*512).astype(np.int32)
                    if label > 1:
                        cv2.circle(image, (box[0], box[1]), 3, (0, 0, 255), 1)
                    else:
                        cv2.circle(image, (box[0], box[1]), 3, (0, 0, 255), -1)
            images_ploted.append(image)
        images_ploted = np.stack(images_ploted, axis=0)
        from ipdb import set_trace; set_trace()
        images2video(images_ploted, "debug.mp4", fps=7)
    
    def debug_check_data(self, images, targets):
        """Check if the data is formatted correctly as we design.

        Args:
            images (tensor): should be normalized, and in the shape of [num_frames, 3, H, W].
            targets (dict): 
                pt_boxes: tensor, [num_points, num_frames, 4], normalized. The unoccluded points should within the image (0~1).
                pt_labels: tensor, [num_points, num_frames], the visibility. 0: should not appear, 1: unoccluded, 2: occluded.
                pt_tracking_mask: tensor, [num_points, num_frames], the tracking mask. 0: should not be tracked, 1: should be tracked (from when it appears).
        """
        def most_frequent_vector(tensor):
            # 将 tensor 转换为一维的 tensor
            tensor_1d = tensor.view(-1, tensor.size(-1)).tolist()

            # 使用 Counter 来计算每个元素的出现次数
            from collections import Counter
            counter = Counter(map(tuple, tensor_1d))

            # 找出出现次数最多的元素
            most_frequent = counter.most_common(1)[0][0]

            return torch.tensor(most_frequent)
        
        def unnormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            # image: 3 H W
            image = image * image.new_tensor(std)[:, None, None] + image.new_tensor(mean)[:, None, None]
            image = image * 255
            image = np.ascontiguousarray(image.permute(1,2,0).long().numpy())
            return image.astype(np.uint8)
        
        def images2video(images, path_video, fps=15):
            '''Transform a series of images into a video.

            Args:
                images (np.array): [num_frame, height, width, 3].
                path_video (string): path to save the resulting video.
            '''
            num_frame, height, width, _ = images.shape
            print(f"Integrating {num_frame} frames into a video with shape H{height}-W{width}...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            videowrite = cv2.VideoWriter(path_video, fourcc, fps=fps, frameSize=(width, height)) 
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                videowrite.write(img)
            videowrite.release()
            print(f"Video saved to {path_video}")

        # self.debug_vis_data(images, targets)
        pt_boxes = targets["pt_boxes"].clone()
        pt_labels = targets["pt_labels"].clone()
        pt_tracking_mask = targets["pt_tracking_mask"].clone()
        # ==============check images
        print("checking images...")
        print("The shape of images should be [num_frames, 3, H, W]. The shape is: ", images.shape)
        # ==============check pt_boxes and pt_labels
        print("checking targets['pt_boxes']...")
        print("The shape of pt_boxes should be [num_points, num_frames, 4]. The shape is: ", pt_boxes.shape)
        occ_pt_boxes = pt_boxes[pt_labels==2]
        occ_pt_box_ptcids, occ_pt_box_frameids = torch.where(pt_labels==2)
        unocc_pt_boxes = pt_boxes[pt_labels==1]
        unocc_pt_box_ptcids, unocc_pt_box_frameids = torch.where(pt_labels==1)
        # for occluded boxes, there are two cases: (without panning)
        #   1. it is occluded at first, where they will have a shared coord.
        #   2. it is occluded because of data aug, where it is outsight of the image or occluded by mask. the first ones should have invalid coord. the after ones should be visualized to checkout.
        occ_pt_boxes_insight_flag = (occ_pt_boxes[..., 0] >= 0) & (occ_pt_boxes[..., 0] <= 1) & (occ_pt_boxes[..., 1] >= 0) & (occ_pt_boxes[..., 1] <= 1)
        occ_pt_box_base = most_frequent_vector(occ_pt_boxes[occ_pt_boxes_insight_flag])
        occ_pt_box_flags_caused_by_maskaug = (occ_pt_boxes[occ_pt_boxes_insight_flag] - occ_pt_box_base).abs().sum(dim=-1) > 1e-5
        print("num of boxes that is occluded at first: ", occ_pt_boxes[occ_pt_boxes_insight_flag].shape[0] - occ_pt_box_flags_caused_by_maskaug.sum())
        occ_pt_box_caused_by_maskaug_ptcids   = occ_pt_box_ptcids[occ_pt_boxes_insight_flag][occ_pt_box_flags_caused_by_maskaug]
        occ_pt_box_caused_by_maskaug_frameids = occ_pt_box_frameids[occ_pt_boxes_insight_flag][occ_pt_box_flags_caused_by_maskaug]
        images_vis_occ_pt_boxes_caused_by_maskaug = [unnormalize_image(img) for img in images]
        for f_id,p_id in zip(occ_pt_box_caused_by_maskaug_frameids, occ_pt_box_caused_by_maskaug_ptcids):
            box = pt_boxes[p_id, f_id]
            box = (box * torch.tensor([self.input_width, self.input_height, self.input_width, self.input_height])).long()
            cv2.circle(images_vis_occ_pt_boxes_caused_by_maskaug[f_id], (box[0].item(), box[1].item()), 3, (0, 0, 255), 1)
        # (without restart)
        # images2video(np.stack(images_vis_occ_pt_boxes_caused_by_maskaug, axis=0), "debug_vis_occ_pt_boxes_caused_by_maskaug.mp4", fps=7)
        # for unoccluded boxes, they should insight
        wrong_unocc_box_flags = ((unocc_pt_boxes[:,0]<0) | (unocc_pt_boxes[:,1]<0) | (unocc_pt_boxes[:,0]>1) | (unocc_pt_boxes[:,1]>1))
        print("wrong unocc boxes: ",unocc_pt_boxes[wrong_unocc_box_flags])
        # ==============check pt_tracking_mask
        # apart for the points that is occluded all the time:
        #   1. Their the tracking mask at the first ~occluded indices should be True.
        #   2. before that, the tracking mask should be False.
        #   3. Tracking mask sould all be True after the first occluded index.
        is_all_occluded = (pt_labels==2).sum(dim=-1) == pt_labels.shape[-1]
        for occluded, tracking_mask, all_occluded in zip(pt_labels, pt_tracking_mask, is_all_occluded):
            if all_occluded:
                if not (torch.where(tracking_mask)[0][0] == 23):
                    print("error in pt_tracking_mask-1")
            else:
                first_indice = torch.where(~(occluded==2))[0][0]
                if not (tracking_mask[:first_indice].sum() == 0):
                    print("error in pt_tracking_mask-2")
                if not (tracking_mask[first_indice:].sum() == tracking_mask[first_indice:].shape[0]):
                    print("error in pt_tracking_mask-3")
        print("Done")


def build_kubric(image_set, args):
    root = Path(args.data_path)
    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "mini_train": (root / "mini_train", root / "annotations" / 'mini_train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
    }
    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    num_groups_per_video = getattr(args, "num_groups_per_video", 1)
    dataset = PointTrackingDataset(
            img_folder, ann_file, 
            transforms=make_temporal_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
            aux_target_hacks=aux_target_hacks_list,
            num_queries_per_video=args.num_queries_per_video,
            num_groups_per_video=num_groups_per_video,
            num_samples_per_video=args.num_samples_per_video,
            sample_visible_at_first_middle_frame=args.sample_visible_at_first_middle_frame,
            sample_continuous_clip=args.sample_continuous_clip,
            input_width=512,
            input_height=512
        )
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='taptr', help='model name')
    parser.add_argument('--mini_box_size', type=int, default=3, help='size of a point')
    parser.add_argument('--num_class', type=int, default=2, help='occluded or not')
    parser.add_argument('--fix_size', action='store_true')
    args = parser.parse_args()
    args.data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    args.data_aug_max_size = 1333
    args.data_aug_scales2_resize = [400, 500, 600]
    args.data_aug_scales2_crop = [384, 600]
    dataset_kubric = PointTrackingDataset(
            Path('datas/kubric_movif_cotracker/r512/val'),
            "None",
            transforms=make_temporal_transforms("mini_train", fix_size=args.fix_size, strong_aug=False, args=args), 
            aux_target_hacks=get_aux_target_hacks_list("mini_train", args),
            num_queries_per_video=256,
            num_samples_per_video=24,
            sample_visible_at_first_middle_frame=True,
            sample_continuous_clip=True,
            input_width=512,
            input_height=512
        )
    images, targets = dataset_kubric[-1]
    import ipdb; ipdb.set_trace()
    debug_visualize(images, targets)
