"""
The TapVid dataset for TaoVid Benchmark, refer from Cotracker2.
"""
import os
import io
import glob
import torch
import pickle
import numpy as np
import mediapy as media
from PIL import Image
from pathlib import Path
from typing import Mapping, Tuple, Union
from datasets.kubric import make_temporal_transforms, get_aux_target_hacks_list


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)  # [first_emerge_frame, y, x]

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_specified(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    start_tracking_frame: int,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """

    query_points = []
    for i in range(target_points.shape[0]):
        index = start_tracking_frame  # start tracking from the specified frame.
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)  # [first_emerge_frame, y, x]

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center = None,
    device = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


class TapVidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_type="davis",
        resize_to_256=True,
        queried_first=True,
        transforms=None,
        aux_target_hacks=None,
        num_queries_per_video=-1,
    ):
        self.dataset_type = dataset_type
        self.resize_to_256 = resize_to_256
        self.queried_first = queried_first
        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset
        elif self.dataset_type == "rgbstacking":
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            self.video_names = list(range(len(self.points_dataset)))
            self.points_dataset={i:self.points_dataset[i] for i in self.video_names}
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            if self.dataset_type == "davis":
                self.video_names = list(self.points_dataset.keys())
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

        # transforms=make_temporal_transforms("val"), 
        # aux_target_hacks=get_aux_target_hacks_list("mini_train", args),
        self.transforms = transforms
        self.aux_target_hacks = aux_target_hacks
        self.num_queries_per_video = num_queries_per_video
        self.grid_size = 5

    def padding_points(self, points_xy, points_occ, height=256, width=256, margin_x=0, margin_y=0, num_padding=-1):
        if (self.num_queries_per_video > 0) and (points_xy.shape[0] < self.num_queries_per_video):
            np.random.seed(0)  # for reproducibility
            # xy
            margin_x = width // 16
            margin_y = height // 16
            min_distance = 1
            if num_padding == -1:
                num_padding = self.num_queries_per_video - points_xy.shape[0]
            # distant points
            def generate_coordinates(original_coords, N, lower_bound=0, upper_bound=255, distance=3):
                coordinates = original_coords.tolist()
                while len(coordinates) < N:
                    new_coordinate = np.random.randint(lower_bound, upper_bound+1, size=2)
                    if all(np.linalg.norm(new_coordinate - c) >= distance for c in coordinates):
                        coordinates.append(new_coordinate)
                return np.array(coordinates[-N:])
            padding_points_xy = generate_coordinates(points_xy[:, 0], num_padding, lower_bound=margin_x, upper_bound=width-1-margin_x, distance=min_distance)
            padding_points_xy = padding_points_xy[:,None].repeat(points_xy.shape[1], axis=1)
            points_xy = np.concatenate((points_xy, padding_points_xy), axis=0)
            # occ
            padding_points_occ = np.ones((num_padding, points_occ.shape[1])) < 0
            # random restart all
            padding_points_occ[:, :12] = \
                padding_points_occ[:, :12] | \
                np.arange(12)[None,:].repeat(padding_points_occ.shape[0], 0) < np.random.randint(0, 12, padding_points_occ.shape[0])[:,None]
            
            points_occ = np.concatenate((points_occ, padding_points_occ), axis=0)
        
        return points_xy, points_occ

    def __getitem__(self, index):
        if self.dataset_type == "davis":
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name]
        frames = video["video"]  # num_frames, height, width, 3

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = video["points"]  # num_points, num_frames, 2
        if self.resize_to_256:
            frames = resize_video(frames, [256, 256])
            target_points *= np.array([255, 255])  # 1 should be mapped to 256-1
        else:
            target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])

        target_occ = video["occluded"]  # num_points, num_frame
        _, height, width, _ = frames.shape
        num_points_, num_frames_ = target_occ.shape
        if self.queried_first:  # all & first query
            target_points, target_occ = self.padding_points(target_points, target_occ, height, width, height//64, height//64)
            converted = sample_queries_first(target_occ, target_points, frames)
            assert converted["target_points"].shape[1] == converted["query_points"].shape[1]
            num_points, num_frames = converted["occluded"][0].shape
            tracking_mask = torch.ones([num_points, num_frames]) > 0
            first_emerge_frames = converted["query_points"][0, :, 0]
            for p_id in range(num_points):
                tracking_mask[p_id, : int(first_emerge_frames[p_id])] = False
            targets ={
                "points": torch.from_numpy(converted["target_points"])[0].float(),
                'occluded': torch.from_numpy(converted["occluded"][0]), 
                'num_frames': num_frames, 
                'sampled_frame_ids': torch.arange(num_frames), 
                'tracking_mask': tracking_mask,
                'query_frames': torch.IntTensor(converted["query_points"][0, :, 0]),
                'sampled_point_ids': torch.arange(num_points),
                "num_real_pt": torch.tensor([num_points_]),
                'seq_name': str(video_name),
            }
            return self.align_format(torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0, targets)
        elif not self.queried_first:  # all & strided query
            count_stride = 0
            query_stride = 5
            target_points_strides = []
            target_occ_strides = []
            start_tracking_frame = []
            for i in range(0, target_occ.shape[1], query_stride):
                mask = target_occ[:, i] == 0
                if mask.sum() > 0:
                    target_points_strides.append(target_points[mask])
                    target_occ_strides.append(target_occ[mask])
                    count_stride += 1
                    start_tracking_frame.append(i)
            aligned_data_list = []
            aligned_inv_data_list = []
            for strid_id in range(count_stride):
                print(f"Preparing Stride {strid_id}")
                num_points_ = target_points_strides[strid_id].shape[0]
                target_points_strides[strid_id], target_occ_strides[strid_id] = self.padding_points(target_points_strides[strid_id], target_occ_strides[strid_id], height, width, height//64, height//64)
                converted = sample_queries_specified(target_occ_strides[strid_id], target_points_strides[strid_id], frames, start_tracking_frame[strid_id])
                assert converted["target_points"].shape[1] == converted["query_points"].shape[1]
                num_points, num_frames = converted["occluded"][0].shape
                tracking_mask = torch.ones([num_points, num_frames]) > 0
                tracking_mask_inv = torch.ones([num_points, num_frames]) > 0
                first_emerge_frames = converted["query_points"][0, :, 0]
                for p_id in range(num_points):
                    tracking_mask[p_id, : int(first_emerge_frames[p_id])] = False
                    tracking_mask_inv[p_id, int(first_emerge_frames[p_id]+1):] = False
                targets ={
                    "points": torch.from_numpy(converted["target_points"])[0].float(),
                    'occluded': torch.from_numpy(converted["occluded"][0]), 
                    'num_frames': num_frames, 
                    'sampled_frame_ids': torch.arange(num_frames), 
                    'tracking_mask': tracking_mask,
                    'query_frames': torch.IntTensor(converted["query_points"][0, :, 0]),
                    'sampled_point_ids': torch.arange(num_points),
                    "num_real_pt": torch.tensor([num_points_]),
                    'seq_name': str(video_name) + f"_stride{strid_id}",
                }
                frames = torch.FloatTensor(frames)
                frames_inv = frames.flip(0)
                targets_inv ={
                    "points": torch.from_numpy(converted["target_points"])[0].float().flip(1),
                    'occluded': torch.from_numpy(converted["occluded"][0]).flip(1), 
                    'num_frames': num_frames, 
                    'sampled_frame_ids': torch.arange(num_frames), 
                    'tracking_mask': tracking_mask_inv.flip(1),
                    'query_frames': num_frames - torch.IntTensor(converted["query_points"][0, :, 0]) - 1,
                    'sampled_point_ids': torch.arange(num_points),
                    "num_real_pt": torch.tensor([num_points_]),
                    'seq_name': str(video_name) + f"_stride{strid_id}_inv",
                }
                aligned_data_list.append(self.align_format(frames.permute(0, 3, 1, 2) / 255.0, targets))
                aligned_inv_data_list.append(self.align_format(frames_inv.permute(0, 3, 1, 2) / 255.0, targets_inv))
            return aligned_data_list, aligned_inv_data_list
        else:
            raise NotImplementedError

    def align_format(self, images, targets):
        if self.transforms is not None:
            images, targets = self.transforms(images, targets)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                targets, images = hack_runner(targets, img=images)
        seq_name = targets.pop("seq_name")
        targets = {
            "pt_boxes": targets.pop("boxes"),
            "pt_labels": targets.pop("labels"),
            "pt_tracking_mask": targets.pop("tracking_mask"),
            "num_real_pt": targets.pop("num_real_pt"),
            "query_frames": targets.pop("query_frames"),
        }
        return images, targets, seq_name
        
    def __len__(self):
        return len(self.points_dataset)



def build_tapvid(image_set, args):
    root = Path(args.data_path)
    PATHS = {
        "tapvid_davis_first": ("davis", root / "tapvid"/ "tapvid_davis" / "tapvid_davis.pkl", True, True),
        "tapvid_davis_strided": ("davis", root / "tapvid"/ "tapvid_davis" / "tapvid_davis.pkl", False, True),
        "tapvid_kinetics_first": ("kinetics", root / "tapvid"/ "tapvid_kinetics" / "Kinetics700_2020_processed", True, True),
        "tapvid_rgb_stacking_first": ("rgb_stacking", root / "tapvid"/ "tapvid_rgb_stacking" / "tapvid_rgb_stacking.pkl", True, True),
    }
    dataset_type, data_root, queried_first, resize_to_256 = PATHS[image_set]
    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list("val", args)
    transforms            = make_temporal_transforms("val", fix_size=args.fix_size, strong_aug=False, args=args)
    
    dataset = TapVidDataset(
        data_root,
        dataset_type=dataset_type,
        resize_to_256=resize_to_256,
        queried_first=queried_first,
        transforms=transforms, 
        aux_target_hacks=aux_target_hacks_list,
        num_queries_per_video=args.num_queries_per_video_eval,
    )
    return dataset

if __name__ == "__main__":
    dataset_name = "tapvid_davis_first"
    dataset_root = "datas/tapvid"
    dataset_type = dataset_name.split("_")[1]
    if dataset_type == "davis":
        data_root = os.path.join(dataset_root, "tapvid_davis", "tapvid_davis.pkl")
    elif dataset_type == "kinetics":
        data_root = os.path.join(
            dataset_root, "/kinetics/kinetics-dataset/k700-2020/tapvid_kinetics"
        )
    tapvid_dataset = TapVidDataset(
            dataset_type=dataset_type,
            data_root=data_root,
            queried_first=not "strided" in dataset_name,
    )
    tapvid_dataset[0]