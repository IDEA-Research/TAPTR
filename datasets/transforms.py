# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")


    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def temporal_resize(images, target, size, max_size=None, compensate=False):
    """Resize images consistently across the temporal dimension

    Args:
        images (torch.tensor): num_frames 3 H W
        target (dict): the targets to be resized together, the supported keys: boxes, points, area, mask
        size (int or list): the expected size. If it is int, it indicates the expected size of the shorter side. If it is a list, it indicase the size of the image.
        max_size (int): the expected size of the longer side. Defaults to None.
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        n_temp, _, w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    if isinstance(images, torch.Tensor):
        image_size = images.permute(0, 1, 3, 2).size()  # bs 3 H W --> bs 3 W H (to align with the PIL image size)
    else:
        image_size = images.size()
    size = get_size(image_size, size, max_size)  # h w
    if len(size) > 2:
        size = size[1:]
    rescaled_images = F.resize(images, size)

    if target is None:
        return rescaled_images, None

    if not compensate:
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_images.size(), images.size()))
    else:
        ratios = tuple(float(s-1) / float(s_orig-1) for s, s_orig in zip(rescaled_images.size(), images.size()))
    _, _, ratio_height, ratio_width = ratios

    target = target.copy()
    if "points" in target:
        points = target["points"]
        scaled_points = points * torch.as_tensor([ratio_width, ratio_height])
        target["points"] = scaled_points
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_images, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)
    

class RandomTemporalConsistentResize(object):
    def __init__(self, sizes, max_size=None, compensate=False):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.compensate = compensate

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return temporal_resize(img, target, size, self.max_size, self.compensate)


class RandomRestartTrajectories(object):
    """Random the start tracking frames of the selected points randomly.

    Args:
        object (_type_): _description_
    """
    def __init__(self, ratio_restart_trajectory) -> None:
        self.ratio_restart_trajectory = ratio_restart_trajectory
    
    def __call__(self, imgs, targets):
        start_tracking_frame_ids = torch.argmax(1-targets["occluded"], axis=1)
        all_occluded = torch.all(targets['occluded'], dim=1)
        start_tracking_frame_ids[all_occluded] = targets['occluded'].shape[-1] - 1
        N = start_tracking_frame_ids.shape[0]
        N_rand = N // 4
        len_temp = targets['occluded'].shape[1]
        # inds of visible points in the 1st frame
        nonzero_inds = [torch.nonzero(1-targets["occluded"][i]) for i in range(N)]
        rand_vis_inds = []
        for nonzero_row in nonzero_inds:
            if len(nonzero_row) > 0:
                rand_vis_inds.append(nonzero_row[torch.randint(len(nonzero_row), size=(1,))])
            else:
                rand_vis_inds.append(torch.IntTensor([[len_temp - 1]]))
        rand_vis_inds = torch.cat(rand_vis_inds, dim=1)[0]
        # rand_vis_inds = torch.cat(
        #     [
        #         nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
        #         for nonzero_row in nonzero_inds
        #     ],
        #     dim=1,
        # )[0]
        indices_to_restart = torch.randperm(N - all_occluded.sum())[:N_rand]
        start_tracking_frame_ids[indices_to_restart] = rand_vis_inds[indices_to_restart]
        targets["occluded"] = targets["occluded"] | (torch.arange(targets['occluded'].shape[1]) < start_tracking_frame_ids.unsqueeze(1))
        targets["tracking_mask"] = targets["tracking_mask"] & (torch.arange(targets['occluded'].shape[1]) >= start_tracking_frame_ids.unsqueeze(1))
        return imgs, self.post_process(targets)
    
    def post_process(self, targets):
        sampled_ids = torch.arange(targets['occluded'].shape[0])
        start_tracking_frame_ids = torch.argmax(1-targets["occluded"], axis=1)
        all_occluded = torch.all(targets['occluded'], dim=1)
        start_tracking_frame_ids[all_occluded] = targets['occluded'].shape[-1] - 1
        start_tracking_frame_ids, sampled_ids = zip(*sorted(zip(start_tracking_frame_ids, sampled_ids)))
        sampled_ids = torch.tensor(sampled_ids)
        
        targets["points"] = targets["points"][sampled_ids]
        targets["occluded"] = targets["occluded"][sampled_ids]
        targets["sampled_point_ids"] = targets["sampled_point_ids"][sampled_ids]
        targets["tracking_mask"] = targets["tracking_mask"][sampled_ids]
        return targets
        

class RandomTemporalCameraPanning(object):
    """ From TAPIR:
        'One subtle difference between the Kubric MOViE [17] dataset and 
        the real world is the lack of panning: although the Kubric camera 
        moves, it is always set to "look at" a single point at the center of the workspace.'
        Make the "look at" point moves along a random linear trajectory.
        The code refers from PIPs's add_spatial_augs
    Args:
        object (_type_): _description_
    """
    def __init__(self, pad_bounds, resize_lim, resize_delta, crop_size, max_crop_offset) -> None:
        self.pad_bounds = pad_bounds
        self.resize_lim = resize_lim
        self.resize_delta = resize_delta
        self.crop_size = crop_size
        self.max_crop_offset = max_crop_offset

    def __call__(self, imgs, targets=None):
        trajs = targets['points'].permute(1, 0, 2).numpy()  # T N 2
        visibles = 1 - targets['occluded'].permute(1, 0).numpy()  # T N 
        rgbs = imgs.permute(0, 2, 3, 1).numpy()  # T H W C
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = (W_new - 1) / float(W - 1)
            scale_y = (H_new - 1) / float(H - 1)
            rgbs_scaled.append(cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        targets['points'] = targets['points'].new_tensor(trajs).permute(1, 0, 2)
        targets['occluded'] = targets['occluded'].new_tensor(1 - visibles).permute(1, 0)
        imgs = imgs.new_tensor(np.stack(rgbs, axis=0)).permute(0, 3, 1, 2)
        targets = self.post_process(targets)
        return imgs, targets
    
    def post_process(self, targets):
        targets['occluded'] = targets['occluded'] | (targets['points'][..., 0] < 0) | (targets['points'][..., 0] > (self.crop_size[1]-1)) | (targets['points'][..., 1] < 0) | (targets['points'][..., 1] > (self.crop_size[0]-1))
        sampled_ids = torch.arange(targets['occluded'].shape[0])
        start_tracking_frame_ids = torch.argmax(1-targets["occluded"], axis=1)
        all_occluded = torch.all(targets['occluded'], dim=1)
        start_tracking_frame_ids[all_occluded] = targets['occluded'].shape[-1] - 1
        start_tracking_frame_ids, sampled_ids = zip(*sorted(zip(start_tracking_frame_ids, sampled_ids)))
        sampled_ids = torch.tensor(sampled_ids)
        tracking_mask = torch.zeros_like(targets["tracking_mask"], dtype=torch.bool)
        # num_invalid = (torch.tensor(start_tracking_frame_ids) == 23).sum()
        # print(num_invalid)
        for point_id, start_tracking_frame_id in enumerate(start_tracking_frame_ids):
            tracking_mask[point_id, start_tracking_frame_id:] = True
        
        targets["points"] = targets["points"][sampled_ids]
        targets["occluded"] = targets["occluded"][sampled_ids]
        targets["sampled_point_ids"] = targets["sampled_point_ids"][sampled_ids]
        targets["tracking_mask"] = tracking_mask
        return targets


class RandomTemporalFlipping(object):
    def __init__(self, h_flip_prob=0.5, v_flip_prob=0.5) -> None:
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
    def __call__(self, imgs, targets):
        trajs = targets['points'].permute(1, 0, 2).numpy()  # T N 2
        rgbs = imgs.permute(0, 2, 3, 1).numpy()  # T H W C
        H_new, W_new = rgbs[0].shape[:2]
        # flip
        h_flipped = False
        v_flipped = False
        # h flip
        if np.random.rand() < self.h_flip_prob:
            h_flipped = True
            rgbs = [rgb[:, ::-1] for rgb in rgbs]
        # v flip
        if np.random.rand() < self.v_flip_prob:
            v_flipped = True
            rgbs = [rgb[::-1] for rgb in rgbs]
        if h_flipped:
            trajs[:, :, 0] = (W_new-1) - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = (H_new-1) - trajs[:, :, 1]
        targets['points'] = targets['points'].new_tensor(trajs).permute(1, 0, 2)
        imgs = imgs.new_tensor(np.stack(rgbs, axis=0)).permute(0, 3, 1, 2)
        return imgs, targets


class RandomPhotometricAug(object):
    """Add ohotometric data augmentation to the input images.
    Args:
        blur_aug_prob (float, optional): probability of applying Gaussian blur. Defaults to 0.25.
        color_aug_prob (float, optional): probability of applying color jitter. Defaults to 0.25.
        eraser_aug_prob (float, optional): probability of applying eraser. Defaults to 0.5.
        eraser_bounds (list, optional): bounds for the eraser. Defaults to [2, 100].
        eraser_max (int, optional): maximum number of eraser to apply. Defaults to 10.
    """
    def __init__(self, blur_aug_prob=0.25, color_aug_prob=0.25, eraser_aug_prob=0.5, replace_aug_prob=0.5, replace_bounds=[2, 100], replace_max=10, eraser_bounds=[2, 100], eraser_max=10, erase=True, replace=True, contrastive_erase=False):
        from torchvision.transforms import ColorJitter, GaussianBlur
        self.blur_aug_prob = blur_aug_prob
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        self.color_aug_prob = color_aug_prob
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.eraser_aug_prob = eraser_aug_prob
        self.eraser_bounds = eraser_bounds
        self.eraser_max = eraser_max
        self.contrastive_erase = contrastive_erase
        self.replace_aug_prob = replace_aug_prob
        self.replace_bounds = replace_bounds
        self.replace_max = replace_max

        self.erase = erase
        self.replace = replace

    def get_contrastive_color(self, color):
        distance = 0
        while distance <= 100:
            random_rgb = np.random.random(3) * 255
            distance = np.linalg.norm(random_rgb - color)
        return random_rgb

    def __call__(self, imgs, targets):
        trajs = targets['points'].permute(1, 0, 2).numpy()  # T N 2
        visibles = 1 - targets['occluded'].permute(1, 0).numpy()  # T N 
        rgbs = imgs.permute(0, 2, 3, 1).numpy() * 255  # T H W C

        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T
        
        if self.erase:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        if self.contrastive_erase:
                            mean_color = self.get_contrastive_color(mean_color)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
                        
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
        if self.replace:
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs
            ]
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
        targets['points'] = targets['points'].new_tensor(trajs).permute(1, 0, 2)
        targets['occluded'] = targets['occluded'].new_tensor(1 - visibles).permute(1, 0)
        imgs = imgs.new_tensor(np.stack(rgbs, axis=0)).permute(0, 3, 1, 2)
        imgs, targets = self.post_process(imgs, targets)
        return imgs, targets

    def post_process(self, imgs, targets):
        height, width = imgs.shape[-2:]
        imgs = imgs / 255.0
        targets['occluded'] = targets['occluded'] | (targets['points'][..., 0] < 0) | (targets['points'][..., 0] > (width-1)) | (targets['points'][..., 1] < 0) & (targets['points'][..., 1] > (height-1))
        sampled_ids = torch.arange(targets['occluded'].shape[0])
        start_tracking_frame_ids = torch.argmax(1-targets["occluded"], axis=1)
        all_occluded = torch.all(targets['occluded'], dim=1)
        start_tracking_frame_ids[all_occluded] = targets['occluded'].shape[-1] - 1
        start_tracking_frame_ids, sampled_ids = zip(*sorted(zip(start_tracking_frame_ids, sampled_ids)))
        sampled_ids = torch.tensor(sampled_ids)
        tracking_mask = torch.zeros_like(targets["tracking_mask"], dtype=torch.bool)
        for point_id, start_tracking_frame_id in enumerate(start_tracking_frame_ids):
            tracking_mask[point_id, start_tracking_frame_id:] = True
        
        targets["points"] = targets["points"][sampled_ids]
        targets["occluded"] = targets["occluded"][sampled_ids]
        targets["sampled_point_ids"] = targets["sampled_point_ids"][sampled_ids]
        targets["tracking_mask"] = tracking_mask
        return imgs, targets


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std, compensate=False):
        self.mean = mean
        self.std = std
        self.compensate = compensate

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "points" in target:
            points = target["points"]
            if self.compensate:
                points = points / torch.tensor([w-1, h-1], dtype=torch.float32)
            else:
                points = points / torch.tensor([w, h], dtype=torch.float32)
            target["points"] = points
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Point2Minibox():
    """Transform Points into mini boxes to fit the pipeline of box detection.
    """
    def __init__(self, mini_box_size=5) -> None:
        self.mini_box_size = mini_box_size

    def __call__(self, target, img=None):
        """turn Points into mini boxes.

        Args:
            target (Dict): the targets to be transfored, there should at least contains:
                points: np.array, [num_queries_per_video, num_samples_per_video, 2]; the coordinates of Points in the image coordinate system.
            img (np.array, optional): _description_. Defaults to None.

        Returns:
            target (Dict): the updated target.
            img (np.array): the original images.
        """
        xy = target.pop("points")
        target["boxes"] = torch.FloatTensor(np.concatenate([xy, np.ones(xy.shape)*self.mini_box_size], axis=-1))
        return target, img


class Visibility2Label():
    """Transform visibilities into class labels to fit the pipeline of detection.
    """
    def __call__(self, target, img=None):
        """

        Args:
            target (Dict): the targets to be transfored, there should at least contains:
                occluded: np.array, [num_queries_per_video, num_samples_per_video]; the flags indicating the visibility of points.
            img (np.array, optional): images. Defaults to None.

        Returns:
            target (Dict): the updated target.
            img (np.array): the original images.
        """
        visibilities = target.pop("occluded")
        target["labels"] = visibilities + 1
        return target, img


class FlattenTemporal():
    """flatten the temporal dimension, to make targets fit the pipeline of data processing.
    """
    def __call__(self, targets, img=None):
        """flatten the temporal dimension, to make targets fit the pipeline of data processing.

        Args:
            targets (dict): 
                boxes: np.array [num_box, num_frames, 4].
                labels: np.array [num_box, num_frames].
            imgs (_type_, optional): _description_. Defaults to None.
        Return:
            targets (dict):
                boxes: np.array [num_box*num_frames, 4].
                labels: np.array [num_box*num_frames].
        """
        targets["boxes"] = targets["boxes"].flatten(0,1)
        targets["labels"] = targets["labels"].flatten(0,1)
        return targets, img


class UnflattenTemporal():
    """unflatten the temporal dimension, to make targets fit the pipeline of data processing.
    """
    def __call__(self, targets, img=None):
        """unflatten the temporal dimension, to make targets fit the pipeline of data processing.

        Args:
            targets (dict): 
                boxes: np.array [num_box*num_frames, 4].
                labels: np.array [num_box*num_frames].
            img (_type_, optional): _description_. Defaults to None.
        Return:
            targets (dict):
                boxes: np.array [num_box, num_frames, 4].
                labels: np.array [num_box, num_frames].
        """
        num_queries_per_video = len(targets["sampled_point_ids"])
        targets["boxes"] = targets["boxes"].reshape(num_queries_per_video, -1, 4)
        targets["labels"] = targets["labels"].reshape(num_queries_per_video, -1)
        return targets, img
