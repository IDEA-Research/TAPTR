# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
from typing import Iterable, Mapping, Tuple, Union

from util.slconfig import DictAction, SLConfig
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.cuda.amp import autocast


def dict2string(dict):
    string = ""
    for k, v in dict.items():
        string += f"{k}: {v} \n"
    return string


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--eval_checkpoint', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--data_path', type=str, default='/path/to/a/specific/datasets')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Evaluate in debug mode, to visualize results for example.")
    
    return parser


def compute_tapvid_metrics(
    query_frame: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_frame: The start tracking frame.  Its size is
         [b, n], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    EPS = 1e-6
    mean = numer / (EPS + denom)
    return mean


class Evaluator:
    """
    A class defining evaluator, refer from CoTrackerv2.
    """

    def __init__(self, exp_dir, ckpt_name) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations", ckpt_name)
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
        self.test_log_dir = os.path.join(exp_dir, "test", ckpt_name)
        if not os.path.exists(self.test_log_dir):
            os.makedirs(self.test_log_dir)
        self.results_AJ_ours = {}
        self.results_AJ_comp = {}

        self.results_DeltaX_ours = {}
        self.results_DeltaX_comp = {}

    def compute_metrics(self, metrics, seq_name, targets, predictions, dataset_name, res_H, res_W):
        """compute metrics for different datasets according to the predictions and targets.

        Args:
            metrics (dict): recording the results of different metrics for each sequence.
            seq_name (str): the name of the currently evaluating sequence.
            targets (dict): 
                pt_boxes (torch.FloatTensor): [num_queries, num_frames, 4], the gt trajectory of queries.
                pt_labels (torch.FloatTensor): [num_queries, num_frames], the gt visibility of queries (1/0).
                query_frames (torch.IntTensor): [num_queries], the first emerge frame.
            predictions (dict): 
                pred_boxes (torch.FloatTensor): [num_queries, num_frames, 4], the predicted trajectory of queries.
                pred_labels (torch.FloatTensor): [num_queries, num_frames], the predicted visibility of queries (1/0).
            dataset_name (_type_): _description_
            res_H, res_W (Int): the resolution of the video.
        """
        num_queries = targets["num_real_pt"]
        pred_trajectory, pred_visibility = predictions["pred_boxes"][:num_queries], predictions["pred_labels"][:num_queries]
        pred_tracks = (pred_trajectory[None, ..., :2] * pred_trajectory.new_tensor([res_W-1, res_H-1])).cpu().numpy()
        pred_occluded = (pred_visibility != 1)[None, ...].cpu().numpy()
        gt_trajectory, gt_visibility, query_frames = targets["pt_boxes"][:num_queries], targets["pt_labels"][:num_queries], targets["query_frames"][:num_queries]
        gt_tracks = (gt_trajectory[None, ..., :2] * pred_trajectory.new_tensor([res_W-1, res_H-1])).cpu().numpy()
        gt_occluded = (gt_visibility != 1)[None, ...].cpu().numpy()
        query_frames = query_frames[None, :].cpu().numpy()

        if "tapvid" in dataset_name:
            out_metrics = compute_tapvid_metrics(
                query_frames,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[seq_name] = out_metrics
            for metric_name in out_metrics.keys():
                out_metrics[metric_name] = out_metrics[metric_name][0]
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )
            print("Sequence Name: ", seq_name)
            print(f"Metrics: {out_metrics}")
            print(f"\nAvg Metric: {metrics['avg']}")
        else:
            raise NotImplementedError


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def summarize_all_points(outputs, targets):
    outputs_sum = {}
    targets_sum = {}
    # full_seq_output
    outputs_sum["full_seq_output"] = {
        "pred_logits": [],
        "pred_boxes": [],
        "dn_meta": None
    }
    targets_sum["full_seq_target"] = [{
        "pt_boxes": [],
        "pt_labels": [],
        "pt_tracking_mask": [],
        "ptq_update_mask": [],
        "query_frames": [],
        "num_real_pt": 0,
    }]

    for output, target in zip(outputs, targets):
        num_real_pt = target["full_seq_target"][0]["num_real_pt"]
        outputs_sum["full_seq_output"]["pred_logits"].append(output["full_seq_output"]["pred_logits"][:, :, :num_real_pt])
        outputs_sum["full_seq_output"]["pred_boxes"].append(output["full_seq_output"]["pred_boxes"][:, :, :num_real_pt])
        targets_sum["full_seq_target"][0]["pt_boxes"].append(target["full_seq_target"][0]["pt_boxes"][:num_real_pt])
        targets_sum["full_seq_target"][0]["pt_labels"].append(target["full_seq_target"][0]["pt_labels"][:num_real_pt])
        targets_sum["full_seq_target"][0]["pt_tracking_mask"].append(target["full_seq_target"][0]["pt_tracking_mask"][:num_real_pt])
        targets_sum["full_seq_target"][0]["ptq_update_mask"].append(target["full_seq_target"][0]["ptq_update_mask"][:num_real_pt])
        targets_sum["full_seq_target"][0]["query_frames"].append(target["full_seq_target"][0]["query_frames"][:num_real_pt])
        targets_sum["full_seq_target"][0]["num_real_pt"] += num_real_pt

    outputs_sum["full_seq_output"]["pred_logits"] = torch.cat(outputs_sum["full_seq_output"]["pred_logits"], dim=2)
    outputs_sum["full_seq_output"]["pred_boxes"] = torch.cat(outputs_sum["full_seq_output"]["pred_boxes"], dim=2)
    targets_sum["full_seq_target"][0]["pt_boxes"] = torch.cat(targets_sum["full_seq_target"][0]["pt_boxes"], dim=0)
    targets_sum["full_seq_target"][0]["pt_labels"] = torch.cat(targets_sum["full_seq_target"][0]["pt_labels"], dim=0)
    targets_sum["full_seq_target"][0]["pt_tracking_mask"] = torch.cat(targets_sum["full_seq_target"][0]["pt_tracking_mask"], dim=0)
    targets_sum["full_seq_target"][0]["ptq_update_mask"] = torch.cat(targets_sum["full_seq_target"][0]["ptq_update_mask"], dim=0)
    targets_sum["full_seq_target"][0]["query_frames"] = torch.cat(targets_sum["full_seq_target"][0]["query_frames"], dim=0)
    return outputs_sum, targets_sum


def main(args):
    # utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k == "dataset_file":
            continue
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, _, _ = build_model_main(args)
    if not args.eval_checkpoint:
        args.eval_checkpoint = os.path.join(args.output_dir, 'checkpoint.pth')
    checkpoint = torch.load(args.eval_checkpoint, map_location='cpu')
    epoch = checkpoint['epoch']
    model.to(device)
    model.eval()
    print("loading checkpoint from {}".format(args.eval_checkpoint))
    use_ema_model = getattr(args, 'use_ema', False)
    if not use_ema_model:
        model.load_state_dict(checkpoint['model'])
    else:
        model_state_dict = {}
        for name, value in checkpoint["ema_model"].items():
            model_state_dict[name.replace("module.", "")] = value
        model.load_state_dict(model_state_dict)

    # dataset
    dataset_val = build_dataset(image_set=args.dataset_file, args=args)  # val

    # Evaluator
    evaluator = Evaluator(args.output_dir, args.eval_checkpoint.split("/")[-1].split(".")[0])

    # Run evaluate.
    metrics = {}
    for data_id in tqdm(range(len(dataset_val))):
        if not ("strided" in args.dataset_file):  # first.
            (samples, targets, seq_name) = dataset_val[data_id]
            H, W = 256, 256
            if isinstance(samples, (list, torch.Tensor)):
                from util.misc import nested_temporal_tensor_from_tensor_list
                samples = nested_temporal_tensor_from_tensor_list(samples[None, ...])
            samples = samples.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            with torch.no_grad():
                outputs, targets = model(samples, [targets])
        else:  # strided
            def fuse_bidir_outputs(outputs_, outputs_inv_, targets_):
                # fuse the bidirectional outputs, to mimic the offline tracker's results.
                outputs_forward_backward = {
                    "full_seq_output": {
                        "dn_meta": None,
                    }
                }
                mask_forward = targets_["full_seq_target"][0]["pt_tracking_mask"].float().transpose(0,1)  # T N
                mask_backward = 1 - mask_forward
                outputs_forward_backward["full_seq_output"]["pred_logits"] = \
                    (outputs_["full_seq_output"]["pred_logits"] * mask_forward[None, ..., None]) + (outputs_inv_["full_seq_output"]["pred_logits"].flip(1) * mask_backward[None, ..., None])
                outputs_forward_backward["full_seq_output"]["pred_boxes"] = \
                    (outputs_["full_seq_output"]["pred_boxes"] * mask_forward[None, ..., None]) + (outputs_inv_["full_seq_output"]["pred_boxes"].flip(1) * mask_backward[None, ..., None])
                return outputs_forward_backward
            
            strided_data_list, strided_inv_data_list = dataset_val[data_id]
            targets_bi = []
            outputs_bi = []
            for ((samples, targets_, seq_name_), (samples_inv, targets_inv_, seq_name_inv_)) in zip(strided_data_list, strided_inv_data_list):
                H, W = 256, 256
                if isinstance(samples, (list, torch.Tensor)):
                    from util.misc import nested_temporal_tensor_from_tensor_list
                    samples = nested_temporal_tensor_from_tensor_list(samples[None, ...])
                    samples_inv = nested_temporal_tensor_from_tensor_list(samples_inv[None, ...])
                samples = samples.to(device)
                samples_inv = samples_inv.to(device)
                targets_ = {k: v.to(device) for k, v in targets_.items()}
                targets_inv_ = {k: v.to(device) for k, v in targets_inv_.items()}
                with torch.no_grad():
                    print("processing: ", seq_name_)
                    outputs_, targets_ = model(samples, [targets_])
                    print("processing: ", seq_name_inv_)
                    outputs_inv_, targets_inv_ = model(samples_inv, [targets_inv_])
                    outputs_bi_ = fuse_bidir_outputs(outputs_, outputs_inv_, targets_)
                    targets_bi_ = targets_
                    # targets_bi_["full_seq_target"][0]["query_frames"] *= 0  # offline tracker. but still need to know which frame is the reference frame.
                    outputs_bi.append(outputs_bi_)
                    targets_bi.append(targets_bi_)
            seq_name = seq_name_
            outputs, targets = summarize_all_points(outputs_bi, targets_bi)
            outputs["window_output_list"] = []
            targets["window_target_list"] = []
        outputs = outputs["full_seq_output"]
        outputs["pred_boxes"] = outputs["pred_boxes"][0].permute(1, 0, 2)
        threshold_occ = 0.5
        pred_occluded = outputs["pred_logits"][0].permute(1, 0, 2)[..., 1].sigmoid() > threshold_occ
        outputs["pred_labels"] = pred_occluded
        evaluator.compute_metrics(metrics, seq_name, targets["full_seq_target"][0], outputs, args.dataset_file, H, W)
    
    # Saving the evaluation results to a .log file
    evaluate_result = metrics.pop("avg")
    result_file = os.path.join(args.output_dir, f"test/{args.eval_checkpoint.split('/')[-1].split('.')[0]}_{args.dataset_file}.log")
    print(f"Dumping eval results to {result_file}.")
    with open(result_file, "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}:\n {dict2string(value)} \n")
        all_result  = "\n========= All Results \n"
        all_result += dict2string(evaluate_result)
        main_result = "\n========= Main Results \n" + f"AJ     : {evaluate_result['average_jaccard']} \nDelta_x: {evaluate_result['average_pts_within_thresh']} \nOA     : {evaluate_result['occlusion_accuracy']}"
        print(main_result)
        f.write(all_result)
        f.write(main_result)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TAPTR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
