# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterions: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None, tb_logger=None, grad_acc_steps=1):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    model.train()
    for criterion_name, criterion in criterions.items():
        if not (criterion is None):
            criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = '\nEpoch: [{}]'.format(epoch)
    print_freq = 10
    print_freq = print_freq * grad_acc_steps
    optimizer.zero_grad()  # to prevent the non-zero initial gradient.

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # ptc_initial = [{k: v.to(device) for k, v in t.items()} for t in ptc_initial]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs, targets = model(samples, targets)
            loss_dict = {}
            if args.activate_det_seg:
                loss_dict.update(criterions["det_seg_criterion"](outputs, targets))
            if args.activate_point_tracking:
                loss_dict.update(criterions["pt_criterion"](outputs, targets))
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses = losses / grad_acc_steps
            losses.backward()
            if (grad_acc_steps > 1 and _cnt > 0 and _cnt % grad_acc_steps == 0) or (grad_acc_steps == 1):
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
                if args.onecyclelr:
                    lr_scheduler.step()

        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if _cnt % print_freq == 0:
            loss_logged = {}
            for loss_name, loss_value in metric_logger.meters.items():
                if len(loss_value.deque) > 0 and loss_value.max > 0:
                    loss_logged[loss_name] = loss_value.median
            if args.rank == 0:
                tb_logger.add_scalars(loss_logged, print_freq//grad_acc_steps)
        _cnt += 1

    optimizer.zero_grad()
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat