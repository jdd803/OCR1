#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:35:33 2017
@author: Kevin Liang (modifications)
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from config import config as cfg
# from .nms.gpu_nms import gpu_nms
# from .nms.cpu_nms import cpu_nms
from lib.nms.py_cpu_nms import py_cpu_nms


def nms(dets, thresh, force_cpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.TRAIN.USE_GPU_NMS and not force_cpu:
        # return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        pass
    else:

    	return py_cpu_nms(dets, thresh)
