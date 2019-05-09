# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import tensorflow as tf


# @tf.function
def py_cpu_nms_v2(i, keep1, dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    xx1 = np.maximum(x1[keep1[i]], x1[:])
    yy1 = np.maximum(y1[keep1[i]], y1[:])
    xx2 = np.minimum(x2[keep1[i]], x2[:])
    yy2 = np.minimum(y2[keep1[i]], y2[:])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[keep1[i]] + areas[:] - inter)

    inds = np.where((ovr >= thresh) & (ovr < 1))[0]
    # keep = [val for val in inds if val not in keep1]
    keep = np.setdiff1d(inds, keep1)

    return keep
