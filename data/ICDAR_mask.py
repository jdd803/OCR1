import cv2
import numpy as np


def generate_mask(im_dims, gt_mask):
    img = np.zeros(im_dims, np.uint8)
    area = np.split(gt_mask, len(gt_mask), axis=0)
    cv2.fillPoly(img, area, (1))
    cv2.imshow("new", img)
    return img


def get_roi_mask(roi, mask):
    return mask[roi[1]:roi[3], roi[0]:roi[2]]
