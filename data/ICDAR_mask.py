import cv2
import numpy as np


def generate_mask(im_dims, gt_mask):
    img = np.zeros(im_dims, np.uint8)
    area = np.split(gt_mask, len(gt_mask), axis=0)
    cv2.fillPoly(img, area, (1))
    cv2.imshow("new", img)
    return img


def box_mask(box, mask):
    return mask[:, :]
