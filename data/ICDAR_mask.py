import cv2
import numpy as np


def generate_mask(im_dims, gt_mask):
    img = np.zeros(im_dims, np.uint8)
    area = np.reshape(gt_mask, (-1, 4, 2))
    cv2.fillPoly(img, np.array(area, 'int32'), 1)
    return img


def get_roi_mask(roi, mask):
    return mask[roi[1]:roi[3], roi[0]:roi[2]]


if __name__ == "__main__":
    generate_mask((320, 320), np.array([[[0, 0], [20, 10], [30, 30], [10, 20]],
                                        [[60, 60], [80, 70], [90, 90], [70, 80]]]))
