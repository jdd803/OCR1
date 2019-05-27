import cv2
import numpy as np


def generate_mask(im_dims, gt_mask, batch_size):
    imgs = np.zeros(im_dims, np.uint8)
    gt0 = np.where(gt_mask[:, 0] == 0)
    mask0 = gt_mask[gt0, 1:]
    area = np.reshape(mask0, (-1, 4, 2))
    cv2.fillPoly(imgs, np.array(area, 'int32'), 1)
    imgs = np.expand_dims(imgs, axis=0)

    for i in range(1, batch_size):
        img = np.zeros(im_dims, np.uint8)
        gt = np.where(gt_mask[:, 0] == i)
        mask = gt_mask[gt, 1:]
        area = np.reshape(mask, (-1, 4, 2))
        cv2.fillPoly(img, np.array(area, 'int32'), 1)
        img = np.expand_dims(img, axis=0)
        imgs = np.concatenate((imgs, img), axis=0)

    return imgs


def get_roi_mask(roi, mask):
    return mask[roi[2]:roi[4], roi[1]:roi[3]]


if __name__ == "__main__":
    generate_mask((320, 320), np.array([[[0, 0], [20, 10], [30, 30], [10, 20]],
                                        [[60, 60], [80, 70], [90, 90], [70, 80]]]))
