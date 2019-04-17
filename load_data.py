import os
import sys
import numpy as np
import cv2
import h5py
import config

train_img_path = './dataset/dataset/ch4_training_images/'
train_gtbox_path = './dataset/dataset/ch4_training_localization_transcription_gt/'
train_mask_path = './dataset/dataset/ch4_training_localization_transcription_mask/'


def next_batch(batch_size, pos):
    images = {}
    gtboxes = {}
    masks = {}
    for i in range(batch_size):
        # read batch_size images
        img_name = 'img_' + str(i + 1 + pos) + '.jpg'
        img_path = train_img_path + img_name
        img = cv2.imread(img_path)
        images[str(i + 1 + pos)] = img

        # read batch_size gt_boxes
        gtbox_name = 'gt_img_' + str(i + 1 + pos) + '.txt'
        gtbox_path = train_gtbox_path + gtbox_name
        file = open(gtbox_path, 'r', encoding='utf-8-sig')
        gtbox = file.readlines()
        for j in range(0, len(gtbox)):
            gtbox[j] = gtbox[j].rstrip('\n')
            gtbox[j] = gtbox[j].split(',')
        file.close()
        gtboxes[str(i + 1 + pos)] = gtbox

        # read batch_size masks
        mask_name = 'gt_img_' + str(i + 1 + pos) + '.txt'
        mask_path = train_mask_path + mask_name
        file = open(mask_path, 'r', encoding='utf-8-sig')
        mask = file.readlines()
        for j in range(0, len(mask)):
            mask[j] = mask[j].rstrip('\n')
            mask[j] = mask[j].split(',')
        file.close()
        masks[str(i + 1 + pos)] = mask

        return images, gtboxes, masks


if __name__ == "__main__":
    images, gtboxes, masks = next_batch(1, 0)
