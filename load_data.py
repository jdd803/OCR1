import tensorflow as tf
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
            gtbox1 = map(int, gtbox[j])
            gtbox[j] = list(gtbox1)
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
            mask1 = map(int, mask[j][:8])
            mask2 = list(mask1)
            mask[j] = mask2
        file.close()
        masks[str(i + 1 + pos)] = mask

        return images, gtboxes, masks


def next_batch1(batch_size, pos):
    img_name = 'img_' + str(1 + pos) + '.jpg'
    img_path = train_img_path + img_name
    img = cv2.imread(img_path)
    img1 = tf.image.resize(img, (432, 768))
    ratio = 0.6
    images = np.expand_dims(img1, axis=0)

    gtbox_name = 'gt_img_' + str(1 + pos) + '.txt'
    gtbox_path = train_gtbox_path + gtbox_name
    file = open(gtbox_path, 'r', encoding='utf-8-sig')
    gtbox = file.readlines()
    for j in range(0, len(gtbox)):
        gtbox[j] = gtbox[j].rstrip('\n')
        gtbox[j] = gtbox[j].split(',')
        gtbox1 = map(int, gtbox[j])
        gtbox[j] = list(gtbox1)
    file.close()
    gtboxes = np.array(gtbox)
    batch_ind = np.zeros((gtboxes.shape[0], 1))
    gtboxes = np.concatenate((batch_ind, gtboxes), axis=-1)

    mask_name = 'gt_img_' + str(1 + pos) + '.txt'
    mask_path = train_mask_path + mask_name
    file = open(mask_path, 'r', encoding='utf-8-sig')
    mask = file.readlines()
    for j in range(0, len(mask)):
        mask[j] = mask[j].rstrip('\n')
        mask[j] = mask[j].split(',')
        mask1 = map(int, mask[j][:8])
        mask2 = list(mask1)
        mask[j] = mask2
    file.close()
    masks = np.array(mask)
    batch_ind = np.zeros((masks.shape[0], 1))
    masks = np.concatenate((batch_ind, masks), axis=-1)

    for i in range(1, batch_size):
        # read batch_size images
        img_name = 'img_' + str(i + 1 + pos) + '.jpg'
        img_path = train_img_path + img_name
        img = cv2.imread(img_path)
        img1 = tf.image.resize(img, (432, 768))
        img2 = np.expand_dims(img1, axis=0)
        images = np.concatenate((images, img2), axis=0)

        # read batch_size gt_boxes
        gtbox_name = 'gt_img_' + str(i + 1 + pos) + '.txt'
        gtbox_path = train_gtbox_path + gtbox_name
        file = open(gtbox_path, 'r', encoding='utf-8-sig')
        gtbox = file.readlines()
        for j in range(0, len(gtbox)):
            gtbox[j] = gtbox[j].rstrip('\n')
            gtbox[j] = gtbox[j].split(',')
            gtbox1 = map(int, gtbox[j])
            gtbox[j] = list(gtbox1)
        file.close()
        gtbox = np.array(gtbox)
        batch_ind = np.zeros((gtbox.shape[0], 1)) + i
        gtbox = np.concatenate((batch_ind, gtbox), axis=-1)
        gtboxes = np.concatenate((gtboxes, gtbox), axis=0)

        # read batch_size masks
        mask_name = 'gt_img_' + str(i + 1 + pos) + '.txt'
        mask_path = train_mask_path + mask_name
        file = open(mask_path, 'r', encoding='utf-8-sig')
        mask = file.readlines()
        for j in range(0, len(mask)):
            mask[j] = mask[j].rstrip('\n')
            mask[j] = mask[j].split(',')
            mask1 = map(int, mask[j][:8])
            mask2 = list(mask1)
            mask[j] = mask2
        file.close()
        mask = np.array(mask)
        batch_ind = np.zeros((mask.shape[0], 1)) + i
        mask = np.concatenate((batch_ind, mask), axis=-1)
        masks = np.concatenate((masks, mask), axis=0)

    gtboxes[:, 1:-1] = gtboxes[:, 1:-1]*ratio
    masks[:, 1:-1] = masks[:, 1:-1] * ratio
    return images, gtboxes, masks


if __name__ == "__main__":
    images, gtboxes, masks = next_batch(1, 0)
