import os,sys

path1 = '..\dataset\ch4_training_localization_transcription_mask'
path2 = '..\dataset\ch4_training_localization_transcription_gt'

for i in range(1000):
    gt_name = '\gt_img_' + str(i + 1) + '.txt'
    mask_name = '\gt_img_' + str(i + 1) + '.txt'
    mask_path = path1 + mask_name
    gt_path = path2 + gt_name
    mask_file = open(mask_path,mode='r',encoding='UTF-8-sig')
    gt_file = open(gt_path,mode='w',encoding='UTF-8-sig')
    lines = mask_file.readlines()
    for line in lines:
        xy = line.split(',')
        x = xy[0::2]
        y = xy[1::2]
        min_x = min(x[0:-1])
        min_y = min(y)
        max_x = max(x[0:-1])
        max_y = max(y)

        box = [min_x,min_y,max_x,max_y]
        for box_i in box:
            gt_file.write(str(box_i)+',')
        gt_file.write('1\n')

    mask_file.close()
    gt_file.close()