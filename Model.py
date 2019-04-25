import tensorflow as tf
import numpy as np
from config import config as cfg
from ResNet50.resnet50 import ResNet50
from lib.bbox.bbox_transform import bbox_transform_inv_tf, clip_boxes_tf
from lib.Inception.Inception_text import inception_text_layer
from lib.RPN.rpn import rpn
from lib.ROI_proposal.roi_proposal import roi_proposal
from lib.deform_psroi_pooling.layer import PS_roi_offset
from lib.nms.nms_wrapper import nms
from lib.nms.nms_v2 import py_cpu_nms_v2 as nms1


def image_mean_subtraction(images, means=[123.68,116.78,103.94]):
    '''
    image normalization by minus images' mean
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.shape[-1]

    img_np = np.reshape(images, (-1, num_channels))
    img_mean = np.mean(img_np, axis=0)
    means = img_mean
    if len(means) != num_channels:
        raise ValueError('The number of means must equal to the number of channels')
    channels = tf.split(value=images, num_or_size_splits=num_channels, axis=-1)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(values=channels, axis=-1)


def upsample_image(images, y1, y2):
    return tf.image.resize(images, size=[y1, y2])
    # return tf.compat.v1.image.resize_bilinear(images, size=[y1, y2])


def model_part1(images, is_training=True):
    '''
    define the model_part1,we used slim's implement of resnet,return rois and ps_score_map and bbox_shift_map
    :param images:
    :param is_training:
    :return:
    '''
    # images = image_mean_subtraction(images)

    fea3, fea4, fea5 = ResNet50(include_top=False, input_tensor=images, input_shape=(224, 224, 3))
    print('Shape of f_{} {}'.format(3, fea3.shape))
    print('Shape of f_{} {}'.format(4, fea4.shape))
    print('Shape of f_{} {}'.format(5, fea5.shape))
    num_outputs = 1024
    f_stage3 = tf.keras.layers.Conv2D(filters=num_outputs, kernel_size=(1, 1),
                                      activation='relu', kernel_initializer='TruncatedNormal')(fea3)
    f_stage3 = tf.keras.layers.BatchNormalization(axis=3, name='fea_stage3')(f_stage3)
    print("Shape of stage3's feature:" + str(f_stage3.shape))
    f_stage4 = upsample_image(fea4, f_stage3.shape[1], f_stage3.shape[2])
    print("Shape of upsampled stage4's feature:" + str(f_stage4.shape))
    fused_fea1 = tf.add(f_stage3, f_stage4, name='fused_feature1')
    print("fused_fea1's shape:" + str(fused_fea1.shape))

    f_stage5_1 = tf.keras.layers.Conv2D(filters=num_outputs, kernel_size=(1, 1),
                                        activation='relu', kernel_initializer='TruncatedNormal')(fea5)
    f_stage5_1 = tf.keras.layers.BatchNormalization(axis=3)(f_stage5_1)
    f_stage5_2 = upsample_image(f_stage5_1, f_stage3.shape[1], f_stage3.shape[2])
    print("Shape of upsampled stage5's feature:" + str(f_stage5_2.shape))
    fused_fea2 = tf.add(f_stage3, f_stage5_2, name='fused_feature2')
    print("fused_fea2's shape:" + str(fused_fea2.shape))

    # Inception_text
    inception_out1 = inception_text_layer(fused_fea1)
    inception_out2 = inception_text_layer(fused_fea2)

    # RPN
    eval_mode = False if is_training else True
    im_info = tf.shape(input=images)
    rpn_net = rpn(featureMaps=inception_out1, im_dims=im_info, feat_stride=8, eval_mode=eval_mode)
    rpn_cls_score = rpn_net.get_rpn_cls_score()
    rpn_bbox_pred = rpn_net.get_rpn_bbox_pred()
    roi = roi_proposal(rpn_net, im_info, eval_mode)  # (n,5)
    rois = roi.get_rois()  # (n,5)

    # compute the score_map for cls_seg and bbox_regression
    ps_score_map = tf.keras.layers.Conv2D(filters=4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,
                                          kernel_size=(1, 1))(inception_out2)
    bbox_shift = tf.keras.layers.Conv2D(filters=4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,
                                        kernel_size=(1, 1))(inception_out2)

    # upsample the score_maps to image's size
    return rois, ps_score_map, bbox_shift, rpn_cls_score, rpn_bbox_pred


def model_part2_1(roi, ps_score_map):
    '''
    deal per_roi on ps_score_map
    :param roi: per_roi(1,5)
    :param ps_score_map:
    :return:mask score,classify score
    '''
    # rois = roi.get_rois()
    ps_roi_layer1 = PS_roi_offset(ps_score_map, roi,
                                  pool_size=cfg.network.PSROI_BINS, pool=False,
                                  feat_stride=8).call(ps_score_map)  # (2*2,h,w)
    # roi_num = tf.shape(rois)[0]
    ps_roi_layer1_shape = tf.shape(input=ps_roi_layer1)
    mask_cls = tf.reshape(ps_roi_layer1, (cfg.dataset.NUM_CLASSES+1, 2,
                                          ps_roi_layer1_shape[-2], ps_roi_layer1_shape[-1]))
    # (2,2,h,w)(inside/outside,class,h,w)

    # mask_cls = tf.reshape(ps_roi_layer1,(cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,2,2,-1))  #(49,2,2,n_points)

    mask_cls = tf.transpose(a=mask_cls, perm=(1, 2, 3, 0))   # (class,h,w,inside/outside)(2,h,w,2)
    mask = tf.nn.softmax(mask_cls, axis=-1)   # (class,h,w,inside/outside)(2,h,w,2)

    cls_max = tf.reduce_max(input_tensor=mask_cls, axis=-1)   # (class,h,w)
    cls_max_r = tf.reshape(cls_max, (cfg.dataset.NUM_CLASSES+1, -1))  # (class,h*w)
    # cls_max_t = tf.transpose(cls_max,(0,2,1,3))   #(2,49,n_points)--(classes,bins,pixel_max)
    # cls_max_r = tf.reshape(cls_max_t,(2,-1))   #(2,49*n_points)--(classes,bins*pixel_max)
    cls_ave = tf.reduce_mean(input_tensor=cls_max_r, axis=-1)   # (2,)

    cls = tf.nn.softmax(cls_ave)
    cls_result = tf.argmax(input=cls, axis=0)    # (n,)
    cls_score = tf.reduce_max(cls, axis=-1)

    mask_result = mask[cls_result]  # (h,w,inside/outside)(h,w,2)
    mask_result = tf.expand_dims(mask_result, axis=0)

    return cls, cls_result, cls_score, mask_result   # (2,1,(h,w,2))


def model_part2_11(roi, ps_score_map):
    '''
    deal per_roi on ps_score_map
    :param roi: per_roi(n,5)
    :param ps_score_map:
    :return:mask score,classify score
    '''
    # rois = roi.get_rois()
    ps_roi_layer1 = PS_roi_offset(ps_score_map, roi,
                                  pool_size=cfg.network.PSROI_BINS, pool=False,
                                  feat_stride=8).call(ps_score_map)  # (2*2,h,w)
    # roi_num = tf.shape(rois)[0]
    ps_roi_layer1_shape = tf.shape(input=ps_roi_layer1)
    mask_cls = tf.reshape(ps_roi_layer1, (cfg.dataset.NUM_CLASSES+1, 2,
                                          ps_roi_layer1_shape[-2], ps_roi_layer1_shape[-1]))
    # (2,2,h,w)(inside/outside,class,h,w)

    # mask_cls = tf.reshape(ps_roi_layer1,(cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,2,2,-1))  #(49,2,2,n_points)

    mask_cls = tf.transpose(a=mask_cls, perm=(1, 2, 3, 0))   # (class,h,w,inside/outside)(2,h,w,2)
    mask = tf.nn.softmax(mask_cls, axis=-1)   # (class,h,w,inside/outside)(2,h,w,2)

    cls_max = tf.reduce_max(input_tensor=mask_cls, axis=-1)   # (class,h,w)
    cls_max_r = tf.reshape(cls_max, (cfg.dataset.NUM_CLASSES+1, -1))  # (class,h*w)
    # cls_max_t = tf.transpose(cls_max,(0,2,1,3))   #(2,49,n_points)--(classes,bins,pixel_max)
    # cls_max_r = tf.reshape(cls_max_t,(2,-1))   #(2,49*n_points)--(classes,bins*pixel_max)
    cls_ave = tf.reduce_mean(input_tensor=cls_max_r, axis=-1)   # (2,)

    cls = tf.nn.softmax(cls_ave)
    cls_result = tf.argmax(input=cls, axis=0)    # (n,)
    cls_score = tf.reduce_max(cls, axis=-1)

    mask_result = mask[cls_result]  # (h,w,inside/outside)(h,w,2)
    mask_result = tf.expand_dims(mask_result, axis=0)

    return cls, cls_result, cls_score, mask_result   # (2,1,(1,h,w,2))


def model_part2_2(roi, bbox_shift):
    '''
    get bbox shift
    :param roi:
    :param bbox_shift:
    :return:
    '''
    # rois = roi.get_rois()
    ps_roi_layer2 = PS_roi_offset(bbox_shift, roi,
                                  pool_size=cfg.network.PSROI_BINS, pool=True,
                                  feat_stride=8).call(bbox_shift)  # (4,k,k)
    # roi_num = tf.shape(rois)[0]
    bbox = tf.reshape(ps_roi_layer2, (4, cfg.network.PSROI_BINS*cfg.network.PSROI_BINS))  # (4,k*k)
    bbox = tf.reduce_mean(input_tensor=bbox, axis=1)  # (4)
    return bbox


result = {}


def model_part2(imdims, rois, ps_score_map, bbox_shift):
    '''
    compute rois' cls_score and mask_score,apply nms on rois by their scores
    :param imdims:
    :param rois:
    :param ps_score_map:
    :param bbox_shift:
    :return:
    '''
    offsets = tf.map_fn(lambda x: model_part2_2(x, bbox_shift), rois)  # (n, 4)
    proposals = bbox_transform_inv_tf(rois[:, -4:], offsets)  # (n, 4)
    proposals = clip_boxes_tf(proposals, imdims)  # (n, 4)
    zero = tf.zeros((proposals.shape[0], 1))
    proposals = tf.concat((zero, proposals), axis=-1)
    rois = tf.concat((rois, proposals), axis=0)   # (2n, 4)

    bbox = model_part2_2(rois[0], bbox_shift)
    bbox = tf.reshape(bbox, (1, -1))
    cls, cls_result, cls_score, mask_result = model_part2_1(rois[0], ps_score_map)
    cls = tf.reshape(cls, (1, -1))
    cls_result = tf.reshape(cls_result, (1,))
    cls_score = tf.reshape(cls_score, (1,))

    for i in tf.range(1, rois.shape[0]):
        bbox1 = model_part2_2(rois[i], bbox_shift)
        bbox1 = tf.reshape(bbox1, (1, -1))
        cls1, cls_result1, cls_score1, mask_result1 = model_part2_1(rois[i], ps_score_map)
        cls1 = tf.reshape(cls1, (1, -1))
        cls = tf.concat((cls, cls1), axis=0)
        cls_result1 = tf.reshape(cls_result1, (1,))
        cls_score1 = tf.reshape(cls_score1, (1,))
        cls_result = tf.concat((cls_result, cls_result1), axis=0)
        cls_score = tf.concat((cls_score, cls_score1), axis=0)
        mask_result = tf.concat((mask_result, mask_result1), axis=0)
        bbox = tf.concat((bbox, bbox1), axis=0)

    return cls, cls_result, cls_score, mask_result, rois, bbox


def model_part3(results):
    '''
    apply nms on the rois
    :param results:
    :return:
    '''
    per_roi = results[4]
    per_roi = per_roi[:, 1:]
    score = tf.reshape(results[2], (-1, 1))
    roi = tf.concat((per_roi, score), axis=-1)

    # apply nms on rois to get boxes with highest scores
    # keep = tf.py_function(nms, [roi, 0.3], tf.int32)
    # keep = nms(roi, 0.3)
    score = tf.reshape(results[2], (-1,))
    keep = tf.image.non_max_suppression(per_roi, score, max_output_size=20, iou_threshold=0.3)

    ovr_threshold = tf.convert_to_tensor(0.5)
    for i in keep:
        keep1 = tf.numpy_function(nms1, [i, keep, roi, ovr_threshold], tf.int32)
        box_up_num = len(keep1) + 1

        # get an unsupressed box
        mask_i = results[3][i]  # (h,w,2)
        roi_i = results[4][i]   # (n,4)
        score_i = results[2][i]  # (n,)
        mask_i, roi_i = tf.py_function(mask_transform, [mask_i, roi_i], [tf.float32, tf.float32])
        mask_i = tf.pad(tensor=mask_i, paddings=[[roi_i[-4], 0], [roi_i[-3], 0], [0, 0]])
        mask_i *= score_i

        # fuse the unsupressed box with supressed boxes by weighted averaging
        for j in keep1:
            mask_j = results[3][j]
            roi_j = results[4][j]
            score_j = results[3][j]
            mask_j, roi_j = tf.py_function(mask_transform, [mask_j, roi_j], [tf.float32, tf.float32])
            rb = tf.maximum(roi_i[-2:], roi_j[-2:])
            pad_rb = rb - roi_j[-2:]
            mask_j = tf.pad(tensor=mask_j, paddings=[[roi_j[-4], pad_rb[0]], [roi_j[-3], pad_rb[1]], [0, 0]])
            mask_i = mask_i + mask_j*score_j
        mask_i /= box_up_num
        mask = mask_i[roi_i[-4]:, roi_i[-3]:, :]
        results[3][i] = mask

    # compute the postive boxes and positive boxes
    # compute the boxes' inside_weights and out_side weights
    roi_item = result[4][keep[0]]  # (4,)
    offset_item = result[5][keep[0]]  # (4,)
    cls_item = result[0][keep[0]]
    rois = tf.reshape(roi_item, (1, -1))
    offset = tf.reshape(offset_item, (1, -1))
    cls = tf.reshape(cls_item, (1, -1))
    roi_mask_item = result[3][keep[0]]
    roi_mask = tf.reshape(roi_mask_item, (1, 5, 5, 2))
    for i in range(1, len(keep)):
        roi_item = result[4][keep[i]]
        roi_item = tf.reshape(roi_item, (1, -1))
        rois = tf.concat((rois, roi_item), axis=0)  # (n,4)
        offset_item = result[5][keep[i]]
        offset_item = tf.reshape(offset_item, (1, -1))
        offset = tf.concat((offset, offset_item), axis=0)  # (n,4)
        cls_item = result[0][keep[i]]
        cls_item = tf.reshape(cls_item, (1, -1))
        cls = tf.concat((cls, cls_item), axis=0)  # (n,k+1)
        roi_mask_item = result[3][keep[i]]
        roi_mask_item = tf.reshape(roi_mask_item, (1, 5, 5, 2))
        roi_mask = tf.concat((roi_mask, roi_mask_item), axis=0)

    return rois, cls, offset, roi_mask


def model_part31(rois, cls_score):
    ''' apply nms on the rois   '''
    per_roi = rois
    per_roi = per_roi[:, 1:]
    score = tf.reshape(cls_score, (-1, 1))
    roi = tf.concat((per_roi, score), axis=-1)

    # apply nms on rois to get boxes with highest scores
    # keep = tf.py_function(nms, [roi, 0.3], tf.int32)
    # keep = nms(roi, 0.3)
    score = tf.reshape(cls_score, (-1,))
    keep = tf.image.non_max_suppression(per_roi, score, max_output_size=300, iou_threshold=0.3)
    return keep


def model_part3_2(roi, keep, i, mask, score):
    ''' upsample and compute the mask '''
    width = roi[keep[i]][2] - roi[keep[i]][0] + 1
    height = roi[keep[i]][3] - roi[keep[i]][1] + 1
    mask1 = tf.image.resize(mask[keep[i]], (height, width))
    ovr_threshold = tf.convert_to_tensor(0.5)
    keep1 = tf.numpy_function(nms1, [i, keep, roi, ovr_threshold], tf.int32)
    box_up_num = len(keep1) + 1

    mask_i = mask1  # (h,w,2)
    roi_i = roi[keep[i]]  # (n,4)
    score_i = score[keep[i]]  # (n,)
    mask_i = tf.pad(tensor=mask_i, paddings=[[roi_i[-4], 0], [roi_i[-3], 0], [0, 0]])
    mask_i *= score_i

    # fuse the unsupressed box with supressed boxes by weighted averaging
    for j in keep1:
        roi_j = roi[j]
        width_j = roi_j[2] - roi_j[0] + 1
        height_j = roi_j[3] - roi_j[1] + 1
        score_j = score[j]
        mask_j = tf.image.resize(mask[j], (height_j, width_j))
        rb = tf.maximum(roi_i[-2:], roi_j[-2:])
        pad_rb = rb - roi_j[-2:]
        mask_j = tf.pad(tensor=mask_j, paddings=[[roi_j[-4], pad_rb[0]], [roi_j[-3], pad_rb[1]], [0, 0]])
        mask_i = mask_i + mask_j * score_j

    mask_i /= box_up_num
    mask = mask_i[roi_i[-4]:, roi_i[-3]:, :]
    return mask


def mask_transform(mask, roi):
    '''
    (k*k,n_points,2)->(height,width,2)
    :param mask: (k*k,n_points,2)
    :param roi:
    :return:
    '''
    feature_boxes = np.round(roi / 8)
    feature_boxes[:, -2:] -= 1
    width = (feature_boxes[:, 3] - feature_boxes[:, 1] + 1)  # (n,1)
    height = (feature_boxes[:, 4] - feature_boxes[:, 2] + 1)  # (n,1)
    mask1 = np.reshape(mask, (cfg.network.PSROI_BINS, cfg.network.PSROI_BINS, -1, 2))
    mask2 = np.split(mask1, cfg.network.PSROI_BINS, axis=2)
    mask3 = np.stack(mask2, axis=1)
    mask4 = np.reshape(mask3, (height, width, 2))
    return mask4, feature_boxes


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, input):
        """Run the model."""
        roi, ps_score, bbox_shift, rpn_cls_score, rpn_bbox_pred = model_part1(images=input, is_training=True)
        imdims = input.shape[0:2]
        result = model_part2(imdims=imdims, rois=roi, ps_score_map=ps_score, bbox_shift=bbox_shift)
        keep = model_part31(result[4], result[2])
        return result, rpn_cls_score, rpn_bbox_pred, keep

