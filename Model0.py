import tensorflow as tf
import numpy as np
from config import config as cfg
from ResNet50.resnet50v1 import ResNet50
from lib.bbox.bbox_transform import bbox_transform_inv_tf, clip_boxes_tf
from lib.Inception.Inception_text1 import InceptionTextLayer
from lib.RPN.rpn1 import RPN
from lib.ROI_proposal.roi_proposal1 import RoiProposal
from lib.deform_psroi_pooling.layer0 import PsRoiOffset, PsRoiOffset1
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
    # channels = np.split(value=images, num_or_size_splits=num_channels, axis=-1)
    channels = np.split(images, num_channels, axis=-1)
    for i in range(num_channels):
        channels[i] = channels[i].astype(np.float32)
        channels[i] -= means[i]
    return np.concatenate(channels, axis=-1)


def upsample_image(images, y1, y2):
    return tf.image.resize(images, size=[y1, y2])


class ModelPart1(tf.keras.Model):
    def __init__(self, img_dims, is_training=True):
        super(ModelPart1, self).__init__()
        self.training = is_training
        self.img_dims = img_dims
        self.feat_stride = 8
        self.num_classes = cfg.dataset.NUM_CLASSES
        self.resnet50 = ResNet50(weights=None, input_shape=None, pooling=None)
        # self.resnet50.load_weights("./path/resnet50_weights/resnet50_weights_tf.h5", by_name=True)
        self.num_outputs = 1024
        self.conv1 = tf.keras.layers.Conv2D(filters=self.num_outputs, kernel_size=(1, 1),
                                            activation='relu', kernel_initializer='TruncatedNormal')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3, name='fea_stage3')
        self.add1 = tf.keras.layers.Add()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.num_outputs, kernel_size=(1, 1),
                                            activation='relu', kernel_initializer='TruncatedNormal')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=3)
        self.add2 = tf.keras.layers.Add()

        self.inception_text_layer = InceptionTextLayer()
        self.rpn = RPN(self.img_dims, self.feat_stride, self.training)
        self.roi_proposal = RoiProposal(self.feat_stride, self.img_dims, not self.training)

        self.conv3 = tf.keras.layers.Conv2D(filters=4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,
                                            kernel_size=(1, 1))
        self.conv4 = tf.keras.layers.Conv2D(filters=4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,
                                            kernel_size=(1, 1))
        # self.resnet50.load_weights("./path/resnet50_weights/resnet50_weights_tf.h5", by_name=True)

    def call(self, inputs):
        fea3, fea4, fea5 = self.resnet50(inputs=inputs)
        # print('Shape of f_{} {}'.format(3, fea3.shape))
        # print('Shape of f_{} {}'.format(4, fea4.shape))
        # print('Shape of f_{} {}'.format(5, fea5.shape))

        f_stage3 = self.conv1(fea3)
        f_stage3 = self.bn1(f_stage3)
        # print("Shape of stage3's feature:" + str(f_stage3.shape))
        f_stage4 = upsample_image(fea4, f_stage3.shape[1], f_stage3.shape[2])
        # print("Shape of upsampled stage4's feature:" + str(f_stage4.shape))
        fused_fea1 = self.add1([f_stage3, f_stage4])
        # print("fused_fea1's shape:" + str(fused_fea1.shape))

        f_stage5_1 = self.conv2(fea5)
        f_stage5_1 = self.bn2(f_stage5_1)
        f_stage5_2 = upsample_image(f_stage5_1, f_stage3.shape[1], f_stage3.shape[2])
        # print("Shape of upsampled stage5's feature:" + str(f_stage5_2.shape))
        fused_fea2 = self.add2([f_stage3, f_stage5_2])
        # print("fused_fea2's shape:" + str(fused_fea2.shape))

        # Inception_text
        inception_out1 = self.inception_text_layer(fused_fea1)
        inception_out2 = self.inception_text_layer(fused_fea2)

        # RPN
        rpn_cls_score, rpn_bbox_pred = self.rpn(inception_out1)
        rois = self.roi_proposal([rpn_cls_score, rpn_bbox_pred])   # (n,5)

        ps_score_map = self.conv3(inception_out2)
        bbox_shift = self.conv4(inception_out2)

        return rois, ps_score_map, bbox_shift, rpn_cls_score, rpn_bbox_pred


class ModelPart21(tf.keras.Model):
    def __init__(self):
        super(ModelPart21, self).__init__()
        self.pool_size = cfg.network.PSROI_BINS
        self.pool = False
        self.feat_stride = 8
        self.ps_roi_pooling = PsRoiOffset(self.pool_size, self.feat_stride)
        pass

    def call(self, inputs):
        roi, ps_score_map = inputs
        ps_roi_layer1 = self.ps_roi_pooling([ps_score_map, roi])  # (n,2*2,k,k)
        ps_roi_layer1_shape = tf.shape(input=ps_roi_layer1)
        mask_cls = tf.reshape(ps_roi_layer1, (ps_roi_layer1_shape[0], cfg.dataset.NUM_CLASSES + 1, 2,
                                              ps_roi_layer1_shape[2], ps_roi_layer1_shape[3]))  # (n,2,2,k,k)
        mask_cls = tf.transpose(a=mask_cls, perm=(0, 2, 3, 4, 1))  # (n,class,h,w,inside/outside)(n,2,k,k,2)
        mask = tf.nn.softmax(mask_cls, axis=-1)  # (n,class,h,w,inside/outside)(n,2,k,k,2)

        cls_max = tf.reduce_max(input_tensor=mask_cls, axis=-1)  # (n,class,k,k)
        cls_max_r = tf.reshape(cls_max, (ps_roi_layer1_shape[0],
                                         cfg.dataset.NUM_CLASSES + 1, -1))  # (n,class,k*k)
        cls_ave = tf.reduce_mean(input_tensor=cls_max_r, axis=-1)  # (n,class)

        cls = tf.nn.softmax(cls_ave, axis=-1)       # (n,c)
        cls_result = tf.argmax(input=cls, axis=-1)  # (n,)
        cls_score = tf.reduce_max(cls, axis=-1)     # (n,)

        # (n,2,k,k,2)->(n,k,k,2)
        mask_ind = tf.reshape(tf.range(ps_roi_layer1_shape[0]), (-1, 1))
        cls_result = tf.cast(tf.reshape(cls_result, (-1, 1)), tf.int32)
        indices = tf.concat((mask_ind, cls_result), axis=-1)
        mask_result = tf.gather_nd(mask, indices=indices)

        return cls, cls_result, cls_score, mask_result  # ((n,c),n,n,(n,k,k,2))


class ModelPart22(tf.keras.Model):
    def __init__(self):
        super(ModelPart22, self).__init__()
        self.pool_size = cfg.network.PSROI_BINS
        self.feat_stride = 8
        self.ps_roi_pooling = PsRoiOffset1(self.pool_size, self.feat_stride)

    def call(self, inputs):
        roi, bbox_shift = inputs
        ps_roi_layer2 = self.ps_roi_pooling([bbox_shift, roi])  # (n,4,k,k)
        bbox = tf.reshape(ps_roi_layer2, (-1, 4, cfg.network.PSROI_BINS * cfg.network.PSROI_BINS))  # (n,4,k*k)
        bbox = tf.reduce_mean(input_tensor=bbox, axis=-1)  # (n,4)
        return bbox


class ModelPart2(tf.keras.Model):
    def __init__(self, img_dims):
        super(ModelPart2, self).__init__()
        self.img_dims = img_dims
        self.model22 = ModelPart22()
        self.model21 = ModelPart21()
        pass

    def call(self, inputs, training=None, mask=None):
        rois, ps_score_map, bbox_shift = inputs
        print("rois num:")
        print(str(rois.shape[0]))
        offsets = self.model22([rois, bbox_shift])
        proposals = bbox_transform_inv_tf(rois[:, -4:], offsets)
        proposals = clip_boxes_tf(proposals, self.img_dims)  # (n, 4)
        zero = tf.zeros((proposals.shape[0], 1))
        proposals = tf.concat((zero, proposals), axis=-1)
        # rois = tf.concat((rois, proposals), axis=0)  # (2n, 4)
        #
        # bbox = self.model22([rois, bbox_shift])
        cls, cls_result, cls_score, mask_result = self.model21([proposals, ps_score_map])

        return cls, cls_result, cls_score, mask_result, rois, offsets


class ModelPart31(tf.keras.layers.Layer):
    def __init__(self):
        super(ModelPart31, self).__init__()

    def call(self, inputs, **kwargs):
        rois, cls_score = inputs
        per_roi = rois[:, 1:]
        score = tf.reshape(cls_score, (-1, 1))

        # apply nms on rois to get boxes with highest scores
        score = tf.reshape(cls_score, (-1,))
        keep = tf.image.non_max_suppression(per_roi, score, max_output_size=300, iou_threshold=0.3)
        return keep


def model_part3_2(roi, keep, i, mask, score):
    ''' upsample and compute the mask '''
    roi = tf.round(roi)
    width = roi[keep[i]][3] - roi[keep[i]][1] + 1
    height = roi[keep[i]][4] - roi[keep[i]][2] + 1
    mask1 = tf.image.resize(mask[keep[i]], (height, width))
    ovr_threshold = tf.constant(0.5)
    keep1 = tf.py_function(nms1, [i, keep, roi[:, 1:], ovr_threshold], tf.int32)
    box_up_num = len(keep1) + 1

    mask_i = mask1  # (h,w,2)
    roi_i = roi[keep[i]]  # (n,5)
    score_i = score[keep[i]]  # (n,)
    mask_i = tf.pad(tensor=mask_i, paddings=[[roi_i[-4], 0], [roi_i[-3], 0], [0, 0]])
    mask_i *= score_i

    # fuse the unsupressed box with supressed boxes by weighted averaging
    for j in keep1:
        roi_j = roi[j]
        width_j = roi_j[3] - roi_j[1] + 1
        height_j = roi_j[4] - roi_j[2] + 1
        score_j = score[j]
        mask_j = tf.image.resize(mask[j], (height_j, width_j))
        rb = tf.maximum(roi_i[-2:], roi_j[-2:])
        pad_rb = rb - roi_j[-2:]
        mask_j = tf.pad(tensor=mask_j, paddings=[[roi_j[-4], pad_rb[0]], [roi_j[-3], pad_rb[1]], [0, 0]])
        mask_i = mask_i + mask_j * score_j

    mask_i /= box_up_num
    mask = mask_i[roi_i[-4]:, roi_i[-3]:, :]
    return mask


class MyModel(tf.keras.Model):
    def __init__(self, img_dims, training=True):
        super(MyModel, self).__init__()
        self.img_dims = img_dims
        self.training = training
        self.model1 = ModelPart1(img_dims, training)
        self.model2 = ModelPart2(img_dims)
        self.model3 = ModelPart31()

    def call(self, input):
        """Run the model."""
        rois, ps_score_map, bbox_shift, rpn_cls_score, rpn_bbox_pred = self.model1(input)
        cls, cls_result, cls_score, mask_result, rois, bbox = self.model2([rois, ps_score_map, bbox_shift])
        keep = self.model3([rois, cls_score])
        result = [cls, cls_result, cls_score, mask_result, rois, bbox]
        return result, rpn_cls_score, rpn_bbox_pred, keep

