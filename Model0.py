import tensorflow as tf
import numpy as np
from config import config as cfg
from tensorflow.contrib import slim
# from tensorflow.contrib.slim.nets import resnet_v2
from ResNet50 import resnet_v2
import tensorflow.contrib.layers as layers
from lib.bbox.bbox_transform import bbox_transform_inv,clip_boxes
from lib.Inception.Inception_text import inception_text_layer
from lib.RPN.rpn import rpn
from lib.ROI_proposal.roi_proposal import roi_proposal
from lib.deform_psroi_pooling.layer import PS_roi_offset
from lib.nms.nms_wrapper import nms
from lib.nms.nms_v2 import py_cpu_nms_v2 as nms1



def image_mean_subtraction(images,means=[123.68,116.78,103.94]):
    '''
    image normalization by minus images' mean
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('The number of means must equal to the number of channels')
    channels = tf.split(value=images,num_or_size_splits=num_channels,axis=3)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(values=channels,axis=3)

def upsample_image(images,y1,y2):
    return tf.image.resize_bilinear(images,size=[y1,y2])

def model_part1(images,weight_decay=1e-5,is_training=True,gt_boxes = None):
    '''
    define the model_part1,we used slim's implement of resnet,return rois and ps_score_map and bbox_shift_map
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    '''
    images = image_mean_subtraction(images)

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        logits,endpoints = resnet_v2.resnet_v2_50(images,is_training=is_training,scope='resnet_v2_50',
                                                  output_stride=16)

    with tf.variable_scope('feature_fusion',values=[endpoints.values]):
        batch_norm_params = {
            'decay':0.997,
            'epsilon':1e-5,
            'scale':True,
            'is_training':is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn = tf.nn.relu,
                            normalizer_fn = slim.batch_norm,
                            normalizer_params = batch_norm_params,
                            ):
            #features extracted from resnet
            fea = [endpoints["model/resnet_v2_50/block2"],endpoints["model/resnet_v2_50/block3"],
                   endpoints["model/resnet_v2_50/block4"]]
            for i in range(3):
                print('Shape of f_{} {}'.format(i,fea[i].shape))
            # The depth of the fused features
            num_outputs = 1024

            #fuse the features
            f_stage3 = slim.conv2d(fea[0],num_outputs=num_outputs,kernel_size=[1,1])
            print("Shape of stage3's feature:" + str(f_stage3.shape))
            f_stage4 = upsample_image(fea[1],tf.shape(f_stage3)[1],tf.shape(f_stage3)[2])
            print("Shape of upsampled stage4's feature:" + str(f_stage4.shape))
            fused_fea1 = tf.add(f_stage3,f_stage4,name='fused_feature1')
            print("fused_fea1's shape:" + str(fused_fea1.shape))

            f_stage5_1 = slim.conv2d(fea[2],num_outputs=num_outputs,kernel_size=[1,1])
            f_stage5_2 = upsample_image(f_stage5_1,tf.shape(f_stage3)[1],tf.shape(f_stage3)[2])
            print("Shape of upsampled stage5's feature:" + str(f_stage5_2.shape))
            fused_fea2 = tf.add(f_stage3,f_stage5_2,name='fused_feature2')
            print("fused_fea2's shape:" + str(fused_fea2.shape))

            #Inception_text
            inception_out1 = inception_text_layer(fused_fea1)
            inception_out2 = inception_text_layer(fused_fea2)


            #RPN
            eval_mode = False if is_training else True
            im_info = tf.shape(images)
            rpn_net = rpn(featureMaps=inception_out1,gt_boxes=gt_boxes,im_dims=im_info,_feat_stride=8,eval_mode=eval_mode)
            rpn_cls_loss = rpn_net.get_rpn_cls_loss()
            rpn_bbox_loss = rpn_net.get_rpn_bbox_loss()
            roi = roi_proposal(rpn_net,gt_boxes,im_info,eval_mode)  #(n,5)
            rois = roi.get_rois()  #(n,5)

            # compute the score_map for cls_seg and bbox_regression
            ps_score_map = slim.conv2d(inception_out2,4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,(1,1),activation_fn=None)  #inside_outside ps_score map
            bbox_shift = slim.conv2d(inception_out2,4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,(1,1),activation_fn=None)   #rfcn_bbox_pred

            #upsample the score_maps to image's size
            return rois,ps_score_map,bbox_shift,rpn_cls_loss,rpn_bbox_loss

def model_part2_1(roi,ps_score_map):
    '''
    deal per_roi on ps_score_map
    :param roi: per_roi(1,5)
    :param ps_score_map:
    :return:mask score,classify score
    '''
    # rois = roi.get_rois()
    ps_roi_layer1 = PS_roi_offset(ps_score_map, roi,
                                  pool_size=cfg.network.PSROI_BINS,pool=False,
                                  feat_stride=8).call(ps_score_map)  #(2*2,h,w)
    # roi_num = tf.shape(rois)[0]
    ps_roi_layer1_shape = tf.shape(ps_roi_layer1)
    mask_cls = tf.reshape(ps_roi_layer1,(cfg.dataset.NUM_CLASSES+1,2,ps_roi_layer1_shape[-2],ps_roi_layer1_shape[-1]))
    #(2,2,h,w)(inside/outside,class,h,w)

    #mask_cls = tf.reshape(ps_roi_layer1,(cfg.network.PSROI_BINS*cfg.network.PSROI_BINS,2,2,-1))  #(49,2,2,n_points)


    mask_cls = tf.transpose(mask_cls,(1,2,3,0))   #(class,h,w,inside/outside)(2,h,w,2)
    mask = tf.nn.softmax(mask_cls,axis=-1)   #(class,h,w,inside/outside)(2,h,w,2)

    cls_max = tf.reduce_max(mask_cls,axis=-1)   #(class,h,w)
    cls_max_r = tf.reshape(cls_max,(cfg.dataset.NUM_CLASSES+1,-1))  #(class,h*w)
    # cls_max_t = tf.transpose(cls_max,(0,2,1,3))   #(2,49,n_points)--(classes,bins,pixel_max)
    # cls_max_r = tf.reshape(cls_max_t,(2,-1))   #(2,49*n_points)--(classes,bins*pixel_max)
    cls_ave = tf.reduce_mean(cls_max_r,axis=-1)   #(2,)

    cls = tf.nn.softmax(cls_ave)
    cls_result = tf.arg_max(cls,dimension=0)    #(n,)
    cls_score = tf.argmax(cls,axis=-1)

    mask_result = mask[cls_result]  #(h,w,inside/outside)(h,w,2)

    return cls, cls_result, cls_score, mask_result   #(2,1,(h,w,2))

def model_part2_2(roi,bbox_shift):
    '''
    get bbox shift
    :param roi:
    :param bbox_shift:
    :return:
    '''
    # rois = roi.get_rois()
    ps_roi_layer2 = PS_roi_offset(bbox_shift, roi,
                                  pool_size=cfg.network.PSROI_BINS, pool=True,
                                  feat_stride=8).call(bbox_shift)  #(4,k,k)
    # roi_num = tf.shape(rois)[0]
    bbox = tf.reshape(ps_roi_layer2, (4,cfg.network.PSROI_BINS*cfg.network.PSROI_BINS))  # (4,k*k)
    bbox = tf.reduce_mean(bbox,axis=1)  #(4)
    return bbox

result = {}

def model_part2(imdims,rois,ps_score_map,bbox_shift):
    '''
    compute rois' cls_score and mask_score,apply nms on rois by their scores
    :param imdims:
    :param rois:
    :param ps_score_map:
    :param bbox_shift:
    :return:
    '''
    # compute orignal roi and roi added with offset

    # def cond1(i, rois):
    #     return i < roi_num
    #
    # def body1(i, rois):
    #     roi = rois[i]
    #     offsets = model_part2_2(rois[i], bbox_shift)  # (4)
    #     offsets = tf.reshape(offsets, (-1, 4))
    #
    #     # compute the boxes with offsets
    #     proposals = tf.py_func(bbox_transform_inv, [rois[i][-4:], offsets], tf.float32)  # (1,4)
    #     proposals = tf.py_func(clip_boxes, [proposals, imdims], tf.float32)  # (1,4)
    #     rois = tf.concat((rois, proposals), axis=0)
    #     return i + 1, rois
    #
    # i, rois = tf.while_loop(cond1, body1, (0, rois))

    offsets = tf.map_fn(lambda x:model_part2_2(x,bbox_shift),rois)  #(n,4)
    proposals = tf.py_func(bbox_transform_inv,[rois[:,-4:],offsets],tf.float32) #(n,4)
    proposals = tf.py_func(clip_boxes,[proposals,imdims],tf.float32)    #(n,4)
    rois = tf.concat((rois,proposals),axis=0)   #(2n,4)

    '''
    for i in range(roi_num):
        i += 1
        offsets = model_part2_2(rois[i], bbox_shift)
        offsets = tf.reshape(offsets, (-1, 4))
        proposals = tf.py_func(bbox_transform_inv, [rois[i][-4:], offsets], tf.float32)
        proposals = tf.py_func(clip_boxes, [proposals, imdims], tf.float32)  # (1,4)
        rois = tf.concat((rois, proposals), axis=0)
    '''
    roi_num = tf.convert_to_tensor(cfg.TRAIN.RPN_POST_NMS_TOP_N*2)

    # deal per roi
    def cond2(i,rois):
        return i<roi_num
    def body2(i,rois):
        bbox = model_part2_2(rois[i],bbox_shift)
        cls, cls_result,cls_score, mask_result = model_part2_1(rois[i], ps_score_map)
        j = tf.py_func(_add_to_dict,[i,cls,cls_result,cls_score,mask_result,rois[i],bbox],tf.int32)
        # global result
        # result[str(i)] = {'cls':cls, 'cls_result':cls_result, 'cls_score':cls_score, 'mask':mask_result, 'roi':rois[i], 'shift':bbox}
        return tf.add(i, 1),rois
    i1,rois = tf.while_loop(cond2, body2, (0,rois))
    '''
    for i in range(roi_num):
        bbox = model_part2_2(rois[i], bbox_shift)
        cls, cls_result, cls_score, mask_result = model_part2_1(rois[i], ps_score_map)
        result[str(i)] = {'cls': cls, 'cls_result': cls_result, 'cls_score': cls_score, 'mask': mask_result,
                          'roi': rois[i], 'shift': bbox}
        pass
    '''
    return result

def _add_to_dict(i,cls, cls_result,cls_score, mask_result,roi,shift):
    global result
    result[str(i)] = {'cls':cls, 'cls_result':cls_result, 'cls_score':cls_score, 'mask':mask_result, 'roi':roi, 'shift':shift}
    return i

def model_part3(results):
    '''
    apply nms on the rois
    :param results:
    :return:
    '''
    per_roi = results['0']['roi']
    score = results['0']['cls_score']
    roi_item = tf.concat((per_roi, score), axis=-1)
    roi = tf.reshape(roi_item,(1,-1))
    for i in range(1,len(results)):
        per_roi = results[str(i)]['roi']
        score = results[str(i)]['cls_score']
        roi_item = tf.concat((per_roi,score),axis=-1)
        roi_item = tf.reshape(roi_item,(1,-1))
        roi = tf.concat((roi,roi_item),axis=0)


    # apply nms on rois to get boxes with highest scores
    keep = tf.py_func(nms,[roi,0.3],tf.int32)   #the keeped boxes' orders
    for i in keep:
        keep1 = tf.py_func(nms1,[i,keep,roi,0.5],tf.int32)
        box_up_num = len(keep1) + 1

        # get an unsupressed box
        mask_i = results[str(i)]['mask']
        roi_i = results[str(i)]['roi']
        score_i = results[str(i)]['cls_score']
        mask_i,roi_i = tf.py_func(mask_transform,[mask_i,roi_i],[tf.float32,tf.float32])
        mask_i = tf.pad(mask_i,[[roi_i[-4],0],[roi_i[-3],0],[0,0]])
        mask_i *= score_i

        # fuse the unsupressed box with supressed boxes by weighted averaging
        for j in keep1:
            mask_j = results[str(j)]['mask']
            roi_j = results[str(j)]['roi']
            score_j = results[str(j)]['cls_score']
            #mask_j,roi_j = mask_transform(mask_j,roi_j)
            mask_j, roi_j = tf.py_func(mask_transform, [mask_j, roi_j], [tf.float32, tf.float32])
            rb = tf.maximum(roi_i[-2:],roi_j[-2:])
            pad_rb = rb - roi_j[-2:]
            mask_j = tf.pad(mask_j,[[roi_j[-4],pad_rb[0]],[roi_j[-3],pad_rb[1]],[0,0]])
            mask_i = mask_i + mask_j*score_j
        mask_i /= box_up_num
        mask = mask_i[roi_i[-4]:,roi_i[-3]:,:]
        results[str(i)]['mask'] = mask

    return keep

def mask_transform(mask,roi):
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
    mask1 = np.reshape(mask,(cfg.network.PSROI_BINS,cfg.network.PSROI_BINS,-1,2))
    mask2 = np.split(mask1,cfg.network.PSROI_BINS,axis=2)
    mask3 = np.stack(mask2,axis=1)
    mask4 = np.reshape(mask3,(height,width,2))
    return mask4,feature_boxes