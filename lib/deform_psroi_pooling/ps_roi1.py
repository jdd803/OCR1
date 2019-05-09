import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


def tf_flatten(inputs):
    """Flatten tensor"""
    return tf.reshape(inputs, [-1])


def tf_repeat(inputs, repeats, axis=0):
    assert len(inputs.get_shape()) == 1

    a = tf.expand_dims(inputs, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def np_flatten(inputs):
    return np.reshape(inputs, [-1])


def np_repeat(inputs, repeats, axis=0):
    assert len(np.shape(inputs)) == 1

    a = np.expand_dims(inputs, -1)
    a = np.tile(a, [1, repeats])
    a = np_flatten(a)
    return a


def get_vals_by_coords(input, coords):
    n_coords = tf.shape(coords)[0]
    indices = tf.stack([
        tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
    ], axis=-1)

    vals = tf.gather_nd(input, indices=indices)
    vals = tf.reshape(vals, (n_coords,))
    return vals  # (n_points)


# PSROI pooling
def ps_roi(features, boxes, k=3, feat_stride=8):
    '''
    Implement the PSROI pooling
    :param features: (1,h,w,k^2*(c+1) or (1,h,w,K^2*4)
    :param boxes: (n,5)->(0,x1,y1,x2,y2)
    :param pool: control whether ave_pool the features
    :param offsets: (n,k*k*(c+1),2)
    :param k: output size,(x,y)
    :return:(n,k,k,c+1)
    '''
    fea_shape = tf.shape(features)
    num_classes = fea_shape[-1] / (k * k)  # channels
    num_classes = tf.cast(num_classes, tf.int32)
    box_num = boxes.shape[0]
    feat_stride1 = tf.cast(feat_stride, tf.float32)
    boxes1 = tf.concat((boxes[:, 1:3], boxes[:, 3:] + 1), axis=1)
    feature_boxes = tf.round(boxes1 / feat_stride1)  # (n,4)
    top_left_point = tf.stack((feature_boxes[:, 0:2], feature_boxes[:, 0:2]), axis=1)
    top_left_point = tf.reshape(top_left_point, (-1, 1, 4))
    boxes_part = tf.zeros((top_left_point.shape[0], k * k, 4))  # (1, k^2,4)
    boxes_part = boxes_part + top_left_point   # (n,k*k,4)
    width = (feature_boxes[:, 2] - feature_boxes[:, 0])   # (n,1)
    height = (feature_boxes[:, 3] - feature_boxes[:, 1])   # (n,1)
    width = tf.clip_by_value(width, clip_value_min=1, clip_value_max=tf.cast(fea_shape[2], tf.float32))
    height = tf.clip_by_value(height, clip_value_min=1, clip_value_max=tf.cast(fea_shape[1], tf.float32))
    width = width / k
    height = height / k

    # split boxes
    shift_x = np.reshape(np.arange(0, k, dtype=np.float32), (1, -1))
    width = np.reshape(width, (-1, 1))
    shift_x = np.matmul(width, shift_x)
    shift_y = np.reshape(np.arange(0, k, dtype=np.float32), (1, -1))
    height = np.reshape(height, (-1, 1))
    shift_y = np.matmul(height, shift_y)

    shift_x1, shift_y1 = np.meshgrid(shift_x[0], shift_y[0])
    shifts = np.stack((np_flatten(shift_x1), np_flatten(shift_y1),
                       np_flatten(shift_x1), np_flatten(shift_y1)), axis=1)
    shifts = np.expand_dims(shifts, axis=0)
    for i in range(1, shift_x.shape[0]):
        shift_x1, shift_y1 = np.meshgrid(shift_x[i], shift_y[i])
        shift = np.stack((np_flatten(shift_x1), np_flatten(shift_y1),
                          np_flatten(shift_x1), np_flatten(shift_y1)), axis=1)
        shift = np.expand_dims(shift, axis=0)
        shifts = np.concatenate((shifts, shift), axis=0)

    boxes_part = boxes_part + shifts
    boxes_part = tf.transpose(boxes_part, (1, 2, 0))    # (k*k, 4, n)
    boxes_r = boxes_part[:, 0, :] + tf.reshape(width, (-1,))
    boxes_b = boxes_part[:, 1, :] + tf.reshape(height, (-1,))
    boxes_r = tf.math.ceil(boxes_r) - 1
    boxes_b = tf.math.ceil(boxes_b) - 1
    boxes_l = tf.floor(boxes_part[:, 0, :])
    boxes_t = tf.floor(boxes_part[:, 1, :])
    boxes_part = tf.stack((boxes_l, boxes_t, boxes_r, boxes_b), axis=1)
    boxes_part = tf.transpose(boxes_part, (2, 0, 1))  # (n, k*k, 4)

    # add offsets to splitted boxes
    boxes_part = tf.expand_dims(boxes_part, axis=1)
    boxes_part = tf.tile(boxes_part, [1, num_classes, 1, 1])
    boxes_part = tf.reshape(boxes_part, (-1, 4))

    # clip split boxes by feature' size
    temp00 = tf.clip_by_value(boxes_part[..., 0], 0., tf.cast(fea_shape[2], tf.float32) - 1)
    temp11 = tf.clip_by_value(boxes_part[..., 1], 0., tf.cast(fea_shape[1], tf.float32) - 1)
    temp22 = tf.clip_by_value(boxes_part[..., 2], 0., tf.cast(fea_shape[2], tf.float32) - 1)
    temp33 = tf.clip_by_value(boxes_part[..., 3], 0., tf.cast(fea_shape[1], tf.float32) - 1)
    boxes_k_offset = tf.stack([temp00, temp11, temp22, temp33], axis=-1)  # (n*c*k*k,4)
    boxes_k_offset = tf.reshape(boxes_k_offset, (box_num, num_classes, k * k, 4))  # (n,c,k*k,4)
    boxes_k_offset = tf.transpose(boxes_k_offset, (0, 2, 1, 3))  # (n,k*k,c,4)

    pooled_response = map_coordinates(features[0], boxes_k_offset, k, num_classes, 1)  # (depth,1)/(depth,h,w)

    return pooled_response  # (depth,k,k)/(depth,height,width)


def ps_roi1(features, boxes, offsets, k=3, feat_stride=8):
    '''
    Implement the PSROI pooling
    :param features: (1,h,w,k^2*(c+1) or (1,h,w,K^2*4)
    :param boxes: (n,5)->(0,x1,y1,x2,y2)
    :param pool: control whether ave_pool the features
    :param offsets: (n,k*k*(c+1),2)
    :param k: output size,(x,y)
    :return:(n,k,k,c+1)
    '''
    fea_shape = tf.shape(features)
    num_classes = fea_shape[-1] / (k * k)  # channels
    num_classes = tf.cast(num_classes, tf.int32)
    box_num = boxes.shape[0]
    feat_stride1 = tf.cast(feat_stride, tf.float32)
    boxes1 = tf.concat((boxes[:, 1:3], boxes[:, 3:] + 1), axis=1)
    feature_boxes = tf.round(boxes1 / feat_stride1)  # (n,4)
    top_left_point = tf.stack((feature_boxes[:, 0:2], feature_boxes[:, 0:2]), axis=1)
    top_left_point = tf.reshape(top_left_point, (-1, 1, 4))
    boxes_part = tf.zeros((top_left_point.shape[0], k * k, 4))  # (1, k^2,4)
    boxes_part = boxes_part + top_left_point   # (n,k*k,4)
    width = (feature_boxes[:, 2] - feature_boxes[:, 0])   # (n,1)
    height = (feature_boxes[:, 3] - feature_boxes[:, 1])   # (n,1)
    width = tf.clip_by_value(width, clip_value_min=1, clip_value_max=tf.cast(fea_shape[2], tf.float32))
    height = tf.clip_by_value(height, clip_value_min=1, clip_value_max=tf.cast(fea_shape[1], tf.float32))
    width = width / k
    height = height / k

    # split boxes
    shift_x = tf.reshape(tf.range(0, k, dtype=tf.float32), (1, -1))
    width = tf.reshape(width, (-1, 1))
    shift_x = tf.matmul(width, shift_x)
    shift_y = tf.reshape(tf.range(0, k, dtype=tf.float32), (1, -1))
    height = tf.reshape(height, (-1, 1))
    shift_y = tf.matmul(height, shift_y)

    shift_x1, shift_y1 = tf.meshgrid(shift_x[0], shift_y[0])
    shifts = tf.stack((tf_flatten(shift_x1), tf_flatten(shift_y1),
                       tf_flatten(shift_x1), tf_flatten(shift_y1)), axis=1)
    shifts = tf.expand_dims(shifts, axis=0)
    for i in range(1, shift_x.shape[0]):
        shift_x1, shift_y1 = tf.meshgrid(shift_x[i], shift_y[i])
        shift = tf.stack((tf_flatten(shift_x1), tf_flatten(shift_y1),
                          tf_flatten(shift_x1), tf_flatten(shift_y1)), axis=1)
        shift = tf.expand_dims(shift, axis=0)
        shifts = tf.concat((shifts, shift), axis=0)

    boxes_part = boxes_part + shifts
    boxes_part = np.transpose(boxes_part, (1, 2, 0))  # (k*k, 4, n)
    boxes_r = boxes_part[:, 0, :] + tf.reshape(width, (-1,))
    boxes_b = boxes_part[:, 1, :] + tf.reshape(height, (-1,))
    boxes_r = tf.math.ceil(boxes_r) - 1
    boxes_b = tf.math.ceil(boxes_b) - 1
    boxes_l = tf.floor(boxes_part[:, 0, :])
    boxes_t = tf.floor(boxes_part[:, 1, :])
    boxes_part = tf.stack((boxes_l, boxes_t, boxes_r, boxes_b), axis=1)
    boxes_part = tf.transpose(boxes_part, (2, 0, 1))  # (n, k*k, 4)

    # add offsets to splitted boxes
    offsets1 = tf.tile(offsets, (1, 1, 1, 2))  # (n,k*k,c,4)
    offsets1 = tf.transpose(offsets1, (0, 2, 1, 3))   # (c,k*k,4)
    boxes_part = tf.expand_dims(boxes_part, axis=1)
    boxes_part = tf.tile(boxes_part, [1, num_classes, 1, 1])
    boxes_part += offsets1  # (c,k*k,4)
    boxes_part = tf.reshape(boxes_part, (-1, 4))  # (n*c*k*k,4)

    # clip split boxes by feature' size
    temp00 = tf.clip_by_value(boxes_part[..., 0], 0., tf.cast(fea_shape[2], tf.float32) - 1)
    temp11 = tf.clip_by_value(boxes_part[..., 1], 0., tf.cast(fea_shape[1], tf.float32) - 1)
    temp22 = tf.clip_by_value(boxes_part[..., 2], 0., tf.cast(fea_shape[2], tf.float32) - 1)
    temp33 = tf.clip_by_value(boxes_part[..., 3], 0., tf.cast(fea_shape[1], tf.float32) - 1)
    boxes_k_offset = tf.stack([temp00, temp11, temp22, temp33], axis=-1)  # (n*c*k*k,4)
    boxes_k_offset = tf.reshape(boxes_k_offset, (box_num, num_classes, k * k, 4))  # (n,c,k*k,4)
    boxes_k_offset = tf.transpose(boxes_k_offset, (0, 2, 1, 3))  # (n,k*k,c,4)

    pooled_response = map_coordinates(features[0], boxes_k_offset, k, num_classes, 1)  # (depth,1)/(depth,h,w)

    return pooled_response  # (depth,k,k)/(depth,height,width)


def map_coordinates1(inputs, boxes, k, num_classes, pool):
    '''
    Get values in the boxes
    :param inputs: feature map (h,w,k*k*2(c+1) or (h,w,K*k*4)
    :param boxes: (n,k*k,depth,4)(x1,y1,x2,y2) May be fraction
    :param num_classes:
    :param pool: whether ave_pool the features
    :return: pooled_box:(depth,k,k)
    '''
    rois_num = boxes.shape[0]
    fea_depth = inputs.shape[-1]
    boxes = tf.reshape(boxes, (-1, 4))  # (n*k*k*c,4)
    boxes_num = boxes.shape[0]  # n*k*k*c

    with tf.device('/cpu:0'):
        box = boxes[0]
        # if box[0] > box[2] or box[1] > box[3]:
        #     pooled_box0 = tf.reshape(0, (-1,))
        #     pooled_box = tf.cast(pooled_box0, tf.float32)
        # else:
        width = tf.floor(box[2] - box[0] + 1)
        height = tf.floor(box[3] - box[1] + 1)
        width = tf.cast(width, 'int32')
        height = tf.cast(height, 'int32')
        tp_lf = tf.reshape(box[0:2], (1, 2))
        grid = tf.meshgrid(tf.range(width), tf.range(height))
        grid = tf.stack(grid, axis=-1)  # (h,w,2)
        grid = tf.reshape(grid, (-1, 2))  # (n_points,2)
        grid = tf.cast(grid, tf.float32)
        coords = grid + tp_lf  # (n,2)
        n_coords = coords.shape[0]

        coords_lt = tf.floor(coords)
        coords_rb = tf.math.ceil(coords)
        coords_lt = tf.cast(coords_lt, tf.int32)
        coords_rb = tf.cast(coords_rb, tf.int32)
        coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
        coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)  # (n_points,2)

        inputs1 = inputs[:, :, 0]  # input's shape is (h,w,1)

        vals_lt = get_vals_by_coords(inputs1, coords_lt)
        vals_rb = get_vals_by_coords(inputs1, coords_rb)
        vals_lb = get_vals_by_coords(inputs1, coords_lb)
        vals_rt = get_vals_by_coords(inputs1, coords_rt)  # (n_points)

        coords_lt = tf.cast(coords_lt, tf.float32)
        # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  # (depth,n_points,2)
        coords_offset_lt = coords - coords_lt  # (n_points,2)
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (n_points)

        temp = tf.reduce_mean(mapped_vals, axis=0)  # (1)
        temp = tf.reshape(temp, (-1,))
        pooled_box = temp

        for i in range(1, boxes_num):
            box = boxes[i]
            # if box[0] > box[2] or box[1] > box[3]:
            #     pooled_box0 = tf.reshape(0, (-1, ))
            #     pooled_box0 = tf.cast(pooled_box0, tf.float32)
            #     pooled_box = tf.concat((pooled_box, pooled_box0), axis=0)
            #     continue

            width = tf.floor(box[2] - box[0] + 1)
            height = tf.floor(box[3] - box[1] + 1)
            width = tf.cast(width, 'int32')
            height = tf.cast(height, 'int32')
            tp_lf = tf.reshape(box[0:2], (1, 2))
            grid = tf.meshgrid(tf.range(width), tf.range(height))
            grid = tf.stack(grid, axis=-1)  # (h,w,2)
            grid = tf.reshape(grid, (-1, 2))  # (n_points,2)
            grid = tf.cast(grid, tf.float32)
            coords = grid + tp_lf  # (n,2)
            n_coords = coords.shape[0]

            coords_lt = tf.floor(coords)
            coords_rb = tf.math.ceil(coords)
            coords_lt = tf.cast(coords_lt, tf.int32)
            coords_rb = tf.cast(coords_rb, tf.int32)
            coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
            coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)   # (n_points,2)

            inputs1 = inputs[:, :, i % fea_depth]   # input's shape is (h,w,1)

            vals_lt = get_vals_by_coords(inputs1, coords_lt)
            vals_rb = get_vals_by_coords(inputs1, coords_rb)
            vals_lb = get_vals_by_coords(inputs1, coords_lb)
            vals_rt = get_vals_by_coords(inputs1, coords_rt)  # (n_points)

            coords_lt = tf.cast(coords_lt, tf.float32)
            # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  # (depth,n_points,2)
            coords_offset_lt = coords - coords_lt  # (n_points,2)
            vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
            vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
            mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (n_points)

            temp = tf.reduce_mean(mapped_vals, axis=0)   # (1)
            temp = tf.reshape(temp, (-1, ))
            pooled_box = tf.concat((pooled_box, temp), axis=0)

    pooled_box = tf.reshape(pooled_box, (rois_num, k, k, num_classes))
    pooled_box = tf.transpose(pooled_box, (0, 3, 1, 2))
    pooled_box = tf.cast(pooled_box, tf.float32)
    return pooled_box  # (n,c,k,k)


import objgraph
# new implement method
# 这里涉及到是高宽还是宽高的问题,要仔细考虑一下
def map_coordinates(inputs, boxes, k, num_classes, pool):
    '''
    Get values in the boxes
    :param inputs: feature map (h,w,k*k*2(c+1) or (h,w,K*k*4)
    :param boxes: (n,k*k,depth,4)(x1,y1,x2,y2) May be fraction
    :param num_classes:
    :param pool: whether ave_pool the features
    :return: pooled_box:(depth,k,k)
    '''
    with tf.device('/gpu:1'):
        rois_num = boxes.shape[0]
        fea_depth = inputs.shape[-1]    # k*k*c
        boxes = tf.reshape(boxes, (-1, 4))  # (n*k*k*c,4)
        boxes_num = boxes.shape[0]  # n*k*k*c

        sample_points = 5
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        width = width / sample_points   # (n*k*k*c, 1)
        height = height / sample_points
        shift_x = np.reshape(np.arange(0, sample_points + 1, dtype=np.float32), (1, -1))
        width = np.reshape(width, (-1, 1))
        shift_x = np.matmul(width, shift_x)  # (n*k*k*c, 5)
        shift_y = np.reshape(np.arange(0, sample_points + 1, dtype=np.float32), (1, -1))
        height = np.reshape(height, (-1, 1))
        shift_y = np.matmul(height, shift_y)

        shift_x1, shift_y1 = np.meshgrid(shift_x[0], shift_y[0])
        shifts = np.stack((np_flatten(shift_x1), np_flatten(shift_y1)), axis=1)
        shifts = np.expand_dims(shifts, axis=0)

        for i in range(1, shift_x.shape[0]):
            # print('----------------------')
            # objgraph.show_growth()  # show the growth of the objects
            shift_x1, shift_y1 = np.meshgrid(shift_x[i], shift_y[i])
            shift = np.stack((np_flatten(shift_x1), np_flatten(shift_y1)), axis=1)
            shift = np.expand_dims(shift, axis=0)
            shifts = np.concatenate((shifts, shift), axis=0)  # (n, k*k, 2)
            # shift_x1, shift_y1 = tf.meshgrid(shift_x[i], shift_y[i])
            # shift = tf.stack((tf_flatten(shift_x1), tf_flatten(shift_y1)), axis=1)
            # shift = tf.expand_dims(shift, axis=0)
            # shifts = tf.concat((shifts, shift), axis=0)  # (n, k*k, 2)
            # objgraph.show_growth()
        tp_lf = boxes[:, 0:2]
        tp_lf = tf.expand_dims(tp_lf, axis=1)
        coords = tp_lf + shifts                     # (n*k*k*c, 36, 2)
        coords = tf.reshape(coords, (-1, 2))        # (n*k*k*c*36, 2)
        inputs1 = tf.transpose(inputs, (2, 1, 0))   # (k*k*c, w, h)
        depth_inds = tf.reshape(tf.range(fea_depth), (-1, 1))
        depth_inds = tf.tile(depth_inds, (rois_num, (sample_points+1)*(sample_points+1)))
        depth_inds = tf_flatten(depth_inds)

        coords_lt = tf.floor(coords)
        coords_rb = tf.math.ceil(coords)
        coords_lt = tf.cast(coords_lt, tf.int32)
        coords_rb = tf.cast(coords_rb, tf.int32)
        coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
        coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)  # (n_points,2)

        def get_vals_by_coords1(input, coords):
            n_coords = tf.shape(coords)[0]
            indices = tf.stack([
                depth_inds, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
            ], axis=-1)

            vals = tf.gather_nd(input, indices=indices)
            vals = tf.reshape(vals, (n_coords,))
            return vals  # (n_points)

        vals_lt = get_vals_by_coords1(inputs1, coords_lt)
        vals_rb = get_vals_by_coords1(inputs1, coords_rb)
        vals_lb = get_vals_by_coords1(inputs1, coords_lb)
        vals_rt = get_vals_by_coords1(inputs1, coords_rt)  # (n_points)

        coords_lt = tf.cast(coords_lt, tf.float32)
        # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  # (depth,n_points,2)
        coords_offset_lt = coords - coords_lt  # (n_points,2)
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (n_points)
        mapped_vals = tf.reshape(mapped_vals, (-1, (sample_points+1)*(sample_points+1)))    # (n*k*k*c,36)
        temp = tf.reduce_mean(mapped_vals, axis=1)  # (n*k*k*c)

        pooled_box = tf.reshape(temp, (rois_num, k, k, num_classes))
        pooled_box = tf.transpose(pooled_box, (0, 3, 1, 2))
        pooled_box = tf.cast(pooled_box, tf.float32)
        return pooled_box  # (n,c,k,k)
