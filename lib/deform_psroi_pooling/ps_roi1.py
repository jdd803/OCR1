import tensorflow as tf
import numpy as np


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


def ps_roi(features, boxes, pool=True, offsets=None, k=3, feat_stride=8):
    # if_pool = 1 if pool else 0
    # if_pool = tf.convert_to_tensor(if_pool)
    if offsets is None:
        offsets1 = tf.convert_to_tensor(0)
    else:
        offsets1 = offsets
    pooled_response = tf.py_function(_ps_roi, [features, boxes, pool, offsets1, k, feat_stride], tf.float32)
    pooled_fea = tf.convert_to_tensor(value=pooled_response)
    return pooled_fea


def _ps_roi(features, boxes, pool, offsets, k, feat_stride):
    '''
    Implement the PSROI pooling
    :param features: (1,h,w,k^2*(c+1) or (1,h,w,K^2*4)
    :param boxes: (n,5)->(0,x1,y1,x2,y2)
    :param pool: control whether ave_pool the features
    :param offsets: (n,k*k*(c+1),2)
    :param k: output size,(x,y)
    :return:(n,k,k,c+1)
    '''
    fea_shape = np.shape(features)
    num_classes = fea_shape[-1] / (k * k)  # channels
    num_classes = tf.cast(num_classes, tf.int32)
    depth = num_classes    # (c+1)
    box_num = tf.shape(boxes)[0]
    feat_stride1 = tf.cast(feat_stride, tf.float32)
    boxes1 = tf.concat((boxes[:, 1:3], boxes[:, 3:] + 1), axis=1)
    feature_boxes = np.round(boxes1 / feat_stride1)  # (n,4)
    top_left_point = np.hstack((feature_boxes[:, 0:2], feature_boxes[:, 0:2])).reshape((-1, 1, 4))
    boxes_part = np.zeros((top_left_point.shape[0], k * k, 4))  # (1, k^2,4)
    boxes_part += top_left_point   # (n,k*k,4)
    width = (feature_boxes[:, 2] - feature_boxes[:, 0])   # (n,1)
    height = (feature_boxes[:, 3] - feature_boxes[:, 1])   # (n,1)
    width = np.clip(width, a_min=1, a_max=None)
    height = np.clip(height, a_min=1, a_max=None)
    # width = max(width, 1)   # scale the too small roi to 1
    # height = max(height, 1)
    width = width / k
    height = height / k
    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)

    # split boxes
    shift_x = np.arange(0, k).reshape((1, -1))
    width = tf.reshape(width, (-1, 1))
    shift_x = np.matmul(width, shift_x)
    shift_y = np.arange(0, k).reshape((1, -1))
    height = tf.reshape(height, (-1, 1))
    shift_y = np.matmul(height, shift_y)
    for i in range(shift_x.shape[0]):
        shift_x1, shift_y1 = np.meshgrid(shift_x[i], shift_y[i])
        shifts = np.vstack((shift_x1.ravel(), shift_y1.ravel(),
                            shift_x1.ravel(), shift_y1.ravel())).transpose()
        boxes_part[i] += shifts  # (n, k*k, 4)
    boxes_part = np.transpose(boxes_part, (1, 2, 0))    # (k*k, 4, n)
    boxes_part[:, 2, :] = boxes_part[:, 0, :] + np.reshape(width, (-1,))
    boxes_part[:, 3, :] = boxes_part[:, 1, :] + np.reshape(height, (-1,))
    boxes_part = np.transpose(boxes_part, (2, 0, 1))  # (n, k*k, 4)
    boxes_part[:, :, 0:2] = np.floor(boxes_part[:, :, 0:2])
    boxes_part[:, :, 2:] = np.ceil(boxes_part[:, :, 2:]) - 1

    # add offsets to splitted boxes
    # add offsets to splitted boxes
    depth1 = tf.cast(depth, tf.int32)
    box_num1 = tf.cast(box_num, tf.int32)
    if offsets.numpy().size == 1:
        offsets0 = np.zeros((box_num1, k * k, depth1, 2))  # (n,k*k,2*(c+1),2)
    else:
        offsets0 = offsets  # (n,k*k,c,2)
    offsets1 = np.tile(offsets0, (1, 1, 1, 2))  # (n,k*k,c,4)
    offsets1 = np.transpose(offsets1, (0, 2, 1, 3))   # (c,k*k,4)
    boxes_part = np.expand_dims(boxes_part, axis=1)
    boxes_part = np.repeat(boxes_part, depth, axis=1)
    boxes_part += offsets1  # (c,k*k,4)
    boxes_part = np.reshape(boxes_part, (-1, 4))  # (n*c*k*k,4)

    # clip split boxes by feature' size
    temp00 = np.clip(boxes_part[..., 0], 0, fea_shape[2] - 1)
    temp11 = np.clip(boxes_part[..., 1], 0, fea_shape[1] - 1)
    temp22 = np.clip(boxes_part[..., 2], 0, fea_shape[2] - 1)
    temp33 = np.clip(boxes_part[..., 3], 0, fea_shape[1] - 1)
    boxes_k_offset = np.stack([temp00, temp11, temp22, temp33], axis=-1)    # (n*c*k*k,4)
    boxes_k_offset = np.reshape(boxes_k_offset, (int(box_num), int(depth), int(k*k), 4))   # (n,c,k*k,4)
    boxes_k_offset = np.transpose(boxes_k_offset, (0, 2, 1, 3))   # (n,k*k,c,4)

    pooled_response = tf.py_function(map_coordinates1, [features[0], boxes_k_offset, k, num_classes, pool],
                                     tf.float32)  # (depth,1)/(depth,h,w)

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
    boxes = np.reshape(boxes, (-1, 4))  # (n*k*k*c,4)
    boxes_num = np.shape(boxes)[0]  # n*k*k*c
    pooled_box = np.zeros((boxes_num,))  # (n*k*k*c)
    for i in range(boxes_num):
        if boxes[i][2] < boxes[i][0] or boxes[i][3] < boxes[i][1]:
            pooled_box[i] = 0
            continue
        box = boxes[i]
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        width = tf.cast(width, 'int32')
        height = tf.cast(height, 'int32')
        tp_lf = np.reshape(box[0:2], (1, 2))
        grid = np.meshgrid(np.array(range(width)), np.array(range(height)))
        grid = np.stack(grid, axis=-1)  # (h,w,2)
        grid = np.reshape(grid, (-1, 2))  # (n_points,2)
        coords = grid + tp_lf  # (n,2)
        n_coords = np.shape(coords)[0]

        coords_lt = np.floor(coords)
        coords_rb = np.ceil(coords)
        coords_lt = coords_lt.astype(np.int32)
        coords_rb = coords_rb.astype(np.int32)
        coords_lb = np.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
        coords_rt = np.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)   # (n_points,2)

        inputs1 = inputs[:, :, i % fea_depth]   # input's shape is (h,w,1)

        def _get_vals_by_coords(input, coords):
            indices = np.stack([
                np_flatten(coords[..., 0]), np_flatten(coords[..., 1])
            ], axis=-1)

            vals = tf.gather_nd(input, indices=indices)
            vals = tf.reshape(vals, (n_coords,))
            return vals  # (n_points)

        vals_lt = _get_vals_by_coords(inputs1, coords_lt)
        vals_rb = _get_vals_by_coords(inputs1, coords_rb)
        vals_lb = _get_vals_by_coords(inputs1, coords_lb)
        vals_rt = _get_vals_by_coords(inputs1, coords_rt)  # (n_points)

        # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  # (depth,n_points,2)
        coords_offset_lt = coords - coords_lt  # (n_points,2)
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (n_points)

        if pool:
            temp = np.mean(mapped_vals, axis=0)   # (1)
            pooled_box[i] = temp
        else:
            pooled_box[i] = np.mean(mapped_vals, axis=0)   # (1)
    pooled_box = np.reshape(pooled_box, (rois_num, k, k, num_classes))
    pooled_box = np.transpose(pooled_box, (0, 3, 1, 2))
    return pooled_box  # (n,c,k,k)
