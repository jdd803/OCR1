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


def ps_roi(features, boxes, pool = True, offsets = [], k = 3, feat_stride = 8):
    pooled_response = tf.py_func(_ps_roi,[features, boxes, pool, offsets, k, feat_stride],tf.float32)
    pooled_fea = tf.convert_to_tensor(pooled_response)
    return pooled_fea

def _ps_roi(features, boxes, pool, offsets, k, feat_stride):
    '''
    Implement the PSROI pooling
    :param features: (1,h,w,2*k^2*(c+1) or (1,h,w,2*K^2*4)
    :param boxes: (5)->(0,x1,y1,x2,y2)
    :param pool: control whether ave_pool the features
    :param offsets: (k*k*(c+1),2)
    :param k: output size,(x,y)
    :return:(b,k,k,c+1)
    '''
    fea_shape = np.shape(features)
    num_classes = fea_shape[-1] / (k * k)  #channels
    depth = num_classes    #(c+1)
    feature_boxes = np.round(boxes / feat_stride)
    feature_boxes[-2:] -= 1  #not include right and bottom edge
    top_left_point = np.hstack((feature_boxes[1:3],feature_boxes[1:3])).reshape((1,4))
    boxes_part = np.zeros(( k * k, 4))  #(k^2,4)
    boxes_part += top_left_point   #(k*k,4)
    width = (feature_boxes[3] - feature_boxes[1]) / k   # (n,1)
    height = (feature_boxes[4] - feature_boxes[2]) / k   # (n,1)

    # split boxes
    shift_x = np.arange(0, k) * width
    shift_y = np.arange(0, k) * height
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    boxes_part += shifts
    boxes_part[:, 2] = boxes_part[:, 0] + width - 1
    boxes_part[:, 3] = boxes_part[:, 1] + height - 1
    boxes_part = np.reshape(np.floor(boxes_part),(k,k,-1,4))  #(k,k,1,4)
    boxes_part[:, -1, 0, 2] = feature_boxes[-2]
    boxes_part[-1, :, 0, 3] = feature_boxes[-1]
    boxes_part = np.reshape(boxes_part, (1, int(k*k), 4))   #(1,k*k,4)

    # add offsets to splitted boxes
    if len(offsets) == 0:
        offsets0 = np.zeros((depth * k * k, 2))  # (k*k*2*(c+1),2)
    else:
        offsets0 = offsets  # (k*k*c,2)
    offsets0 = np.reshape(offsets0, (int(k * k), int(depth), 2))  #(x,y,x,y,x,y)(k*k,c,2)
    # offsets1 = tf.stack((offsets0, offsets0),axis=3)
    # offsets1 = tf.reshape(offsets1,(boxes_num, k * k, depth, 4))
    offsets1 = np.tile(offsets0, (1, 1, 2))  #(k*k,c,4)
    offsets1 = np.transpose(offsets1,(1,0,2))   #(c,k*k,4)
    boxes_part = np.repeat(boxes_part,depth,axis=0)
    boxes_part += offsets1  #(c,k*k,4)
    boxes_part = np.reshape(boxes_part,(int(k*k*depth),4)) #(c*k*k,4)

    # clip split boxes by feature' size
    temp00 = np.clip(boxes_part[..., 0], 0, fea_shape[2] - 1)
    temp11 = np.clip(boxes_part[..., 1], 0, fea_shape[1] - 1)
    temp22 = np.clip(boxes_part[..., 2], 0, fea_shape[2] - 1)
    temp33 = np.clip(boxes_part[..., 3], 0, fea_shape[1] - 1)
    boxes_k_offset = np.stack([temp00,temp11,temp22,temp33],axis=-1)    #(c*k*k,4)
    boxes_k_offset = np.reshape(boxes_k_offset,(int(depth), int(k*k), 4))   #(c,k*k,4)
    boxes_k_offset = np.transpose(boxes_k_offset,(1,0,2))   #(k*k,c,4)

    # num of classes
    all_boxes_num = k * k
    for i in range(all_boxes_num):
        part_k = i % (k * k)
        pooled_fea = map_coordinates(features[0],boxes_k_offset[i],part_k,num_classes,pool)  #(depth,1)/(depth,h,w)
        if (part_k % k) == 0:
            pooled_row = pooled_fea
        elif (part_k % k) == (k - 1) and part_k != (k - 1):
            pooled_row = np.concatenate((pooled_row, pooled_fea), axis=2)
            pooled_response = np.concatenate((pooled_response, pooled_row), axis=1)
        elif (part_k % k) == (k - 1) and part_k == (k - 1):
            pooled_row = np.concatenate((pooled_row,pooled_fea), axis=2)
            pooled_response = pooled_row
        else:
            pooled_row = np.concatenate((pooled_row,pooled_fea), axis=2)
        # try:
        #     pooled_response = np.concatenate((pooled_response, pooled_fea), 0)
        # except UnboundLocalError:
        #     pooled_response = pooled_fea

    return pooled_response  #(depth,k,k)/(depth,height,width)

def map_coordinates(inputs,boxes,k,num_classes,pool):
    '''
    Get values in the boxes
    :param inputs: feature map (h,w,2*k^2*(c+1) or (h,w,2*K^2*2)
    :param boxes: (depth,4)(x1,y1,x2,y2) May be fraction
    :param k: relative position
    :param num_classes:
    :param pool: whether ave_pool the features
    :return:
    '''
    # compute box's width and height, both are integer
    width = boxes[0][2] - boxes[0][0] + 1
    height = boxes[0][3] - boxes[0][1] + 1

    depth = np.shape(boxes)[0]
    tp_lf = np.reshape(boxes[:,0:2],(-1,1,2))   #(depth,1,2)
    grid = np.meshgrid(np.array(range(int(height))), np.array(range(int(width))))
    grid = np.stack(grid, axis=-1)  #(h,w,2)
    #grid = np.expand_dims(grid,axis=0)  #(1,h,w,2)
    grid = np.reshape(grid, (1,-1, 2))  #(1,n_points,2)
    coords = grid + tp_lf   #(depth,n,2)
    n_coords = np.shape(coords)[1]

    # coords_lt = tf.cast(tf.floor(coords), 'int32')  #(depth,n_points,2)
    # coords_rb = tf.cast(tf.ceil(coords), 'int32')
    # coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    # coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)


    coords_lt = np.floor(coords)
    coords_rb = np.ceil(coords)
    coords_lt = coords_lt.astype(np.int32)
    coords_rb = coords_rb.astype(np.int32)
    coords_lb = np.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = np.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)   #(depth,n_points,2)

    idx = np_repeat(range(depth), n_coords)

    def _get_vals_by_coords(input, coords):
        inputs1 = input[:,:,int(k*num_classes):int((k+1)*num_classes)]  #(h,w,depth)
        inputs2 = np.transpose(inputs1,(2,0,1))  #(depth,h,w)
        indices = np.stack([
            idx,np_flatten(coords[..., 0]), np_flatten(coords[..., 1])
        ], axis=-1)
        inputs_shape = np.shape(inputs2)
        temp1 = inputs_shape[1]*inputs_shape[2]
        temp2 = inputs_shape[2]
        indices1 = [i[0]*temp1+i[1]*temp2+i[2] for i in indices]

        vals = np.take(inputs2,indices1)
        vals = np.reshape(vals, (int(depth),int(n_coords)))
        return vals  #(depth,n_points)

    vals_lt = _get_vals_by_coords(inputs, coords_lt)
    vals_rb = _get_vals_by_coords(inputs, coords_rb)
    vals_lb = _get_vals_by_coords(inputs, coords_lb)
    vals_rt = _get_vals_by_coords(inputs, coords_rt)  #(depth,n_points)

    # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  #(depth,n_points,2)
    coords_offset_lt = coords - coords_lt  # (depth,n_points,2)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (depth,n_points)

    if pool:
        pooled_box = np.mean(mapped_vals, axis=1)   #(depth,1)
        pooled_box = np.reshape(pooled_box, (depth, 1, 1))    #(depth,1,1)
    else:
        pooled_box = np.reshape(mapped_vals, (depth, int(height), int(width)))    #(depth,h,w)


    return pooled_box #(depth,1)/(depth,h,w)