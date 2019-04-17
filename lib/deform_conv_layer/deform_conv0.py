from __future__ import absolute_import, division

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
import tensorflow as tf

def tf_flatten(inputs):
    """Flatten tensor"""
    return tf.reshape(inputs, [-1])

def tf_repeat(inputs, repeats, axis=0):
    assert len(inputs.get_shape()) == 1

    a = tf.expand_dims(inputs, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def tf_repeat_2D(inputs,repeat_num,axis = 0):
    assert len(inputs.get_shape()) == 2

    inputs = tf.expand_dims(inputs,0)
    inputs = tf.tile(inputs,[repeat_num,1,1])
    return inputs

def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals

def tf_batch_map_coordinates(inputs,coords,order = 1):
    """
    Implement bilinear interpolation
    :param inputs:tf.tensor shape=(b,h,w)
    :param coords:tf.tensor shape=(b,n_points,2)
    :param order:
    :return:
    """
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    height = inputs_shape[1]
    weight = inputs_shape[2]
    n_coords = tf.shape(coords)[1]

    temp1 = tf.clip_by_value(coords[...,0],0,tf.cast(height,'float32') - 1)
    temp2 = tf.clip_by_value(coords[...,1],0,tf.cast(weight,'float32') - 1)
    coords = tf.stack([temp1,temp2],axis=-1)
    coords_lt = tf.cast(tf.floor(coords),'int32')
    coords_rb = tf.cast(tf.ceil(coords),'int32')
    coords_lb = tf.stack([coords_lt[...,0],coords_rb[...,1]],axis=-1)
    coords_rt = tf.stack([coords_rb[...,0],coords_lt[...,1]],axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(inputs, coords_lt)
    vals_rb = _get_vals_by_coords(inputs, coords_rb)
    vals_lb = _get_vals_by_coords(inputs, coords_lb)
    vals_rt = _get_vals_by_coords(inputs, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals

def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals

def tf_batch_map_offsets(input,offsets,order = 1):
    """
    Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, h, w)
    offsets: tf.Tensor. shape = (b, h, w, 2)
    """
    pass
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    height = input_shape[1]
    weight = input_shape[2]
    offsets = tf.reshape(offsets,(batch_size,-1,2))
    grid = tf.meshgrid(tf.range(height),tf.range(weight),indexing='ij')
    grid = tf.stack(grid,axis=-1)
    grid = tf.cast(grid,dtype='float32')
    grid = tf.reshape(grid,(-1,2))
    grid = tf_repeat_2D(grid,batch_size)
    coords = grid + offsets

    mapped_vals = tf_batch_map_coordinates(input, coords, order)
    return mapped_vals