# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 03:03:29 2017

@author: Castiel
"""


import lasagne
import theano.tensor as T
import numpy as np
Layer = lasagne.layers.Layer
as_tuple = lasagne.utils.as_tuple

class Upscale2DLayer(Layer):

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate','bilinear'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(upscaled[:, :, ::a, ::b], input)
        elif self.mode == 'bilinear':
            # only for a = b for now
            assert a == b, "scale factor a!=b"
            upscaled = T.nnet.abstract_conv.bilinear_upsampling(input,a)
        
        return upscaled



class ElementwiseProductLayer(Layer):

    def __init__(self, incoming, B, mode='repeat', **kwargs):
        super(ElementwiseProductLayer, self).__init__(incoming, **kwargs)

        self.B = B

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input * self.B
        

def centerMask(n,d):
    mask = np.ones((n,d)).astype('float32')
    qn = n/4
    qd = d/4
    mask[qn:-qn,qd:-qd] = 0
    return mask
    
        
    


        
        
        