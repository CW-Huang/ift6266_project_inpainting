# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:10:04 2017

@author: Chin-Wei
"""

import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, softmax, sigmoid
from lasagne.layers import get_output, get_all_params

from PIL import Image
import scipy.misc
floatX = theano.config.floatX


Conv2DLayer = lasagne.layers.Conv2DLayer
Pool2DLayer = lasagne.layers.Pool2DLayer
Deconv2DLayer = lasagne.layers.Deconv2DLayer

data_path = r'../dataset/organized/'
data_path = r'../organized_all/'

fnames = ['10020','20045','30065','40085',
          '50105','60122','70142','77871']


def load_data(path=data_path,filename='train2014_inputs.npy'):
    """
        x = load_data()
        y = load_data(filename='train2014_targets.npy')    
    """
    return np.load(open(path+filename,'r'))
    




def get_all_data(path,fname1,fname2):
    
    x = load_data(path,filename=fname1).astype(floatX).transpose(0,3,1,2) / 255.
    y = load_data(path,filename=fname2).astype(floatX).transpose(0,3,1,2) / 255.
    
    # get the original
    for j in [0,1,2]:
        x[:,j,16:48,16:48] = y[:,j]
    print 'Datasize: {}'.format(x.shape[0])
    
    return x


def MLP(net=None,input_shape=None,n_layers=None,
        n_nodes=None,activations=tanh):
    
    if net is None:
        net = lasagne.layers.InputLayer(shape=np.append([None],input_shape)) 
    
    if n_layers:
        
        for i in range(n_layers):
            
            if type(n_nodes) == list:
                n_node = n_nodes[i]
            else:
                n_node = n_nodes
                
            if type(activations) == list:
                act = activations[i]
            else:
                act = activations

            
            net = lasagne.layers.DenseLayer(incoming=net,
                                            num_units=n_node,
                                            nonlinearity=act)
    else:
        raise(Exception())                    
    return net
    

def ConvNet2D(net=None,input_shape=None,n_layers=None,
              n_filters=None,f_sizes=None,strides=None,pads=None,pools=None,
              activations=tanh):
    
    if net is None:
        net = lasagne.layers.InputLayer(
            shape=np.append([None],input_shape).tolist()
        ) 
    
    if n_layers:
        
        for i in range(n_layers):
            
            n_filter = n_filters[i]
            f_size = f_sizes[i]
            stride = strides[i]
            pad = pads[i]
            pool = pools[i]
            act = activations[i]

            net = Conv2DLayer(net,n_filter,f_size,stride,
                              pad,nonlinearity=act)
            if pool:
                net = Pool2DLayer(net,
                                  pool_size=pool['size'],
                                  mode=pool['mode'])

            print net.output_shape
    return net


def Deconv2D(net=None,input_shape=None,n_layers=None,
              n_filters=None,f_sizes=None,strides=None,pads=None,pools=None,
              activations=tanh):
    
    if net is None:
        net = lasagne.layers.InputLayer(
            shape=np.append([None],input_shape).tolist()
        ) 
    
    if n_layers:
        
        for i in range(n_layers):
            
            n_filter = n_filters[i]
            f_size = f_sizes[i]
            stride = strides[i]
            pad = pads[i]
            pool = pools[i]
            act = activations[i]

            if pool:
                net = lasagne.layers.Upscale2DLayer(net,
                                                    scale_factor=pool['size'])
            net = Deconv2DLayer(net,n_filter,f_size,stride,
                                               pad,nonlinearity=act)
            print net.output_shape
    return net



    