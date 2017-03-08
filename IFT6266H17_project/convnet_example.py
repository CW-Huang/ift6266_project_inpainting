# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:17:29 2017

@author: Chin-Wei
"""

import os
import numpy as np
import scipy.misc
from utils import load_data, MLP, ConvNet2D, data_path
from lasagne.nonlinearities import rectify, tanh, sigmoid
from lasagne.layers import get_output, get_all_params
import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX


if __name__ == '__main__':
    
    
#==============================================================================
#   Toy model using conv-mlp-deconv 
#==============================================================================
    
    import argparse
    parser = argparse.ArgumentParser()
    # XXX using sample size of one
    parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='number of epochs')
    
    args = parser.parse_args()
    print(args)
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    
    x = load_data(data_path).astype(floatX).transpose(0,3,1,2) / 255.
    y = load_data(data_path,filename='train2014_targets.npy').astype(floatX).transpose(0,3,1,2) / 255.
    
    
    x_v = x[-3:]
    y_v = y[-3:]
    x = x[:-3]
    y = y[:-3]
    
    print 'Datasize: {}'.format(x.shape[0])

    net = ConvNet2D(None,
        input_shape=[3,64,64],
        n_layers=5,
        n_filters=[64,64,64,64,64],
        f_sizes=[3,3,3,3,3],
        strides=[1,1,2,2,3],
        pads=[0,0,0,0,0],
        pools=[None,None,None,None,None],
        activations=[tanh,rectify,rectify,rectify,tanh]
    )
    
    net = MLP(net,None,1,[1024],[tanh])
    
    net = lasagne.layers.ReshapeLayer(
        net,
        [[0]]+[64,4,4]
    )
    
    net = lasagne.layers.Upscale2DLayer(net,2,'dilate')
    
    net = ConvNet2D(net,None,
        n_layers=2,
        n_filters=[64,64],
        f_sizes=[5,4],
        strides=[1,1],
        pads=[4,3],
        pools=[None,None],
        activations=[tanh,rectify]
    )
    
    net = lasagne.layers.Upscale2DLayer(net,2,'repeat')
    
    net = ConvNet2D(net,None,
        n_layers=3,
        n_filters=[64,64,3],
        f_sizes=[3,3,3],
        strides=[1,1,1],
        pads=[2,1,1],
        pools=[None,None,None],
        activations=[rectify,rectify,sigmoid]
    )
    
    
    
    
    input_var = T.tensor4('in')
    target_var = T.tensor4('target')
    
    
    
    pred = get_output(net,input_var)
    loss = lasagne.objectives.squared_error(pred,target_var).mean()
    params = get_all_params(net)    
    updates = lasagne.updates.adam(loss,params,0.001)
    
    train_func = theano.function([input_var,target_var],
                                 loss,
                                 updates=updates)
    predict = theano.function([input_var],pred) 
    
    
    save_dir = "./tmp/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print 'start training! \n'
    t = 0
    for e in range(n_epochs):
        
        for j in range(np.floor(len(x)/float(batch_size)).astype(int)):
            begin = batch_size*j
            end = min(batch_size*(j+1),len(x))
            x_, y_ = x[begin:end], y[begin:end]
            loss = train_func(x_,y_)
            
            t+=1
            
        print t,loss            
        
        if e%20==0:
            top3 = predict(x_v[:3])
            for i in range(3):
                top_i = top3[i].transpose(1,2,0)*255
                img = x_v[i].transpose(1,2,0).copy() * 255
                img[16:48,16:48] = top_i.copy()
                
                print '{}{}-{}-tmp.jpg'.format(save_dir,e,i)
                scipy.misc.imsave('{}{}-{}-tmp.jpg'.format(save_dir,e,i), img)
    
    
    top3 = predict(x_v[:3])
    for i in range(3):
        top_i = top3[i].transpose(1,2,0)*255
        img = x_v[i].transpose(1,2,0).copy() * 255
        img[16:48,16:48] = top_i.copy()
        
        print '{}{}-{}-tmp.jpg'.format(save_dir,e,i)
        scipy.misc.imsave('{}{}-{}-tmp.jpg'.format(save_dir,e,i), img)
        scipy.misc.imsave('{}{}-{}-tmp_.jpg'.format(save_dir,e,i), top_i)
    
    for i in range(3):
        img = x_v[i].transpose(1,2,0).copy() * 255
        img[16:48,16:48] = y_v[i].transpose(1,2,0).copy() * 255
        scipy.misc.imsave('realout{}-tmp.jpg'.format(i), img)
    
