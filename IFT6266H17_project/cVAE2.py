# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:53:40 2017

@author: Chin-Wei
"""



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



import os
import numpy as np
from lasagne.nonlinearities import linear, sigmoid
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import squared_error as se
import lasagne
tnc = lasagne.updates.total_norm_constraint
nonl = lasagne.nonlinearities.tanh
import theano
import theano.tensor as T
floatX = theano.config.floatX

from utils import get_all_data_border, data_path, fnames
from modules import Upscale2DLayer, ElementwiseProductLayer, centerMask


k = 512
max_norm = 10
lr_default = 0.001

def get_encoder1():
    
    print '\tgetting encoder'
    enc = lasagne.layers.InputLayer(shape=(None,3,64,64)) 
    
    enc = lasagne.layers.Conv2DLayer(enc,32,3,
                                     stride=1,pad=0,nonlinearity=nonl)
    print enc.output_shape
    
    enc = lasagne.layers.Conv2DLayer(enc,32,3,
                                     stride=1,pad=0,nonlinearity=nonl)
    print enc.output_shape
    
    enc = lasagne.layers.Conv2DLayer(enc,64,5,
                                     stride=2,pad=0,nonlinearity=nonl)
    print enc.output_shape

    enc = lasagne.layers.Conv2DLayer(enc,64,5,
                                     stride=1,pad=0,nonlinearity=nonl) 
    print enc.output_shape
    
    enc = lasagne.layers.Conv2DLayer(enc,128,5,
                                     stride=2,pad=0,nonlinearity=nonl) 
    print enc.output_shape
    
    enc = lasagne.layers.Conv2DLayer(enc,256,5,
                                     stride=1,pad=0,nonlinearity=nonl) 
    print enc.output_shape
    
    enc = lasagne.layers.Conv2DLayer(enc,512,3,
                                     stride=1,pad=0,nonlinearity=nonl) 
    print enc.output_shape

    # mean
    enc_m = lasagne.layers.Conv2DLayer(enc,512,3,
                                       stride=1,pad=0,nonlinearity=nonl) 
    print enc_m.output_shape
    
    enc_m = lasagne.layers.DenseLayer(enc_m,k,
                                      nonlinearity=linear,
                                      W=lasagne.init.Normal(std=0.001),
                                      b=lasagne.init.Normal(std=0.001))
    print enc_m.output_shape
    
    #enc_m = lasagne.layers.ReshapeLayer(enc_m,[[0],2048])
    #print enc_m.output_shape
    
    
    # variance
    enc_s = lasagne.layers.Conv2DLayer(enc,512,3,
                                       stride=1,pad=0,nonlinearity=nonl)
    print enc_s.output_shape
    
    enc_s = lasagne.layers.DenseLayer(enc_s,k,
                                      nonlinearity=linear,
                                      W=lasagne.init.Normal(std=0.001),
                                      b=lasagne.init.Normal(std=0.001))
    print enc_s.output_shape
    #enc_s = lasagne.layers.ReshapeLayer(enc_s,[[0],2048])
    #print enc_s.output_shape
    
    return enc_m, enc_s



    

def get_decoder():
    
    print '\tgetting decoder'
    
    # inference
    dec1_input = lasagne.layers.InputLayer(shape=np.append([None],k)) 
    dec1 = lasagne.layers.DenseLayer(dec1_input,2048)
    print dec1.output_shape
    
    dec1 = lasagne.layers.ReshapeLayer(
        dec1,
        [[0]]+[512,2,2]
    )
    print dec1.output_shape       
    
    dec1_1 = Upscale2DLayer(dec1,2)
    dec1_1 = lasagne.layers.Conv2DLayer(dec1_1,256,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec1_1.output_shape, 1
    #4x4#
    
    dec1_2 = Upscale2DLayer(dec1_1,2)
    dec1_2 = lasagne.layers.Conv2DLayer(dec1_2,128,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec1_2.output_shape, 1
    #8x8#
    
    dec1_3 = Upscale2DLayer(dec1_2,2)
    dec1_3 = lasagne.layers.Conv2DLayer(dec1_3,64,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec1_3.output_shape, 1
    #16x16#
    
    # conditional
    dec2_input = lasagne.layers.InputLayer(shape=(None,3,64,64)) 

    dec2_3 = lasagne.layers.Conv2DLayer(dec2_input,64,3,
                                        stride=1,pad=1,nonlinearity=nonl)
    dec2_3 = lasagne.layers.Pool2DLayer(dec2_3,2)
    print dec2_3.output_shape, 3
    
    dec2_2 = lasagne.layers.Conv2DLayer(dec2_3,128,3,
                                        stride=1,pad=1,nonlinearity=nonl)
    dec2_2 = lasagne.layers.Pool2DLayer(dec2_2,2)
    print dec2_2.output_shape, 2
    
    dec2_1 = lasagne.layers.Conv2DLayer(dec2_2,256,3,
                                        stride=1,pad=1,nonlinearity=nonl)
    dec2_1 = lasagne.layers.Pool2DLayer(dec2_1,2)
    print dec2_1.output_shape, 1
    
    
    ## merging 
    
    dec_1 = lasagne.layers.ElemwiseSumLayer(
        [ElementwiseProductLayer(dec2_1,centerMask(8,8)),
         lasagne.layers.PadLayer(dec1_1,2)]
    )
    dec_1 = Upscale2DLayer(dec_1,2)
    dec_1 = lasagne.layers.Conv2DLayer(dec_1,128,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec_1.output_shape
    
    dec_2 = lasagne.layers.ElemwiseSumLayer(
        [ElementwiseProductLayer(dec2_2,centerMask(16,16)),
         lasagne.layers.PadLayer(dec1_2,4)]
    )
    dec_2 = lasagne.layers.ConcatLayer([dec_2,dec_1])
    dec_2 = lasagne.layers.Conv2DLayer(dec_2,128,3,
                                       stride=1,pad=1,nonlinearity=nonl) 
    dec_2 = Upscale2DLayer(dec_2,2)
    dec_2 = lasagne.layers.Conv2DLayer(dec_2,64,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec_2.output_shape
    
    
    dec_3 = lasagne.layers.ElemwiseSumLayer(
        [ElementwiseProductLayer(dec2_3,centerMask(32,32)),
         lasagne.layers.PadLayer(dec1_3,8)]
    )
    dec_3 = lasagne.layers.ConcatLayer([dec_3,dec_2])
    dec_3 = lasagne.layers.Conv2DLayer(dec_3,64,3,
                                       stride=1,pad=1,nonlinearity=nonl) 
    dec_3 = Upscale2DLayer(dec_3,2)
    dec_3 = lasagne.layers.Conv2DLayer(dec_3,32,3,
                                       stride=1,pad=1,nonlinearity=nonl)    
    print dec_3.output_shape
    
    dec_3 = lasagne.layers.Conv2DLayer(dec_3,3,3,
                                       stride=1,pad=1,nonlinearity=sigmoid)
    print dec_3.output_shape
    
 
    return dec_3, dec1_input, dec2_input

    

 

class cVAE(object):
    
    def __init__(self):
        
        # inpv -> input variable, i.e. full image
        # ep -> std normal noise
        # beta -> border, p(z|beta)
        # w -> annealing weight
        self.inpv = T.tensor4('inpv')
        self.ep = T.matrix('ep')
        self.beta = T.tensor4('beta') 
        self.sample = T.matrix('sample')
        self.w = T.scalar('w') 
        self.lr = T.scalar('lr')
        
        self.enc_m, self.enc_s = get_encoder1()
        self.dec, self.dec1_input, self.dec2_input = get_decoder()
        
        self.k = np.prod(self.enc_m.output_shape[1:])
        
        self.qm = get_output(self.enc_m,self.inpv)
        self.qlogs = get_output(self.enc_s,self.inpv)
        self.qlogv = 2*self.qlogs
        self.qs = T.exp(self.qlogs)
        self.qv = T.exp(self.qlogs*2)
        
        
        self.z = self.qm + self.qs * self.ep
        self.rec = get_output(self.dec,
                              {self.dec1_input:self.z,
                               self.dec2_input:self.beta})
        self.ancestral = get_output(self.dec,
                                    {self.dec1_input:self.sample,
                                     self.dec2_input:self.beta})
        
        self.log_px_z = - se(self.rec,self.inpv)
        self.log_pz   = - 0.5 * (self.qm**2 + self.qv)
        self.log_qz_x = - 0.5 * (1+self.qlogv)
        
        
        self.kls = T.sum(self.log_qz_x - self.log_pz,1)
        self.rec_errs = T.sum(-self.log_px_z,axis=[1,2,3])
        
        self.kl = T.mean(self.kls)
        self.rec_err = T.mean(self.rec_errs)
        self.loss = self.w*self.kl+self.rec_err
        
        self.params = np.concatenate([get_all_params(ly) for ly in
            [self.enc_m,
             self.enc_s,
             self.dec]        
        ]).tolist()
        
        self.grads = T.grad(self.loss, self.params)
        self.scaled_grads = tnc(self.grads, max_norm)
        self.updates = lasagne.updates.adam(self.scaled_grads, self.params, 
                                            self.lr)
        self.train_func = theano.function([self.inpv,self.beta,
                                           self.ep,self.w,
                                           self.lr],
                                          [self.loss,self.rec_err,self.kl],
                                          updates=self.updates)
        self.recons_func = theano.function([self.inpv,self.beta,self.ep],
                                           self.rec)
        self.sample_func = theano.function([self.beta,self.sample],
                                           self.ancestral)
        
    def train_function(self,x,beta,w,lr=lr_default):
        n = x.shape[0]
        ep = np.random.randn(n,self.k).astype(floatX)
        return self.train_func(x,beta,ep,w,lr)
    
    def sample_function(self,beta):
        n = beta.shape[0]
        sp = np.random.randn(n,self.k).astype(floatX)
        return self.sample_func(beta,sp)
    
    def recons_function(self,x,beta):
        n = x.shape[0]
        ep = np.random.randn(n,self.k).astype(floatX)
        return self.recons_func(x,beta,ep)
    

        
if __name__ == '__main__':
    
    args = dict(
        lr = 0.0001,
        batch_size = 64,
        n_epochs = 50,
        n_rec = 5,
        note = 'anneal'
    )
    
    print args
    
    lr = args['lr']
    batch_size = args['batch_size']
    n_epochs = args['n_epochs']
    n_rec = args['n_rec']

    print '\tloading data'
    train_fn1 = ['train2014_inputs_{}.npy'.format(fname) for fname in fnames]
    train_fn2 = ['train2014_targets_{}.npy'.format(fname) for fname in fnames]

    val_fn1 = 'val2014_inputs_{}.npy'.format('10011')
    val_fn2 = 'val2014_targets_{}.npy'.format('10011')
    
    b_v, x_v = get_all_data_border(data_path,val_fn1,val_fn2)
    b_v = b_v[:1000]
    x_v = x_v[:1000]
    

    model=cVAE()

    print('\tdone building model')

    reconstruct = model.recons_function
    train_func = model.train_function
    ancestral = model.sample_function


    import datetime
    now = str(datetime.datetime.today())[:-7]
    save_dir = now.replace('-','').replace(' ','_').replace(':','_')
    print save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print 'start training! \n'
    t = 0
    for e in range(n_epochs):
        
        print '='*20, 'Epoch', e, '='*20, '\n'
        
        
        for fn1,fn2 in zip(train_fn1,train_fn2):
            print 'get files: {} and {}'.format(fn1,fn2)
            b,x = get_all_data_border(data_path,fn1,fn2)
            
            for j in range(np.floor(len(x)/float(batch_size)).astype(int)):
                begin = batch_size*j
                end = min(batch_size*(j+1),len(x))
                x_ = x[begin:end]
                b_ = b[begin:end]
                w = np.cast[floatX]( np.min([1,0.0001+0.00005*t]) )
                
                loss,rec_err,kl = train_func(x_,b_,w)
                if t%20==0:
                    print e,j,t,loss,rec_err,kl  
                
                t+=1
                
                 
            
            del x
            
         
        if e%1 == 0:
            print e
            ind = np.random.randint(0,1000,n_rec) # plot five images
            xx = x_v[ind]
            bb = b_v[ind]
            
            re_x = reconstruct(xx,bb)
            cd_x = ancestral(bb)

            fig = plt.figure()
            tt=0
            for i in range(4):
                for j in range(n_rec):
                    tt+=1
                    if i == 0:
                        bb_j = bb[j]
                        ax = fig.add_subplot(4,n_rec,tt)
                        ax.imshow(bb_j.transpose(1,2,0))
                        ax.axis('off')
                    elif i == 1:
                        cd_x_j = cd_x[j]
                        ax = fig.add_subplot(4,n_rec,tt)
                        ax.imshow(cd_x_j.transpose(1,2,0))
                        ax.axis('off')
                    elif i == 2:
                        re_x_j = re_x[j]
                        ax = fig.add_subplot(4,n_rec,tt)
                        ax.imshow(re_x_j.transpose(1,2,0))
                        ax.axis('off')
                    elif i == 3:
                        xx_j = xx[j]
                        ax = fig.add_subplot(4,n_rec,tt)
                        ax.imshow(xx_j.transpose(1,2,0))
                        ax.axis('off')
            
            fig.savefig('{}/temp_{}.png'.format(save_dir,e))
            plt.close()
        
            
            

    
       