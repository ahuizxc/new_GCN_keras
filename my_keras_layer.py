#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:01:59 2018

@author: liushenghui
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pdb
import copy
import types as python_types
import warnings
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.layers import Layer
import copy


    
class My_conv2d(Layer):
    """
    """
    def __init__(self, units, depth, channel, **kwargs):
        super(My_conv2d, self).__init__(**kwargs)
        self.units = units
        self.depth = depth
        self.channel = channel
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='my_conv',shape=(self.units, self.depth, self.channel),dtype=tf.float64,
                              initializer='glorot_uniform',
                              trainable=True)

    def call(self, inputs):       
        def f1():      
            return tf.constant(1.0)
        def f2():
            return tf.constant(0.0)
        array_train = inputs[0]
        array_neighbor = inputs[1]
        length = array_train.shape[1]
        inchannels = array_train.shape[2]-1 
        idx = tf.constant([i for i in range(length)],dtype=tf.float64)
        train_list = array_train[:,:,0]
        lbb = tf.constant([-1.0 for u in range(self.channel-1)],dtype=tf.float64)
        for i in range(length):
            lst = array_neighbor[:,:,0]
            ot = []
            index = array_train[0,i,0]
            newp = tf.where(tf.equal(train_list,index))[0][0]
            pp = tf.where(tf.equal(lst,index))[0][0]
            nnn = array_neighbor[0,pp,self.units:]
            not_ = tf.where(tf.equal(nnn,lbb))
            not_ = tf.cast(not_,tf.float64)
            not_ = tf.tile(not_,[1,inchannels])
            for out in range(self.units):
                tp = tf.Variable(0.0, dtype=tf.float64)
                tmp = tf.reduce_sum(tf.multiply(array_train[0,i,1:],self.kernel[out,:,-1]))
                tf.assign_add(tp,tmp)
                re = tf.multiply(self.kernel[out,:,-1],array_train[0,newp,:-1])
                re = tf.tile(tf.reshape(re,(1,inchannels)),[self.channel-1,1])
                tf.assign_add(tp,tf.reduce_sum(tf.multiply(re, not_)))
#                for c in range(inchannels):
#                    re = tf.multiply(self.kernel[out,c,-1],array_train[0,newp,c])
#                    update_value = tf.reduce_sum(tf.multiply(not_,re))
#                    tf.assign_add(tp, update_value)                                                  
                ot.append(tp)                
            tmp_tensor = tf.reshape(ot,(-1,self.units))
            print(i)
            if i == 0:
                tensor_no_index = tmp_tensor
            else:
                tensor_no_index = tf.concat([tensor_no_index,tmp_tensor], axis=0)        
        idx = tf.reshape(idx,(-1,1))
        ot_train = tf.concat([idx, tensor_no_index], axis=1)
        ot_train = tf.cast(ot_train,dtype=tf.float64)
        self.out_train_data = ot_train
        return ot_train

    def compute_output_shape(self, input_shape):
        return self.out_train_data.shape
class My_maxpool1(Layer):
    """
    """
    def __init__(self, n, m, **kwargs):
        
        super(My_maxpool1, self).__init__(**kwargs)
        self.n = n
        self.m = m       
        
    def call(self, inputs):
        
        def f1(maxmin,maxindex,neighbormaxindex,max_,tp):
            tf.assign(maxmin, max_)
            tf.assign(maxindex,tp)
            tf.assign(neighbormaxindex,4*i+j)
            return maxmin,maxindex,neighbormaxindex
        def f2(maxmin,maxindex,neighbormaxindex):
            return maxmin,maxindex,neighbormaxindex
        traindata = inputs[0]
        neighbor = tf.squeeze(inputs[1])
        bt = neighbor[:,self.m]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        data = tf.gather(neighbor, indices, validate_indices=None, name=None)
        new_neighbor = data[neighbor.shape[0]-self.n:]  
        len_new_neighbor = neighbor.shape[1]
        shengyu = data[0:neighbor.shape[0]-self.n]
        
        insertchannels = tf.where(tf.equal(new_neighbor[:,4],0))[:,0]  
#        obj = tf.zeros((insertchannels.shape[0],traindata.shape[1]-1))
        rev = tf.reshape(tf.gather(neighbor[:,0],insertchannels,validate_indices=None, name=None),(-1,1))
        rev = tf.pad(rev,paddings=[[0,0],[0,3]])
        traindata = tf.concat([traindata,rev],axis=0)   
        traindatalist = traindata[:,0]   
        baoliutrain_array = []
        baoliuneighbor_array = []
        
        for i in range(int(self.n/4)):    
            print(i)
            maxmin = tf.Variable(-100000.0,dtype=tf.float64)
            maxindex = tf.Variable(-100,dtype=tf.int64)
            neighbormaxindex = tf.Variable(-100,dtype=tf.int64)
#            maxmin = tf.Variable(maxmin)
#            maxindex = tf.Variable(maxindex)
#            neighbormaxindex = tf.Variable(neighbormaxindex)
            for j in range(4):
#                maxmin = tf.Variable(maxmin)
#                maxindex = tf.Variable(maxindex)
#                neighbormaxindex = tf.Variable(neighbormaxindex)
                re1 = tf.not_equal(new_neighbor[4*i+j,4],0)
                tp = tf.where(tf.equal(traindatalist,new_neighbor[4*i+j,0]))[0,0]
                max_ = tf.reduce_max(traindata[tp,1:])
                re = tf.equal(re1, max_>maxmin)
#                maxmin,maxindex,neighbormaxindex = 
                tf.cond(re,lambda: f1(maxmin,maxindex,neighbormaxindex,max_,tp),lambda:f2(maxmin,maxindex,neighbormaxindex))
            baoliutrain_array.append(maxindex)
            baoliuneighbor_array.append(neighbormaxindex)
        baoliutrain_array = tf.stack(baoliutrain_array)
        baoliuneighbor_array = tf.stack(baoliuneighbor_array)
        new_neighbor = tf.gather(new_neighbor,baoliuneighbor_array)
        traindata = tf.gather(traindata,baoliutrain_array)
        traindata = tf.cast(traindata,dtype=tf.float64)
        self.traindata = traindata
        return traindata
    def compute_output_shape(self, input_shape):
        return self.out_train_data.shape
class My_maxpool2(Layer):
    """
    """
    def __init__(self, n, m, **kwargs):
        
        super(My_maxpool2, self).__init__(**kwargs)
        self.n = n
        self.m = m       
        
    def call(self, inputs):        
        def f1(neighbormaxindex):
#            tf.assign(maxmin, max_)
#            tf.assign(maxindex,tp)
            tf.assign(neighbormaxindex,4*i+j)
            return neighbormaxindex
        def f2(neighbormaxindex):
            return neighbormaxindex
        traindata = inputs[0]
        neighbor = tf.squeeze(inputs[1])
        bt = neighbor[:,self.m]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        data = tf.gather(neighbor, indices, validate_indices=None, name=None)
        new_neighbor = data[neighbor.shape[0]-self.n:]  
        shengyu = data[0:neighbor.shape[0]-self.n]        
        insertchannels = tf.where(tf.equal(new_neighbor[:,4],0))[:,0]  
#        obj = tf.zeros((insertchannels.shape[0],traindata.shape[1]-1))
        rev = tf.reshape(tf.gather(neighbor[:,0],insertchannels,validate_indices=None, name=None),(-1,1))
        rev = tf.pad(rev,paddings=[[0,0],[0,3]])
        traindata = tf.concat([traindata,rev],axis=0)   
        traindatalist = traindata[:,0]   
#        baoliutrain_array = []
        baoliuneighbor_array = []
#        
        for i in range(int(self.n/4)):    
            print(i)
            maxmin = tf.Variable(-100000.0,dtype=tf.float64)
#            maxindex = tf.Variable(-100,dtype=tf.int64)
            neighbormaxindex = tf.Variable(-100,dtype=tf.int64)
#            maxmin = tf.Variable(maxmin)
#            maxindex = tf.Variable(maxindex)
            neighbormaxindex = tf.Variable(neighbormaxindex)
            for j in range(4):
                maxmin = tf.Variable(maxmin)
#                maxindex = tf.Variable(maxindex)
                neighbormaxindex = tf.Variable(neighbormaxindex)
                re1 = tf.not_equal(new_neighbor[4*i+j,4],0)
                tp = tf.where(tf.equal(traindatalist,new_neighbor[4*i+j,0]))[0,0]
                max_ = tf.reduce_max(traindata[tp,1:])
                re = tf.equal(re1, max_>maxmin)
                tf.cond(re,lambda: f1(neighbormaxindex),lambda:f2(neighbormaxindex))
            baoliuneighbor_array.append(neighbormaxindex)
        baoliuneighbor_array = tf.stack(baoliuneighbor_array)
        new_neighbor = tf.gather(new_neighbor,baoliuneighbor_array)
        new_neighbor = tf.concat([new_neighbor,shengyu],axis=0)
        set_all = [i for i in range(new_neighbor.shape[1]) if i not in [i for i in range(5,22)]]
        new_neighbor = tf.gather(new_neighbor,set_all, axis=1)  
        bt = new_neighbor[:,self.m]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        new_neighbor = tf.gather(new_neighbor, indices, validate_indices=None, name=None)
        len_s = tf.shape(tf.where(tf.equal(new_neighbor[:,1+self.m],-1)))[0]
        tmp = []
        def ff1(new_neighbor,self,g,v):
            p = tf.where(tf.equal(new_neighbor[:,1+self.m],new_neighbor[g][5+v]))[0][0]
            return new_neighbor[p][0]
        def ff2(new_neighbor,g,v):
            return new_neighbor[g,5+v]
        def fs1(td):            
            return td
        def fs2(new_neighbor,g):
            tf.reshape(new_neighbor[g,:],(1,-1))
            return tf.reshape(new_neighbor[g,:],(1,-1))
        for g in range(new_neighbor.shape[0]):
            print(g)
            tp_ = tf.reshape(new_neighbor[g,:5],(1,-1))
            tp_2 = tf.reshape(new_neighbor[g,22:],(1,-1))
            qq = []
            for v in range(17):
                con = tf.greater(new_neighbor[g,5+v],0)
                ww = tf.cond(con, lambda: ff1(new_neighbor,self,g,v), lambda:ff2(new_neighbor,g,v))
                qq.append(ww)
            ttt = tf.reshape(tf.stack(qq),(1,-1))
            tff = tf.concat([tp_,ttt],axis=1)
            td = tf.concat([tff,tp_2],axis=1)
            con2 = tf.greater(len_s,g)
            fdd = tf.cond(con2,lambda:fs2(new_neighbor,g),lambda:fs1(td))
            tmp.append(fdd)
        tmp = tf.cast(tf.squeeze(tf.stack(tmp)),tf.float64)
        self.tmp = tmp
        return tmp
    def compute_output_shape(self, input_shape):
        return self.tmp.shape
    
class My_deconv2d(Layer):
    """
    """
    def __init__(self, out_channel, m, n, m_,  kernel_shape, **kwargs):
        super(My_deconv2d, self).__init__(**kwargs)
        self.out_channel = out_channel
        self.m  = m
        self.kernel_shape = kernel_shape
        self.m_ = m_
        self.n = n
    def build(self, input_shape):
        self.kernel = self.add_weight(name='my_deconv',shape=self.kernel_shape,dtype=tf.float64,
                              initializer='glorot_uniform',
                              trainable=True)

    def call(self, inputs):            
        def f1(maxmin,maxindex,neighbormaxindex,max_,tp):
            tf.assign(maxmin, max_)
            tf.assign(maxindex,tp)
            tf.assign(neighbormaxindex,4*i+j)
            return maxmin,maxindex,neighbormaxindex
        def f2(maxmin,maxindex,neighbormaxindex):
            return maxmin,maxindex,neighbormaxindex
        traindata = inputs[0]
        neighbor = tf.squeeze(inputs[1])
        bt = neighbor[:,self.m_]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        data = tf.gather(neighbor, indices, validate_indices=None, name=None)
        new_neighbor = data[neighbor.shape[0]-self.n:]  
        len_new_neighbor = neighbor.shape[1]
        shengyu = data[0:neighbor.shape[0]-self.n]
        insertchannels = tf.where(tf.equal(new_neighbor[:,4],0))[:,0]  
#        obj = tf.zeros((insertchannels.shape[0],traindata.shape[1]-1))
        rev = tf.reshape(tf.gather(neighbor[:,0],insertchannels,validate_indices=None, name=None),(-1,1))
        rev = tf.pad(rev,paddings=[[0,0],[0,3]])
        traindata = tf.concat([traindata,rev],axis=0)   
        traindatalist = traindata[:,0]   
        baoliutrain_array = []
        baoliuneighbor_array = []
        
        for i in range(int(self.n/4)):    
            print(i)
            maxmin = tf.Variable(-100000.0,dtype=tf.float64)
            maxindex = tf.Variable(-100,dtype=tf.int64)
            neighbormaxindex = tf.Variable(-100,dtype=tf.int64)
            for j in range(4):
#                maxmin = tf.Variable(maxmin)
#                maxindex = tf.Variable(maxindex)
#                neighbormaxindex = tf.Variable(neighbormaxindex)
                re1 = tf.not_equal(new_neighbor[4*i+j,4],0)
                tp = tf.where(tf.equal(traindatalist,new_neighbor[4*i+j,0]))[0,0]
                max_ = tf.reduce_max(traindata[tp,1:])
                re = tf.equal(re1, max_>maxmin)
#                maxmin,maxindex,neighbormaxindex = 
                tf.cond(re,lambda: f1(maxmin,maxindex,neighbormaxindex,max_,tp),lambda:f2(maxmin,maxindex,neighbormaxindex))
            baoliutrain_array.append(maxindex)
            baoliuneighbor_array.append(neighbormaxindex)
        baoliutrain_array = tf.stack(baoliutrain_array)
        baoliuneighbor_array = tf.stack(baoliuneighbor_array)
        new_neighbor = tf.gather(new_neighbor,baoliuneighbor_array)
        traindata = tf.gather(traindata,baoliutrain_array)
        traindata = tf.cast(traindata,dtype=tf.float64)
        new_neighbor = tf.concat([new_neighbor,shengyu],axis=0)
        set_all = [i for i in range(new_neighbor.shape[1]) if i not in [i for i in range(5,22)]]
        new_neighbor = tf.gather(new_neighbor,set_all, axis=1)  
        bt = new_neighbor[:,self.m]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        new_neighbor = tf.gather(new_neighbor, indices, validate_indices=None, name=None)
        len_s = tf.shape(tf.where(tf.equal(new_neighbor[:,1+self.m],-1)))[0]
        tmp = []
        def ff1(new_neighbor,self,g,v):
            p = tf.where(tf.equal(new_neighbor[:,1+self.m],new_neighbor[g][5+v]))[0][0]
            return new_neighbor[p][0]
        def ff2(new_neighbor,g,v):
            return new_neighbor[g,5+v]
        def fs1(td):            
            return td
        def fs2(new_neighbor,g):
            tf.reshape(new_neighbor[g,:],(1,-1))
            return tf.reshape(new_neighbor[g,:],(1,-1))
        for g in range(new_neighbor.shape[0]):
            print(g)
            tp_ = tf.reshape(new_neighbor[g,:5],(1,-1))
            tp_2 = tf.reshape(new_neighbor[g,22:],(1,-1))
            qq = []
            for v in range(17):
                con = tf.greater(new_neighbor[g,5+v],0)
                ww = tf.cond(con, lambda: ff1(new_neighbor,self,g,v), lambda:ff2(new_neighbor,g,v))
                qq.append(ww)
            ttt = tf.reshape(tf.stack(qq),(1,-1))
            tff = tf.concat([tp_,ttt],axis=1)
            td = tf.concat([tff,tp_2],axis=1)
            con2 = tf.greater(len_s,g)
            fdd = tf.cond(con2,lambda:fs2(new_neighbor,g),lambda:fs1(td))
            tmp.append(fdd)
        tmp = tf.cast(tf.squeeze(tf.stack(tmp)),tf.float64)
        poolinput = traindata
        neighbor = tmp
        lastneighbor = tf.squeeze(inputs[1])
        deletelength = tf.where(tf.equal(neighbor[:,self.m],-1))[0].shape[0]
        bt = neighbor[:,self.m_]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        data = tf.gather(neighbor, indices, validate_indices=None, name=None)
        data = data[deletelength:]
        new_traindata = tf.zeros((3,poolinput.shape[1]),dtype=tf.float64)
        tmp = tf.zeros((3,poolinput.shape[1]),dtype=tf.float64)
        channellist = poolinput[:, 0]
        for i in range(poolinput.shape[0]):
            p1 = tf.where(tf.equal(channellist,data[i][0]))[0][0]
            new_traindata = tf.concat([new_traindata, tf.reshape(poolinput[p1,:],(1,-1))],axis=0)
            new_traindata = tf.concat([new_traindata,tmp],axis=0)
        outsize = (poolinput.shape[0] - 1) * 4 + 4
        tl_l = []
        for out in range(self.out_channel):
            we = self.kernel[out]
            tm_l = []
            for k in range(outsize):
                opt = tf.reduce_sum(tf.multiply(new_traindata[k:k+4,1:],we))
                tm_l.append(opt)
            tm_l = tf.stack(tm_l)
            tl_l.append(tm_l)
        tl_l = tf.transpose(tf.stack(tl_l))
        bt = lastneighbor[:,self.m_]
        values, indices = tf.nn.top_k(bt, k=bt.shape[0], sorted=True, name=None)
        lastneighbor = tf.gather(lastneighbor, indices, validate_indices=None, name=None)
        l = lastneighbor.shape[0] - tl_l.shape[0]
        outputs = tf.concat([lastneighbor[l:,0:1],tl_l],axis=1)
        self.outputs = outputs
        return outputs

    def compute_output_shape(self, input_shape):
        return self.outputs.shape