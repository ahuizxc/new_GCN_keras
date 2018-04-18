 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:00:33 2018

@author: liushenghui
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import Mycon as con
import random
import pickle
import os
import Readdata as Rd
import glob
import tensorflow as tf
import keras
from keras.layers import LSTM, Activation
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import datetime
import pdb
QUADRANT_NUM = 16
FEATURE_NUM = 5
POINT_NUM = 153
SKIP = QUADRANT_NUM + 1
LENGTH=152
learning_rate=1e-2
max_iter=1
sess=tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def get_data(map_path):
    map_list=listdir_nohidden(map_path)
    map_list_arr=[]
    channels_list_arr=[]
    label_list_arr=[]
    centerpoint_list_arr=[]
    for file in map_list:
        neighbor, centerpoint, channels = Rd.preprocess(Rd.load_data(file))
        map_list_arr.append(neighbor)
        channels_list_arr.append(channels)
        label_list_arr.append(centerpoint)
        centerpoint_list_arr.append(centerpoint)
    return (map_list_arr,channels_list_arr,label_list_arr,centerpoint_list_arr)


# 读取下一个batch数据
def next_batch(maps, batch_size):
    indices = np.arange(len(maps))
    for start_idx in range(0, len(maps) - batch_size + 1, batch_size):
        exceprt = indices[start_idx : start_idx + batch_size]
        yield np.array(maps)[exceprt]





path = 'data'
maps, channels, labels, centerpoints=get_data(path)
maps=np.array(maps)
array= np.loadtxt('neighbor.txt')
array = np.expand_dims(array, axis=0)
print (array.shape)
#np.savetxt('channles.txt', channles)
channles = np.loadtxt('channles.txt')
channles = np.expand_dims(channles,axis = 0)

#maps[:, 3]
centerpoints= np.array(centerpoints[0])
channels = np.array(channels)
print (channels.shape)
labels = np.array(labels)
#train_data1 = tf.convert_to_tensor(channels)
#neighbor1 = tf.convert_to_tensor(maps)
#centerpoints = tf.convert_to_tensor(centerpoints)
#centerpoints=tf.cast(centerpoints,tf.float32)
target_data = np.random.rand(5, 153, 2)
lll = np.random.rand(5,153,4)
lab = np.random.rand(153,4)
sess = tf.Session()
K.set_session(sess)
channel_ipt = tf.placeholder(tf.float64, shape=(1,153,6))
maps_ipt = tf.placeholder(tf.float64, shape=(1,187,73))
mp_i =  tf.placeholder(tf.float64, shape=(187,73))
labs = tf.placeholder(tf.float64, shape=(176,3))
#labs = tf.placeholder(tf.float64, shape=(55,56))
insert_cha = tf.placeholder(tf.float64,shape=(None))
from my_keras_layer import My_conv2d, My_maxpool1, My_maxpool2, My_deconv2d
starttime = datetime.datetime.now()
b = My_conv2d(3,5,18)([channel_ipt,maps_ipt])
#a = My_maxpool1(176,1)([b,maps_ipt])
#c = My_maxpool2(176,1)([b,maps_ipt])
ot = My_deconv2d(2,1,176,1, (2,4,1))([b,maps_ipt])
tf_loss = tf.reduce_mean(tf.square(tf.reshape(labs,[-1])-tf.reshape(ot, [-1])))
opt = tf.train.GradientDescentOptimizer(0.01)
train_op = opt.minimize(tf_loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()    
sess.run(init_op)
epochs = 1
with sess.as_default():
    losses = []
    for epoch in range(epochs):
        ct = 0 
        loss = 0
        for i in range(5):  
            s_train_channel = channels[i:i+1,:,:]
            s_train_maps = array
            s_train_y = lll[i,:,:]
            _, loss_, conv_b= sess.run([train_op, tf_loss, b],
                                feed_dict={channel_ipt: s_train_channel,
                                           maps_ipt: s_train_maps,
                                           labs:s_train_y})
            print('epoch: '+str(epoch)+'-----step: '+str(i)+'----loss: '+str(loss_))
endtime = datetime.datetime.now()

print('running time: '+str((endtime - starttime).seconds))


'''
# None 代表样本数量不固定
input_ph = tf.placeholder(shape=(5, 153, 6), dtype=tf.float32, name="input_map")
neighbor_ph = tf.placeholder(shape=(5, 153, 20), dtype=tf.float32, name="neighbor_map")
centerpoints_ph=tf.placeholder(shape=(5, 153, 2), dtype=tf.float32, name="centerpoints_map")
target_ph = tf.placeholder(shape=(5, 153, 2), dtype=tf.float32, name="annotation")

weight1 = weight_variable([3,5,18])
h_conv1 = conv2d(5,input_ph,neighbor_ph,weight1,3)
neighbor2,h_pool1,addneighbor2=max_pool_2x2(5,h_conv1,neighbor_ph,centerpoints,72,37)
weight2 = weight_variable([8,3,18])
h_conv2=conv2d(5,h_pool1,neighbor2,weight2,1)
neighbor3,h_pool2,addneighbor3=max_pool_2x2(5,h_conv2,neighbor2,centerpoints,21,11)
# 反卷积
#反卷积1
W_decon1 = weight_variable([2, 4, 1])
de_con1 = Decon_2d(5,h_pool2,W_decon1,neighbor3,addneighbor3)
#反卷积2
W_decon2 = weight_variable([2, 4, 1])
de_con2 = Decon_2d(5, de_con1, W_decon2, neighbor2, addneighbor2)

init = tf.global_variables_initializer()
sess.run(init)
sess.run(h_conv1,feed_dict={input_ph: train_data1, neighbor_ph:neighbor1})
loss = tf.square(de_con2 - target_ph)
 # 声明变量的优化器
my_opt = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
 # 训练算法
for i in range(100):
    for c in range(5):
        train_data1 = tf.convert_to_tensor(channels)
        neighbor1 = tf.convert_to_tensor(maps)
        centerpoints = tf.convert_to_tensor(centerpoints)
        centerpoints = tf.cast(centerpoints, tf.float32)
        target_data = np.random.rand(5, 153, 2)
        loss = sess.run([loss,my_opt],feed_dict={input_ph: train_data1, neighbor_ph:neighbor1,target_ph: target_data})
        if (i + 1) % 25 == 0:
            print('Loss = ' + str(sess.run(loss, feed_dict={input_ph: train_data1,
                                                            neighbor_ph:neighbor1,
                                                            centerpoints_ph:centerpoints,
                                                            target_ph: target_data})))
sess.close()

#if __name__=='__main__':
#    main()
'''