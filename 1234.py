#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
poolput=np.array([[  13,   6.81285954],[  20,  -1.7875793 ],[  47,   5.58511353], [  77,  -1.87969208],
 [  83,  32.27759552],[ 93,  24.01549911],[ 103,  -9.24335861], [ 108,  22.62213326],
 [ 128,  29.62034798],[ 131,  16.72520256],[ 144,  22.80354881],[ 148,  32.6521225 ]]
)

neighbor=np.array([[  47,  2,  1, -1, -1, -1, 20, -1,  131, 13, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1],
 [ 148,  6,  1,  144, -1, 93, -1, -1, -1, -1,  108, -1,-1, -1, -1, -1, 83, -1, -1, -1],
 [ 128, 12,  1, -1, -1, -1,  131, 13, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1,  148, -1],
 [ 131, 16,  1, -1, -1, 20, -1, -1, -1, 13, -1, -1,-1, -1,  128,  108, 47,  103, -1, -1],
 [  13, 20,  1, 20, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1,  108, -1, 47, -1, -1],
 [  20, 23,  1, -1, -1, -1, -1, -1, -1, -1, -1, 13,-1,  131, 47, -1,  103, -1, -1, -1],
 [ 108, 28,  1, -1, -1, 93, -1, 13, -1, -1,  128, -1,-1, -1, -1, -1, 83, -1,  148, -1],
 [  83, 29,  1, -1, -1,  144, -1, -1,  148, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1],
 [  77, 35,  1, -1, -1, -1,  103, -1, -1, 93, -1, -1, -1,  148, -1, -1, -1, -1,  144, -1],
 [ 103, 37,  1, -1, -1, -1, -1, -1, 20,  131, -1, -1,-1, 93, 77, -1, -1,  144, -1, -1],
 [  93, 44,  1, -1, -1,  103, -1, -1, -1, -1, -1, -1,-1,  108, -1, -1, -1, 77, -1, -1],
 [ 144, 47,  1, -1, -1, -1, -1, -1, -1,  103, 77,  148,-1, 83, -1, -1, -1, -1, -1, -1]]
)

neighbor2=np.array([[129,   1,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,86,  -1,  16,  -1,  -1,  47,  -1,  46,  -1,  -1],
 [ 47,   2,   1,  -1,  46,  -1,  -1, 129, 131,  -1,-1,  -1,  23,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [ 1002,   3,   0,   0,   0,   0,   0,   0,   0,   0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [ 86,   4,   1,  20,  -1,  -1,  -1,  -1,  -1,  -1,-1,  -1,  -1,  -1,  16,  -1,  -1,  46,  66,  -1],
 [  137,   5,   1,  -1,  -1,  -1,  -1,  62,  -1, 108,-1,  -1,  -1,  -1,  80,  -1,  83,  -1,  -1,  -1],
 [  148,   6,   1,  -1,  -1,  11,  -1,  -1,  -1,  81,108,  -1,  -1,  -1,  -1, 137,  83, 118,  -1,  -1],
 [  102,   7,   1,  69, 144,  -1, 121,  -1,  11, 148,-1,  -1,  -1,  -1,  -1,  83,  -1,  -1,  -1,  -1],
 [ 69,   8,   1, 125, 144,  -1,  95,  99,  -1,  -1,121, 102,  -1,  83,  -1,  -1,  -1,  -1,  -1,  -1],
 [ 1000,   9,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [ 1001,  10,   0,   0,   0,   0,   0,   0,   0,   0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [ 15,  11,   1,  23, 131,  -1,  -1,  -1,  33,  -1,-1,  -1,  -1,  -1,  -1, 128,  38,  -1,  -1,  -1],
 [  128,  12,   1,  -1,  -1,  62,  -1,  15,  -1,  -1,-1,  -1,  -1,  -1,  -1,  -1,  -1, 137, 108,  -1],
 [ 29,  13,   1,  -1,  -1,  -1,  -1,  -1,  38,  -1,-1,  62,  -1,  -1,  81,  -1,  11,  -1,  -1,  -1],
 [ 38,  14,   1,  -1,  23,  -1,  -1,  -1,  15,  -1,-1,  -1,  -1,  -1,  -1,  62,  29,  -1, 103,  -1],
 [ 23,  15,   1,  -1,  47,  -1, 131,  -1,  -1,  -1,-1,  15,  38,  -1,  -1,  -1,  -1, 103,  -1,  -1],
 [  131,  16,   1,  -1,  -1,  -1,  -1,  -1,  16,  33,-1,  -1,  15,  -1,  23,  -1,  47,  -1,  -1,  -1],
 [ 33,  17,   1,  -1,  13,  -1,  -1,  -1,  -1,  -1,-1,  -1,  14,  -1,  -1, 108,  15,  16,  -1,  -1],
 [ 14,  18,   1,  -1,  33,  -1,  -1,  -1,  -1,  -1,-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [ 16,  19,   1,  -1,  20,  -1,  86,  -1,  -1,  13,-1,  -1,  -1,  -1,  -1,  -1,  47,  -1,  -1,  -1],
 [ 13,  20,   1,  20,  -1,  -1,  -1,  -1,  -1,  -1,-1,  -1,  33,  -1,  -1,  -1,  -1,  16,  -1,  -1],
 [ 46,  21,   1,  -1,  -1,  -1,  -1,  -1,  -1,  86,-1,  -1,  23,  -1,  -1, 103,  56,  -1,  66,  -1],
 [ 96,  22,   1,  -1,  -1,  -1,  66,  -1,  46,  59,-1,  56,  -1,  -1,  -1,  -1,  -1, 123,  -1,  -1],
 [ 20,  23,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  86,  16,  -1,  -1,  -1,  -1,  -1,  66,  -1],
 [ 66,  24,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,86,  -1,  -1,  -1,  96, 123,  -1,  -1,  -1,  -1],
 [ 81,  25,   1,  11,  -1,  -1,  29,  62,  -1,  -1,-1,  -1,  -1, 108,  -1,  -1,  -1, 148,  -1,  -1],
 [ 11,  26,   1,  -1,  -1,  -1,  30,  -1,  29,  -1,-1,  81,  -1, 148,  -1,  -1, 102,  -1, 121,  -1],
 [ 62,  27,   1,  29,  -1,  -1,  -1,  38,  -1,  -1,-1,  -1,  -1, 128, 108,  81,  -1,  -1,  -1,  -1],
 [  108,  28,   1,  -1,  -1,  81,  62,  33,  -1,  -1,128,  -1,  -1, 110,  80,  -1,  83, 137, 148,  -1],
 [ 83,  29,   1, 118, 126, 144,  -1, 102, 148,  -1,-1,  80,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [  118,  30,   1,  -1,  -1,  -1,  -1, 126,  -1, 148,-1,  83,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [  110,  31,   1,  -1,  -1, 108,  -1,  -1,  -1,  -1,-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  80,  -1],
 [ 80,  32,   1, 118,  -1,  -1, 108,  -1,  -1,  -1,110,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
 [ 1003,  33,   0,   0,   0,   0,   0,   0,   0,   0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [ 95,  34,   1,  -1,  -1,  -1,  -1,  -1,  59,  -1,-1,  77,  -1,  26,  69,  -1,  -1, 144,  -1,  -1],
 [ 77,  35,   1,  95,  -1,  59,  -1,  -1,  -1,  93, -1,  -1,  -1,  30,  -1,  -1,  -1,  99,  26,  -1],
 [ 26,  36,   1,  -1,  -1,  95,  -1,  -1,  -1,  -1,77,  -1,  -1,  99,  -1,  -1,  -1,  -1,  -1,  -1],
 [  103,  37,   1,  -1,  -1,  -1,  -1,  46,  -1,  23,38,  -1,  -1,  30,  11,  -1,  -1,  56,  -1,  -1],
 [ 56,  38,   1,  96,  -1,  59,  -1,  -1,  46, 103,-1,  93,  -1,  77,  -1,  -1,  -1,  -1,  -1,  -1],
 [  123,  39,   1,  -1,  -1,  -1,  -1,  66,  -1,  59,-1,  -1,  -1,  -1, 144,  70,  -1,  -1,  -1,  -1],
 [ 59,  40,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,-1,  -1,  -1,  56,  -1,  -1,  95, 123,  -1,  -1],
 [  121,  41,   1,  -1,  -1,  99,  -1,  30,  -1,  -1,11,  -1,  -1,  -1, 102,  -1,  -1,  -1,  69,  -1],
 [ 99,  42,   1,  -1,  -1,  95,  -1,  -1,  -1,  77,30,  -1,  -1, 121,  -1,  69,  -1,  -1,  -1,  -1],
 [ 30,  43,   1,  -1,  -1,  77,  93,  -1,  -1,  -1,-1,  -1,  -1,  -1,  11, 121,  -1,  -1,  99,  -1],
 [ 93,  44,   1,  56,  -1, 103,  -1,  -1,  -1,  -1,-1,  -1,  -1,  -1,  30,  -1,  -1,  77,  -1,  -1],
 [  125,  45,   1,  -1,  -1,  70, 144,  -1,  -1,  -1,-1,  69,  83,  -1, 126, 118,  -1,  -1,  -1,  -1],
 [  126,  46,   1,  -1,  -1,  -1, 125,  -1,  -1,  -1,-1,  -1,  83,  -1,  -1, 118,  -1,  -1,  -1,  -1],
 [  144,  47,   1,  -1,  -1,  -1, 123,  -1,  59,  95,-1,  -1,  69,  83, 125,  70,  -1,  -1,  -1,  -1],
 [ 70,  48,   1,  -1,  -1,  -1,  -1, 123,  -1,  -1,-1,  -1,  -1, 125,  -1,  -1,  -1,  -1,  -1,  -1]]
)


#print poolput.shape
#print neighbor.shape
#print neighbor2.shape
inputheight=(len(poolput)+1)*3+len(poolput)
heigth=4*(len(poolput)-1)+4
weigth=len(poolput[0])
#训练通道按照邻接矩阵重新排序
newtraindata = np.zeros((inputheight,weigth))
outdata=np.zeros((heigth,weigth))
#print newtraindata.shape
channellist = poolput[:, 0]
#补充空节点
outsize = (len(poolput) - 1) * 4 + 4
outputs = np.zeros([outsize], np.float32)
s=len(neighbor)
obj = np.zeros(20)
obj[0:] = -2
for i in range(s):
    for j in range(3):
        neighbor = np.insert(neighbor, j+i*4, obj, 0)
        newtraindata[j+i*4][0]=-1
    p1 = np.where(channellist == neighbor[(i*4+3)][0])[0][0]
    newtraindata[i*4+3]=poolput[p1]
obj1 = np.zeros(20)
obj1[0:] = -2
neighbor = np.insert(neighbor, s*4, obj1, 0)
neighbor = np.insert(neighbor, s*4+1, obj1, 0)
neighbor = np.insert(neighbor, s*4+2, obj1, 0)
newtraindata[s * 4][0] = -1
newtraindata[s * 4+1][0] = -1
newtraindata[s * 4+2][0] = -1
#print neighbor.shape
#卷积运算
weigh=np.random.rand(4,1)
#print weigh
for k in range(heigth):
    outdata[k][1]=np.dot(newtraindata[:, 1][k:k+4],weigh)
#for m in range(48):
outdata[:,0]=neighbor2[:, 0]
print outdata[:,0]
deletelist=np.where(neighbor2[:, 2] == 0)
outdata= np.delete(outdata,deletelist , 0)
print outdata[:,0]
print neighbor2[:, 0]
print neighbor2[:, 2]
# print newtraindata[:,0]
#i=48,k=4,s=4,p=0
#o′=s(i′−1)+k−2p  4*11+4
#o=⌊i+2p−k/s⌋+1.   51-4+1=48