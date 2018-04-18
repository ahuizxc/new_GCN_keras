#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import Mycon as con
import random
import pickle
import os
import Readdata as Rd
import glob
import copy
QUADRANT_NUM = 16
FEATURE_NUM = 5
POINT_NUM = 153
SKIP = QUADRANT_NUM + 1
LENGTH=152
learning_rate=1e-2
max_iter=1



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

def conv2d(train_data,neighbor,weight,outsize):

    length = train_data.shape[0]
    inchannels = train_data.shape[1] - 1
    out_train_data = np.zeros((length, outsize + 1), np.float32)
    list = neighbor[:, 0]
    train_list = train_data[:, 0]
    for i in range(length):
        for out in range(outsize):
            index = train_data[i][0]
            p = np.where(list == index)[0][0]
            out_train_data[i][0] = index
            for c in range(inchannels):
                out_train_data[i][out + 1] += train_data[i][c + 1] * weight[out][c][-1]
                for k in range(17):
                    nn = neighbor[p][k + 5]
                    if nn != -1:
                        newp = np.where(train_list == nn)[0][0]
                        out_train_data[i][out + 1] += train_data[newp][c+1] * weight[out][c][k]

    print (out_train_data.shape)
    return out_train_data


def insernodetoneighbor(b,neighbor,m):
    #print neighbor[:, 4]
    for index, i in enumerate(b.nodes()):
        if type(i) == str:
            obj=np.zeros((1,73))-1
            obj[0][0]=int(filter(str.isdigit, i))
            obj[0][m]=b.node[i]['name']
            obj[0][4] = 0
            hangshu = neighbor.shape[0]
            neighbor = np.insert(neighbor, hangshu, obj, 0)

        else:
            p1 = np.where(neighbor[:, 0] == i)[0][0]
            neighbor[p1][m] = int(b.node[i]['name'])
    # 按照第二列数据排序
    data = neighbor
    #print neighbor.shape,sorted(neighbor[:,m])
    #print sorted(neighbor[:, m-1])
    return data
# def insernodetoneighbor2(b,neighbor):
#     #print neighbor.shape
#     for index, i in enumerate(b.nodes()):
#         if type(i) == str:
#             hangshu = neighbor.shape[0]
#             obj = [int(filter(str.isdigit, i)), -1, b.node[i]['name'], -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                    0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             neighbor = np.insert(neighbor, hangshu, obj, 0)
#         else:
#             p1 = np.where(neighbor[:, 0] == i)[0][0]
#             neighbor[p1][2] = int(b.node[i]['name'])
#
#     # 按照第三列数据排序
#     data = neighbor[neighbor[:, 2].argsort()]
#     return data
# def insernodetoneighbor3(b,neighbor):
#     for index, i in enumerate(b.nodes()):
#         if type(i) == str:
#             hangshu = neighbor.shape[0]
#             obj = [int(filter(str.isdigit, i)), -1, -1, b.node[i]['name'], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,
#                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             neighbor = np.insert(neighbor, hangshu, obj, 0)
#         else:
#             p1 = np.where(neighbor[:, 0] == i)[0][0]
#             neighbor[p1][3] = int(b.node[i]['name'])
#
#     # 按照第三列数据排序
#     data = neighbor[neighbor[:, 3].argsort()]
#     return data

def gengxinbianhao(data2):
    list2=data2[:, 2]
    list1=data2[:, 1]
    for index,i in enumerate(list2):
        if i > 0 and data2[index][1] > 0:
            n = int(data2[index][1] - 1) / 4
            for j in range(4):
                p1 = np.where(list1 == 4 * n + 1 + j)[0][0]
                data2[p1][2] = i
    return data2

def gengxinbianhao3(data2):
    list3=data2[:, 3]
    list2=data2[:, 2]
    for index,i in enumerate(list3):
        if i > 0 and data2[index][2] > 0:
             n = int(data2[index][2] - 1) / 4
             for j in range(4):
                 p = np.where(list2 == 4 * n + 1 + j)
                 for temp in p[0]:
                     data2[temp][3] = i
    return data2

def cacluate_neighbor(data,graph,center,l):
    for node in graph:
        uu = graph.neighbors(node)
        for m in uu:
            # 计算角度
            jiaodu = Rd.countjiaodu(center[m][0], center[m][1], center[node][0],center[node][1])
            # 计算象限
            xiangxian = Rd.Sortpoly(jiaodu)
            # 该分组区间的所有点的该象限都要记录下对应的二级编号
            p1=np.where(data[:, 0]==m)[0][0]
            bianhao=data[p1][l+1]
            cijibianhao = graph.node[node]['name']
            n = cijibianhao / 4
            for j in range(4):
                p = np.where(data[:, l] == (n - 1) * 4 + j + 1)[0]
                for temp in p:
                    data[temp][5+l*17 + xiangxian] = bianhao

    return data
def cacluate_neighbor2(data,graph,center,l):
    for node in graph:
        uu = graph.neighbors(node)
        for m in uu:
            # 计算角度
            jiaodu = Rd.countjiaodu(center[m][0], center[m][1], center[node][0],center[node][1])
            # 计算象限
            xiangxian = Rd.Sortpoly(jiaodu)
            p1 = np.where(data[:, 0] == m)[0][0]
            # 找到改节点对应的三级编号
            bianhao = data[p1][l+1]
            # 该点对应二级分组区间的所有该象限都要记录下对应的三级编号
            cijibianhao = graph.node[node]['name']
            n = cijibianhao / 4
            for j in range(4):
                p = np.where(data[:, l] == (n - 1) * 4 + j + 1)
                for temp in p[0]:
                    data[temp][5+17*l + xiangxian] = bianhao


def predata(path):
    path = path
    maps, channels, labels, centerpoints = get_data(path)
    maps = np.array(maps)
    centerpoints = np.array(centerpoints[0])
    channels = np.array(channels)
    g = Rd.readnodes(maps[0])
    tempresult1, firstaddnodes = Rd.julei(g, 72, 37)
    # 记录第一次编号
    #print maps[0][:, 4]
    data1 = insernodetoneighbor(firstaddnodes, maps[0],1)
    xx = copy.deepcopy(tempresult1)
    yy = copy.deepcopy(firstaddnodes)
    # 第一次分组
    poolgaraph1 = Rd.max_pool2X2(xx, yy)
    copypoolgraph1 = copy.deepcopy(poolgaraph1)
    tempresult2, secondaddnodes = Rd.julei(copypoolgraph1, 21, 10)
    # 记录第二次编号
    data2 = insernodetoneighbor(secondaddnodes, data1,2)
    data2 = gengxinbianhao(data2)
    ww = copy.deepcopy(tempresult2)
    zz = copy.deepcopy(secondaddnodes)
    # 第二次分组
    poolgaraph2 = Rd.max_pool2X2(ww, zz)

    copypoolgraph2 = copy.deepcopy(poolgaraph2)
    tempresult2, thirdaddnodes= Rd.julei(copypoolgraph2, 6, 3)
    # 记录第三次编号
    data3 = insernodetoneighbor(thirdaddnodes, data2,3)
    data3 = gengxinbianhao3(data3)
    qq = copy.deepcopy(tempresult2)
    mm = copy.deepcopy(thirdaddnodes)
    # 第三次分组
    poolgaraph3 = Rd.max_pool2X2(qq, mm)
    # 记录第一次池化后邻居节点
    data3 = cacluate_neighbor(data3, poolgaraph1, centerpoints, 1)
    #记录第二次池化邻居节点
    data3 = cacluate_neighbor(data3, poolgaraph2, centerpoints, 2)

   #记录第三次池化邻居节点
    data3 = cacluate_neighbor(data3, poolgaraph3, centerpoints, 3)

    #需要增大channel
    #记录第三次池化后节点

    return channels[0],data3



path = 'data'
channles,array=predata(path)
weigh1=np.random.randint(0,2,(3,5,18))
#第一层
traindata=conv2d(channles, array, weigh1, 3)

a,b=con.max_pool2X2(traindata,array,176,1)

#第二层
#weigh2 = np.random.randint(0, 2, (4, 3, 18))
#traindata2 = conv2d(a, b, weigh2, 4)
#c,d=con.max_pool2X2(traindata2,b,52,2)
## print c.shape,d.shape
##第三层
#weigh3 = np.random.randint(0, 2, (2, 4, 18))
#traindata3 = conv2d(c,d, weigh3, 2)
#e,f=con.max_pool2X2(traindata3,d,16,3)
## print f

