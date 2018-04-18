#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import Readdata as Rd
import random
import copy
import networkx
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
def graclusmatch(G):
    deletenodelist = []
    randomlist = G.nodes()
    slice=-9
    while len(randomlist)>0:
        if slice==-9:
            slice = random.sample(randomlist, 1)[0]
        if len(G.neighbors(slice)) > 0:
            min = 999
            nodeindex = -10
            for index in G.neighbors(slice):
                if index==slice:
                    print (slice)
                if G.degree(index) < min and index!=slice:
                    min = G.degree(index)
                    nodeindex = index
            deletenodelist.append([slice, nodeindex])
            G.remove_node(slice)
            if nodeindex not in G.nodes():
                print (nodeindex)
                print (deletenodelist)
                #print G.neighbors(slice)
                print (G.nodes())
            G.remove_node(nodeindex)
            randomlist = G.nodes()
            while len(randomlist) > 0:
                slice = random.sample(randomlist, 1)[0]
                if len(G.neighbors(slice)) > 0:
                    slice = slice
                    break
                else:
                    randomlist.remove(slice)
        else:
            randomlist.remove(slice)
            while len(randomlist)>0:
                slice=random.sample(randomlist, 1)[0]
                if len(G.neighbors(slice)) > 0:
                    slice=slice
                    break
                else:
                    randomlist.remove(slice)
    return deletenodelist

def coarsen(G,new_g,deltetlist):
    for ii in deltetlist:
        G.remove_node(ii[1])
        ss=new_g.neighbors(ii[1])
        for aa in ss:
            if aa not in [x[1] for x in deltetlist]:
                G.add_edge(ii[0],aa)
            else:
                for i in range(len(deltetlist)):
                    if(aa == deltetlist[i][1]):
                        G.add_edge(ii[0],deltetlist[i][0])
                        break
    for j in G.nodes():
        if (j, j) in G.edges():
            G.remove_edge(j, j)
    return G
#diyicibudian
def paixu1(result1,result2,matchlist2):
    pp = 1
    oo = 1
    kk = 1
    add =1
    for xx in result2.nodes():
        result2.node[xx]['name'] = copy.deepcopy(pp)
        pp = pp + 1
        if xx in [x[0] for x in matchlist2]:
            for i in range(len(matchlist2)):
                if (xx == matchlist2[i][0]):
                    result1.node[xx]['name'] = copy.deepcopy(oo)
                    result1.node[matchlist2[i][1]]['name'] = copy.deepcopy(oo + 1)
                    oo = oo + 2
                    break
        else:
            result1.add_node('none'+bytes(add))
            result1.node['none'+bytes(add)]['shuxing'] = 0
            result1.node['none'+bytes(add)]['name'] = copy.deepcopy(oo)
            result1.node[xx]['name'] = copy.deepcopy(oo + 1)
            add=add+1
            oo = oo + 2

    return result2,result1
#diercibudian
def paixu2(G,result1,matchlist1):
    #k=len(G)
    k=1000
    for point in result1.nodes():
        if point in [x[0] for x in matchlist1]:
            for i in range(len(matchlist1)):
                if(point==matchlist1[i][0]):
                    G.node[point]['name']=result1.node[point]['name']*2-1
                    G.node[matchlist1[i][1]]['name'] = result1.node[point]['name'] * 2
                    break
        elif(result1.node[point]['shuxing']==0):
            G.add_node('none'+bytes(k))
            G.node['none'+bytes(k)]['name'] = result1.node[point]['name']*2-1
            k=k+1
            G.add_node('none'+bytes(k))
            G.node['none'+bytes(k)]['name'] = result1.node[point]['name'] * 2
            k=k+1
        else:
            G.add_node('none'+bytes(k))
            G.node['none'+bytes(k)]['name'] = result1.node[point]['name'] * 2-1
            k = k + 1
            G.node[point]['name'] = result1.node[point]['name'] * 2
    return result1,G
def julei(graph,a,b):
    graph1_0 = copy.deepcopy(graph)
    graph1_1 = copy.deepcopy(graph)
    graph1_2 = copy.deepcopy(graph)
    deltetlist = graclusmatch(graph1_0)
    while len(deltetlist)!=a:
        graph1_0=copy.deepcopy(graph1_2)
        deltetlist = graclusmatch(graph1_0)
    result1 = coarsen(graph1_1, graph1_2, deltetlist)
    graph2_0=copy.deepcopy(result1)
    graph2_1=copy.deepcopy(result1)
    graph2_2 = copy.deepcopy(result1)
    #second
    deltetlist2 = graclusmatch(graph2_0)
    while len(deltetlist2)!=b:
        graph2_0=copy.deepcopy(graph2_2)
        deltetlist2 = graclusmatch(graph2_0)
    result2 = coarsen(graph2_1, graph2_2, deltetlist2)
    result2_1, addresult1 = paixu1(result1, result2, deltetlist2)
    result1_1, addG = paixu2(graph, result1, deltetlist)
    return result2_1,addG

def max_pool2X2(graph,addG,traindata,neighbor,centerpoint):
    #print neighbor[:, 3]
    for index,i in enumerate (addG.nodes()):
        if type(i)==str:
            hangshu=len(neighbor)
            obj = np.zeros(20)
            obj[0]=int(filter(str.isdigit, i))
            obj[1]=addG.node[i]['name']
            neighbor=np.insert(neighbor, hangshu, obj, 0)
            width=len(traindata[0])
            obj2=np.zeros(width)
            obj2[0]=int(filter(str.isdigit, i))
            obj2[1:]=-1000
            traindata=np.insert(traindata,hangshu,obj2,0)
        else:
            neilist = neighbor[:, 0]
            weizhi = np.where(neilist == i)[0][0]
            neighbor[weizhi][1]=addG.node[i]['name']
    #按照第二列数据排序
    data = neighbor[neighbor[:, 1].argsort()]
    decondata=data[:]
    resultnodes = graph.nodes()
    for node in resultnodes:
        temp = graph.node[node]['name']
        neighborlist = data[:, 1]
        p1 = np.where(neighborlist == (4 * temp - 3))[0][0]
        p2 = np.where(neighborlist == (4 * temp - 2))[0][0]
        p3 = np.where(neighborlist == (4 * temp - 1))[0][0]
        p4 = np.where(neighborlist == (4 * temp ))[0][0]
        #邻接矩阵索引数组
        F_index=[p1,p2,p3,p4]

        traindatalist = traindata[:, 0]
        nnweizhi1 = np.where(traindatalist == data[p1][0])[0][0]
        nnweizhi2 = np.where(traindatalist == data[p2][0])[0][0]
        nnweizhi3 = np.where(traindatalist == data[p3][0])[0][0]
        nnweizhi4 = np.where(traindatalist == data[p4][0])[0][0]
        T_index=[nnweizhi1,nnweizhi2,nnweizhi3,nnweizhi4]
        #取通道最大值
        F_tongdao=[traindata[nnweizhi1][1],traindata[nnweizhi2][1],traindata[nnweizhi3][1],traindata[nnweizhi4][1]]
        M_index=F_tongdao.index(max(F_tongdao))
        p=F_index[M_index]
        newdata=data[p][0]
        del F_index[M_index]
        del T_index[M_index]
        #print[F_index[0], F_index[1], F_index[2]]
        #print [data[F_index[0]][1], data[F_index[1]][1], data[F_index[2]][1]]
        #print [data[F_index[0]][0],data[F_index[1]][0],data[F_index[2]][0]]
        traindata = np.delete(traindata,T_index , 0)
        data = np.delete(data, F_index, 0)
        if newdata!=node:
            uu = graph.neighbors(node)
            graph.remove_node(node)
            graph.add_node(newdata)
            for e in uu:
                graph.add_edge(newdata, e)
    newnodelist=graph.nodes()
    for k in newnodelist:
        list = data[:, 0]
        p = np.where(list ==k)[0][0]
        for s in range(17):
            data[p][3+s]=-1
        #print len(graph)
        N_neighbors = graph.neighbors(k)
        for s in N_neighbors:
            s=int(s)
            k=int(k)
            jiaodu = countjiaodu(centerpoint[s][1], centerpoint[s][0],centerpoint[k][1], centerpoint[k][0])
            xiangxian = int(Sortpoly(jiaodu))
            data[p][xiangxian+3] = s
    #print decondata[:, 3]
    return data,traindata,decondata

def Deconv(poolinput,weights,neighbor,lastneighbor):
    # i=48,k=4,s=4,p=0
    # o′=s(i′−1)+k−2p  4*11+4
    # o=⌊i+2p−k/s⌋+1.   51-4+1=48
    inputheight = (len(poolinput) + 1) * 3 + len(poolinput)
    heigth = 4 * (len(poolinput) - 1) + 4
    weigth = len(poolinput[0])
    # 训练通道按照邻接矩阵重新排序
    newtraindata = np.zeros((inputheight, weigth))
    outdata = np.zeros((heigth, weigth))
    # print newtraindata.shape
    channellist = poolinput[:, 0]
    # 补充空节点
    outsize = (len(poolinput) - 1) * 4 + 4
    outputs = np.zeros([outsize], np.float32)
    s = len(neighbor)
    obj = np.zeros(20)
    obj[0:] = -2
    for i in range(s):
        for j in range(3):
            neighbor = np.insert(neighbor, j + i * 4, obj, 0)
            newtraindata[j + i * 4][0] = -1
        p1 = np.where(channellist == neighbor[(i * 4 + 3)][0])[0][0]
        newtraindata[i * 4 + 3] = poolinput[p1]
    obj1 = np.zeros(20)
    obj1[0:] = -2
    neighbor = np.insert(neighbor, s * 4, obj1, 0)
    neighbor = np.insert(neighbor, s * 4 + 1, obj1, 0)
    neighbor = np.insert(neighbor, s * 4 + 2, obj1, 0)
    newtraindata[s * 4][0] = -1
    newtraindata[s * 4 + 1][0] = -1
    newtraindata[s * 4 + 2][0] = -1
    # 卷积运算
    for k in range(heigth):
        outdata[k][1] = np.dot(newtraindata[:, 1][k:k + 4], weights)
    outdata[:, 0] = lastneighbor[:, 0]
    deletelist = np.where(lastneighbor[:, 2] == 0)
    outdata = np.delete(outdata, deletelist, 0)
    return outdata



def Sortpoly(jiaodu):
    d=999
    chushu=np.pi/8
    d=jiaodu/chushu
    return d

def countjiaodu(pointx,pointy,centerx,centery):
    x=pointx-centerx
    y=pointy-centery
    if x==0:
        x=x+0.001
    normal=np.arctan(y/x)
    if(x>=0 and y>=0):
        jiaodu= normal
    elif(x<0 and y<0):
        jiaodu=np.pi+normal
    elif(x<0 and y>0):
        jiaodu=np.pi+normal
    elif(x>0,y<0):
        jiaodu = 2*np.pi+normal
    return jiaodu

