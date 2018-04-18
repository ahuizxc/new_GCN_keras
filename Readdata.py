# -*-coding:utf8-*-#
import networkx as nx
import random
import glob
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
FEATURE_NUM = 5
POINT_NUM = 153
LENGTH=152


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    return None

def preprocess(x):
    centerpoint = []
    neighbor = np.zeros(( POINT_NUM, 20), np.float32)
    channels = np.zeros((POINT_NUM, 6), np.float32)
    for j in range(POINT_NUM):
        neighbor[j][0] = j
        neighbor[j][1] = -1
        neighbor[j][2] = 1
        for s in range(17):
            neighbor[j][s+3]=-1
    def get_info(k):
        return x["polygon" + str(k)]
    for i in range(POINT_NUM):
        #质心
        info=get_info(i)
        info_center = info[2]
        centerpoint.append(info_center)
        #邻居节点
        info_neighbors = info[1][0]
        for match in info_neighbors:
            neighbor[i][match[0] + 3] = match[1]
        #属性通道
        info_channels=info[0]
        #channels=np.zeros((POINT_NUM, 6), np.float32)
        channels[i][0]=i
        for channel in range(5):
            channels[i][channel+1]=info_channels[channel]
    return neighbor,centerpoint,channels

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

def readnodes(pointlist):
    Graph = nx.Graph()
    edges = []
    for i in range(len(pointlist)):
        for j in range(len(pointlist[i]) - 3):
            if pointlist[i][j + 3] != -1:
                edge = (pointlist[i][0], pointlist[i][j + 3], {'weight': 1})
                edges.append(edge)
    for ii in edges:
        Graph.add_node(ii[0])
        Graph.add_node(ii[1])
        #print ii
    Graph.add_edges_from(edges)
    nx.set_node_attributes(Graph, 'shuxing', 1)
    #print sorted(pointlist[:, 0])
    #print sorted(Graph.nodes())
    return Graph