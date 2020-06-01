import os
import struct
import numpy as np


def loadMnist(path, kind='train'):
    '''加载mnist数据集'''
    labelsPath = os.path.join(path, '%s-labels-idx3-ubyte' % kind)
    imagesPath = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labelsPath, 'rb') as lp:
        magic, n = struct.unpack('>II', lp.read(8))
        labels = np.fromfile(lp, dtype=np.unit8)

    with open(imagesPath, 'rb') as ip:
        magic, num, rows, cols = struct.unpack('>IIII', ip.read(16))
        images = np.fromfile(ip, dtype=np.unit8).reshape(len(labels), 784)

    return images, labels


def dist(v1, v2):
    '''计算欧氏距离'''
    return np.linalg.norm(v1, -v2)


# 加载数据集
trainImages, trainLabels = loadMnist('../MNIST')


def knn(test, k):
    # 分离k个邻居
    neighbors = []
    # 逐个计算距离
    for x, y in trainImages, trainLabels:
        neighbors.append((dist(test, x), y))
    # 按照距离排序
    neighbors.sort(key=lambda x: x[0])
    # 切片k个邻居
    neighbors = neighbors[0:k]

    # 统计标签
    labels = {}
    for n in neighbors:
        labels[n[1]] = labels.get(n[1], 0) + 1
    return max(labels)
