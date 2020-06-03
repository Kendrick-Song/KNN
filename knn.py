import os
import struct
import numpy as np


def loadMnist(path, kind='train'):
    '''加载mnist数据集'''
    labelsPath = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    imagesPath = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labelsPath, 'rb') as lp:
        magic, n = struct.unpack('>II', lp.read(8))
        labels = np.fromfile(lp, dtype=np.uint8)

    with open(imagesPath, 'rb') as ip:
        magic, num, rows, cols = struct.unpack('>IIII', ip.read(16))
        images = np.fromfile(ip, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def dist(v1, v2):
    '''计算欧氏距离'''
    return np.linalg.norm(v2-v1)


'''加载数据集'''
trainImages, trainLabels = loadMnist('./mnist')


def knn(test, k):
    '''knn分类算法'''
    # 分离k个邻居
    neighbors = []
    # 逐个计算距离
    for x, y in zip(trainImages, trainLabels):
        d = dist(x, test)
        neighbors.append((d, y))
    # 按照距离排序
    neighbors.sort(key=lambda x: x[0])
    # 切片k个邻居
    neighbors = neighbors[0:k]

    # 统计标签
    labels = {}
    for n in neighbors:
        l = n[1]
        labels[l] = labels.get(l, 0) + 1
    print('Predict:', max(labels, key=labels.get))
    return max(labels, key=labels.get)


'''测试集分类'''
testImages, testLabels = loadMnist('./mnist', 't10k')  # 测试集读取
testImages, testLabels = testImages[:1000], testLabels[:1000]  # 测试集切片
k = int(input('Please enter the value of K: '))
knnLabels = np.array([knn(x, k) for x in testImages])

'''计算准确率'''
trueNum = 0
for n in range(len(knnLabels)):
    if knnLabels[n] == testLabels[n]:
        trueNum += 1
accuracy = trueNum/len(testLabels)
print('Accuracy: %.3f%%' % (accuracy*100))
