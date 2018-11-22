'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#K-means算法支持函数

#文本数据解析函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))  #将每一行的数据映射成float型
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#生成k个随机质心(质心满足数据边界之内)
def randCent(dataSet, k):
    # 得到数据样本的维度
    n = shape(dataSet)[1]
    # 初始化为一个(k,n)的矩阵
    centroids = mat(zeros((k,n)))
    # 遍历数据集的每一维度
    for j in range(n):
        # 得到该列数据的最小值
        minJ = min(np.array(dataSet)[:,j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# #测试
# datMat=loadDataSet('testSet.txt') #加载数据
# centroids=randCent(datMat, 3) #随机产生三个质心
# print(centroids) #打印质心

'''
所有支持函数正常运行之后，就可以准备实现完整的K-均值算法了。该算法会创建k个质心，
然后将每个点分配到最近的质心，再重新计算质心。这个过程重复数次，直到数据点的簇分
配结果不再改变为止。
'''


#k-均值聚类算法
#@dataSet:聚类数据集
#@k:用户指定的k个类
#@distMeas:距离计算方法，默认欧氏距离distEclud()
#@createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = shape(dataSet)[0]
    # 初始化一个(m,2)的矩阵
    clusterAssment = mat(zeros((m,2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为false
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离最正无穷；最小距离对应索引为-1
            minDist = inf; minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(np.array(centroids)[j,:],np.array(dataSet)[i,:])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                    minDist = distJI; minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i,:] = minIndex,minDist**2
        # 打印k-均值聚类的质心
        print (centroids)
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = np.array(dataSet)[nonzero(clusterAssment[:,0].A==cent)[0]]
            # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment

# #测试
# datMat=loadDataSet('testSet.txt')
# myCentroids, clustAssing =kMeans(datMat, 4)
#
# #画图
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# #画出四个质心
# for data in myCentroids:
#     x = np.array(data)[0, 0]
#     y = np.array(data)[0, 1]
#
#     ax1.scatter(x, y, c='r', marker='x')
#
# #按类别画出点
# for i in range(len(datMat)):
#
#     x = datMat[i][0]
#     y = datMat[i][1]
#     label=np.array(clustAssing[i])[0, 0]
#     if (label == 0):
#         ax1.scatter(x, y, c='r', marker='^')
#     if (label == 1):
#         ax1.scatter(x, y, c='g', marker='s')
#     if (label == 2):
#         ax1.scatter(x, y, c='b', marker='o')
#     if (label == 3):
#         ax1.scatter(x, y, c='y', marker='D')
#
# #显示所画的图
# plt.show()

'''
为了克服K-均值算法收敛于局部最小值的问题，有人提出了另一个称为二分K-均值的算法。
该算法首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，
选择哪个簇进行划分取决于对其划分是否可最大程度降低SSE的值。上述基于SSE的划分过程
不断重复，直到用户指定的簇数目为止。

二分K-均值算法的伪代码如下：
将所有点看成一个簇
当簇数目小于k时
对于每一个簇
    计算总误差
    在给定的簇上进行K-均值聚类（k=2)
    计算将该簇一分为二之后的总误差
选择使得误差最小的那个簇进行划分操作
'''

#二分K-均值聚类算法
#@dataSet:待聚类数据集
#@k：用户指定的聚类个数
#@distMeas:用户指定的距离计算方法，默认为欧式距离计算
def biKmeans(dataSet,k,distMeas=distEclud):
    #获得数据集的样本数
    m=shape(dataSet)[0]
    #初始化一个元素均值0的(m,2)矩阵
    clusterAssment=mat(zeros((m,2)))
    #获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0=mean(dataSet,axis=0).tolist()[0]
    #当前聚类列表为将数据集聚为一类
    centList=[centroid0]
    #遍历每个数据集样本
    for j in range(m):
        #计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    #循环，直至二分k-均值达到k类为止
    while (len(centList)<k):
        #将当前最小平方误差置为正无穷
        lowerSSE=inf
        #遍历当前每个聚类
        for i in range(len(centList)):
            #通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster=\
                dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat,splitClustAss=\
                kMeans(ptsInCurrCluster,2,distMeas)
            #计算该类划分后两个类的误差平方和
            sseSplit=sum(splitClustAss[:,1])
            #计算数据集中不属于该类的数据的误差平方和
            sseNotSplit=\
                sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印这两项误差值
            print('sseSplit,and notSplit:',%(sseSplit,sseNotSplit))
            #划分第i类后总误差小于当前最小总误差
            if(sseSplit+sseNotSplit)<lowerSSE:
                #第i类作为本次划分类
                bestCentToSplit=i
                #第i类划分后得到的两个质心向量
                bestNewCents=centroidMat
                #复制第i类中数据点的聚类结果即误差值
                bestClustAss=splitClustAss.copy()
                #将划分第i类后的总误差作为当前最小误差
                lowerSSE=sseSplit+sseNotSplit
        #数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        #当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=\
                len(centList)
        #同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        #连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=\
                bestCentToSplit
        #打印本次执行2-均值聚类算法的类
        print('the bestCentToSplit is:',bestCentToSplit)
        #打印被划分的类的数据个数
        print('the len of bestClustAss is:',(len(bestClustAss)))
        #更新质心列表中的变化后的质心向量
        centList[bestCentToSplit]=bestNewCents[0,:]
        #添加新的类的质心向量
        centList.append(bestNewCents[1,:])
        #更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A==\
                bestCentToSplit)[0],:]=bestClustAss
        #返回聚类结果
        return mat(centList),clusterAssment



import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print (yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
