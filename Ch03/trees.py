'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

#产生数据集合标签
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算信息熵 sum=-p1*logp1-p2*logp2-p3*logp3...
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #数据的大小
    labelCounts = {}           #字典，键=不同的标签，值=该标签出现次数
    for featVec in dataSet:    #统计不同元素的数量及其出现次数
        currentLabel = featVec[-1] #获取最后一列类别标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #如果字典里没有该类别，新建
        labelCounts[currentLabel] += 1  #已有该类别，则计数加1
    shannonEnt = 0.0   #信息熵变量初始化
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries    #计算概率
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# dataSet, labels=createDataSet()
# shannonEnt=calcShannonEnt(dataSet)
# print("原数据为：",dataSet)
# print("标签为：",labels)
# print("香农熵为：",shannonEnt)

'''
函数功能:按照给定特征划分数据集
dataSet :待划分的数据集
axis    :划分依据的特征所在下标
value   :划分依据的特征
返回结果:返回axis处，所有值为value的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: #查找特征数据指定维度中等于value的值
            #下面两句主要功能就是剔除原数据中axis轴的特征数据
            reducedFeatVec = featVec[:axis]         #取0到aixs（不包括axis）的数据
            reducedFeatVec.extend(featVec[axis+1:]) #取axis+1到最后的数据
            retDataSet.append(reducedFeatVec)       #加入到列表中
    return retDataSet

# #测试
# dataSet, labels = createDataSet()
# print("原数据为：",dataSet)
# print("标签为：",labels)
# split = splitDataSet(dataSet,0,1)  #找第0维为1的数据
# print("划分后的结果为:",split)

#选择最好的数据集划分方式，返回该特征所在列
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #最后一列用于标签
    baseEntropy = calcShannonEnt(dataSet)  #计算总的信息熵
    bestInfoGain = 0.0; bestFeature = -1   #最好的信息增益，最好的特征
    for i in range(numFeatures):           #迭代所有特征
        featList = [example[i] for example in dataSet]#获取数据集中指定列所有特征
        uniqueVals = set(featList)         #保留唯一值，去掉重复的
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #计算信息增益; 即减少熵
        if (infoGain > bestInfoGain):           #将此与迄今为止的最佳收益进行比较
            bestInfoGain = infoGain             #如果优于当前最佳，设置为最佳
            bestFeature = i
    return bestFeature                          #返回最佳特征所在列

# #测试
# dataSet, labels=createDataSet()
# bestFeature=chooseBestFeatureToSplit(dataSet)
# print(bestFeature)

#统计各类的数量，按条件返回指定值
def majorityCnt(classList):
    classCount={}  #创建字典
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0  #如果该类不在字典中，则创建该类，初始值为0
        classCount[vote] += 1 #对应类值加1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)#按指定维度反向排序
    return sortedClassCount[0][0] #返回数最多的那类标签

#创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]    #获取数据集类别列表
    if classList.count(classList[0]) == len(classList): #如果所有的类别都一样
        return classList[0]  #返回该类别
    if len(dataSet[0]) == 1: #如果数据集中只有1个数据
        return majorityCnt(classList)  #返回该类标签
    bestFeat = chooseBestFeatureToSplit(dataSet) #计算信息熵最大的特征
    bestFeatLabel = labels[bestFeat]   #获取该特征的类别标签
    myTree = {bestFeatLabel:{}}        #创建树的一个根节点
    del(labels[bestFeat])              #删除信息熵最高的标签
    featValues = [example[bestFeat] for example in dataSet] #取出数据集中信息熵最高的特征
    uniqueVals = set(featValues)       #去除重复部分
    for value in uniqueVals:
        subLabels = labels[:]       #复制所有标签，因此树木不会弄乱现有标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

# #测试
# dataSet, labels=createDataSet()
# myTree=createTree(dataSet,labels)
# print(myTree)

# 决策树的分类函数，返回当前节点的分类标签
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


###决策树的分类函数，返回当前节点的分类标签
def classify(inputTree, featLabels, testVec):  ##传入的数据为dict类型
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    ##这里表明了python3和python2版本的差别，上述两行代码在2.7中为：firstStr = inputTree.key()[0]
    secondDict = inputTree[firstStr]  ##建一个dict
    # print(secondDict)
    featIndex = featLabels.index(firstStr)  # 找到在label中firstStr的下标
    for i in secondDict.keys():
        print(i)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:  ###判断一个变量是否为dict，直接type就好
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel  ##比较测试数据中的值和树上的值，最后得到节点

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

# 测试
myData, labels = createDataSet()
print(labels)
mytree = retrieveTree(0)
print(mytree)
classify = classify(mytree, labels, [1, 0])
print(classify)