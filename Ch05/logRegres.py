'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

#加载数据
def loadDataSet():
    dataMat = []; labelMat = []  #数据，标签
    fr = open('testSet.txt')
    for line in fr.readlines():  #逐行读取
        lineArr = line.strip().split()  #去除字符串首尾空格后拆分
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #转换成Numpy矩阵
    labelMat = mat(classLabels).transpose() #转换成Numpy矩阵
    m,n = shape(dataMatrix)  #m是dataMatrix的行，n是dataMatrix的列
    alpha = 0.001            #学习率
    maxCycles = 500          #计算轮数
    weights = ones((n,1))    #3行一列
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)     #矩阵相乘，100行*3列乘以3行1列
        error = (labelMat - h)              #矢量减法，实际值-计算值
        weights = weights + alpha * dataMatrix.transpose()* error #更新权值
    return weights

# #测试
# import logRegres
# dataArr,labelMat=loadDataSet()   #加载数据
# rst=gradAscent(dataArr,labelMat) #梯度上升算法
# print(rst)

#上面已经解出了一组回归系数，它确定了不同类别数据之间的分割线。那么怎样画出该分割线，
#从而使得优化的过程便于理解呢？下面将解决这个问题。
#
#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# #测试
# dataArr,labelMat=loadDataSet()   #加载数据
# weights=gradAscent(dataArr,labelMat) #梯度上升算法
# plotBestFit(weights.getA())  # getA()将numpy矩阵转换为数组

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# #测试
# dataArr,labelMat=loadDataSet()   #加载数据
# weights=stocGradAscent0(array(dataArr),labelMat) #随机梯度上升算法
# plotBestFit(weights)  # getA()将numpy矩阵转换为数组

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# #测试
# dataArr,labelMat=loadDataSet()   #加载数据
# weights=stocGradAscent1(array(dataArr),labelMat) #随机梯度上升算法
# plotBestFit(weights)  # getA()将numpy矩阵转换为数组

#使用Logistic回归方法进行分类并不需要做很多工作，所需做的只是把测试集上每个
#特征向量以最优化方法得来的回归系数，再将该乘积结果求和，最后输入到Sigmoid函数
#中即可。如果对应的Sigmoid值大于0.5就预测类别标签为1，否则为0

#Logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#训练并测试数据
def colicTest():
    #读取训练集和测试集
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines(): #按行读取
        currLine = line.strip().split('\t') #去除空格后拆分数据
        lineArr =[]
        for i in range(21):  #提取特征数据
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr) #将特征加入训练集
        trainingLabels.append(float(currLine[21])) #将标签加入标签集
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) #随机梯度上升算法训练
    errorCount = 0; numTestVec = 0.0  #错误数量，测试集数量
    for line in frTest.readlines():   #按行读取数据
        numTestVec += 1.0             #测试集数量加1
        currLine = line.strip().split('\t') #去除空格后拆分数据
        lineArr =[]
        for i in range(21):  #提取特征数据
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]): #判断类别
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) #错误数除以总数，得到正确率
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

#多次测试，求平均结果
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

#测试
multiTest()

'''
    Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以最优化算法
完成。在最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机梯度上升算法。
    随机梯度上升算法与梯度上升算法效果相当，但占用更少的计算资源。此外，随机梯度上升是一个
在线算法，它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批处理运算。
    机器学习的一个重要问题就是如何处理缺失数据。这个问题没有标准答案，取决于实际应用中的需
求。现有一些解决方案，每种解决方案都各有优缺点。
'''