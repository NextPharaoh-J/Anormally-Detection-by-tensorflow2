import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

def a_rawDataSet(nRowread=170000,index=0):#未处理的一个VNF数据
    path = "E:/Study/Thesis/Project_items/Final/dataSet/"
    dataset = pd.read_csv(path + str(index) + '.csv', nrows=nRowread, index_col=0)  # 读取数据集行数
    return np.array(dataset)
def a_anomalyDataSet(nRowread=170000,num=10000,u=0,sigma=1,index=0):
    path = "E:/Study/Thesis/Project_items/Final/dataSet/"
    dataset = pd.read_csv(path + str(index) + '.csv', nrows=nRowread, index_col=0)  # 读取数据集行数
    dataset = np.array(dataset)
    flag = [0]*nRowread
    for i in range(num):
        #随机寻找num个时刻,Num到num+100出现噪声
        random_seed = np.random.randint(0,dataset.shape[0]-60)
        for j in range(60):
            dataset[random_seed+j] += np.random.normal(u,sigma,dataset[random_seed].shape)
            flag[random_seed+j] = 1
    dataset = (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))  # 最值归一化
    dataset[np.isnan(dataset)] = 0
    return dataset,flag
def a_normDataSet(nRowread=170000,index=0):#按列进行归一化
    path = "E:/Study/Thesis/Project_items/Final/dataSet/"
    dataset = pd.read_csv(path + str(index) + '.csv', nrows=nRowread, index_col=0)  # 读取数据集行数
    dataset = (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))  # 最值归一化
    dataset[np.isnan(dataset)] = 0
    return np.array(dataset)
def distance(array1, array2):#矩阵拉值再计算欧氏距离
    data_ = array1.flatten()
    data = array2.flatten()
    dist = np.sqrt(np.sum(np.square(data - data_)))
    return dist
def dataWindowing(datasets,time_step=60,stride=1,):
    # 在数组前面添加 time_step-1 个所有元素都为零的新数组，每个新数组的长度为 26
    pad_width = [(time_step-1, 0), (0, 0)]
    datasets = np.pad(datasets, pad_width, mode='constant', constant_values=0)
    # 计算新数组的形状
    new_shape = ((len(datasets) - time_step) // stride + 1, time_step, datasets.shape[-1])
    # 计算新数组的步幅
    new_strides = (datasets.strides[0] * stride,) + datasets.strides
    # 使用as_strided函数创建新数组
    new_data = as_strided(datasets, shape=new_shape, strides=new_strides)
    return new_data
def plotDataset(datasets,index=0):#画出数据集中第i列
    df = pd.DataFrame(datasets)
    plt.figure()
    plt.legend()
    # df[index].plot()
    plt.plot(df[index],'b')
    plt.xlabel('time')
    plt.title('')
    # plt.ylabel('数据集第'+str(index)+'列')
    plt.ylabel('CPU idle percentage')
    plt.show()

    # name = ['cpu_idle', 'cpu_stolen', 'cpu_system', 'cpu_wait_perc', 'disk_inode_used_perc','dis_space_used_perc']
def custom_similarity(matrix1, matrix2):
    """
    计算两个矩阵的余弦相似度
    :param matrix1: 一个n*26的矩阵
    :param matrix2: 另一个n*26的矩阵
    :return: 矩阵1和矩阵2的余弦相似度
    """
    dot_product = np.sum(np.multiply(matrix1, matrix2), axis=1)
    norm_matrix1 = np.linalg.norm(matrix1, axis=1)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1)
    return dot_product / (norm_matrix1 * norm_matrix2)
def AnomalyDection_GEN(model,nRowread=17000,errors=10,index=0):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=0, sigma=10,index=index)
    reconData = model.predict([abnormalData, dataWindowing(abnormalData)])
    similarity = custom_similarity(reconData, abnormalData)
    TP = FN = FP = TN = 0
    threshold_l = np.mean(similarity) * 0.9
    abnormal = [0] * nRowread  #检测异常下标
    for index in range(nRowread):
        # 将余弦相似度低于0.9的数据标记为异常
        abnormal[index] = 1 if similarity[index] < threshold_l else 0
        if abnormal[index] == flag[index] == 1: #TP
            TP += 1
        elif abnormal[index] == 1 and flag[index] == 0: #FP
            FP += 1
        elif abnormal[index] == 0 and flag[index] == 1: #FN
            FN += 1
        elif abnormal[index] == flag[index] == 0: #TN
            TN += 1
    TPR = TP/(TP+FN)    #预测成功的异常
    FPR = FP/(FP+TN)    #误判异常的正常
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/nRowread
    print(TP,FP,FN,TN,nRowread)
    print("真阳率:%f,假阳率:%f,精确率:%f,正确率:%f" % (TPR,FPR,precision,accuracy))
    ####################################
    # 画出异常与检测的异常大致分布
    ####################################
    indices = np.where(flag)
    x1 = indices[0]
    y1 = np.ones_like(x1)  # 纵坐标固定为 1
    indices = np.where(abnormal)
    x2 = indices[0]
    y2 = np.ones_like(x2)*1.5
    # 绘制散点图
    plt.figure()
    plt.ylim(0.5,2)
    plt.title('anomaly distribution')
    plt.scatter(x1, y1, alpha=1,s=3, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1,s=3, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
def AnomalyDection_AE(model,nRowread=17000,errors=10,index=0):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=0, sigma=10,index=index)
    reconData = model.predict(abnormalData)
    similarity = custom_similarity(reconData, abnormalData)
    TP = FN = FP = TN = 0
    threshold_l = np.mean(similarity) * 0.9
    abnormal = [0] * nRowread  #检测异常下标
    for index in range(nRowread):
        # 将余弦相似度低于0.9的数据标记为异常
        abnormal[index] = 1 if similarity[index] < threshold_l else 0
        if abnormal[index] == flag[index] == 1: #TP
            TP += 1
        elif abnormal[index] == 1 and flag[index] == 0: #FP
            FP += 1
        elif abnormal[index] == 0 and flag[index] == 1: #FN
            FN += 1
        elif abnormal[index] == flag[index] == 0: #TN
            TN += 1
    TPR = TP/(TP+FN)    #预测成功的异常
    FPR = FP/(FP+TN)    #误判异常的正常
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/nRowread
    print(TP,FP,FN,TN,nRowread)
    print("真阳率:%f,假阳率:%f,精确率:%f,正确率:%f" % (TPR,FPR,precision,accuracy))
    ####################################
    # 画出异常与检测的异常大致分布
    ####################################
    indices = np.where(flag)
    x1 = indices[0]
    y1 = np.ones_like(x1)  # 纵坐标固定为 1
    indices = np.where(abnormal)
    x2 = indices[0]
    y2 = np.ones_like(x2)*1.5
    # 绘制散点图
    plt.figure()
    plt.ylim(0.5,2)
    plt.title('anomaly distribution')
    plt.scatter(x1, y1, alpha=1,s=3, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1,s=3, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
def AnomalyDection_LSTM(model,nRowread=17000,errors=10,index=0):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=0, sigma=10,index=index)
    reconData = model.predict(dataWindowing(abnormalData))
    similarity = custom_similarity(reconData, abnormalData)
    TP = FN = FP = TN = 0
    threshold_l = np.mean(similarity) * 0.9
    abnormal = [0] * nRowread  #检测异常下标
    for index in range(nRowread):
        # 将余弦相似度低于0.9的数据标记为异常
        abnormal[index] = 1 if similarity[index] < threshold_l else 0
        if abnormal[index] == flag[index] == 1: #TP
            TP += 1
        elif abnormal[index] == 1 and flag[index] == 0: #FP
            FP += 1
        elif abnormal[index] == 0 and flag[index] == 1: #FN
            FN += 1
        elif abnormal[index] == flag[index] == 0: #TN
            TN += 1
    TPR = TP/(TP+FN)    #预测成功的异常
    FPR = FP/(FP+TN)    #误判异常的正常
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/nRowread
    print(TP,FP,FN,TN,nRowread)
    print("真阳率:%f,假阳率:%f,精确率:%f,正确率:%f" % (TPR,FPR,precision,accuracy))
    ####################################
    # 画出异常与检测的异常大致分布
    ####################################
    indices = np.where(flag)
    x1 = indices[0]
    y1 = np.ones_like(x1)  # 纵坐标固定为 1
    indices = np.where(abnormal)
    x2 = indices[0]
    y2 = np.ones_like(x2)*1.5
    # 绘制散点图
    plt.figure()
    plt.ylim(0.5,2)
    plt.title('anomaly distribution')
    plt.scatter(x1, y1, alpha=1,s=3, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1,s=3, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()