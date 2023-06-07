import os.path
from model import LSTM_model,AutoEncoder_model,AutoEncoder_CNN_model,AE_LSTM_model,LSTM_GANomaly_model
from datasets import dataWindowing,a_rawDataSet,a_normDataSet,distance,a_anomalyDataSet,custom_similarity,AnomalyDection_LSTM,AnomalyDection_GEN,AnomalyDection_AE
import numpy as np
import tensorflow as tf
import datetime as time
# from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
tf.data.experimental.enable_debug_mode()
logdir="logs/fit/" + time.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
def LSTM_(trainData, testData, time_step=60, features=26, epochs=20, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = LSTM_model(time_step=time_step, features=features)
    if readModel:
        if os.path.exists('./save/LSTM.h5'):
            tf.print('--------------Loading Model--------------')
            model = tf.keras.models.load_model('./save/LSTM.h5')
        else:
            tf.print('--------------Model is not exist--------------')
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy','mse'])
    ####################################
    # train
    ####################################
    if train:
        windowData = dataWindowing(trainData)
        history = model.fit(windowData,trainData,epochs=epochs,batch_size=batch_size,callbacks=[tensorboard_callback])
        plt.plot(history.history['mse'],'b',label='mse')
        plt.plot(history.history['accuracy'],'r',label='accuracy')
        plt.title('train loss and accuracy of LSTM')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.show()
        if save:
            model.save('./save/LSTM.h5')
    ####################################
    # test
    ####################################
    if test:
        testData1 = dataWindowing(testData)
        reconData = model.predict(testData1)
        #20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData,testData)
        difference = np.abs(testData-reconData)
        plt.plot(difference[:,0])
        plt.title('distance of row[0]:LSTM')
        plt.show()
        plt.plot(cos)
        plt.title('reconData cosine_similarity:LSTM')
        plt.show()
        print('LSTM,测试数据余弦相似度：',np.mean(cos),'    欧氏距离：',dist)
    ####################################
    # anomaly
    ####################################
    if anomaly:
        noise = np.random.normal(0,0.1,testData.shape)
        testData = noise + testData
        testData1 = dataWindowing(testData)
        reconData = model.predict(testData1)
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        difference = np.abs(testData - reconData)
        plt.plot(difference[:, 0])
        plt.title('anomaly,distance of row[0]:LSTM')
        plt.show()
        plt.plot(cos)
        plt.title('anomaly,reconData cosine_similarity:LSTM')
        plt.show()
        print('LSTM,异常数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
def AutoEncoderCNN(trainData, testData, features=26, epochs=20, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = AutoEncoder_CNN_model(features=features)
    if readModel:
        if os.path.exists('./save/AutoEncoder.h5'):
            tf.print('--------------Loading Model--------------')
            model = tf.keras.models.load_model('./save/AutoEncoderCNN.h5')
        else:
            tf.print('--------------Model is not exist--------------')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
    ####################################
    # train
    ####################################
    if train:
        history = model.fit(trainData,trainData,epochs=epochs,batch_size=batch_size,callbacks=[tensorboard_callback])
        plt.plot(history.history['mse'],'b',label='mse')
        plt.plot(history.history['accuracy'],'r',label='accuracy')
        plt.title('train loss and accuracy of AutoEncoderCNN')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.show()
        if save:
            model.save('./save/AutoEncoderCNN.h5')
    ####################################
    # test
    ####################################
    if test:
        reconData = model.predict(testData)
        #20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData,testData)
        difference = np.abs(testData-reconData)
        plt.plot(difference[:,0])
        plt.title('distance of row[0]:AutoEncoderCNN')
        plt.show()
        plt.plot(cos)
        plt.title('reconData cosine_similarity:AutoEncoderCNN')
        plt.show()
        print('AutoEncoderCNN,测试数据余弦相似度：',np.mean(cos),'    欧氏距离：',dist)
    ####################################
    # anomaly
    ####################################
    if anomaly:
        noise = np.random.normal(0,0.1,testData.shape)
        testData = noise + testData
        reconData = model.predict(testData)
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        # difference = np.abs(testData - reconData)
        # plt.plot(difference[:, 0])
        # plt.title('anomaly，index0:difference of Data and reconData')
        # plt.show()
        plt.plot(cos)
        plt.title('anomaly,distance of row[0]:AutoEncoderCNN')
        plt.show()
        print('AutoEncoderCNN,异常数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
def AutoEncoder(trainData, testData, features=26, epochs=20, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = AutoEncoder_model(features=features)
    if readModel:
        if os.path.exists('./save/AutoEncoder.h5'):
            tf.print('--------------Loading Model--------------')
            model = tf.keras.models.load_model('./save/AutoEncoder.h5')
        else:
            tf.print('--------------Model is not exist--------------')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
    ####################################
    # train
    ####################################
    if train:
        history = model.fit(trainData, trainData, epochs=epochs, batch_size=batch_size,
                            callbacks=[tensorboard_callback])
        plt.plot(history.history['mse'], 'b', label='mse')
        plt.plot(history.history['accuracy'], 'r', label='accuracy')
        plt.title('train loss and accuracy of AutoEncoder')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.show()
        if save:
            model.save('./save/AutoEncoder.h5')
    ####################################
    # test
    ####################################
    if test:
        reconData = model.predict(testData)
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        difference = np.abs(testData - reconData)
        plt.plot(difference[:, 0])
        plt.title('distance of row[0]:AutoEncoder')
        plt.show()
        plt.plot(cos)
        plt.title('reconData cosine_similarity:AutoEncoder')
        plt.show()
        print('AutoEncoder,测试数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
    ####################################
    # anomaly
    ####################################
    if anomaly:
        noise = np.random.normal(0, 0.1, testData.shape)
        testData = noise + testData
        reconData = model.predict(testData)
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        # difference = np.abs(testData - reconData)
        # plt.plot(difference[:, 0])
        # plt.title('anomaly，index0:difference of Data and reconData')
        # plt.show()
        plt.plot(cos)
        plt.title('anomaly:reconData cosine_similarity,AutoEncoder')
        plt.show()
        print('AutoEncoder,异常数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
def AE_LSTM(trainData, testData, features=26, epochs=20, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = AE_LSTM_model(features=features)
    if readModel:
        if os.path.exists('./save/AE_LSTM.h5'):
            tf.print('--------------Loading Model--------------')
            model = tf.keras.models.load_model('./save/AE_LSTM.h5')
        else:
            tf.print('--------------Model is not exist--------------')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
    ####################################
    # train
    ####################################
    if train:
        windowdata = dataWindowing(trainData)
        history = model.fit([trainData,windowdata], trainData, epochs=epochs, batch_size=batch_size,
                            callbacks=[tensorboard_callback])
        plt.plot(history.history['mse'], 'b', label='mse')
        plt.plot(history.history['accuracy'], 'r', label='accuracy')
        plt.title('train loss and accuracy of AE_LSTM')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.show()
        if save:
            model.save('./save/AE_LSTM.h5')
    ####################################
    # test
    ####################################
    if test:
        windowdata = dataWindowing(testData)
        reconData = model.predict([testData,windowdata])
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        difference = np.abs(testData - reconData)
        plt.plot(difference[:, 0])
        plt.title('distance of row[0]:AE_LSTM')
        plt.show()
        plt.plot(cos)
        plt.title('AE_LSTM:reconData cosine_similarity')
        plt.show()
        print('AE_LSTM,测试数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
    ####################################
    # anomaly
    ####################################
    if anomaly:
        noise = np.random.normal(0, 0.1, testData.shape)
        testData = noise + testData
        windowdata = dataWindowing(testData)
        reconData = model.predict([testData,windowdata])
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        # difference = np.abs(testData - reconData)
        # plt.plot(difference[:, 0])
        # plt.title('anomaly，index0:difference of Data and reconData')
        # plt.show()
        plt.plot(cos)
        plt.title('anomaly:reconData cosine_similarity,AE_LSTM')
        plt.show()
        print('AE_LSTM,异常数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
def LSTM_GAN(trainData, testData, epochs=10, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = LSTM_GANomaly_model()
    if readModel:
        if os.path.exists('./save/LSTM_GAN.h5'):
            tf.print('--------------Loading Model--------------')
            model = tf.keras.models.load_model('./save/LSTM_GAN.h5')
        else:
            tf.print('--------------Model is not exist--------------')
    model.compile()
    ####################################
    # train
    ####################################
    if train:
        windowdata = dataWindowing(trainData)
        history = model.fit([trainData, windowdata], trainData, epochs=epochs, batch_size=batch_size,
                            callbacks=[tensorboard_callback],validation_data=[testData,dataWindowing([testData])])
        plt.plot(history.history['mse'], 'b', label='mse')
        plt.plot(history.history['accuracy'], 'r', label='accuracy')
        plt.title('train loss and accuracy of LSTM_GAN')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.show()
        if save:
            model.save('./save/LSTM_GAN.h5')
    ####################################
    # test
    ####################################
    if test:
        windowdata = dataWindowing(testData)
        reconData = model.predict([testData, windowdata])
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        difference = np.abs(testData - reconData)
        plt.plot(difference[:, 0])
        plt.title('distance of row[0]:LSTM_GAN')
        plt.show()
        plt.plot(cos)
        plt.title('LSTM_GAN:reconData cosine_similarity')
        plt.show()
        print('LSTM_GAN,测试数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
    ####################################
    # anomaly
    ####################################
    if anomaly:
        noise = np.random.normal(0, 0.1, testData.shape)
        testData = noise + testData
        windowdata = dataWindowing(testData)
        reconData = model.predict([testData, windowdata])
        # 20000个向量，代表26个维度上的余弦相似度，可以用来计算重构误差
        cos = np.mean(custom_similarity(reconData, testData), axis=1)
        dist = distance(reconData, testData)
        # difference = np.abs(testData - reconData)
        # plt.plot(difference[:, 0])
        # plt.title('anomaly，index0:difference of Data and reconData')
        # plt.show()
        plt.plot(cos)
        plt.title('anomaly:reconData cosine_similarity,LSTM_GAN')
        plt.show()
        print('LSTM_GAN,异常数据余弦相似度：', np.mean(cos), '    欧氏距离：', dist)
def train(trainData, features=26, epochs=20, batch_size=256, train=True, test=True, readModel=True, save=True, anomaly=True):
    model = AE_LSTM_model(features=features)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse',custom_similarity])
    windowdata = dataWindowing(trainData)
    history = model.fit([trainData,windowdata], trainData, epochs=epochs, batch_size=batch_size,
                        callbacks=[tensorboard_callback])
    plt.plot(history.history['mse'], 'b', label='mse')
    plt.plot(history.history['accuracy'], 'r', label='accuracy')

    plt.title('train loss and accuracy of AE_LSTM')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.show()


def distribution(nRowread=170000, errors=100, index=5, u=0.1, sigma=0.1):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=u, sigma=sigma, index=index)

    lstm = tf.keras.models.load_model('E:\Study\Thesis\Project_items\Final\model_trained_servers\save\LSTM.h5',compile=False)
    reconData1 = lstm.predict(dataWindowing(abnormalData))
    similarity1 = custom_similarity(reconData1, abnormalData)
    threshold1 = np.mean(similarity1) * 0.9

    ae = tf.keras.models.load_model('E:\Study\Thesis\Project_items\Final\model_trained_servers\save\AutoEncoder.h5',compile=False)
    reconData2 = ae.predict(abnormalData)
    similarity2 = custom_similarity(reconData2, abnormalData)
    threshold2 = np.mean(similarity2) * 0.9

    aecnn = tf.keras.models.load_model('E:\Study\Thesis\Project_items\Final\model_trained_servers\save\AutoEncoderCNN.h5', compile=False)
    reconData3 = aecnn.predict(abnormalData)
    similarity3 = custom_similarity(reconData3, abnormalData)
    threshold3 = np.mean(similarity3) * 0.9

    ae_lstm = tf.keras.models.load_model('E:\Study\Thesis\Project_items\Final\model_trained_servers\save\AE_LSTM.h5',compile=False)
    reconData4 = ae_lstm.predict([abnormalData, dataWindowing(abnormalData)])
    similarity4 = custom_similarity(reconData4, abnormalData)
    threshold4 = np.mean(similarity4) * 0.9

    abnormal1 =[0] * nRowread  # 检测异常下标
    abnormal2 =[0] * nRowread
    abnormal3 =[0] * nRowread
    abnormal4 =[0] * nRowread
    for index in range(nRowread):
        # 将余弦相似度低于0.9的数据标记为异常
        abnormal1[index] = 1.1 if similarity1[index] < threshold1 else 0
    for index in range(nRowread):
        abnormal2[index] = 1 if similarity2[index] < threshold2 else 0
    for index in range(nRowread):
        abnormal3[index] = 1 if similarity3[index] < threshold3 else 0
    for index in range(nRowread):
        abnormal4[index] = 1 if similarity4[index] < threshold4 else 0

    indices0 = np.where(flag)
    x0 = indices0[0]
    y0 = np.ones_like(x0)  # 纵坐标固定为 1

    indices1 = np.where(abnormal1)
    x1 = indices1[0]
    y1 = np.ones_like(x1) * 1.1

    indices2 = np.where(abnormal2)
    x2 = indices2[0]
    y2 = np.ones_like(x2) * 1.2

    indices3 = np.where(abnormal3)
    x3 = indices3[0]
    y3 = np.ones_like(x3) * 1.3

    indices4 = np.where(abnormal4)
    x4 = indices4[0]
    y4 = np.ones_like(x4) * 1.4

    # 绘制散点图
    plt.figure()
    # plt.ylim(0.9, 1.6)
    plt.title('anomaly distribution')
    plt.scatter(x0, y0, alpha=1, marker='o', c='r', label='flag')  # 异常位置
    plt.scatter(x1, y1, alpha=1, marker='*', c='c', label='LSTM')  # 异常位置
    plt.scatter(x2, y2, alpha=1, marker='^', c='g', label='AE')  # 检测出的异常位置
    plt.scatter(x3, y3, alpha=1, marker='v', c='b', label='AECNN')  # 检测出的异常位置
    plt.scatter(x4, y4, alpha=1, marker='x', c='y', label='LG')  # 检测出的异常位置
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.show()

distribution()



# if __name__ == '__main__':
#     dataSets = a_normDataSet(17000)
#     trainData = dataSets[:-2000]
#     testData = dataSets[-2000:]
#     # LSTM_(trainData,testData,readModel=True,train=False,test=True,anomaly=True,epochs=20,batch_size=256,)
#     # AutoEncoder(trainData,testData,readModel=True,train=False,test=True,anomaly=True,epochs=20,batch_size=256,)
#     # AutoEncoderCNN(trainData,testData,readModel=True,train=False,test=True,anomaly=True,epochs=20,batch_size=256,)
#     # LSTM_GAN(trainData,testData,readModel=False,train=True,test=False,anomaly=False,epochs=20,batch_size=32,)
#     AE_LSTM(trainData,testData,readModel=False,train=True,test=True,anomaly=True,epochs=20,batch_size=32,)
#     # lstm = tf.keras.models.load_model('/kaggle/input/model/LSTM.h5')
#     # ae = tf.keras.models.load_model('/kaggle/input/model/AutoEncoderCNN.h5')
#     # ae_lstm = tf.keras.models.load_model('/kaggle/input/model/AE_LSTM.h5')
#     # AnomalyDection_LSTM(lstm,nRowread=17000,errors=5,index=2)
#     # AnomalyDection_GEN(ae_lstm, nRowread=170000, errors=10, index=5)
#     # AnomalyDection_AE(ae,nRowread=170000, errors=10, index=5)



'''
def AnomalyDection_AE(model,nRowread=17000,errors=10,index=0 ,u=0, sigma=1):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=u, sigma=sigma,index=index)
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
    plt.title('AE anomaly distribution')
    plt.scatter(x1, y1, alpha=1,s=10, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1,s=10, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('AE score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
def AnomalyDection_AECNN(model,nRowread=17000,errors=10,index=0 ,u=0, sigma=1):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=u, sigma=sigma,index=index)
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
    plt.title('AECNN anomaly distribution')
    plt.scatter(x1, y1, alpha=1,s=10, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1,s=10, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('AECNN score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
def AnomalyDection_LSTM(model,nRowread=17000,errors=10,index=0,u=0.1,sigma=0.05):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors,u=u, sigma=sigma,index=index)
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
    plt.title('LSTM anomaly distribution')
    plt.scatter(x1, y1, alpha=1, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('LSTM score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
def AnomalyDection_GEN(model,nRowread=17000,errors=10,index=0,u=0.1, sigma=0.05):
    abnormalData, flag = a_anomalyDataSet(nRowread=nRowread, num=errors, u=u, sigma=sigma,index=index)
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
    plt.title('LG anomaly distribution')
    plt.scatter(x1, y1, alpha=1, c='b')#异常位置
    plt.scatter(x2, y2, alpha=1, marker='*',c='r')#检测出的异常位置
    plt.show()
    ####################################
    # 画出重构数据相似度
    ####################################
    plt.figure()
    plt.title('LG score and threshold')
    plt.xlabel('time')
    plt.xlabel('similarity')
    plt.plot(flag, 'red', label='flag', alpha=0.2)
    plt.plot(similarity, 'blue', label='similarity')
    plt.plot([0, nRowread], [threshold_l, threshold_l], 'purple', alpha=0.5, label='threshold')
    plt.legend()
    plt.show()
if __name__ == '__main__':

    lstm = tf.keras.models.load_model('/kaggle/input/model/LSTM.h5')
    aecnn = tf.keras.models.load_model('/kaggle/input/model/AutoEncoderCNN.h5')
    ae = tf.keras.models.load_model('/kaggle/input/ae-lstm/AutoEncoder.h5')
    ae_lstm = tf.keras.models.load_model('/kaggle/input/model/AE_LSTM.h5')
#     old_ae_lstm = tf.keras.models.load_model('/kaggle/input/ae-lstm/AE_LSTM.h5')
    AnomalyDection_LSTM(lstm,nRowread=170000,errors=10,index=5,u=0.1,sigma=0.05)
    AnomalyDection_AE(ae,nRowread=170000,errors=10,index=5,u=0.1,sigma=0.05)
    AnomalyDection_AECNN(aecnn,nRowread=170000,errors=10,index=5,u=0.1,sigma=0.05)
    AnomalyDection_GEN(ae_lstm,nRowread=170000,errors=10,index=5,u=0.1,sigma=0.05)
'''