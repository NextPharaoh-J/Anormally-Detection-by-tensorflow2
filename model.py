import tensorflow as tf
from tensorflow.keras.layers import Input,LSTM,Dense,Conv1DTranspose,UpSampling1D,MaxPooling1D,\
    concatenate,Conv1D,Flatten,Reshape,Dropout,LeakyReLU
from tensorflow.keras import Model,regularizers
from tensorflow.keras.optimizers import Adam,RMSprop
import numpy as np

def LSTM_model(time_step=60,features=26):
    input = Input(shape=(60,26,))
    x = LSTM(64,activation='tanh',input_shape=(time_step,features),return_sequences=True,dropout=0.2)(input)
    x = LSTM(32,activation='tanh',dropout=0.2)(x)
    output = Dense(features,activation='linear')(x)
    return Model(input,output)
def Encoder_model(latent_dim=16,features=26):
    input = Input(shape=(features,))
    x = Dense(128, activation='tanh',kernel_regularizer=regularizers.l2(0.001))(input)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='tanh',kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='tanh',kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    output = Dense(latent_dim, activation='tanh')(x)
    return Model(input,output)
def Decoder_model(latent_dim=16,features=26):
    input = Input(shape=(latent_dim,))
    x = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(input)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    output = Dense(features, activation='linear')(x)
    return Model(input, output)
def AutoEncoder_model(features=26):
    encoder = Encoder_model()
    decoder = Decoder_model()
    input = Input(shape=(features,))
    lantentVector = encoder(input)
    reconData = decoder(lantentVector)
    return Model(input,reconData)
def Encoder_CNN_model(latent_dim=16,features=26):
    input = Input(shape=(features,))
    x = Reshape((features, 1))(input)
    x = Conv1D(16, 3, activation='tanh')(x)
    x = Conv1D(16, 3, activation='tanh')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='tanh')(x)
    x = Conv1D(64, 3, activation='tanh')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    output = Dense(latent_dim, activation='tanh')(x)
    return Model(input, output)
def Decoder_CNN_model(latent_dim=16, features=26):
    input = Input(shape=(latent_dim,))
    x = Dense(64, activation='tanh')(input)
    x = Reshape((8,8))(x)
    x = Conv1DTranspose(64, 3, activation='tanh', padding='same')(x)
    x = Conv1DTranspose(64, 3, activation='tanh', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(16, 3, activation='tanh', padding='same')(x)
    x = Conv1DTranspose(16, 3, activation='tanh', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(features, 3, activation='tanh', padding='same')(x)
    x = Flatten()(x)
    output = Dense(features, activation='linear')(x)
    # output = Reshape((feature,))(x)
    return Model(input, output)
def AutoEncoder_CNN_model(features=26):
    encoder = Encoder_CNN_model()
    decoder = Decoder_CNN_model()
    input = Input(shape=(features,))
    lantentVector = encoder(input)
    reconData = decoder(lantentVector)
    return Model(input, reconData)
def encoder_LSTM_model(features=26,time_step=60,latent_dim1=16,latent_dim2=8):

    input1 = Input(shape=(features,))
    encoder1 = Encoder_CNN_model(features=features,latent_dim=latent_dim1)
    latent_Vector = encoder1(input1)

    input2 = Input(shape=(time_step, features,))
    x = LSTM(64, activation='tanh', return_sequences=True, dropout=0.2)(input2)
    x = LSTM(32, activation='tanh', dropout=0.2)(x)
    output2 = Dense(latent_dim2, activation='linear')(x)
    encoder2 = Model(input2,output2)
    time_Vector = encoder2(input2)
    zippedVector = concatenate([latent_Vector,time_Vector])
    return Model([input1,input2],zippedVector)
def decoder_LSTM_model(features=26,latent_dim=24):
    decoder = Decoder_CNN_model(features=features,latent_dim=latent_dim)
    input = Input(shape=(latent_dim,))
    output = decoder(input)
    return Model(input,output)
def AE_LSTM_model(features=26,time_step=60):
    input1 = Input(shape=(features,))
    input2 = Input(shape=(time_step, features,))
    encoder = encoder_LSTM_model()
    decoder = decoder_LSTM_model()
    zippedVector = encoder([input1,input2])
    reconData = decoder(zippedVector)
    return Model([input1,input2],reconData)
def discriminator_model(features=26):
    rawdata = Input(shape=(features,))

    windowdata = Input(shape=(60,26))
    x1 = LSTM(64, activation='tanh',return_sequences=True)(windowdata)
    x1 = LSTM(32, activation='tanh')(x1)
    output1 = Dense(26, activation='sigmoid')(x1)

    input = rawdata + output1*0.1

    x = Dense(8*8,activation='tanh')(input)
    x = Reshape((8,8))(x)
    x = Conv1DTranspose(256, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(256, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(features, 3, padding='same')(x)
    x = Flatten()(x)
    x = Dense(64,activation='tanh')(x)
    output2 = Dense(1, activation='sigmoid')(x)
    return Model([rawdata,windowdata], output2)
def critic_model(features=26):
    rawdata = Input(shape=(features,))

    windowdata = Input(shape=(60,26))
    x1 = LSTM(64, activation='tanh',return_sequences=True)(windowdata)
    x1 = LSTM(32, activation='tanh')(x1)
    output1 = Dense(26,activation='tanh')(x1)

    input = rawdata + output1*0.5

    x = Dense(8*8,activation='tanh')(input)
    x = Reshape((8,8))(x)
    x = Conv1DTranspose(256, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(256, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1DTranspose(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(features, 3, padding='same')(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    output2 = Dense(1,activation='tanh')(x)
    return Model([rawdata,windowdata], output2)
class LSTM_GANomaly_model(Model):
    def __init__(self):
        super(LSTM_GANomaly_model, self).__init__()
        self.discriminator = discriminator_model()
        self.generator = AE_LSTM_model()
        self.clip_value = 0.001
        self.counter = 0
    def call(self, inputs, **kwargs):
        return self.generator(inputs)
    def similarity(self, y_true, y_pred):
        cos_similarity = tf.keras.losses.CosineSimilarity(axis=1)
        return cos_similarity(y_true, y_pred)
    def compile(self, loss=None, **kwargs):
        super(LSTM_GANomaly_model, self).compile()
        self.d_optimizer = Adam()
        self.g_optimizer = Adam()
    def train_step(self, data):
        data = data[0]
        rawData = data[0]
        windowdata = data[1]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            reconData = self.generator(data)

            y_real = self.discriminator(data)
            y_fake = self.discriminator([reconData,windowdata])

            valid = tf.ones_like(y_real, dtype=tf.float32)
            fake = tf.zeros_like(y_fake,dtype=tf.float32)

            fake_loss = tf.keras.backend.binary_crossentropy(fake, y_fake)
            real_loss = tf.keras.backend.binary_crossentropy(valid, y_real)
            d_loss = tf.reduce_mean(fake_loss*0.5) + tf.reduce_mean(real_loss*0.5)

            similarity = tf.reduce_mean(self.similarity(reconData, rawData))
            dist = tf.reduce_mean(tf.square(reconData - rawData))

            similarityLoss = 1-similarity
            distanceLoss = dist

            reconloss = similarityLoss+distanceLoss
            # loss :余弦相似度，欧式距离，交叉熵
            g_loss = tf.keras.backend.binary_crossentropy(valid, y_fake)

            # 计算梯度
            gen_gradients = gen_tape.gradient(g_loss+reconloss, self.generator.trainable_variables)
            disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

            # 设置每更新 5 次判别器再更新一次生成器
            if self.counter % 5 == 0:
                # 裁剪梯度
                disc_gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value)) for grad in disc_gradients]
                self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

            self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        # 更新计数器
        self.counter += 1
        return {'d_loss': tf.reduce_mean(d_loss), 'g_loss': tf.reduce_mean(g_loss),'y_real':tf.reduce_mean(y_real),'y_fake':tf.reduce_mean(y_fake)
            ,'recon_loss':tf.reduce_mean(reconloss),'similarity':tf.reduce_mean(similarity) , 'dist':tf.reduce_mean(dist) }