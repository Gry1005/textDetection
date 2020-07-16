import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, Add
from keras.layers import Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate,UpSampling2D, Lambda
from keras.initializers import glorot_uniform
from keras.models import Model
import keras as K




def model_U_ResNet50_Centerline_Localheight():

    def identity_block(X, f, filters, stage, block):
        """
        实现图3的恒等块
        参数：
            X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
            f - 整数，指定主路径中间的CONV窗口的维度
            filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
            stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
            block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        返回：
            X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)
        """

        # 定义命名规则
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        # 获取过滤器
        F1, F2, F3 = filters

        # 保存输入数据，将会用于为主路径添加捷径
        X_shortcut = X

        # 主路径的第一部分
        ##卷积层
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                   name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
        ##归一化
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        ##使用ReLU激活函数
        X = Activation("relu")(X)

        # 主路径的第二部分
        ##卷积层
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                   name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
        ##归一化
        X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        ##使用ReLU激活函数
        X = Activation("relu")(X)

        # 主路径的第三部分
        ##卷积层
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                   name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
        ##归一化
        X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
        ##没有ReLU激活函数

        # 最后一步：
        ##将捷径与输入加在一起
        X = Add()([X, X_shortcut])
        ##使用ReLU激活函数
        X = Activation("relu")(X)

        return X

    def convolutional_block(X, f, filters, stage, block, s=2):
        """
        实现图5的卷积块
        参数：
            X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
            f - 整数，指定主路径中间的CONV窗口的维度
            filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
            stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
            block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
            s - 整数，指定要使用的步幅
        返回：
            X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
        """

        # 定义命名规则
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        # 获取过滤器数量
        F1, F2, F3 = filters

        # 保存输入数据
        X_shortcut = X

        # 主路径
        ##主路径第一部分
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
                   name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = Activation("relu")(X)

        ##主路径第二部分
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                   name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        X = Activation("relu")(X)

        ##主路径第三部分
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                   name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

        # 捷径
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                            name=conv_name_base + "1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

        # 最后一步
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)

        return X

    def ResNet50(input_shape=(512, 512, 3), classes=6):
        """
        实现ResNet50
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        参数：
            input_shape - 图像数据集的维度
            classes - 整数，分类数
        返回：
            model - Keras框架的模型
        """

        # 定义tensor类型的输入数据

        X_input = Input(input_shape,name='input')

        inputs=X_input

        # 0填充
        X = ZeroPadding2D((3, 3))(X_input)

        # stage1
        x0 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
                   kernel_initializer=glorot_uniform(seed=0))(X)
        x0 = BatchNormalization(axis=3, name="bn_conv1")(x0)
        x0 = Activation("relu")(x0)
        x0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x0)

        # stage2
        x1 = convolutional_block(x0, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
        x1 = identity_block(x1, f=3, filters=[64, 64, 256], stage=2, block="b")
        x1 = identity_block(x1, f=3, filters=[64, 64, 256], stage=2, block="c")

        # stage3
        x2 = convolutional_block(x1, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
        x2 = identity_block(x2, f=3, filters=[128, 128, 512], stage=3, block="b")
        x2 = identity_block(x2, f=3, filters=[128, 128, 512], stage=3, block="c")
        x2 = identity_block(x2, f=3, filters=[128, 128, 512], stage=3, block="d")

        # stage4
        x3 = convolutional_block(x2, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
        x3 = identity_block(x3, f=3, filters=[256, 256, 1024], stage=4, block="b")
        x3 = identity_block(x3, f=3, filters=[256, 256, 1024], stage=4, block="c")
        x3 = identity_block(x3, f=3, filters=[256, 256, 1024], stage=4, block="d")
        x3 = identity_block(x3, f=3, filters=[256, 256, 1024], stage=4, block="e")
        x3 = identity_block(x3, f=3, filters=[256, 256, 1024], stage=4, block="f")

        # stage5
        x4 = convolutional_block(x3, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
        x4 = identity_block(x4, f=3, filters=[512, 512, 2048], stage=5, block="b")
        x4_take = identity_block(x4, f=3, filters=[512, 512, 2048], stage=5, block="c")

        # 均值池化层
        x4 = AveragePooling2D(pool_size=(2, 2), padding="same")(x4)

        '''
        # 输出层
        X = Flatten()(X)
        X = Dense(classes, activation="softmax", name="fc" + str(classes),
                  kernel_initializer=glorot_uniform(seed=0))(X)
        
        '''

        # ------以上为ResNet50的部分

        f1 = x4_take

        #unpool
        f1=Lambda(lambda x: tf.image.resize(x, (32,32)))(f1)

        f2 = x3

        h1 = Concatenate()([f2, f1])

        h1 = layers.Conv2D(128, (1, 1),
                           activation='relu',
                           padding='same',
                           name='up1_1')(h1)

        h1 = layers.Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same',
                           name='up1_2')(h1)

        h2 = Concatenate()([x2, UpSampling2D((2, 2))(h1)])

        h2 = layers.Conv2D(64, (1, 1),
                           activation='relu',
                           padding='same',
                           name='up2_1')(h2)
        h2 = layers.Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           name='up2_2')(h2)

        #unpool
        h3 = Concatenate()([Lambda(lambda x: tf.image.resize(x, (128,128)))(x1), UpSampling2D((2, 2))(h2)])

        h3 = layers.Conv2D(32, (1, 1),
                           activation='relu',
                           padding='same',
                           name='up3_1')(h3)
        h3 = layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same',
                           name='up3_2')(h3)

        #unpool
        h4_take = Concatenate()([Lambda(lambda x: tf.image.resize(x, (256,256)))(x0), UpSampling2D((2, 2))(h3)])

        h4 = layers.Conv2D(32, (1, 1),
                           activation='relu',
                           padding='same',
                           name='up4_1')(h4_take)
        h4 = layers.Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same',
                           name='up4_2')(h4)

        h5 = Concatenate()([inputs, UpSampling2D((2, 2))(h4)])
        h5 = layers.Conv2D(16, (1, 1),
                           activation='relu',
                           padding='same',
                           name='up5_1')(h5)
        ################## output for TEXT/NON-TEXT ############

        o1 = layers.Conv2D(3, (3, 3),
                           activation='softmax',
                           padding='same',
                           name='up5_2')(h5)
        ################## output for centerline /other ###########
        h41 = layers.Conv2D(32, (1, 1),
                            activation='relu',
                            padding='same',
                            name='up41_1')(h4_take)
        h41 = layers.Conv2D(32, (3, 3),
                            activation='relu',
                            padding='same',
                            name='up41_2')(h41)

        h51 = Concatenate()([inputs, UpSampling2D((2, 2))(h41)])
        h51 = layers.Conv2D(16, (1, 1),
                            activation='relu',
                            padding='same',
                            name='up51_1')(h51)

        o11 = layers.Conv2D(2, (3, 3),
                            activation='softmax',
                            padding='same',
                            name='up51_2')(h51)

        # ------ local height regression ------

        #change!!!!
        o5 = Concatenate()([inputs, UpSampling2D((2, 2))(h41)])
        o5 = layers.Conv2D(16, (1, 1),
                            activation='relu',
                            padding='same',
                            name='up61_1')(o5)

        o5 = layers.Conv2D(2, (3, 3),
                            activation='relu',
                            padding='same',
                            name='up61_2')(o5) #512,512,2

        o5 = layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same',
                                    activation='relu', name='regress-4-7')(o5)  # 512,512, 1


        # 创建模型
        model = Model(inputs=inputs, outputs=[o1, o11, o5], name='U-ResNet50-model-Localheight')

        return model

    return ResNet50()

