import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose
from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.layers import Lambda,TimeDistributed
from keras import layers

def model_U_VGG_Centerline_Localheight():
    # input_shape = (720, 1280, 3)
    # input_shape = (512,512,3)
    input_shape = (None, None, 3)
    inputs = Input(shape=input_shape, name='input')

    # Block 1
    x0 = layers.Conv2D(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(inputs)
    x0 = layers.Conv2D(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x0)
    x0 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x0)

    # Block 2
    x1 = layers.Conv2D(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x0)
    x1 = layers.Conv2D(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x1)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x1)

    # Block 3
    x2 = layers.Conv2D(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv1')(x1)
    x2 = layers.Conv2D(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv2')(x2)
    x2_take = layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3')(x2)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x2_take)

    # Block 4
    x3 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x2)
    x3 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(x3)
    x3_take = layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block4_conv3')(x3)
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x3_take)

    # Block 5
    x4 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x3)
    x4 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(x4)
    x4_take = layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block5_conv3')(x4)
    x4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x4_take)

    # f1 = UpSampling2D((2,2))(x4)
    # if TASK_4:
    #    f1 = ZeroPadding2D(padding=((1,0), (0,0)), name = 'f1')(f1)
    f1 = x4_take
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

    h3 = Concatenate()([x1, UpSampling2D((2, 2))(h2)])
    h3 = layers.Conv2D(32, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up3_1')(h3)
    h3 = layers.Conv2D(32, (3, 3),
                       activation='relu',
                       padding='same',
                       name='up3_2')(h3)

    h4_take = Concatenate()([x0, UpSampling2D((2, 2))(h3)])

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

    ################ Regression ###########################
    '''
    b1 = Concatenate(name='agg_feat-1')([x4_take, h1])  # block_conv3, up1_2 # 32,32,630
    b1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='agg_feat-2')(b1)  # 64,64,128
    '''

    # ------ local height regression ------
    h42 = layers.Conv2D(32, (1, 1),
                        activation='relu',
                        padding='same',
                        name='up42_1')(h4_take)
    h42 = layers.Conv2D(32, (3, 3),
                        activation='relu',
                        padding='same',
                        name='up42_2')(h42)

    h52 = Concatenate()([inputs, UpSampling2D((2, 2))(h42)])
    h52 = layers.Conv2D(16, (1, 1),
                        activation='relu',
                        padding='same',
                        name='up52_1')(h52)

    o5 = layers.Conv2D(1, (3, 3),
                        activation='relu',
                        padding='same',
                        name='regress-4-1')(h52)

    # o1: t/nt, o11:centerline, o2:x,y, o3:sin,cos, o4:bounding box width,height, o5:localheight
    # model =  Model(inputs, [o1,o11, o2,o3,o4], name = 'U-VGG-model')
    model = Model(inputs, [o1, o11, o5], name='U-VGG-model-Localheight')

    return model

