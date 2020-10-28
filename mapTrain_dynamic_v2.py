import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import tensorflow as tf
print(tf.__file__)
print(tf.__version__)
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
#from sklearn.model_selection import train_test_split
from keras.models import load_model

import numpy as np
import cv2
import argparse
import glob
import math
from loss import weighted_categorical_crossentropy, mean_squared_error_mask, mean_absolute_error_localheight
from loss import mean_absolute_error_mask, mean_absolute_percentage_error_mask
from mymodel import model_U_VGG, model_U_VGG_Centerline_Localheight


from generator_dynamic_v2_GB import SynthMap_DataGenerator_Centerline_Localheight_Dynamic


#saved_weights = '../weights/synthText_model_bsize8_w1_spe100_ep233.hdf5'
model = model_U_VGG_Centerline_Localheight()
#model.load_weights(saved_weights)

#image_root_path = '/data/zekunl/synthMap_application/generate_data/concat_out_text_space/'
image_root_path = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset4/backGround3/*/'
fonts_path = 'E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset4/fonts/'
GB_path="E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset4/GB.txt"

prefix = 'map_dynamic_'

nb_epochs = 5
steps_per_epoch = 5
weight_ratio = 1.


#regress_loss1 = regress_loss2 = mean_absolute_error_mask
regress_loss1 = mean_absolute_error_localheight

weights1 = np.array([weight_ratio,weight_ratio, 1.]) # probability map for text region, border and background
weights2 = np.array([1., weight_ratio]) # probability map for background and centerline

adam = keras.optimizers.Adam(lr=0.0001)
ith = prefix + 'bsize8_w1_spe'+str(steps_per_epoch)+'_ep' + str(nb_epochs)
log_name = '../logs/'+ ith + '.log'
csv_logger = keras.callbacks.CSVLogger(log_name)
ckpt_filepath = '../weights/'+ prefix + 'finetune_map_model_bsize8_w1_spe'+str(steps_per_epoch)+'_ep{epoch:02d}.hdf5'
model_ckpt = keras.callbacks.ModelCheckpoint(ckpt_filepath,period = 10)
myvalidation_steps = 5


train_datagen = SynthMap_DataGenerator_Centerline_Localheight_Dynamic(image_root_path = image_root_path, fonts_path=fonts_path,GB_path=GB_path,batch_size= 8,  seed = 3333, mode = 'training',overlap=True,showPicDir='../dynamicPics/')

#output: t/nt, centerline,
#model.compile(adam, loss = [weighted_categorical_crossentropy(weights1),weighted_categorical_crossentropy(weights2)  ,regress_loss2,regress_loss1,regress_loss2])
model.compile(adam, loss = [weighted_categorical_crossentropy(weights1),weighted_categorical_crossentropy(weights2)  ,regress_loss1])
callbacks = [csv_logger,model_ckpt]

#model.fit_generator(train_datagen, steps_per_epoch = 250, epochs = 100, validation_data = val_datagen, validation_steps = myvalidation_steps, callbacks = callbacks, shuffle=True,max_queue_size=5 )

#model.fit_generator(train_datagen, steps_per_epoch = steps_per_epoch, epochs = nb_epochs,  callbacks = callbacks, shuffle=True,max_queue_size=5 )

model.fit_generator(iter(train_datagen), steps_per_epoch = steps_per_epoch, epochs = nb_epochs,  callbacks = callbacks)

model.save('../weights/finetune_map_model_' + ith + '.hdf5')


'''
#model.load_weights('weights/pretrain_ugg_model_29_bsize8_w10_ep50.hdf5')
model.load_weights('weights/pretrain_ugg_model_32_bsize8_w1_ep100_inited.hdf5')
#model.compile('adam', loss = 'binary_crossentropy')
#model.compile('adam' ,loss = ['categorical_crossentropy'])
#model.summary()
ith = prefix + 'bsize8_w1_ep400'
log_name = ith + '.log'
csv_logger = keras.callbacks.CSVLogger(log_name)
filepath = prefix + 'pretrain_ugg_model_bsize8_w1_ep{epoch:02d}_inited.hdf5'
model_ckpt = keras.callbacks.ModelCheckpoint(filepath,period = 20)
callbacks = [csv_logger,model_ckpt]

model.fit_generator(train_datagen, steps_per_epoch = 100, epochs = 400,  callbacks = callbacks, shuffle=True,max_queue_size=20 )
model.save('pretrain_ugg_model_' + ith + '_inited.hdf5')

'''