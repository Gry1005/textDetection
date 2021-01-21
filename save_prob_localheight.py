import glob
import cv2
import math
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import tensorflow as tf

print(tf.__file__)
print(tf.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, AveragePooling2D, Reshape
from keras.layers import UpSampling2D, ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.layers import Lambda, TimeDistributed
from keras import layers

import numpy as np
import cv2
import argparse
import glob

from loss import weighted_categorical_crossentropy, mean_squared_error_mask
from loss import mean_absolute_error_mask, mean_absolute_percentage_error_mask
from mymodel import model_U_VGG_Centerline_Localheight
#from mymodel_v2 import model_U_VGG_Centerline_Localheight
from mymodel_resNet50 import model_U_ResNet50_Centerline_Localheight

#map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/sub_maps_masks_grid1000/[0-9]*.jpg')
map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/sub_maps_masks_grid1000/USGS*.jpg')
#map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/synthMap_curved_os_z16_768/*.jpg')


saved_weights = '../weights/finetune_map_model_concat_out_fontsW_IA_model1.0_w1e50_e0_bsize8_spe200_ep50.hdf5'
model = model_U_VGG_Centerline_Localheight()
#model = model_U_ResNet50_Centerline_Localheight()
model.load_weights(saved_weights)

outputdir='../concat_out_fontsW_IA_model1.0_w1e50/'

idx = 0
all_boxes = []
all_confs = []
sin_list = []
cos_list = []
for map_path in map_images[0:30]:
    print(map_path)
    base_name = os.path.basename(map_path)

    map_img = cv2.imread(map_path)
    map_img = cv2.resize(map_img, (512, 512))

    in_map_img = map_img / 255.

    #cv2.imshow('in',in_map_img)
    #cv2.waitKey()

    img = np.expand_dims(in_map_img, axis=0)
    if saved_weights.split('_')[3] == '21':
        out = model.predict([img, np.expand_dims(np.indices((64, 64)).transpose(1, 2, 0), axis=0)])
    else:
        out = model.predict(img)

    prob_map = out[0]

    center_map = out[1]

    localheight_map = out[2]


    localheight_result = np.zeros((512, 512, 3), np.uint8)

    #localheight_map=(localheight_map*math.sqrt(2*512*512))

    #print(localheight_map)

    prob_map = (prob_map[0] * 255).astype(np.uint8)

    center_map = (center_map[0][:, :, 1] * 255).astype(np.uint8)

    #num_c,connected_map=cv2.connectedComponents(center_map)

    #print(num_c,connected_map)



    # 画圆
    for i in range(0, 512):
        for j in range(0, 512):
            #if (localheight_map[0][i][j] > 0 ) and center_map[i][j]>0 and prob_map[i][j][0]>0:
            if localheight_map[0][i][j] > 0:
                #print('localheight:',localheight_map[0][i][j])
                cv2.circle(localheight_result, (j, i), localheight_map[0][i][j]*0.4, (0, 0, 255), -1)

    # 标记多边形边框
    img_gray = cv2.cvtColor(localheight_result, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(map_img, contours, -1, (255, 0, 255), 1)

    localheight_map = (localheight_map[0]*255).astype(np.uint8)

    cv2.imwrite(outputdir+'prob_' + base_name, prob_map)
    cv2.imwrite(outputdir+'cent_' + base_name, center_map)
    cv2.imwrite(outputdir+'localheight_map_' + base_name, localheight_map)
    cv2.imwrite(outputdir+'localheight_' + base_name, map_img)
    cv2.imwrite(outputdir+'localheight_result_' + base_name, localheight_result)



