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

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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

map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/sub_maps_masks_grid1000/[0-9]*.jpg')

# saved_weights = 'weights/finetune_map_model_map_4_2_bsize8_w1_spe200_ep50.hdf5'
saved_weights = '../weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5'
model = model_U_VGG_Centerline_Localheight()
model.load_weights(saved_weights)

idx = 0
all_boxes = []
all_confs = []
sin_list = []
cos_list = []
for map_path in map_images[0:10]:
    print(map_path)
    base_name = os.path.basename(map_path)

    map_img = cv2.imread(map_path)

    x_o=map_img.shape[0]
    y_o=map_img.shape[1]

    map_img_original=map_img.copy()
    map_img = cv2.resize(map_img, (512, 512))

    x_r=float(x_o/512.)

    y_r=float(y_o/512.)

    in_map_img = map_img / 255.
    img = np.expand_dims(in_map_img, axis=0)
    if saved_weights.split('_')[3] == '21':
        out = model.predict([img, np.expand_dims(np.indices((64, 64)).transpose(1, 2, 0), axis=0)])
    else:
        out = model.predict(img)

    prob_map = out[0]

    center_map = out[1]

    localheight_map = out[2]


    localheight_result = np.zeros((512, 512, 3), np.uint8)

    #map_zero=np.zeros((512, 512, 3), np.uint8)

    #localheight_map=(localheight_map*math.sqrt(2*512*512))

    #print(localheight_map)

    prob_map = (prob_map[0] * 255).astype(np.uint8)

    center_map = (center_map[0][:, :, 1] * 255).astype(np.uint8)

    num_c,connected_map=cv2.connectedComponents(center_map)

    #print(num_c,connected_map)

    txt_name='../mapW1ResultsTxt/' + base_name[0:len(base_name)-4]+'.txt'

    f=open(txt_name,'w+')

    for k in range(1,num_c):
        # 画圆
        for i in range(0, 512):
            for j in range(0, 512):
                if connected_map[i][j]==k and localheight_map[0][i][j] > 0 and center_map[i][j] > 0 and prob_map[i][j][0] > 0:
                    cv2.circle(localheight_result, (j, i), localheight_map[0][i][j], (0, 0, 255), -1)

        #cv2.imshow('localheight_result',localheight_result)

        #cv2.waitKey()

        # 标记多边形边框
        img_gray = cv2.cvtColor(localheight_result, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.drawContours(map_img, contours, -1, (255, 0, 255), 1)

        #cv2.polylines(img=map_img, pts=contours, isClosed=True, color=(255, 0, 255), thickness=1)

        #cv2.fillPoly(img=map_img, pts=contours[0], color=(255, 0, 255))

        #print(contours[0])

        new_context=''

        if len(contours)==0:
            continue

        for i in range(0,len(contours[0])):

            #print(type(contours[0][i][0][0].item()))
            if i<len(contours[0])-1:
                new_context=new_context+str(contours[0][i][0][0].item()*x_r)+','+str(contours[0][i][0][1].item()*y_r)+','
            else:
                new_context = new_context + str(contours[0][i][0][0].item() * x_r) + ',' + str(contours[0][i][0][1].item() * y_r)


        new_context=new_context+'\n'

        f.writelines(new_context)

        localheight_result = np.zeros((512, 512, 3), np.uint8)


        #解析txt

        points=new_context.split(',')

        polyPoints=[]

        for i in range(0,len(points)):
            if i%2==0:
                polyPoints.append([float(points[i]),float(points[i+1])])

        #print(polyPoints)

        polyPoints = np.array([polyPoints], dtype=np.int32)

        cv2.polylines(map_img_original, polyPoints, True, (0, 255, 0), 1)

        cv2.imshow("map_img_original",map_img_original)

        cv2.waitKey()
        


    f.close()

    localheight_map = (localheight_map[0]*255).astype(np.uint8)

    cv2.imwrite('../mapW1ResultsTxt/prob_' + base_name, prob_map)
    cv2.imwrite('../mapW1ResultsTxt/cent_' + base_name, center_map)
    cv2.imwrite('../mapW1ResultsTxt/localheight_map_' + base_name, localheight_map)
    cv2.imwrite('../mapW1ResultsTxt/localheight_' + base_name, map_img)
    #cv2.imwrite('../mapW1ResultsTxt/localheight_result_' + base_name, localheight_result)



