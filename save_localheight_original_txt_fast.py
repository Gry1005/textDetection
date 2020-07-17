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

map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/sub_maps_masks_grid1000/[0-9]*.jpg')
#map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/sub_maps_masks_grid1000/USGS*.jpg')
#map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/synthMap_curved_os_z16_768/*.jpg')
#map_images = glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/weinman19-maps/D0042-1070002.tiff')
output_path='../mapW1_OS_ResultsTxt/'
# saved_weights = 'weights/finetune_map_model_map_4_2_bsize8_w1_spe200_ep50.hdf5'
saved_weights = '../weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5'
model = model_U_VGG_Centerline_Localheight()
model.load_weights(saved_weights)

for map_path in map_images[1:6]:

    base_name = os.path.basename(map_path)

    txt_name = output_path + base_name[0:len(base_name) - 4] + '.txt'
    #txt_name = output_path + base_name[0:len(base_name) - 5] + '.txt'

    f = open(txt_name, 'w+')

    print(map_path)

    base_name = os.path.basename(map_path)

    map_img = cv2.imread(map_path)

    width=map_img.shape[1] #dimension2
    height=map_img.shape[0] #dimension1

    in_map_img = map_img / 255.

    #img = np.expand_dims(in_map_img, axis=0)

    localheight_map_o=np.zeros((height,width, 1), np.float64)

    prob_map_o=np.zeros((height, width, 3), np.float64)

    center_map_o=np.zeros((height, width, 2), np.float64)

    y=0

    y_break_flag=True

    while y_break_flag:

        if y+512>=height:
            y=height-512
            y_break_flag=False

        x_break_flag=True

        x=0

        while x_break_flag:

            if x+512>=width:

                x=width-512

                x_break_flag=False

            print('y: ', y, ' x: ', x)

            x0=x
            x1=x+512
            y0=y
            y1=y+512

            img_clip=in_map_img[y0:y1,x0:x1]

            img_clip = np.expand_dims(img_clip, axis=0)

            if saved_weights.split('_')[3] == '21':
                out = model.predict([img_clip, np.expand_dims(np.indices((64, 64)).transpose(1, 2, 0), axis=0)])
            else:
                out = model.predict(img_clip)

            prob_map_clip = out[0]

            center_map_clip = out[1]

            localheight_map_clip = out[2]

            for i in range(y0,y1):
                for j in range(x0,x1):
                    if prob_map_o[i][j][0]==0 and prob_map_o[i][j][1]==0 and prob_map_o[i][j][2]==0:
                        prob_map_o[i][j][0]=prob_map_clip[0][i-y0][j-x0][0]
                        prob_map_o[i][j][1] = prob_map_clip[0][i - y0][j - x0][1]
                        prob_map_o[i][j][2] = prob_map_clip[0][i - y0][j - x0][2]

                    if center_map_o[i][j][0]==0 and center_map_o[i][j][1]==0:
                        center_map_o[i][j][0]=center_map_clip[0][i-y0][j-x0][0]
                        center_map_o[i][j][1] = center_map_clip[0][i - y0][j - x0][1]

                    if localheight_map_o[i][j]==0:
                        localheight_map_o[i][j]=localheight_map_clip[0][i-y0][j-x0]

            x=x+500

        y=y+500


    #localheight_map=(localheight_map*math.sqrt(2*512*512))

    #print(localheight_map)

    prob_map_o = (prob_map_o * 255).astype(np.uint8)

    center_map_o = (center_map_o[:, :, 1] * 255).astype(np.uint8)

    localheight_map = (localheight_map_o * 255).astype(np.uint8)

    #num_c,connected_map=cv2.connectedComponents(center_map)

    #print(num_c,connected_map)

    #localheight_result_o=np.zeros((height, width, 3), np.uint8)

    #划分不同的centerline
    num_c, connected_map = cv2.connectedComponents(center_map_o)

    print('num_c:',num_c)

    exist_k=set()

    for i in range(0,height):
        for j in range(0,width):
            if connected_map[i][j] not in exist_k:
                exist_k.add(connected_map[i][j])


    print('all k:',len(exist_k))

    count = 0

    for k in exist_k:

        count+=1

        print('count:', count)

        centerPoints=[]

        mini=float('inf')
        minj=float('inf')
        maxi=0
        maxj=0

        for i in range(0, height):
            for j in range(0, width):
                if connected_map[i][j]==k and localheight_map_o[i][j] > 0 and center_map_o[i][j] > 0 and prob_map_o[i][j][0] > 0:
                    # if localheight_map_o[0][i][j] > 0:
                    #cv2.circle(localheight_result_o, (j, i), localheight_map_o[i][j], (0, 0, 255), -1)
                    mini=min(mini,i)
                    maxi=max(maxi,i)
                    minj=min(minj,j)
                    maxj=max(maxj,j)
                    centerPoints.append((i,j))

        if len(centerPoints)==0:
            continue

        localheight_result_o = np.zeros((maxi-mini+100, maxj-minj+100, 3), np.uint8)

        # 画圆

        '''
        for i in range(0, height):
            for j in range(0, width):
                if connected_map[i][j]==k and localheight_map_o[i][j] > 0 and center_map_o[i][j] > 0 and prob_map_o[i][j][0] > 0:
                    # if localheight_map_o[0][i][j] > 0:
                    cv2.circle(localheight_result_o, (j, i), localheight_map_o[i][j], (0, 0, 255), -1)
        '''
        for i,j in centerPoints:
            cv2.circle(localheight_result_o, (j-minj+50, i-mini+50), localheight_map_o[i][j]*0.4, (0, 0, 255), -1)

        # 标记多边形边框
        img_gray = cv2.cvtColor(localheight_result_o, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_context = ''

        if len(contours) == 0:
            continue

        for i in range(0, len(contours[0])):

            # print(type(contours[0][i][0][0].item()))
            if i < len(contours[0]) - 1:
                new_context = new_context + str(contours[0][i][0][0].item()+minj-50 ) + ',' + str(contours[0][i][0][1].item()+mini-50 ) + ','
            else:
                new_context = new_context + str(contours[0][i][0][0].item()+minj-50 ) + ',' + str(contours[0][i][0][1].item()+mini-50)

        new_context = new_context + '\n'

        f.writelines(new_context)


    cv2.imwrite(output_path+'prob_' + base_name[0:len(base_name) - 4] + '.jpg', prob_map_o)
    cv2.imwrite(output_path+'cent_' + base_name[0:len(base_name) - 4] + '.jpg', center_map_o)
    cv2.imwrite(output_path+'localheight_map_' + base_name[0:len(base_name) - 4] + '.jpg', localheight_map_o)
    #cv2.imwrite('../original_os_test_txt/localheight_' + base_name, map_img)
    #cv2.imwrite('../original_os_test_txt/localheight_result_' + base_name, localheight_result_o)

    f.close()


    #txt parse
    with open(txt_name, 'r') as f:
        data = f.readlines()

    polyList=[]

    for line in data:

        polyStr=line.split(',')

        poly=[]

        for i in range(0,len(polyStr)):
            if i%2==0:
                poly.append([int(polyStr[i]),int(polyStr[i+1])])

        polyList.append(poly)

    print('all: ',len(polyList))

    txt_pixel_result=np.zeros((height, width, 3), np.uint8)

    for i in range(0,len(polyList)):

        polyPoints=np.array([polyList[i]],dtype=np.int32)

        cv2.polylines(map_img, polyPoints, True, (0, 0, 255), 1)

        cv2.fillPoly(txt_pixel_result, polyPoints, (0, 0, 255))

        print('i: ',i)

    cv2.imwrite(output_path+'parse_result_'+base_name[0:len(base_name) - 4] + '.jpg',map_img)
    cv2.imwrite(output_path+'txt_pixel_result_' + base_name[0:len(base_name) - 4] + '.jpg', txt_pixel_result)




