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
                                                                                                              
import numpy as np  
import cv2
import argparse     
import glob

from loss import weighted_categorical_crossentropy, mean_squared_error_mask
from loss import mean_absolute_error_mask, mean_absolute_percentage_error_mask
from mymodel import model_U_VGG, model_U_VGG_Centerline


map_images = glob.glob('../original/*.jpg')

#saved_weights = 'weights/finetune_map_model_map_4_2_bsize8_w1_spe200_ep50.hdf5'
#saved_weights = '../weights/synthText_model_bsize8_w1_spe100_ep233.hdf5'
saved_weights = '../weights/finetune_map_model_map_textinit233_bsize8_w1_spe200_ep50.hdf5'
model = model_U_VGG_Centerline()
model.load_weights(saved_weights)



idx = 0
all_boxes = []
all_confs = []
sin_list = []
cos_list = []
for map_path in map_images[0:5]:
    print(map_path)
    base_name = os.path.basename(map_path)
    
    map_img = cv2.imread(map_path)
    map_img = cv2.resize (map_img, (512,512))

    
    in_map_img = map_img / 255.
    img = np.expand_dims(in_map_img, axis = 0)
    if saved_weights.split('_')[3] == '21':
        out = model.predict([img, np.expand_dims(np.indices((64,64)).transpose(1,2,0),axis = 0)])
    else:
        out = model.predict(img)

    prob_map = out[0]

    center_map = out[1]
    
    
    prob_map = (prob_map[0]*255).astype(np.uint8)
    cent_map = (center_map[0][:,:,1]*255).astype(np.uint8)
    
    cv2.imwrite('../results1/prob_'+base_name,prob_map)
    cv2.imwrite('../results1/cent_'+base_name,cent_map)
    
  
