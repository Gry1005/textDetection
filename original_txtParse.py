import glob
import os
import cv2
import numpy as np

image_list=glob.glob('E:/Spatial Computing & Informatics Laboratory/CutTextArea/dataset/original_size_OS_USGS_map/historical-map-groundtruth-25/102903919/*.jpg')


for image_path in image_list[0:1]:

    base_name = os.path.basename(image_path)

    print('base_name:',base_name)

    txt_path = '../original_os_test_txt/' + base_name[0:len(base_name) - 4] + '.txt'

    image_x = cv2.imread(image_path)

    height=image_x.shape[0]

    width=image_x.shape[1]

    with open(txt_path, 'r') as f:
        data = f.readlines()

    bbox_idx = 0

    polyList=[]

    for line in data:

        polyStr=line.split(',')

        poly=[]

        for i in range(0,len(polyStr)):
            if i%2==0:
                poly.append([int(polyStr[i]),int(polyStr[i+1])])

        polyList.append(poly)

    print('all: ',len(polyList))

    txt_pixel_result = np.zeros((height, width, 3), np.uint8)

    for i in range(0,len(polyList)):

        polyPoints=np.array([polyList[i]],dtype=np.int32)

        cv2.polylines(image_x, polyPoints, True, (0, 0, 255), 1)

        cv2.fillPoly(txt_pixel_result,polyPoints,(0,0,255))

        print('i: ',i)

    cv2.imwrite('../original_os_test_txt/parse_result_'+base_name,image_x)

    cv2.imwrite('../original_os_test_txt/txt_pixel_result_' + base_name, txt_pixel_result)
