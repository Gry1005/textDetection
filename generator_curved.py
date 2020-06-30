import numpy as np
import cv2
import argparse
import glob
import math
import os


class SynthMap_DataGenerator_Centerline_Localheight(object):
    # image_root_path : map and mask images dir
    # list_path points to the file-list of train/val/test split
    def __init__(self, image_root_path, batch_size=128, seed=1234, mode='training', border_percent=0.15):
        img_path_list = glob.glob(image_root_path + '/*.jpg')

        # get the full path of map & maks files
        X, Y = [], []
        for img_name in img_path_list:
            img_id = os.path.basename(img_name).split('.jpg')[0]
            label_path = image_root_path + '/' + img_id + '.txt'

            Y.append(label_path)

        X = img_path_list

        print('num_samples = ', len(X))

        self.idx = 0
        self.nb_samples = len(X)
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.border_percent = border_percent

        self.seed = seed
        self.mode = mode
        np.random.seed(seed)

    def __len__(self):
        return self.nb_samples

    # 把bbox存储的数据格式，把边框的四个点坐标算出来
    def bbox_transform_angle(self, bbox):
        """convert a bbox of form [cx, cy, SIN,COS, w, h] to points. Works
        for numpy array or list of tensors.
        """
        cx, cy, sin, cos, w, h = bbox
        # y in img coord system yifan

        out_box = [[]] * 8

        out_box[0] = cx + w / 2 * cos + h / 2 * sin
        out_box[1] = cy - h / 2 * cos + w / 2 * sin
        out_box[2] = cx - w / 2 * cos + h / 2 * sin
        out_box[3] = cy - h / 2 * cos - w / 2 * sin
        out_box[4] = cx - w / 2 * cos - h / 2 * sin
        out_box[5] = cy + h / 2 * cos - w / 2 * sin
        out_box[6] = cx + w / 2 * cos - h / 2 * sin
        out_box[7] = cy + h / 2 * cos + w / 2 * sin

        return out_box

    def __getitem__(self, batch_idx):
        # randomly shuffle for training split
        # sequentiallyy take the testing data
        # print self.mode, batch_idx
        if (self.mode == 'training'):
            sample_indices = np.random.randint(0, self.nb_samples, self.batch_size)
            # print 'training, sample_indices',sample_indices
        else:
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates),
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range(batch_idx * self.batch_size, min(self.nb_samples, (batch_idx + 1) * self.batch_size))
            # print 'val, sample_indices',sample_indices

        # get the file paths
        subset_X = [self.X[i] for i in sample_indices]
        subset_Y = [self.Y[i] for i in sample_indices]

        std_size = 512
        # get the images
        batch_X = []
        batch_Y_mask = []
        batch_Y_border = []
        #batch_Y_regress = []
        batch_Y_centerline = []

        #加入一个图片数组作为最终结果：local height

        batch_Y_localheight = []


        for map_path, label_path in zip(subset_X, subset_Y):

            image_x = cv2.imread(map_path)
            o_height, o_width = image_x.shape[0], image_x.shape[1]
            image_x = cv2.resize(image_x, (std_size, std_size))

            h_sca = std_size * 1. / o_height
            w_sca = std_size * 1. / o_width

            height, width = image_x.shape[0], image_x.shape[1]
            # print 'height,width',height,width
            image_x = image_x / 255.
            batch_X.append(image_x)
            # init variables
            prob_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)
            border_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)
            centerline_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)

            #加入一张新的结果图片：local height

            localheight_img = np.zeros((height, width, 3), np.uint8)

            inside_Y_regress = np.zeros((height, width, 6), np.float32)
            border_p = self.border_percent
            centers = []
            thetas = []
            whs = []

            # read and precess file: 读取数据要修改
            with open(label_path, 'r') as f:
                data = f.readlines()

            bbox_idx = 0
            for line in data:

                bbox_idx += 1

                polyStr = line.split('/')[0]
                pointListStr = line.split('/')[1]
                localheight = line.split('/')[2].split('\n')[0]

                polyStr = polyStr.split(';')
                pointListStr = pointListStr.split(';')

                poly2 = []
                pointList2 = []

                for i in range(0, len(polyStr)):
                    x = float(polyStr[i].split(',')[0])*h_sca
                    y = float(polyStr[i].split(',')[1])*w_sca
                    poly2.append([x, y])

                for i in range(0, len(pointListStr)):
                    x1 = float(pointListStr[i].split(',')[0])*h_sca
                    y1 = float(pointListStr[i].split(',')[1])*w_sca
                    pointList2.append([x1, y1])

                # print(poly2)
                # print(pointList2)
                # print('localheight:'+localheight)

                poly2 = np.array([poly2], dtype=np.int32)

                pointList2 = np.array([pointList2], dtype=np.int32)

                # create mask image
                #points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                # print (points)
                # 填充多边形
                #cv2.fillConvexPoly(prob_img, points.astype(np.int), (0, bbox_idx, 0))
                cv2.fillPoly(prob_img, poly2, (0, bbox_idx, 0))
                #centers.append([x_c, y_c])

                # which one is the longer edge；实际上用的都是e12
                #e14 = (x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1)
                #e12 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
                #e_diag = np.sqrt(e12)

                # create centerline img
                # two cases: mid(1,2)->mid(3,4) or mid(2,3)->mid(1,4)

                #算完centerline之后，算local height

                '''
                if e_diag != 0.0:
                    # 1 -- 2
                    # 4 -- 3
                    si = (y2 - y1) / e_diag
                    co = (x2 - x1) / e_diag

                    w = np.sqrt(e12)
                    h = np.sqrt(e14)

                    #local height
                    lh=np.sqrt(e14)

                    left_pt = (int((x1 + x4) / 2.), int((y1 + y4) / 2.))
                    right_pt = (int((x2 + x3) / 2.), int((y2 + y3) / 2.))

                else:
                    # 4 -- 1
                    # 3 -- 2
                    e_diag = np.sqrt(e14)
                    si = (y4 - y1) / e_diag
                    co = (x4 - x1) / e_diag

                    w = np.sqrt(e14)
                    h = np.sqrt(e12)

                    #local height
                    lh=np.sqrt(e12)

                    left_pt = (int((x3 + x4) / 2.), int((y3 + y4) / 2.))
                    right_pt = (int((x1 + x2) / 2.), int((y1 + y2) / 2.))

                
                if e12 >= e14:
                    atan = np.arctan((y2-y1)/(x2-x1 + 1e-5))
                    w = np.sqrt(e12)
                    h = np.sqrt(e14)
                else:
                    atan = np.arctan((y4-y1)/(x4-x1 + 1e-5))
                    w = np.sqrt(e14)
                    h = np.sqrt(e12)
                
                thetas.append([si, co])
                whs.append([w, h])

                # create border img
                delta = thickness = int(max(border_p * h, 3))
                w_border = w + delta  # centerline
                h_border = h + delta  # centerline
                outer_bbox = [x_c, y_c, si, co, w_border, h_border]
                box = self.bbox_transform_angle(outer_bbox)
                x11, y11, x22, y22, x33, y33, x44, y44 = [int(b) for b in box]

                points = [(x11, y11), (x22, y22), (x33, y33), (x44, y44)]
                '''
                cv2.polylines(border_img,poly2,True,(0,1,0),2)

                #cv2.line(border_img, points[0], points[1], (0, 1, 0), thickness)
                #cv2.line(border_img, points[1], points[2], (0, 1, 0), thickness)
                #cv2.line(border_img, points[2], points[3], (0, 1, 0), thickness)
                #cv2.line(border_img, points[3], points[0], (0, 1, 0), thickness)

                # create centerline image
                #cv2.line(centerline_img, left_pt, right_pt, (0, 1, 0), thickness * 2)
                cv2.polylines(centerline_img, pointList2, False, (0, 1, 0), 4)

                #create localheight image
                #cv2.line(localheight_img, left_pt, right_pt, (lh, 0, 0), thickness * 2)
                #print('localheight:',float(localheight))
                cv2.polylines(localheight_img, pointList2, False, (float(localheight), 0, 0), 4)

                '''
                # regressions
                this_bbox_mask = (prob_img[:, :, 1] == bbox_idx)
                index_array = np.indices((height, width))  # (2, h, w)

                this_bbox_regress_x = np.expand_dims(np.array(x_c - index_array[1]) / std_size,
                                                     axis=-1)  # h, w, 1 ###### NOrmalization needs attention
                this_bbox_regress_y = np.expand_dims(np.array(y_c - index_array[0]) / std_size,
                                                     axis=-1)  # h, w, 1 ####### Normalization needs attention
                this_bbox_regress_twh = np.array([si, co, w / std_size, h / std_size])[:, np.newaxis,
                                        np.newaxis] * np.ones((height, width))  # 4, h, w
                this_bbox_regress_twh = this_bbox_regress_twh.transpose(1, 2, 0)  # h, w, 4
                this_bbox_regress = np.concatenate([this_bbox_regress_x, this_bbox_regress_y, this_bbox_regress_twh],
                                                   axis=-1)

                inside_Y_regress[this_bbox_mask > 0] = this_bbox_regress[this_bbox_mask > 0]
                '''

            prob_img[prob_img > 0] = 1
            prob_img = prob_img.astype(np.float32)
            prob_img = prob_img[:, :, 1]
            prob_img = np.expand_dims(prob_img, axis=-1)
            border_img = border_img.astype(np.float32)
            border_img = border_img[:, :, 1]
            border_img = np.expand_dims(border_img, axis=-1)

            centerline_img[:, :, 0] = 1 - centerline_img[:, :, 1]
            centerline_img = centerline_img[:, :, 0:2]  # take 2 channels

            #local_height
            localheight_img = localheight_img[:, :, 0:1] #take 1 channel
            #normalization
            #localheight_img = localheight_img.astype(np.float32)
            #localheight_img = localheight_img / 2

            centers = np.array(centers)  # num_gt_boxes, 2
            thetas = np.array(thetas)  # num_gt_boxes
            whs = np.array(whs)  # num_gt_boxes, 2

            batch_Y_mask.append(prob_img)
            batch_Y_border.append(border_img)
            batch_Y_centerline.append(centerline_img)

            #local height
            batch_Y_localheight.append(localheight_img)

            '''
            if centers.shape[0] == 0:
                batch_Y_regress.append(np.zeros((height, width, 6)))
                continue

            dist_square = (index_array[0, :, :] - centers[:, 1, np.newaxis, np.newaxis]) ** 2 + (
                        index_array[1, :, :] - centers[:, 0, np.newaxis, np.newaxis]) ** 2
            args = np.argmin(dist_square, axis=0)

            # batch_Y_regress.append(dist_square) # to visualize center assignments
            # batch_Y_regress.append(args)

            temp_1 = (centers[args] - index_array[::-1, :, :].transpose(1, 2, 0)) / np.array(
                [1.0 * std_size, 1.0 * std_size])  # h, w, 2  # center = (h * (index + pred), w * (index + pred))
            temp_2 = thetas[args]  # h, w, 2
            temp_3 = whs[args] / np.array([1.0 * std_size, 1.0 * std_size])  # h, w, 2
            temp_Y_regress = np.concatenate([temp_1, temp_2, temp_3], axis=-1)

            # print inside_Y_regress.shape
            # print prob_img.shape
            # print temp_Y_regress.shape

            Y_regress = inside_Y_regress * prob_img + temp_Y_regress * (1 - prob_img)
            batch_Y_regress.append(Y_regress)
            '''

        batch_X = np.array(batch_X)
        batch_Y_mask = np.array(batch_Y_mask)
        batch_Y_border = np.array(batch_Y_border)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border, 0, 1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1 - mask_and_border], axis=-1).astype(
            np.float32)

        batch_Y_centerline = np.array(batch_Y_centerline).astype(np.float32)

        #local_height
        batch_Y_localheight = np.array(batch_Y_localheight).astype(np.float32)

        #batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        #batch_Y_regress1 = batch_Y_regress[:, :, :, 0:2]
        #batch_Y_regress2 = batch_Y_regress[:, :, :, 2:4]
        #batch_Y_regress3 = batch_Y_regress[:, :, :, 4:]

        #batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis=-1)
        #batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis=-1)
        #batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis=-1)

        #return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_regress1, batch_Y_regress2, batch_Y_regress3])
        return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_localheight])

    def next(self):
        idx = self.idx
        self.idx = (1 + idx) % self.nb_samples
        return self[idx]
