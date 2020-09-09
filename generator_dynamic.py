import numpy as np
import cv2
import argparse
import glob
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random, math, copy
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
from matplotlib import font_manager
import re


# Notes for inserting text on images:
# if no rotation or customized font is required, then cv2.putText and cv2.getTextsize would work perfectly
# But one reason NOT to use  cv2 to insert text is its incapability to insert customized font styles
# even loadFontData did not work for me in python


def fig2data(fig):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in RGB mode. Roll the ALPHA channel to have it in RGB mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf


def fig2img(fig):
    # from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGB format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer( "RGB", ( w ,h ), buf.tostring( ) )


def visualize_points(text_layer, pts):
    '''
    text_layer is of PIL image type
    pts is an array
    '''
    text_layer = np.array(text_layer).astype(np.uint8).copy()

    for pt in pts:
        pt = [int(a) for a in pt]
        cv2.circle(text_layer, (pt[0], pt[1]), 5, (255, 0, 0), 10)
    plt.imshow(text_layer)
    plt.show()


# generate text layer
def text_on_canvas(text, myf, ro, color=(0.5, 0.5, 0.5), margin=1):
    axis_lim = 1

    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.axis([0, axis_lim, 0, axis_lim])

    # place the top left corner at (axis_lim/20,axis_lim/2) to avoid clip during rotation
    aa = plt.text(axis_lim / 20., axis_lim / 2., text, color=color, ha='left', va='top', fontproperties=myf,
                  rotation=ro, wrap=False)
    plt.axis('off')
    text_layer = fig2img(fig)  # convert to image
    plt.close()

    we = aa.get_window_extent()
    min_x, min_y, max_x, max_y = we.xmin, 500 - we.ymax, we.xmax, 500 - we.ymin
    box = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)

    # return coordinates to further calculate the bbox of rotated text
    return text_layer, min_x, min_y, max_x, max_y


def geneText(text, font_family, font_size, font_color, rot_angle, style):
    # if font size too big, then put.text automatically adjust the position, which makes computed position errorous.
    myf = font_manager.FontProperties(fname=font_family, size=font_size)

    if style < 8:  # rotated text
        fcolor = tuple(a / 255. for a in font_color)  # convert from [0,255] to [0,1]
        # no rotation, just to get the minimum bbox
        htext_layer, min_x, min_y, max_x, max_y = text_on_canvas(text, myf, ro=0, color=fcolor)

        # print min_x,min_y,max_x,  max_y
        M = cv2.getRotationMatrix2D((min_x, min_y), rot_angle, 1)
        # pts is 4x3 matrix
        pts = np.array([[min_x, min_y, 1], [max_x, min_y, 1], [max_x, max_y, 1], [min_x, max_y, 1]])  # clockwise
        affine_pts = np.dot(M, pts.T).T

        if (affine_pts <= 0).any() or (affine_pts >= 500).any():
            return 0, 0  # exceed boundary. skip
        else:
            text_layer = htext_layer.rotate(rot_angle, center=(min_x, min_y), fillcolor='white')

            # visualize_points(htext_layer, pts)
            # visualize_points(text_layer, affine_pts)
            return text_layer, affine_pts

    else:
        raise NotImplementedError


# add to map
def zk_addToMap(img_bg, text_layer, p):
    text_bi = np.array(text_layer)[:, :, 0].copy()
    text_bina = text_bi.copy()

    # binarizatoin. if text region, set to 1, if not text region, set to 0.
    # since no text region has color intensity greater than 128. thus we use 255 as the threshold

    #text_bi[text_bina != 255] = 255
    #text_bi[text_bina == 255] = 0

    #text_bi[text_bina <= 130] = 255
    #text_bi[text_bina > 130] = 0

    text_bi[text_bina <= 195] = 255
    text_bi[text_bina > 195] = 0

    text_bi = Image.fromarray(text_bi)
    #text_bi.filter(ImageFilter.GaussianBlur(4))

    img_bg.paste(text_layer, (p[0], p[1]), mask=text_bi)

    return img_bg, text_bina



class SynthMap_DataGenerator_Centerline_Localheight_Dynamic(object):
    # image_root_path : map and mask images dir
    # list_path points to the file-list of train/val/test split
    def __init__(self, image_root_path,fonts_path, GB_path, batch_size=128, seed=1234, mode='training', border_percent=0.15):

        self.fonts_path=fonts_path
        self.GB_path=GB_path

        img_path_list = glob.glob(image_root_path + '/*.jpg')

        # get the full path of map & maks files
        X=[]

        '''
        X, Y = [], []
        for img_name in img_path_list:
            img_id = os.path.basename(img_name).split('.jpg')[0]
            label_path = image_root_path + '/' + img_id + '.txt'

            Y.append(label_path)
        '''

        X = img_path_list

        print('num_samples = ', len(X))

        self.idx = 0
        self.nb_samples = len(X)
        self.X = X
        #self.Y = Y
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
        #subset_Y = [self.Y[i] for i in sample_indices]

        std_size = 512
        # get the images
        batch_X = []
        batch_Y_mask = []
        batch_Y_border = []
        batch_Y_regress = []
        batch_Y_centerline = []

        #加入一个图片数组作为最终结果：local height

        batch_Y_localheight = []

        #subset_X_2存储粘贴文字之后的图片
        subset_X_2=[]
        #subset_Y存储每张图片上的文字坐标
        subset_Y=[]

        #把文字粘贴到图片上!!!
        fonts_path = self.fonts_path

        # load the words
        word_set = set()
        geoname_f = open(self.GB_path, "r", encoding='utf-8')
        for line in geoname_f:
            cols = re.split(r'\t+', line.rstrip('\t'))
            words = cols[1].split(' ')
            for w in words:
                word_set.add(w.strip('()'))
        geoname_f.close()
        set_len = len(word_set)
        #print(len(word_set), ' words in total')
        #print('eg:', list(word_set)[0:10])

        # process
        word_set = list(word_set)

        fonts = glob.glob(fonts_path + '/*.ttf')

        words_size = len(word_set)
        #cnt = 0  # process on #cnt images
        text_num_thresh = 10

        for i, patch_path in enumerate(subset_X):

            #对每张图片记录下每个文字的四个坐标
            text_lis=[]

            # if i < 26738:
            # continue
            #print('processing', patch_path)
            img_bg_pil = Image.open(patch_path).convert('RGB')

            W, H = img_bg_pil.size
            # print(W,H)

            img_bg_cv = cv2.imread(patch_path)

            # for i in range(text_num):
            text_num = 0
            text_region = []
            while (text_num < text_num_thresh):
                # get input text string
                text = word_set[random.randint(1, words_size - 1)].strip()
                text = re.sub('[^0-9A-Za-z]+', '', text)  # remove symbols
                while (len(text) < 1):  # text length is zero. no character except for spaces
                    text = word_set[random.randint(1, words_size - 1)].strip()
                    text = re.sub('[^0-9A-Za-z]+', '', text)  # remove symbols

                # font specification
                font_face = fonts[random.randint(0, len(fonts) - 1)]

                # font_size加入比较大的字体(100,150)
                font_size = random.randint(10, 80)
                # font_size = random.randint(5, 30)
                font_size2 = random.randint(110, 160)

                r = random.randint(0, 9)
                if r == 6:
                    font_size = font_size2

                ro = random.randint(-90, 90)
                fcolor = random.randint(0, 128) * np.ones((3))

                # some variations to the original input text
                if np.random.randint(0, 2):  # 50% chance to capitalize the text
                    text = text.upper()

                # 50% chance to insert blank space
                if np.random.randint(0, 2):
                    insert_type = np.random.randint(0, 3)
                    if insert_type == 0:  # 1/3 of the chance to insert ONE blank space between chars
                        text = " ".join(text)
                    if insert_type == 1:  # 1/3 of the chance to insert TWO blank space bween chars
                        text = "  ".join(text)
                    if insert_type == 2:  # 1/3 of the chance to insert FIVE blank space bween chars
                        text = "     ".join(text)

                # print text
                text_layer, af_pts = geneText(text, font_face, font_size, font_color=fcolor, rot_angle=ro, style=1)

                # text region

                if text_layer != 0:
                    text_w, text_h = text_layer.size

                    successFlag = True
                    while True:
                        breakflag = True
                        upper_left_pos = (
                            random.randint(0, W - text_w / 2),
                            random.randint(0, H - text_h / 2))  # upper-left pos on bg img
                        # these are the text region positions on original text layer
                        left = int(np.min(af_pts[:, 0]))
                        up = int(np.min(af_pts[:, 1]))
                        right = int(np.max(af_pts[:, 0]))
                        bottom = int(np.max(af_pts[:, 1]))

                        x1, y1 = af_pts[0]
                        x2, y2 = af_pts[1]
                        x3, y3 = af_pts[2]
                        x4, y4 = af_pts[3]
                        x1, x2 = int(x1 - left + upper_left_pos[0]), int(x2 - left + upper_left_pos[0])
                        x3, x4 = int(x3 - left + upper_left_pos[0]), int(x4 - left + upper_left_pos[0])
                        y1, y2 = int(y1 - up + upper_left_pos[1]), int(y2 - up + upper_left_pos[1])
                        y3, y4 = int(y3 - up + upper_left_pos[1]), int(y4 - up + upper_left_pos[1])
                        bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

                        if len(text_region) == 0:
                            im = np.zeros([W, H], dtype="uint8")
                            text_region.append(bbox)
                            mask = cv2.fillConvexPoly(im, np.array(bbox), 10)
                            break
                        else:
                            mask_copy = mask
                            im1 = np.zeros([W, H], dtype="uint8")
                            mask1 = cv2.fillConvexPoly(im1, np.array(bbox), 10)
                            masked_and = mask_copy + mask1
                            and_area = np.sum(
                                np.float32(
                                    np.greater(masked_and, 10)))  # use and_are to check if masked_and has overlap area
                            if and_area > 1.0:
                                successFlag = False
                                break
                            elif x1 > H or x2 > H or x3 > H or x4 > H or y1 > W or y2 > W or y3 > W or y4 > W:  # not exceed the boundary
                                successFlag = False
                                break
                            else:
                                text_region.append(bbox)
                                mask = mask + mask1
                                break

                    if successFlag:
                        text_layer = text_layer.crop((left, up, right, bottom))  # crop out the text region
                        img_bg_pil, _ = zk_addToMap(img_bg_pil, text_layer, upper_left_pos)  # place on the bg image

                        # calculate the tight bbox position on the bg image ( infer from original text layer)
                        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = text_region[-1]

                        # draw =ImageDraw.Draw(img_bg_pil)
                        # draw.line(((x2,y2),(x3,y3)), fill=128)
                        c_x, c_y = 0.5 * (x1 + x3), 0.5 * (y1 + y3)

                        text_lis.append([x1, y1, x2, y2, x3, y3, x4, y4])

                    text_num += 1

            #PIL转换为CV2
            subset_X_2.append(cv2.cvtColor(np.asarray(img_bg_pil.convert('RGB')),cv2.COLOR_RGB2BGR))
            subset_Y.append(text_lis)

            #cnt += 1
            #if cnt == 20:
                #break

        for image, text_lis in zip(subset_X_2, subset_Y):

            x = image
            o_height, o_width = x.shape[0], x.shape[1]
            x = cv2.resize(x, (std_size, std_size))

            h_sca = std_size * 1. / o_height
            w_sca = std_size * 1. / o_width

            height, width = x.shape[0], x.shape[1]
            # print 'height,width',height,width
            x = x / 255.
            batch_X.append(x)
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

            bbox_idx = 0
            for line in text_lis:
                bbox_idx += 1
                # if len(line)!=9:
                #    print line
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                x3 = line[4]
                y3 = line[5]
                x4 = line[6]
                y4 = line[7]

                ## check h_scale and w_scale when they are not the same
                x1 = float(x1) * w_sca
                y1 = float(y1) * h_sca
                x2 = float(x2) * w_sca
                y2 = float(y2) * h_sca
                x3 = float(x3) * w_sca
                y3 = float(y3) * h_sca
                x4 = float(x4) * w_sca
                y4 = float(y4) * h_sca
                x_c = 0.5 * (x1 + x3)
                y_c = 0.5 * (y1 + y3)

                # create mask image
                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                # print (points)
                # 填充多边形
                cv2.fillConvexPoly(prob_img, points.astype(np.int), (0, bbox_idx, 0))
                centers.append([x_c, y_c])

                # which one is the longer edge；实际上用的都是e12
                e14 = (x4 - x1) * (x4 - x1) + (y4 - y1) * (y4 - y1)
                e12 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
                e_diag = np.sqrt(e12)

                # create centerline img
                # two cases: mid(1,2)->mid(3,4) or mid(2,3)->mid(1,4)

                #算完centerline之后，算local height和angle

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

                '''
                if e12 >= e14:
                    atan = np.arctan((y2-y1)/(x2-x1 + 1e-5))
                    w = np.sqrt(e12)
                    h = np.sqrt(e14)
                else:
                    atan = np.arctan((y4-y1)/(x4-x1 + 1e-5))
                    w = np.sqrt(e14)
                    h = np.sqrt(e12)
                '''
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

                cv2.line(border_img, points[0], points[1], (0, 1, 0), thickness)
                cv2.line(border_img, points[1], points[2], (0, 1, 0), thickness)
                cv2.line(border_img, points[2], points[3], (0, 1, 0), thickness)
                cv2.line(border_img, points[3], points[0], (0, 1, 0), thickness)

                # create centerline image
                cv2.line(centerline_img, left_pt, right_pt, (0, 1, 0), thickness * 2)

                #create localheight image
                #cv2.line(localheight_img, left_pt, right_pt, (lh, 0, 0), thickness * 2)
                cv2.line(localheight_img, left_pt, right_pt, (lh, 0, 0), 1)

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

        batch_X = np.array(batch_X)
        batch_Y_mask = np.array(batch_Y_mask)
        batch_Y_border = np.array(batch_Y_border)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border, 0, 1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1 - mask_and_border], axis=-1).astype(
            np.float32)

        batch_Y_centerline = np.array(batch_Y_centerline).astype(np.float32)

        #local_height
        batch_Y_localheight = np.array(batch_Y_localheight).astype(np.float32)

        batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        batch_Y_regress1 = batch_Y_regress[:, :, :, 0:2]
        batch_Y_regress2 = batch_Y_regress[:, :, :, 2:4]
        batch_Y_regress3 = batch_Y_regress[:, :, :, 4:]

        batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis=-1)
        batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis=-1)
        batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis=-1)

        #return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_regress1, batch_Y_regress2, batch_Y_regress3])
        return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_localheight])

    def next(self):
        idx = self.idx
        self.idx = (1 + idx) % self.nb_samples
        return self[idx]
