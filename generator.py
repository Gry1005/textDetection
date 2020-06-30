import numpy as np  
import cv2
import argparse     
import glob
import math
import os


def get_angle(sin, cos):
    if cos >= 0:
        # angle in [-90, 90]
        angle = np.arcsin(sin)
    elif sin >= 0:
        # angle in [90, 180]
        angle = math.pi - np.arcsin(sin)
    else:
        # angle in [-180, 90]
        angle = - math.pi - np.arcsin(sin)

    # radian to degree
    return angle * 180/math.pi

class SynthText_DataGenerator_Centerline(object):   
    # image_root_path : PATH_TO_SynthText/
    def __init__(self, image_root_path, batch_size = 128,  seed = 1234, mode = 'training', border_percent = 0.15):  
        
        meta_file = image_root_path  + '/gt.mat'
        import scipy.io
        mat_data = scipy.io.loadmat(meta_file)
        
        wordBB = mat_data['wordBB'][0] # bounding boxes
        txt = mat_data['txt'][0] # text content
        imnames = mat_data['imnames'][0] # image names
        charBB = mat_data['charBB'][0] # character level boxes
        
        ############### debug info ###############
        '''
        print wordBB[0].shape
        print txt[0]
        print imnames[0]
        print charBB[0].shape
        
        (2, 4, 15)
        [u'Lines:\nI lost\nKevin ' u'will                '
         u'line\nand            ' u'and\nthe             ' u'(and                '
         u'the\nout             ' u'you                 ' u"don't\n pkg          "]
        [u'8/ballet_106_0.jpg']
        (2, 4, 54)
        
        '''
        
        print('num_samples = ',len(wordBB))
        # X: map file list, Y: binary mask file list                                                                                                             
        self.idx = 0 
        self.nb_samples =  len(wordBB)                  
        self.batch_size = batch_size 
        
        self.wordBB = wordBB
        self.txt = txt
        self.imnames = imnames
        self.charBB = charBB
        self.image_root_path = image_root_path
        self.border_percent = border_percent

        self.seed = seed             
        self.mode = mode                                    
        np.random.seed(seed)  
        
    def bbox_transform_angle(self, bbox):
        """convert a bbox of form [cx, cy, SIN,COS, w, h] to points. Works
        for numpy array or list of tensors.
        """
        cx, cy, sin, cos, w, h  = bbox
        # y in img coord system yifan

        out_box = [[]]*8

        out_box[0] = cx + w/2 * cos + h/2 * sin
        out_box[1] = cy - h/2 * cos + w/2 * sin
        out_box[2] = cx - w/2 * cos + h/2 * sin
        out_box[3] = cy - h/2 * cos - w/2 * sin
        out_box[4] = cx - w/2 * cos - h/2 * sin
        out_box[5] = cy + h/2 * cos - w/2 * sin
        out_box[6] = cx + w/2 * cos - h/2 * sin
        out_box[7] = cy + h/2 * cos + w/2 * sin

        return out_box
 
    def __getitem__(self, batch_idx):   
        # randomly shuffle for training split
        # sequentiallyy take the testing data
        #print self.mode, batch_idx
        if (self.mode == 'training'):     
            sample_indices = np.random.randint( 0, self.nb_samples, self.batch_size )            
        else:                                                                                    
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates), 
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range( batch_idx * self.batch_size, min( self.nb_samples, (batch_idx+1) * self.batch_size ) )    

        # get the file paths
        subset_X = [self.imnames[i][0].decode("utf-8-sig").encode("utf-8").strip() for i in sample_indices]
        subset_Y = [self.wordBB[i] for i in sample_indices]
                                                                                                                                                                  
        
        # get the images
        batch_X = []    
        batch_Y_mask = []
        batch_Y_border = []
        batch_Y_regress = []
        batch_Y_centerline = []
        for map_path,bboxes in zip(subset_X,subset_Y): 
            map_path = self.image_root_path + '/' + map_path
            ori_img = cv2.imread(map_path)
            height, width, c = ori_img.shape
            
            
            ############################################################################\
            # w,h and thetas gt on original image
            
            if bboxes.ndim == 2:
                #print bboxes.shape
                #print map_path
                bboxes = np.expand_dims(bboxes, axis = -1)
            x1_array = bboxes[0,0,:]# bboxes is of shape (2, 4, N) or (x/y, 1-4, # bbox)
            x2_array = bboxes[0,1,:]
            x3_array = bboxes[0,2,:]
            x4_array = bboxes[0,3,:]
            y1_array = bboxes[1,0,:]
            y2_array = bboxes[1,1,:]
            y3_array = bboxes[1,2,:]
            y4_array = bboxes[1,3,:]

            x_c_array = 0.5 *(x1_array + x3_array )
            y_c_array = 0.5 *(y1_array + y3_array )

            # find which size is the longer size
            e14_array = (x4_array-x1_array)**2 + (y4_array -y1_array)**2
            e12_array = (x2_array-x1_array)**2 + (y2_array -y1_array)**2
            thetas = []
            whs = []
            lr_pts = []
            for e12,e14,x1,x2,x3,x4,y1,y2,y3,y4 in zip(e12_array, e14_array,x1_array,x2_array,x3_array,x4_array,y1_array,y2_array,y3_array,y4_array):
                #if e12 >= e14:
                
                e_diag = np.sqrt(e12)
                if e_diag !=0.0:
                    si = (y2-y1)/e_diag
                    co = (x2-x1)/e_diag

                    w = np.sqrt(e12)
                    h = np.sqrt(e14)
                    
                    left_pt = (int((x1+x4)/2.),int((y1+y4)/2.))
                    right_pt= (int((x2+x3)/2.),int((y2+y3)/2.))
                    
                    
                else:
                    e_diag = np.sqrt(e14)
                    si = (y4-y1)/e_diag
                    co = (x4-x1)/e_diag

                    w = np.sqrt(e14)
                    h = np.sqrt(e12)
                    
                    left_pt = (int((x3+x4)/2.),int((y3+y4)/2.))
                    right_pt= (int((x1+x2)/2.),int((y1+y2)/2.))
                    
                '''
                if e_diag != 0.0:
                #print [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [np.sqrt(e12), np.sqrt(e14)], [si, co]]
                    thetas.append([si,co])
                    whs.append([w,h])
                '''
                thetas.append([si,co])
                whs.append([w,h])
                lr_pts.append([left_pt, right_pt])


            ############################################################################## 
            # prob_map gt on original image

            prob_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            for i in range(0,bboxes.shape[-1]):
                bb = bboxes[:,:,i]
                points = bb.transpose(1,0)
                # mark text regions with non-zero numbers 
                cv2.fillConvexPoly(prob_img, points.astype(np.int32),(0,i+1,0)) 

            ##########################################################
            # border gt on original image
            border_img = np.zeros((height,width,3), np.uint8) # regression target for y1
            border_p = self.border_percent
            centerline_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            
            
            for x_c,y_c, theta, wh, lr_pt in zip(x_c_array, y_c_array, thetas, whs, lr_pts):
                sin, cos = theta
                w,h = wh
                left_pt, right_pt = lr_pt
                delta =  thickness = int(max(border_p * h,3))
                w_border = w + delta # centerline
                h_border = h + delta # centerline

                outer_bbox = [x_c,y_c,sin, cos, w_border, h_border]
                box = self.bbox_transform_angle(outer_bbox)
                
                
                x11,y11,x22,y22,x33,y33,x44,y44 = [int(b) for b in box]

                points = [(x11,y11),(x22,y22),(x33,y33),(x44,y44)]
                
                cv2.line(border_img, points[0], points[1], (0,1,0), thickness)
                cv2.line(border_img, points[1], points[2], (0,1,0), thickness)
                cv2.line(border_img, points[2], points[3], (0,1,0), thickness)
                cv2.line(border_img, points[3], points[0], (0,1,0), thickness)
                
                # create centerline image
                cv2.line(centerline_img, left_pt, right_pt, (0,1,0), thickness*2 )
                


            std_size = 512
            padding = 0

            # rescale and padding to 512 x 512 of ori_img and prob_map
            ########################################################################
            if height > width:
                scale = 1.0 * std_size / height
                padding = std_size - width * scale
            else:
                scale = 1.0 * std_size / width
                padding = std_size - height * scale

            scale_img = cv2.resize(ori_img, dsize=(0,0), fx = scale, fy = scale)
            scale_prob_int = cv2.resize(prob_img, dsize = (0,0), fx = scale, fy= scale,  interpolation = cv2.INTER_NEAREST)
            scale_border_img = cv2.resize(border_img,dsize = (0,0), fx = scale, fy= scale,  interpolation = cv2.INTER_NEAREST)
            scale_centerline_img = cv2.resize(centerline_img,dsize = (0,0), fx = scale, fy= scale,  interpolation = cv2.INTER_NEAREST)
            
            
            #print 'padding',padding
            padding = np.int(round(padding))
            #print 'int padding',padding
            padsize_1 = np.int(np.floor( padding / 2 ))
            padsize_2 = np.int(padding - padsize_1)

            if height > width:
                padding_1 = np.zeros((std_size, padsize_1, c), dtype = np.uint8) # h, w, c
                padding_2 = np.zeros((std_size, padsize_2, c), dtype = np.uint8)
                std_img = np.concatenate((padding_1, scale_img, padding_2), axis = 1)
                std_prob_int = np.concatenate((padding_1, scale_prob_int, padding_2), axis = 1)
                std_border_img = np.concatenate((padding_1, scale_border_img, padding_2),axis = 1)
                std_centerline_img = np.concatenate((padding_1, scale_centerline_img, padding_2),axis = 1)
            else:
                padding_1 = np.zeros((padsize_1, std_size, c), dtype= np.uint8) # h, w, c
                padding_2 = np.zeros((padsize_2, std_size, c) , dtype = np.uint8)
                #print scale_img.shape, padding_1.shape, padding_2.shape
                std_img = np.concatenate((padding_1, scale_img, padding_2), axis = 0)
                std_prob_int = np.concatenate((padding_1, scale_prob_int, padding_2), axis = 0)
                std_border_img = np.concatenate((padding_1, scale_border_img, padding_2), axis = 0)
                std_centerline_img = np.concatenate((padding_1, scale_centerline_img, padding_2), axis = 0)

            std_prob_img = np.zeros((std_size, std_size, c))
            std_prob_img[std_prob_int > 0] = 1 # binarization

            # border image and prob_img might overlap
            # if pixel belongs to border image, then remove it from prob image
            # this guarantees the summation of probabilities equals to 1 ( not bigger than 1)
            std_prob_img = std_prob_img.astype(np.uint8) - (std_prob_img.astype(np.uint8) & std_border_img.astype(np.uint8))
            std_prob_img = std_prob_img.astype(np.float32)
            x = std_img/ 255.
            
            std_centerline_img[:,:,0] = 1- std_centerline_img[:,:,1]
            std_centerline_img = std_centerline_img[:,:,0:2] # take 2 channels
            std_centerline_img = std_centerline_img.astype(np.float32)
            ##############################################################################

            batch_X.append(x)

            if height > width:
                x_c_array = scale * x_c_array + padsize_1
                y_c_array = scale * y_c_array
            else:
                x_c_array = scale * x_c_array
                y_c_array = scale * y_c_array + padsize_1

            centers = np.array([x_c_array,y_c_array]) # (2, num_gt_boxes)
            centers = centers.transpose(1,0) # (num_gt_boxes, 2)


            #regression target for the pixels lying inside gt boxes
            index_array = np.indices((std_size, std_size)) # (2, h, w)
            inside_Y_regress = np.zeros((std_size, std_size, 6), np.float32)
            bbox_idx = 1
            for x_c, y_c, wh, theta in zip(x_c_array, y_c_array, whs, thetas):
                w, h = wh
                si, co = theta
                this_bbox_mask = (std_prob_int[:,:,1] == bbox_idx)

                this_bbox_regress_x =  np.expand_dims(np.array(x_c - index_array[1] )/ std_size, axis = -1) # h, w, 1 ###### NOrmalization needs attention
                this_bbox_regress_y = np.expand_dims( np.array(y_c - index_array[0])/ std_size, axis = -1)

                this_bbox_regress_twh = np.array([si,co, scale * w/std_size,  scale * h/std_size])[:, np.newaxis, np.newaxis] * np.ones((std_size, std_size)) # 3, h, w
                this_bbox_regress_twh = this_bbox_regress_twh.transpose(1,2,0) # h, w, 4
                this_bbox_regress = np.concatenate([this_bbox_regress_x, this_bbox_regress_y, this_bbox_regress_twh], axis = -1)

                inside_Y_regress[this_bbox_mask>0] = this_bbox_regress[this_bbox_mask>0]
                bbox_idx += 1

            prob_img = prob_img.astype(np.float32)
            prob_img = np.expand_dims(prob_img, axis = -1)
            centers = np.array(centers) # num_gt_boxes, 2
            thetas = np.array(thetas) # num_gt_boxes, 2
            whs = np.array(whs) # num_gt_boxes, 2

            dist_square = (index_array[0,:,:] - centers[:,1, np.newaxis, np.newaxis] )**2 + (index_array[1,:,:] - centers[:,0, np.newaxis, np.newaxis] )**2
            args = np.argmin(dist_square, axis = 0)

            #batch_Y_regress.append(dist_square) # to visualize center assignments
            # batch_Y_regress.append(args)

            temp_1 = (centers[args] - index_array[::-1,:,:].transpose(1,2,0))/ np.array([1.0 * std_size, 1.0 * std_size]) # h, w, 2  # center = (h * (index + pred), w * (index + pred))
            temp_2 = thetas[args] # h, w, 2
            temp_3 = whs[args] * scale / std_size # h, w, 2
            # <x,y,sin,cos,w,h>
            temp_Y_regress = np.concatenate([temp_1, temp_2, temp_3], axis = -1)


            std_prob_img = np.expand_dims(std_prob_img[:,:,1], axis = -1)
            std_border_img = np.expand_dims(std_border_img[:,:,1], axis = -1)

            Y_regress = inside_Y_regress * std_prob_img  + temp_Y_regress * (1-std_prob_img)
            
            # append
            batch_Y_mask.append(std_prob_img)
            batch_Y_border.append(std_border_img)
            batch_Y_regress.append(Y_regress)
            batch_Y_centerline.append(std_centerline_img)

        batch_X = np.array(batch_X).astype(np.float32)
        batch_Y_mask = np.array(batch_Y_mask).astype(np.float32)
        batch_Y_border = np.array(batch_Y_border).astype(np.float32)
        batch_Y_centerline = np.array(batch_Y_centerline).astype(np.float32)
        
        #batch_Y_mask_yield = np.concatenate([batch_Y_mask, 1- batch_Y_mask], axis = -1)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border,0,1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1-mask_and_border], axis = -1).astype(np.float32)

        batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        batch_Y_regress1 = batch_Y_regress[:,:,:,0:2]
        batch_Y_regress2 = batch_Y_regress[:,:,:,2:4]
        batch_Y_regress3 = batch_Y_regress[:,:,:,4:]

        batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis = -1)
        batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis = -1)
        batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis = -1)
        
        #print 'batch_x.shape',batch_X.shape
        return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_regress1,batch_Y_regress2, batch_Y_regress3])
        #return (batch_X, batch_Y_mask)
                                                                                                  
                                                     
                                                                                                  
    def next(self):                                                                               
        idx = self.idx                                                                            
        self.idx = ( 1 + idx ) % self.nb_samples                                                  
        return  self[idx]
    


    
    

class SynthMap_DataGenerator(object):   
    # image_root_path : map and mask images dir
    # list_path points to the file-list of train/val/test split 
    def __init__(self, image_root_path, batch_size = 128,  seed = 1234, mode = 'training', border_percent =0.15 ):  
        img_path_list = glob.glob(image_root_path + '/*.jpg') 
        
        # get the full path of map & maks files
        X,Y = [],[]
        for img_name in img_path_list:
            img_id = os.path.basename(img_name).split('.jpg')[0]
            label_path = image_root_path + '/' + img_id + '.txt'
            
            Y.append(label_path)
            
        X = img_path_list
        

        print('num_samples = ',len(X))
        # X: map file list, Y: binary mask file list                                                                                                             
        self.idx = 0 
        self.nb_samples =  len(X)   
        self.X = X                 
        self.Y = Y                 
        self.batch_size = batch_size 
        self.border_percent = border_percent
                                                                                                                                                                        
        self.seed = seed             
        self.mode = mode                                    
        np.random.seed(seed)
         
    def __len__(self):
        return self.nb_samples
    
    
    def bbox_transform_angle(self, bbox):
        """convert a bbox of form [cx, cy, SIN,COS, w, h] to points. Works
        for numpy array or list of tensors.
        """
        cx, cy, sin, cos, w, h  = bbox
        # y in img coord system yifan

        out_box = [[]]*8

        out_box[0] = cx + w/2 * cos + h/2 * sin
        out_box[1] = cy - h/2 * cos + w/2 * sin
        out_box[2] = cx - w/2 * cos + h/2 * sin
        out_box[3] = cy - h/2 * cos - w/2 * sin
        out_box[4] = cx - w/2 * cos - h/2 * sin
        out_box[5] = cy + h/2 * cos - w/2 * sin
        out_box[6] = cx + w/2 * cos - h/2 * sin
        out_box[7] = cy + h/2 * cos + w/2 * sin

        return out_box




    def __getitem__(self, batch_idx):   
        # randomly shuffle for training split
        # sequentiallyy take the testing data
        #print self.mode, batch_idx
        if (self.mode == 'training'):     
            sample_indices = np.random.randint( 0, self.nb_samples, self.batch_size )            
            #print 'training, sample_indices',sample_indices
        else:                                                                                    
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates), 
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range( batch_idx * self.batch_size, min( self.nb_samples, (batch_idx+1) * self.batch_size ) )    
            #print 'val, sample_indices',sample_indices
        # get the file paths
        subset_X = [self.X[i] for i in sample_indices]
        subset_Y = [self.Y[i] for i in sample_indices]
                                                                                                                                                                  
        std_size = 512
        # get the images
        batch_X = []    
        batch_Y_mask = []
        batch_Y_border = []
        batch_Y_regress = []
        for map_path,label_path in zip(subset_X,subset_Y): 

            x = cv2.imread(map_path)
            o_height, o_width = x.shape[0], x.shape[1] 
            x = cv2.resize (x, (512,512))
            
            h_sca = 512. / o_height
            w_sca = 512. / o_width
            
            
            height, width = x.shape[0], x.shape[1] 
            #print 'height,width',height,width
            x = x/255.
            batch_X.append(x)
            # init variables
            prob_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            border_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            inside_Y_regress = np.zeros((height, width, 6), np.float32)
            border_p = self.border_percent
            centers = []
            thetas = []
            whs = []
            
            # read and precess file
            with open(label_path, 'r') as f:
                data = f.readlines()
            
            bbox_idx = 0
            for line in data:
                bbox_idx += 1
                line = line.decode("utf-8-sig").encode("utf-8").strip().split(',')
                #if len(line)!=9:
                #    print line
                x1 = line[0]
                y1 = line[1] 
                x2 = line[2]
                y2 = line[3]
                x3  = line[4]
                y3  = line[5]
                x4 = line[6]
                y4 = line[7]
                transcription = line[-1]
                ## check h_scale and w_scale when they are not the same
                x1 = float(x1) * w_sca
                y1 = float(y1) * h_sca
                x2 = float(x2) * w_sca
                y2 = float(y2) * h_sca
                x3 = float(x3) * w_sca
                y3 = float(y3) * h_sca
                x4 = float(x4) * w_sca
                y4 = float(y4) * h_sca
                x_c = 0.5*(x1+x3)    
                y_c = 0.5*(y1+y3)

                if transcription == '###':
                    continue
                
                # create mask image
                points = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                #print (points)
                cv2.fillConvexPoly(prob_img, points.astype(np.int),(0,bbox_idx ,0))
                centers.append([x_c,y_c])
                
                # which one is the longer edge
                e14 = (x4-x1)*(x4-x1) + (y4 -y1)*(y4 -y1)
                e12 = (x2-x1)*(x2-x1) + (y2 -y1)*(y2 -y1)
                e_diag = np.sqrt(e12)
                if e_diag !=0.0:
                    si = (y2-y1)/e_diag
                    co = (x2-x1)/e_diag

                    w = np.sqrt(e12)
                    h = np.sqrt(e14)
                else:
                    e_diag = np.sqrt(e14)
                    si = (y4-y1)/e_diag
                    co = (x4-x1)/e_diag

                    w = np.sqrt(e14)
                    h = np.sqrt(e12)
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
                thetas.append([si,co])
                whs.append([w,h])
    
                # create border img
                delta = thickness = int(max(border_p * h,3))
                w_border = w + delta # centerline
                h_border = h + delta # centerline
                outer_bbox = [x_c, y_c, si, co, w_border, h_border]
                box = self.bbox_transform_angle(outer_bbox)
                x11,y11,x22,y22,x33,y33,x44,y44 = [int(b) for b in box]

                points = [(x11,y11),(x22,y22),(x33,y33),(x44,y44)]
                
                cv2.line(border_img, points[0], points[1], (0,1,0), thickness)
                cv2.line(border_img, points[1], points[2], (0,1,0), thickness)
                cv2.line(border_img, points[2], points[3], (0,1,0), thickness)
                cv2.line(border_img, points[3], points[0], (0,1,0), thickness)


                #this_bbox_mask = np.zeros((height,width,3), np.uint8) # hightlight current bbox
                #cv2.fillConvexPoly(this_bbox_mask, points.astype(np.int),(0,1,0))
                #this_bbox_mask = this_bbox_mask[:,:,1] # take the green channel, shape (h, w)
                this_bbox_mask = (prob_img[:,:,1] == bbox_idx)
                
                index_array = np.indices((height,width)) # (2, h, w)
                
                this_bbox_regress_x =  np.expand_dims(np.array(x_c - index_array[1] )/ std_size, axis = -1) # h, w, 1 ###### NOrmalization needs attention
                this_bbox_regress_y = np.expand_dims( np.array(y_c - index_array[0])/ std_size, axis = -1) # h, w, 1 ####### Normalization needs attention
                this_bbox_regress_twh = np.array([si, co , w/std_size, h/std_size])[:, np.newaxis, np.newaxis] * np.ones((height,width)) # 4, h, w
                this_bbox_regress_twh = this_bbox_regress_twh.transpose(1,2,0) # h, w, 4
                this_bbox_regress = np.concatenate([this_bbox_regress_x, this_bbox_regress_y, this_bbox_regress_twh], axis = -1)
                
                inside_Y_regress[this_bbox_mask>0] = this_bbox_regress[this_bbox_mask>0]


            prob_img[prob_img>0] =1
            prob_img = prob_img.astype(np.float32)
            prob_img = prob_img[:,:,1]
            prob_img = np.expand_dims(prob_img, axis = -1)
            border_img = border_img.astype(np.float32)
            border_img = border_img[:,:,1]
            border_img = np.expand_dims(border_img, axis = -1)

            centers = np.array(centers) # num_gt_boxes, 2
            thetas = np.array(thetas) # num_gt_boxes
            whs = np.array(whs) # num_gt_boxes, 2
            
            
            batch_Y_mask.append(prob_img)
            batch_Y_border.append(border_img)
            

            if centers.shape[0] == 0:
                batch_Y_regress.append(np.zeros((height, width, 6)))
                continue
 
            dist_square = (index_array[0,:,:] - centers[:,1, np.newaxis, np.newaxis] )**2 + (index_array[1,:,:] - centers[:,0, np.newaxis, np.newaxis] )**2
            args = np.argmin(dist_square, axis = 0)
            
            #batch_Y_regress.append(dist_square) # to visualize center assignments
            # batch_Y_regress.append(args)
            
            temp_1 = (centers[args] - index_array[::-1,:,:].transpose(1,2,0))/ np.array([1.0 * std_size, 1.0 * std_size]) # h, w, 2  # center = (h * (index + pred), w * (index + pred))
            temp_2 = thetas[args] # h, w, 2
            temp_3 = whs[args] / np.array([1.0 * std_size, 1.0 * std_size]) # h, w, 2
            temp_Y_regress = np.concatenate([temp_1, temp_2, temp_3], axis = -1)
            
            #print inside_Y_regress.shape
            #print prob_img.shape
            #print temp_Y_regress.shape
            
            Y_regress = inside_Y_regress * prob_img + temp_Y_regress * (1-prob_img)
            batch_Y_regress.append(Y_regress)


        batch_X = np.array(batch_X)  
        batch_Y_mask = np.array(batch_Y_mask) 
        batch_Y_border = np.array(batch_Y_border)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border, 0, 1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1-mask_and_border], axis = -1).astype(np.float32)

        
        batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        batch_Y_regress1 = batch_Y_regress[:,:,:,0:2]
        batch_Y_regress2 = batch_Y_regress[:,:,:,2:4]
        batch_Y_regress3 = batch_Y_regress[:,:,:,4:]

        batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis = -1)
        batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis = -1)
        batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis = -1)
        

        
        return (batch_X, [batch_Y_mask_yield, batch_Y_regress1, batch_Y_regress2, batch_Y_regress3])
                                                                                                  
                                                     
                                                                                                  
    def next(self):                                                                               
        idx = self.idx                                                                            
        self.idx = ( 1 + idx ) % self.nb_samples                                                  
        return  self[idx]


#mapTrain_2里使用的
class SynthMap_DataGenerator_Centerline(object):   
    # image_root_path : map and mask images dir
    # list_path points to the file-list of train/val/test split 
    def __init__(self, image_root_path, batch_size = 128,  seed = 1234, mode = 'training', border_percent =0.15 ):  
        img_path_list = glob.glob(image_root_path + '/*.jpg') 
        
        # get the full path of map & maks files
        X,Y = [],[]
        for img_name in img_path_list:
            img_id = os.path.basename(img_name).split('.jpg')[0]
            label_path = image_root_path + '/' + img_id + '.txt'
            
            Y.append(label_path)
            
        X = img_path_list
        

        print('num_samples = ',len(X))
                                                                                                           
        self.idx = 0 
        self.nb_samples =  len(X)   
        self.X = X                 
        self.Y = Y                 
        self.batch_size = batch_size 
        self.border_percent = border_percent
                                                                                                                                             
        self.seed = seed             
        self.mode = mode                                    
        np.random.seed(seed)
         
    def __len__(self):
        return self.nb_samples
    
    #把bbox存储的数据格式，把边框的四个点坐标算出来
    def bbox_transform_angle(self, bbox):
        """convert a bbox of form [cx, cy, SIN,COS, w, h] to points. Works
        for numpy array or list of tensors.
        """
        cx, cy, sin, cos, w, h  = bbox
        # y in img coord system yifan

        out_box = [[]]*8

        out_box[0] = cx + w/2 * cos + h/2 * sin
        out_box[1] = cy - h/2 * cos + w/2 * sin
        out_box[2] = cx - w/2 * cos + h/2 * sin
        out_box[3] = cy - h/2 * cos - w/2 * sin
        out_box[4] = cx - w/2 * cos - h/2 * sin
        out_box[5] = cy + h/2 * cos - w/2 * sin
        out_box[6] = cx + w/2 * cos - h/2 * sin
        out_box[7] = cy + h/2 * cos + w/2 * sin

        return out_box




    def __getitem__(self, batch_idx):   
        # randomly shuffle for training split
        # sequentiallyy take the testing data
        #print self.mode, batch_idx
        if (self.mode == 'training'):     
            sample_indices = np.random.randint( 0, self.nb_samples, self.batch_size )            
            #print 'training, sample_indices',sample_indices
        else:                                                                                    
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates), 
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range( batch_idx * self.batch_size, min( self.nb_samples, (batch_idx+1) * self.batch_size ) )    
            #print 'val, sample_indices',sample_indices

        # get the file paths
        subset_X = [self.X[i] for i in sample_indices]
        subset_Y = [self.Y[i] for i in sample_indices]
                                                                                                                                                                  
        std_size = 512
        # get the images
        batch_X = []    
        batch_Y_mask = []
        batch_Y_border = []
        batch_Y_regress = []
        batch_Y_centerline = []

        for map_path,label_path in zip(subset_X,subset_Y): 

            x = cv2.imread(map_path)
            o_height, o_width = x.shape[0], x.shape[1] 
            x = cv2.resize (x, (std_size,std_size))
            
            h_sca = std_size * 1. / o_height
            w_sca = std_size * 1. / o_width
            
            
            height, width = x.shape[0], x.shape[1] 
            #print 'height,width',height,width
            x = x/255.
            batch_X.append(x)
            # init variables
            prob_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            border_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            centerline_img = np.zeros((height,width,3), np.uint8) # regression target for y1 (segmentation task)
            
            inside_Y_regress = np.zeros((height, width, 6), np.float32)
            border_p = self.border_percent
            centers = []
            thetas = []
            whs = []
            
            # read and precess file
            with open(label_path, 'r') as f:
                data = f.readlines()
            
            bbox_idx = 0
            for line in data:
                bbox_idx += 1
                #line = line.decode("utf-8-sig").encode("utf-8").strip().split(',')
                line = line.strip().split(',')
                #if len(line)!=9:
                #    print line
                x1 = line[0]
                y1 = line[1] 
                x2 = line[2]
                y2 = line[3]
                x3  = line[4]
                y3  = line[5]
                x4 = line[6]
                y4 = line[7]
                transcription = line[-1]
                ## check h_scale and w_scale when they are not the same
                x1 = float(x1) * w_sca
                y1 = float(y1) * h_sca
                x2 = float(x2) * w_sca
                y2 = float(y2) * h_sca
                x3 = float(x3) * w_sca
                y3 = float(y3) * h_sca
                x4 = float(x4) * w_sca
                y4 = float(y4) * h_sca
                x_c = 0.5*(x1+x3)    
                y_c = 0.5*(y1+y3)

                if transcription == '###':
                    continue
                
                # create mask image
                points = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                #print (points)
                #填充多边形
                cv2.fillConvexPoly(prob_img, points.astype(np.int),(0,bbox_idx ,0))
                centers.append([x_c,y_c])
                
                # which one is the longer edge
                e14 = (x4-x1)*(x4-x1) + (y4 -y1)*(y4 -y1)
                e12 = (x2-x1)*(x2-x1) + (y2 -y1)*(y2 -y1)
                e_diag = np.sqrt(e12)
                
                # create centerline img
                # two cases: mid(1,2)->mid(3,4) or mid(2,3)->mid(1,4)
                
                
                if e_diag !=0.0:
                    # 1 -- 2 
                    # 4 -- 3
                    si = (y2-y1)/e_diag
                    co = (x2-x1)/e_diag

                    w = np.sqrt(e12)
                    h = np.sqrt(e14)
                    
                    left_pt = (int((x1+x4)/2.),int((y1+y4)/2.))
                    right_pt= (int((x2+x3)/2.),int((y2+y3)/2.))
                    
                else:
                    # 4 -- 1
                    # 3 -- 2
                    e_diag = np.sqrt(e14)
                    si = (y4-y1)/e_diag
                    co = (x4-x1)/e_diag

                    w = np.sqrt(e14)
                    h = np.sqrt(e12)
                    
                    left_pt = (int((x3+x4)/2.),int((y3+y4)/2.))
                    right_pt= (int((x1+x2)/2.),int((y1+y2)/2.))
                    
                    
                
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
                thetas.append([si,co])
                whs.append([w,h])
                
                
    
                # create border img
                delta = thickness = int(max(border_p * h,3))
                w_border = w + delta # centerline
                h_border = h + delta # centerline
                outer_bbox = [x_c, y_c, si, co, w_border, h_border]
                box = self.bbox_transform_angle(outer_bbox)
                x11,y11,x22,y22,x33,y33,x44,y44 = [int(b) for b in box]

                points = [(x11,y11),(x22,y22),(x33,y33),(x44,y44)]
                
                cv2.line(border_img, points[0], points[1], (0,1,0), thickness)
                cv2.line(border_img, points[1], points[2], (0,1,0), thickness)
                cv2.line(border_img, points[2], points[3], (0,1,0), thickness)
                cv2.line(border_img, points[3], points[0], (0,1,0), thickness)
                
                # create centerline image
                cv2.line(centerline_img, left_pt, right_pt, (0,1,0), thickness*2 )
                

                # regressions
                this_bbox_mask = (prob_img[:,:,1] == bbox_idx)
                index_array = np.indices((height,width)) # (2, h, w)
                
                this_bbox_regress_x =  np.expand_dims(np.array(x_c - index_array[1] )/ std_size, axis = -1) # h, w, 1 ###### NOrmalization needs attention
                this_bbox_regress_y = np.expand_dims( np.array(y_c - index_array[0])/ std_size, axis = -1) # h, w, 1 ####### Normalization needs attention
                this_bbox_regress_twh = np.array([si, co , w/std_size, h/std_size])[:, np.newaxis, np.newaxis] * np.ones((height,width)) # 4, h, w
                this_bbox_regress_twh = this_bbox_regress_twh.transpose(1,2,0) # h, w, 4
                this_bbox_regress = np.concatenate([this_bbox_regress_x, this_bbox_regress_y, this_bbox_regress_twh], axis = -1)
                
                inside_Y_regress[this_bbox_mask>0] = this_bbox_regress[this_bbox_mask>0]


            prob_img[prob_img>0] =1
            prob_img = prob_img.astype(np.float32)
            prob_img = prob_img[:,:,1]
            prob_img = np.expand_dims(prob_img, axis = -1)
            border_img = border_img.astype(np.float32)
            border_img = border_img[:,:,1]
            border_img = np.expand_dims(border_img, axis = -1)
            
            centerline_img[:,:,0] = 1- centerline_img[:,:,1]
            centerline_img = centerline_img[:,:,0:2] # take 2 channels
            

            centers = np.array(centers) # num_gt_boxes, 2
            thetas = np.array(thetas) # num_gt_boxes
            whs = np.array(whs) # num_gt_boxes, 2
            
            
            batch_Y_mask.append(prob_img)
            batch_Y_border.append(border_img)
            batch_Y_centerline.append(centerline_img)
            

            if centers.shape[0] == 0:
                batch_Y_regress.append(np.zeros((height, width, 6)))
                continue
 
            dist_square = (index_array[0,:,:] - centers[:,1, np.newaxis, np.newaxis] )**2 + (index_array[1,:,:] - centers[:,0, np.newaxis, np.newaxis] )**2
            args = np.argmin(dist_square, axis = 0)
            
            #batch_Y_regress.append(dist_square) # to visualize center assignments
            # batch_Y_regress.append(args)
            
            temp_1 = (centers[args] - index_array[::-1,:,:].transpose(1,2,0))/ np.array([1.0 * std_size, 1.0 * std_size]) # h, w, 2  # center = (h * (index + pred), w * (index + pred))
            temp_2 = thetas[args] # h, w, 2
            temp_3 = whs[args] / np.array([1.0 * std_size, 1.0 * std_size]) # h, w, 2
            temp_Y_regress = np.concatenate([temp_1, temp_2, temp_3], axis = -1)
            
            #print inside_Y_regress.shape
            #print prob_img.shape
            #print temp_Y_regress.shape
            
            Y_regress = inside_Y_regress * prob_img + temp_Y_regress * (1-prob_img)
            batch_Y_regress.append(Y_regress)


        batch_X = np.array(batch_X)  
        batch_Y_mask = np.array(batch_Y_mask) 
        batch_Y_border = np.array(batch_Y_border)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border, 0, 1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1-mask_and_border], axis = -1).astype(np.float32)

        batch_Y_centerline = np.array(batch_Y_centerline).astype(np.float32)
        
        batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        batch_Y_regress1 = batch_Y_regress[:,:,:,0:2]
        batch_Y_regress2 = batch_Y_regress[:,:,:,2:4]
        batch_Y_regress3 = batch_Y_regress[:,:,:,4:]

        batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis = -1)
        batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis = -1)
        batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis = -1)
        

        
        return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_regress1, batch_Y_regress2, batch_Y_regress3])
                                                                                                  
                                                     
                                                                                                  
    def next(self):                                                                               
        idx = self.idx                                                                            
        self.idx = ( 1 + idx ) % self.nb_samples                                                  
        return  self[idx]


# 修改之后的函数
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
        batch_Y_regress = []
        batch_Y_centerline = []

        #加入一个图片数组作为最终结果：local height

        batch_Y_localheight = []


        for map_path, label_path in zip(subset_X, subset_Y):

            x = cv2.imread(map_path)
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

            # read and precess file
            with open(label_path, 'r') as f:
                data = f.readlines()

            bbox_idx = 0
            for line in data:
                bbox_idx += 1
                # line = line.decode("utf-8-sig").encode("utf-8").strip().split(',')
                line = line.strip().split(',')
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
                transcription = line[-1]
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

                if transcription == '###':
                    continue

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
                cv2.line(localheight_img, left_pt, right_pt, (lh, 0, 0), thickness * 2)

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

#SynthText Localheight
class SynthText_DataGenerator_Centerline_Localheight(object):
    # image_root_path : PATH_TO_SynthText/
    def __init__(self, image_root_path, batch_size=128, seed=1234, mode='training', border_percent=0.15):

        meta_file = image_root_path + '/gt.mat'
        import scipy.io
        mat_data = scipy.io.loadmat(meta_file)

        wordBB = mat_data['wordBB'][0]  # bounding boxes
        txt = mat_data['txt'][0]  # text content
        imnames = mat_data['imnames'][0]  # image names
        charBB = mat_data['charBB'][0]  # character level boxes

        ############### debug info ###############
        '''
        print wordBB[0].shape
        print txt[0]
        print imnames[0]
        print charBB[0].shape

        (2, 4, 15)
        [u'Lines:\nI lost\nKevin ' u'will                '
         u'line\nand            ' u'and\nthe             ' u'(and                '
         u'the\nout             ' u'you                 ' u"don't\n pkg          "]
        [u'8/ballet_106_0.jpg']
        (2, 4, 54)

        '''

        print('num_samples = ', len(wordBB))
        # X: map file list, Y: binary mask file list
        self.idx = 0
        self.nb_samples = len(wordBB)
        self.batch_size = batch_size

        self.wordBB = wordBB
        self.txt = txt
        self.imnames = imnames
        self.charBB = charBB
        self.image_root_path = image_root_path
        self.border_percent = border_percent

        self.seed = seed
        self.mode = mode
        np.random.seed(seed)

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
        else:
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates),
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range(batch_idx * self.batch_size, min(self.nb_samples, (batch_idx + 1) * self.batch_size))

            # get the file paths
        subset_X = [self.imnames[i][0].strip() for i in sample_indices]
        subset_Y = [self.wordBB[i] for i in sample_indices]

        # get the images
        batch_X = []
        batch_Y_mask = []
        batch_Y_border = []
        batch_Y_regress = []
        batch_Y_centerline = []

        #localheight
        batch_Y_localheight=[]

        for map_path, bboxes in zip(subset_X, subset_Y):
            map_path = self.image_root_path + '/' + map_path
            ori_img = cv2.imread(map_path)
            height, width, c = ori_img.shape

            ############################################################################\
            # w,h and thetas gt on original image

            if bboxes.ndim == 2:
                # print bboxes.shape
                # print map_path
                bboxes = np.expand_dims(bboxes, axis=-1)
            x1_array = bboxes[0, 0, :]  # bboxes is of shape (2, 4, N) or (x/y, 1-4, # bbox)
            x2_array = bboxes[0, 1, :]
            x3_array = bboxes[0, 2, :]
            x4_array = bboxes[0, 3, :]
            y1_array = bboxes[1, 0, :]
            y2_array = bboxes[1, 1, :]
            y3_array = bboxes[1, 2, :]
            y4_array = bboxes[1, 3, :]

            x_c_array = 0.5 * (x1_array + x3_array)
            y_c_array = 0.5 * (y1_array + y3_array)

            # find which size is the longer size
            e14_array = (x4_array - x1_array) ** 2 + (y4_array - y1_array) ** 2
            e12_array = (x2_array - x1_array) ** 2 + (y2_array - y1_array) ** 2
            thetas = []
            whs = []
            lr_pts = []

            #localheight

            lh=0

            for e12, e14, x1, x2, x3, x4, y1, y2, y3, y4 in zip(e12_array, e14_array, x1_array, x2_array, x3_array,
                                                                x4_array, y1_array, y2_array, y3_array, y4_array):
                # if e12 >= e14:

                e_diag = np.sqrt(e12)
                if e_diag != 0.0:
                    si = (y2 - y1) / e_diag
                    co = (x2 - x1) / e_diag

                    w = np.sqrt(e12)
                    h = np.sqrt(e14)

                    lh=np.sqrt(e14)

                    left_pt = (int((x1 + x4) / 2.), int((y1 + y4) / 2.))
                    right_pt = (int((x2 + x3) / 2.), int((y2 + y3) / 2.))


                else:
                    e_diag = np.sqrt(e14)
                    si = (y4 - y1) / e_diag
                    co = (x4 - x1) / e_diag

                    w = np.sqrt(e14)
                    h = np.sqrt(e12)

                    lh=np.sqrt(e12)

                    left_pt = (int((x3 + x4) / 2.), int((y3 + y4) / 2.))
                    right_pt = (int((x1 + x2) / 2.), int((y1 + y2) / 2.))

                '''
                if e_diag != 0.0:
                #print [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [np.sqrt(e12), np.sqrt(e14)], [si, co]]
                    thetas.append([si,co])
                    whs.append([w,h])
                '''
                thetas.append([si, co])
                whs.append([w, h])
                lr_pts.append([left_pt, right_pt])

            ##############################################################################
            # prob_map gt on original image

            prob_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)

            for i in range(0, bboxes.shape[-1]):
                bb = bboxes[:, :, i]
                points = bb.transpose(1, 0)
                # mark text regions with non-zero numbers
                cv2.fillConvexPoly(prob_img, points.astype(np.int32), (0, i + 1, 0))

                ##########################################################
            # border gt on original image
            border_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1
            border_p = self.border_percent

            centerline_img = np.zeros((height, width, 3), np.uint8)  # regression target for y1 (segmentation task)

            #local height

            localheight_img = np.zeros((height, width, 3), np.uint8)

            for x_c, y_c, theta, wh, lr_pt in zip(x_c_array, y_c_array, thetas, whs, lr_pts):
                sin, cos = theta
                w, h = wh
                left_pt, right_pt = lr_pt
                delta = thickness = int(max(border_p * h, 3))
                w_border = w + delta  # centerline
                h_border = h + delta  # centerline

                outer_bbox = [x_c, y_c, sin, cos, w_border, h_border]
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
                cv2.line(localheight_img, left_pt, right_pt, (int(lh), 0, 0), thickness * 2)

            std_size = 512
            padding = 0

            # rescale and padding to 512 x 512 of ori_img and prob_map
            ########################################################################
            if height > width:
                scale = 1.0 * std_size / height
                padding = std_size - width * scale
            else:
                scale = 1.0 * std_size / width
                padding = std_size - height * scale

            scale_img = cv2.resize(ori_img, dsize=(0, 0), fx=scale, fy=scale)
            scale_prob_int = cv2.resize(prob_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            scale_border_img = cv2.resize(border_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            scale_centerline_img = cv2.resize(centerline_img, dsize=(0, 0), fx=scale, fy=scale,
                                              interpolation=cv2.INTER_NEAREST)

            #localheight
            scale_localheight_img=cv2.resize(localheight_img, dsize=(0, 0), fx=scale, fy=scale,
                                              interpolation=cv2.INTER_NEAREST)

            # print 'padding',padding
            padding = np.int(round(padding))
            # print 'int padding',padding
            padsize_1 = np.int(np.floor(padding / 2))
            padsize_2 = np.int(padding - padsize_1)

            if height > width:
                padding_1 = np.zeros((std_size, padsize_1, c), dtype=np.uint8)  # h, w, c
                padding_2 = np.zeros((std_size, padsize_2, c), dtype=np.uint8)
                std_img = np.concatenate((padding_1, scale_img, padding_2), axis=1)
                std_prob_int = np.concatenate((padding_1, scale_prob_int, padding_2), axis=1)
                std_border_img = np.concatenate((padding_1, scale_border_img, padding_2), axis=1)
                std_centerline_img = np.concatenate((padding_1, scale_centerline_img, padding_2), axis=1)

                #lh
                std_localheight_img=np.concatenate((padding_1, scale_localheight_img, padding_2), axis=1)

            else:
                padding_1 = np.zeros((padsize_1, std_size, c), dtype=np.uint8)  # h, w, c
                padding_2 = np.zeros((padsize_2, std_size, c), dtype=np.uint8)
                # print scale_img.shape, padding_1.shape, padding_2.shape
                std_img = np.concatenate((padding_1, scale_img, padding_2), axis=0)
                std_prob_int = np.concatenate((padding_1, scale_prob_int, padding_2), axis=0)
                std_border_img = np.concatenate((padding_1, scale_border_img, padding_2), axis=0)
                std_centerline_img = np.concatenate((padding_1, scale_centerline_img, padding_2), axis=0)

                # lh
                std_localheight_img = np.concatenate((padding_1, scale_localheight_img, padding_2), axis=0)

            std_prob_img = np.zeros((std_size, std_size, c))
            std_prob_img[std_prob_int > 0] = 1  # binarization

            # border image and prob_img might overlap
            # if pixel belongs to border image, then remove it from prob image
            # this guarantees the summation of probabilities equals to 1 ( not bigger than 1)
            std_prob_img = std_prob_img.astype(np.uint8) - (
                        std_prob_img.astype(np.uint8) & std_border_img.astype(np.uint8))
            std_prob_img = std_prob_img.astype(np.float32)
            x = std_img / 255.

            std_centerline_img[:, :, 0] = 1 - std_centerline_img[:, :, 1]
            std_centerline_img = std_centerline_img[:, :, 0:2]  # take 2 channels
            std_centerline_img = std_centerline_img.astype(np.float32)

            #lh
            std_localheight_img = std_localheight_img[:, :, 0:1]
            std_localheight_img = std_localheight_img.astype(np.float32)

            ##############################################################################

            batch_X.append(x)

            if height > width:
                x_c_array = scale * x_c_array + padsize_1
                y_c_array = scale * y_c_array
            else:
                x_c_array = scale * x_c_array
                y_c_array = scale * y_c_array + padsize_1

            centers = np.array([x_c_array, y_c_array])  # (2, num_gt_boxes)
            centers = centers.transpose(1, 0)  # (num_gt_boxes, 2)

            # regression target for the pixels lying inside gt boxes
            index_array = np.indices((std_size, std_size))  # (2, h, w)
            inside_Y_regress = np.zeros((std_size, std_size, 6), np.float32)
            bbox_idx = 1
            for x_c, y_c, wh, theta in zip(x_c_array, y_c_array, whs, thetas):
                w, h = wh
                si, co = theta
                this_bbox_mask = (std_prob_int[:, :, 1] == bbox_idx)

                this_bbox_regress_x = np.expand_dims(np.array(x_c - index_array[1]) / std_size,
                                                     axis=-1)  # h, w, 1 ###### NOrmalization needs attention
                this_bbox_regress_y = np.expand_dims(np.array(y_c - index_array[0]) / std_size, axis=-1)

                this_bbox_regress_twh = np.array([si, co, scale * w / std_size, scale * h / std_size])[:, np.newaxis,
                                        np.newaxis] * np.ones((std_size, std_size))  # 3, h, w
                this_bbox_regress_twh = this_bbox_regress_twh.transpose(1, 2, 0)  # h, w, 4
                this_bbox_regress = np.concatenate([this_bbox_regress_x, this_bbox_regress_y, this_bbox_regress_twh],
                                                   axis=-1)

                inside_Y_regress[this_bbox_mask > 0] = this_bbox_regress[this_bbox_mask > 0]
                bbox_idx += 1

            prob_img = prob_img.astype(np.float32)
            prob_img = np.expand_dims(prob_img, axis=-1)
            centers = np.array(centers)  # num_gt_boxes, 2
            thetas = np.array(thetas)  # num_gt_boxes, 2
            whs = np.array(whs)  # num_gt_boxes, 2

            dist_square = (index_array[0, :, :] - centers[:, 1, np.newaxis, np.newaxis]) ** 2 + (
                        index_array[1, :, :] - centers[:, 0, np.newaxis, np.newaxis]) ** 2
            args = np.argmin(dist_square, axis=0)

            # batch_Y_regress.append(dist_square) # to visualize center assignments
            # batch_Y_regress.append(args)

            temp_1 = (centers[args] - index_array[::-1, :, :].transpose(1, 2, 0)) / np.array(
                [1.0 * std_size, 1.0 * std_size])  # h, w, 2  # center = (h * (index + pred), w * (index + pred))
            temp_2 = thetas[args]  # h, w, 2
            temp_3 = whs[args] * scale / std_size  # h, w, 2
            # <x,y,sin,cos,w,h>
            temp_Y_regress = np.concatenate([temp_1, temp_2, temp_3], axis=-1)

            std_prob_img = np.expand_dims(std_prob_img[:, :, 1], axis=-1)
            std_border_img = np.expand_dims(std_border_img[:, :, 1], axis=-1)

            Y_regress = inside_Y_regress * std_prob_img + temp_Y_regress * (1 - std_prob_img)

            # append
            batch_Y_mask.append(std_prob_img)
            batch_Y_border.append(std_border_img)
            batch_Y_regress.append(Y_regress)
            batch_Y_centerline.append(std_centerline_img)

            #lh
            batch_Y_localheight.append(std_localheight_img)

        batch_X = np.array(batch_X).astype(np.float32)
        batch_Y_mask = np.array(batch_Y_mask).astype(np.float32)
        batch_Y_border = np.array(batch_Y_border).astype(np.float32)
        batch_Y_centerline = np.array(batch_Y_centerline).astype(np.float32)

        #lh
        batch_Y_localheight = np.array(batch_Y_localheight).astype(np.float32)

        # batch_Y_mask_yield = np.concatenate([batch_Y_mask, 1- batch_Y_mask], axis = -1)
        mask_and_border = np.clip(batch_Y_mask + batch_Y_border, 0, 1)
        batch_Y_mask_yield = np.concatenate([batch_Y_mask, batch_Y_border, 1 - mask_and_border], axis=-1).astype(
            np.float32)

        batch_Y_regress = np.array(batch_Y_regress).astype(np.float32)
        batch_Y_regress1 = batch_Y_regress[:, :, :, 0:2]
        batch_Y_regress2 = batch_Y_regress[:, :, :, 2:4]
        batch_Y_regress3 = batch_Y_regress[:, :, :, 4:]

        batch_Y_regress1 = np.concatenate([batch_Y_mask, batch_Y_regress1], axis=-1)
        batch_Y_regress2 = np.concatenate([batch_Y_mask, batch_Y_regress2], axis=-1)
        batch_Y_regress3 = np.concatenate([batch_Y_mask, batch_Y_regress3], axis=-1)

        # print 'batch_x.shape',batch_X.shape
        #return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_regress1, batch_Y_regress2, batch_Y_regress3])
        return (batch_X, [batch_Y_mask_yield, batch_Y_centerline, batch_Y_localheight])
        # return (batch_X, batch_Y_mask)

    def next(self):
        idx = self.idx
        self.idx = (1 + idx) % self.nb_samples
        return self[idx]







