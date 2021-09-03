#!/usr/bin/env python
# Author: Dong Guo

import cv2
import os
import numpy as np

def fix_img(filepath):
    roi_dict={}
    
    # read image with cv2
    im = cv2.imread(filepath)
    # convert the image from BGR into gray
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # image binarization, the default threshold is 127
    ret, im_inv = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
    
    # image denoising using the gaussian blur 
    kernel = 1/16*np.array([[1,2,1], [2,4,2], [1,2,1]])
    im_blur = cv2.filter2D(im_inv,-1,kernel)
    
    # image binarization, after debugging, found that 185 is a better value for the blurred CAPTCHAs
    ret, im_res = cv2.threshold(im_blur,185,255,cv2.THRESH_BINARY)
    
    # after debugging, found that all CAPTCHAs characters are same size and positions are quite fit
    # so just find out the position values of each CAPTCHA character and get the images data
    roi_dict[0] = im_res[4:25, 8:28]
    roi_dict[1] = im_res[4:25, 38:58]
    roi_dict[2] = im_res[4:25, 68:88]
    roi_dict[3] = im_res[4:25, 98:118]
    
    # return all CAPTCHAs characters as a dictionary
    return roi_dict

def cut_img(train_dir,cut_dir,suffix):
    # walk through the directory
    for root,dirs,files in os.walk(train_dir):
        for f in files:
            # get the file path
            filepath = os.path.join(root,f)
            # check the file suffix
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in suffix:
                # get the images data of each CAPTCHA character
                roi_dict = fix_img(filepath)
                # cut each CAPTCHA character with the filename incluing the label
                for i in sorted(roi_dict.keys()):
                    cv2.imwrite("{0}/{1}_{2}.jpg".format(cut_dir,f.split('.')[0],f[i]),roi_dict[i])
    
    # close cv2 write operation
    cv2.waitKey(0)
    
    return True

def train_model(cut_dir,suffix):
    # create an empty dataset to store the information of CAPTCHAs characters
    samples = np.empty((0, 420))
    # create an empty lables list
    labels = []
    
    # walk through the directory
    for root,dirs,files in os.walk(cut_dir):
        for f in files:
            filepath = os.path.join(root,f)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in suffix:
                filepath = os.path.join(root,f)
                # read the label of each CAPTCHA character
                label = f.split(".")[0].split("_")[-1]
                labels.append(label)
                # store the CAPTCHA character data into samples dataset
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                sample = im.reshape((1, 420)).astype(np.float32)
                samples = np.append(samples, sample, 0)
                samples = samples.astype(np.float32)

    # labels-label_id mapping
    unique_labels = list(set(labels))
    unique_ids = list(range(len(unique_labels)))
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    label_ids = list(map(lambda x: label_id_map[x], labels))
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
    
    # train the model with KNN
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)
    
    # return the model and labels-label_id mapping
    return {'model':model,'id_label_map':id_label_map}

def rek_img(model_dict,rek_dir,suffix,results_csv):
    # get the model and labels-label_id mapping 
    model = model_dict['model']
    id_label_map = model_dict['id_label_map']
    label_dict = {}
    
    # walk through the directory
    for root,dirs,files in os.walk(rek_dir):
        for f in files:
            filepath = os.path.join(root,f)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in suffix:
                # get the images data of each CAPTCHA character
                roi_dict = fix_img(filepath)
                # get the value of each CAPTCHA character from the model
                for i in sorted(roi_dict.keys()):
                    sample = roi_dict[i].reshape((1, 420)).astype(np.float32)
                    ret, results, neighbours, distances = model.findNearest(sample, k = 3)
                    label_id = int(results[0,0])
                    label = id_label_map[label_id]               
                    label_dict[i] = label
                
                # convert all CAPTCHA characters values into a string
                result_str = ''.join(str(v) for k,v in sorted(label_dict.items()))
                # append the result into a csv
                with open(results_csv, "a") as myfile:
                    myfile.write("{0},{1}\n".format(f,result_str))
                myfile.close()

    return True

if __name__ == '__main__':
    # suffix list of all images
    suffix = ['jpg','png']
    # directory of images for training
    train_dir = './imgs/train'
    # directory of images for the tailored CAPTCHAs characters  
    cut_dir = './imgs/cut'
    # directory of images for testing
    rek_dir = './imgs/test'
    # result csv file path
    results_csv = './results.csv'

    print('INFO: Cutting images...')
    cut_img(train_dir,cut_dir,suffix)
    print('INFO: Training model...')
    model_dict = train_model(cut_dir,suffix)
    print('INFO: Recognizing images in directory {0}...'.format(rek_dir))
    rek_img(model_dict,rek_dir,suffix,results_csv)
    print('INFO: See results in {0}'.format(results_csv))
