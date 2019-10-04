'''
This script usesthe json files provided with the PubLayNet dataset to create 
the segmentation masks (png, 1 channel, gray scale) for the training and val sets.

Download the dataset from https://github.com/ibm-aur-nlp/PubLayNet
Unzip the files and organise them in the following directory structure:

  + datasets (current working directory)
    - create_PubLayNet_segmentation_mask_png_files.py
    - build_data.py
    - build_PubLayNet_tfrecords.py
    + PubLayNet
        - dev.json
        - train.json
        + RawImages
            + val_data (contains unzipped dev images)
            + train_data
                + train_0 (contains unzipped train_0 images)
                ...
                + train_6 (contains unzipped train_6 images)
        + SegmentationClass
            + train (this script will save training segmentation masks here)
            + val (this script will save validation segmentation masks here)
        + tfrecords (empty)

September 25th, 2019
Navid Lambert-Shirzad
'''

from PIL import Image
import json
from PIL import ImageFont, ImageDraw
from glob import glob
import numpy as np
import tensorflow as tf

###############################################################
#Use dev.json file to create val set's segmentation masks#
###############################################################

with open('PubLayNet/dev.json', 'r') as fp:
    devsamples = json.load(fp)
# organise the dictionary into something more usable
images = {}
for image in devsamples['images']:
    images[image['id']] = {'file_name': image['file_name'], 
                           'width': image['width'], 
                           'height': image['height'], 
                           'annotations': []}
for ann in devsamples['annotations']:
    images[ann['image_id']]['annotations'].append(ann)

num_val_images = len(images.keys())
for counter, img_id in enumerate(list(images)):
    if counter%100 == 0:
        print('Creating segmentation masks for the validation set: image {} of {}'.format(counter, num_val_images))
    #print(images[img_id])
    
    seg_mask = np.zeros((images[img_id]['width'],images[img_id]['height']), dtype=int)
    for ann in images[img_id]['annotations']:
        current_bbox = np.asarray(ann['bbox'], dtype = np.int32)
        x1, x2 = current_bbox[0], current_bbox[0] + current_bbox[2]
        y1, y2 = current_bbox[1], current_bbox[1] + current_bbox[3]
        #the object's pixels are updated to its class_id
        seg_mask[x1:x2, y1:y2] = ann['category_id']
        #the object's border pixels are updated to 255 (unknown) to create contrast and aid with learning
        seg_mask[x1, y1:y2] = 255
        seg_mask[x2, y1:y2] = 255
        seg_mask[x1:x2, y1] = 255
        seg_mask[x1:x2, y2] = 255


    seg_mask = seg_mask.T    
    
    seg_img = Image.fromarray(seg_mask.astype(dtype=np.uint8))
    segfilename = 'PubLayNet/SegmentationClass/val/'+images[img_id]['file_name'][:-4]+'.png'
    with tf.gfile.Open(segfilename, mode='w') as f:
        seg_img.save(f, 'PNG')

###############################################################
#Use train.json file to create train set's segmentation masks#
###############################################################

with open('PubLayNet/train.json', 'r') as fp:
    trainsamples = json.load(fp)
# organise the dictionary into something more usable
images = {}
for image in trainsamples['images']:
    images[image['id']] = {'file_name': image['file_name'], 
                           'width': image['width'], 
                           'height': image['height'], 
                           'annotations': []}
for ann in trainsamples['annotations']:
    images[ann['image_id']]['annotations'].append(ann)

num_train_images = len(images.keys())
for counter, img_id in enumerate(list(images)[:110003]):
    if counter%100 == 0:
        print('Creating segmentation masks for the training set: image {} of {}'.format(counter, num_train_images))
    #print(images[img_id])
    
    seg_mask = np.zeros((images[img_id]['width'],images[img_id]['height']), dtype=int)
    for ann in images[img_id]['annotations']:
        current_bbox = np.asarray(ann['bbox'], dtype = np.int32)
        x1, x2 = current_bbox[0], current_bbox[0] + current_bbox[2]
        y1, y2 = current_bbox[1], current_bbox[1] + current_bbox[3]
        #the object's pixels are updated to its class_id
        seg_mask[x1:x2, y1:y2] = ann['category_id']
        #the object's border pixels are updated to 255 (unknown) to create contrast and aid with learning
        seg_mask[x1, y1:y2] = 255
        seg_mask[x2, y1:y2] = 255
        seg_mask[x1:x2, y1] = 255
        seg_mask[x1:x2, y2] = 255
        
        

    seg_mask = seg_mask.T    
    
    seg_img = Image.fromarray(seg_mask.astype(dtype=np.uint8))
    segfilename = 'PubLayNet/SegmentationClass/train/'+images[img_id]['file_name'][:-4]+'.png'
    with tf.gfile.Open(segfilename, mode='w') as f:
        seg_img.save(f, 'PNG')
