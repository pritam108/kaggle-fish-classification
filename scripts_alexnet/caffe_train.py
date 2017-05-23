
# coding: utf-8

# In[14]:

import numpy as np
import matplotlib.pyplot as plt
import caffe
import lmdb
import os
import cv2
import numpy as np
import imutils
from random import randint


# In[6]:

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
trainingFolders = ["../train/ALB/","../train/BET/","../train/DOL/","../train/LAG/","../train/NoF/","../train/OTHER/","../train/SHARK/","../train/YFT/"]
testFolder = "../test_stg1/"
train_lmdb = "fishing_train_lmdb_after_augmentation"
validation_lmdb = "fishing_val_lmdb_after_augmentation"
train_full_lmdb = "fishing_train_full_lmdb_after_augmentation"


# In[7]:

def horizontalFlip(img):
    width = len(img[0])
    height = len(img)
    for y in range(height):
        for x in range(width/2):
            tmp = np.copy(img[y,x])
            img[y,x][:] = img[y,width-x-1][:]
            img[y,width-x-1][:] = tmp[:]
    return img


# In[8]:

def verticalFlip(img):
    width = len(img[0])
    height = len(img)
    for x in range(width):
        for y in range(height/2):
            tmp = np.copy(img[y,x])
            img[y,x][:] = img[height-y-1,x][:]
            img[height-y-1,x][:] = tmp[:]
    return img


# In[9]:

def addRandomJitter(img):
    img_0_1 = np.random.rand(len(img),len(img[0]),3)*50
    img_0_1 =  img_0_1.astype('uint8')
    img_new =  img + img_0_1
    return img_new


# In[10]:

def randomImageRotate(img):
    rotationAngle = randint(0,360)
    return imutils.rotate_bound(img,rotationAngle)


# In[11]:

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


# In[12]:

def make_datum(img, label):

    return caffe.proto.caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,

        data=np.rollaxis(img, 2).tostring())


# In[16]:

# Train Data Pre-Processing
env = lmdb.open(train_lmdb, map_size=int(1e12))
with env.begin(write=True,) as txn:
    labelNo = 0
    for trainingFolder in trainingFolders:
        files = os.listdir(trainingFolder)
        label = trainingFolder.split("/")[2]
        images = []
        count = 0
        for imageFile in files:
            if(count % 6 == 0):
                count += 1
                continue
            img = cv2.imread(trainingFolder + imageFile)
            
            img_mod1 = horizontalFlip(np.copy(img))
            img_mod2 = verticalFlip(np.copy(img))
            img_mod3 = randomImageRotate(np.copy(img))
            img_mod4 = addRandomJitter(np.copy(img))
            
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
            
            if(labelNo == 0 and count % 10!= 0):
                continue
            
            img = transform_img(img_mod1, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
            
            img = transform_img(img_mod2, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
            
            img = transform_img(img_mod3, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
            
            img = transform_img(img_mod4, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
            
            
        print(labelNo)
        labelNo += 1
env.close()


# In[17]:

# Validation Data Pre-Processing
env = lmdb.open(validation_lmdb, map_size=int(1e12))
with env.begin(write=True,) as txn:
    labelNo = 0
    for trainingFolder in trainingFolders:

        files = os.listdir(trainingFolder)
        label = trainingFolder.split("/")[2]
        images = []
        count = 0

        for imageFile in files:
            if(count % 6 != 0):
                count += 1
                continue
            img = cv2.imread(trainingFolder + imageFile)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            datum = make_datum(img, labelNo)
            str_id = label + "_" + str(count)
            txn.put(str_id,datum.SerializeToString())
            count += 1
        print(labelNo)
        labelNo += 1
env.close()


# In[ ]:



