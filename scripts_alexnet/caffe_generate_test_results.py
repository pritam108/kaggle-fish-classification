
# coding: utf-8

# In[29]:

import numpy as np
import matplotlib.pyplot as plt
import caffe
import lmdb
import os
import cv2
import numpy as np
import imutils
from random import randint


# In[30]:

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
trainingFolders = ["../train/ALB/","../train/BET/","../train/DOL/","../train/LAG/","../train/NoF/","../train/OTHER/","../train/SHARK/","../train/YFT/"]
testFolder = "../test_stg2/"
train_lmdb = "fishing_train_lmdb"
validation_lmdb = "fishing_val_lmdb"
train_full_lmdb = "fishing_train_full_lmdb"


# In[31]:

def horizontalFlip(img):
    width = len(img[0])
    height = len(img)
    for y in range(height):
        for x in range(width/2):
            tmp = np.copy(img[y,x])
            img[y,x][:] = img[y,width-x-1][:]
            img[y,width-x-1][:] = tmp[:]
    return img


# In[32]:

def verticalFlip(img):
    width = len(img[0])
    height = len(img)
    for x in range(width):
        for y in range(height/2):
            tmp = np.copy(img[y,x])
            img[y,x][:] = img[height-y-1,x][:]
            img[height-y-1,x][:] = tmp[:]
    return img


# In[33]:

def addRandomJitter(img):
    img_0_1 = np.random.rand(len(img),len(img[0]),3)*50
    img_0_1 =  img_0_1.astype('uint8')
    img_new =  img + img_0_1
    return img_new


# In[34]:

def randomImageRotate(img):
    rotationAngle = randint(0,360)
    return imutils.rotate_bound(img,rotationAngle)


# In[35]:

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    img = np.rollaxis(img, 2)
    
    return img


# In[36]:

#Read mean image
mean_blob = caffe.proto.caffe_pb2.BlobProto()
with open('imagenet_mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
caffe.set_mode_gpu()
caffe.set_device(0)


# In[58]:

#Read model architecture and trained model's weights
net = caffe.Net('models/bvlc_alexnet_after_augmentation/deploy.prototxt',
                'models/bvlc_alexnet_after_augmentation/caffe_alexnet_train_iter_10000.caffemodel',
                caffe.TEST)


# In[52]:

#Define image transformers
transformer = caffe.io.Transformer({'data': (1,3, 256, 256)})
transformer.set_mean('data', mean_array)
#transformer.set_input_scale('data',227)
#transformer.set_transpose('data', (2,0,1))


# In[ ]:




# In[53]:

mean_array.shape


# In[54]:

'''
for imageFile in os.listdir("../test_stg1/"):
    img = cv2.imread("../test_stg1/" + imageFile,cv2.IMREAD_COLOR)
    img = transform_img(img,img_width=IMAGE_WIDTH,img_height=IMAGE_HEIGHT)
    img = img.astype('float32')
    img -= mean_array
    img = np.rollaxis(img,2)
    img = np.rollaxis(img,2)
    print(img.shape)
    img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_CUBIC)
    print(img.shape)
    img = np.rollaxis(img,2)
    print(img.shape)
    img = img[np.newaxis,:,:,:]
    print(img.shape)
    net.blobs['data'].data[...] = img
    out = net.forward()
    pred_probas = out['prob']
    #xx = transformer.preprocess('data', img)
    break
'''


# In[59]:

results = []
count = 0
for imageFile in os.listdir("../test_stg1/"):
    img = cv2.imread("../test_stg1/" + imageFile,cv2.IMREAD_COLOR)
    img = transform_img(img,img_width=IMAGE_WIDTH,img_height=IMAGE_HEIGHT)
    img = img.astype('float32')
    img -= mean_array
    img = np.rollaxis(img,2)
    img = np.rollaxis(img,2)
    img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_CUBIC)
    img = np.rollaxis(img,2)
    img = img[np.newaxis,:,:,:]
    net.blobs['data'].data[...] = img
    out = net.forward()
    pred_probas = out['prob']
    outputRecord = []
    outputRecord += [imageFile]
    outputRecord += ["{:f}".format(prob) for prob in pred_probas[0]]
    results += [outputRecord]
    count += 1
    if count % 100 == 0:
        print(count)


# In[60]:

count = 0
for imageFile in os.listdir("../test_stg2/"):
    img = cv2.imread("../test_stg2/" + imageFile,cv2.IMREAD_COLOR)
    img = transform_img(img,img_width=IMAGE_WIDTH,img_height=IMAGE_HEIGHT)
    img = img.astype('float32')
    img -= mean_array
    img = np.rollaxis(img,2)
    img = np.rollaxis(img,2)
    img = cv2.resize(img, (227, 227), interpolation = cv2.INTER_CUBIC)
    img = np.rollaxis(img,2)
    img = img[np.newaxis,:,:,:]
    net.blobs['data'].data[...] = img
    out = net.forward()
    pred_probas = out['prob']
    outputRecord = []
    outputRecord += ["test_stg2/"+imageFile]
    outputRecord += ["{:f}".format(prob) for prob in pred_probas[0]]
    results += [outputRecord]
    count += 1
    if count % 100 == 0:
        print(count)


# In[61]:

import csv
with open("EightOutput.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)


# In[22]:

#anotherResult = [["test_stg2/" + result[0]] + result[1:] for result in results]


# In[24]:

'''with open("FourthOutput.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerows(anotherResult)'''


# In[ ]:



