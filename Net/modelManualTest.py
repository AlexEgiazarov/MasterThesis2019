from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from skimage.transform import pyramid_gaussian
import glob

import cv2
import time

from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array #Imports data generator and image functions

from PIL import Image
import numpy as np
import os


def sendToModel(image, model):
    #cv2.imshow("Roi",image)
    #cv2.waitKey(0)
    netInput = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA) #resizing input for nets
    x = img_to_array(netInput) #Numpy array with shape 3 200 200
    x = np.resize(x, (200, 200, 3))
    x = x.reshape((1,)+x.shape) #adding extra dimention

    #classification part
    #result = model.predict_classes(x)
    result = model.predict_proba(x)
    return result

def batchTest(model, path):
    resultTrue = 0
    resultFalse = 0
    resultUnknown = 0
    path = path + "/*"
    for filename in glob.glob(path):
        image = cv2.imread(filename)
        x = img_to_array(image) #Numpy array with shape 3 200 200
        x = np.resize(x, (200, 200, 3))
        x = x.reshape((1,)+x.shape) #adding extra dimention

        #classification part
        #result = model.predict_classes(x)
        result = model.predict(x)
        print(result)
        
        if result[0] == 1:
            resultTrue += 1
        elif result[0] == 0:
            resultFalse += 1
        else:
            resultUnknown += 1
        
        '''
        if result[0] > 0:
            resultTrue += 1
        elif result[0] == 0:
            resultFalse += 1
        else:
            resultUnknown += 1

        '''
    print("---RESULTS---")
    print("---for ", path)
    print("Positive : ", resultTrue)
    print("Negative : ", resultFalse)
    print("Undecided : ", resultUnknown)
    
def manualTest(filename, model):
    image = cv2.imread(filename)
    image = cv2.resize(image,None,fx=0.7,fy=0.7)

    flag = True
    while flag:
        r = cv2.selectROI("Kek", image, False, True) #selecting ROI   
        imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #cropping initial image
        size = min(imCrop.shape[:2])-1 #getting shape
        imCrop = imCrop[0:size, 0:size] #resising to rectangular
        resized = cv2.resize(imCrop, (200,200), interpolation = cv2.INTER_AREA)

        result = sendToModel(resized, model)
        print(result)

        

if __name__ == "__main__":
    model = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-receivers-big-set-small-model.h5')
    pathNegative = 'data/receivers/testing/0notrec'
    pathPositive = 'data/receivers/testing/1rec'
    #name = 'test/test3/pic1.jpg'
    #name = 'test/testWork/output_000005.jpg'
    #manualTest(name, model)
    batchTest(model, pathPositive)
    #batchTest(model, pathNegative)