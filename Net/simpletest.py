from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from skimage.transform import pyramid_gaussian

import sys

import glob
import random
from imutils import contours
import imutils
from skimage import measure
import cv2
import time

from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array #Imports data generator and image functions

from PIL import Image
import numpy as np
import os

def netReshape(window): #Reshaping window to feed the network
    netInput = cv2.resize(window, (200, 200), interpolation = cv2.INTER_AREA) #resizing
    netInput = img_to_array(netInput) #numpy array of shape 3 200 200
    netInput = np.resize(netInput, (200, 200, 3)) #resize
    netInput = netInput.reshape((1,)+netInput.shape) #adding extra dimention
    return netInput

def main():
    print("Start test")
    model = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/model-mags(5epochs-binary).h5')

    #labels =["Barrel", "Mag", "Receiver", "Stock", "Random"]
    labels = ["Barrel", "Random"]
    path = "/home/theinfernal/MasterProject/Net/NetModel/test/testMulticlass/"

    for i in glob.glob(path+"*"):
        image = cv2.imread(i)
        netInput = netReshape(image)
        predictionResult = model.predict_proba(netInput)
        #print("RESULT")
        print(predictionResult)
        index = np.argmax(predictionResult)
        cv2.imshow(labels[index], image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()