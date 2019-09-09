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

'''This is a sliding window that accepts an image and slides a rectangle across'''
def sliding_window(image, stepSize, windowSize): #sliding window function
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



'''Model cluster'''
model_stock = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-stocks1.h5')
model_barrel = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-barrels1-85.h5')
model_mags = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-mags2-85%accuracy.h5')

'''Statisctics counters'''
image_counter = 0
stock_counter = 0
barrel_counter = 0
mag_counter = 0


'''Classification'''
image = cv2.imread("data/test/test9.jpg") #test image
(winW, winH) = (128, 128) #Sliding window dimentions

for (i, resized) in enumerate(pyramid_gaussian(image, downscale=1.5)): #target pyramid
    if resized.shape[0]<30 or resized.shape[1] < 30: #stops if the resized image is too small
        break

    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)): #sliding
        if window.shape[0] != winH or window.shape[1] != winW: #rejecting non consistent windows
            continue

        #debugging part
        cv2.imshow("Test window", window)
        
        #Processing part
        netInput = cv2.resize(window, (200, 200), interpolation = cv2.INTER_AREA) #resizing input for nets
        x = img_to_array(netInput) #Numpy array with shape 3 200 200
        x = np.resize(x, (200, 200, 3))
        x = x.reshape((1,)+x.shape) #adding extra dimention

        #classification part
        result_stock = model_stock.predict_classes(x)
        result_barrel = model_barrel.predict_classes(x)
        result_mags = model_mags.predict_classes(x)

        if result_stock[0] == 1:
            print('Result stock : {}'.format(result_stock))
            stock_counter += 1
        if result_barrel[0] == 1:
            print('Result barrel : {}'.format(result_barrel))
            barrel_counter += 1
        if result_mags[0] == 1:
            print('Result mag : {}'.format(result_mags))
            mag_counter += 1

        image_counter += 1
        #cv2.waitKey(0)
        


'''
#OLD TYPE
for filename in glob.glob('data/stocks/testing/*'):
    img = load_img(filename) #loading image
    x = img_to_array(img) #Numpy array with shape 3 200 200
    x = np.resize(x, (200,200,3))
    x = x.reshape((1,)+x.shape) #adding extra dimention
    result = new_model.predict_classes(x)
    print('{}{}'.format(result, filename))
'''

print("--------STATISTICS---------")
if stock_counter != 0:
    print("STOCK ----- {}".format(stock_counter/image_counter))
else:
    print("STOCK ----- NONE")
if barrel_counter != 0:
    print("BARREL ----- {}".format(barrel_counter/image_counter))
else:
    print("BARREL ----- NONE")
if mag_counter != 0:
    print("MAGS ----- {}".format(mag_counter/image_counter))
else:
    print("MAGS ----- NONE")
    
print('DONE')
