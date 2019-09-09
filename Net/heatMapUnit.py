from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from skimage.transform import pyramid_gaussian

import glob
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

def calcStep(image, window): #finding the appropriate step size to iterate over whole image
    dims = image.shape[:2] #getting shape of the image
    dim = dims[0] #getting the top
    win = window[0] #window size
    winTemp = win #temporary window
    step = 0.1 #starting step size of 20%
    flag = True
    while flag:
        print("Current step is {}".format(step))
        print("Current window is {}".format(winTemp))
        winTemp = winTemp + (win*step) #calculating overlap: window + percentage of overlap
        #print(winTemp)
        #print(dim)
        if int(winTemp) == int(dim): #if final window slide equals image dimention
            print("Found step")
            flag = False
        elif step>0.5:
            winTemp = win
            step = 0.1
        elif int(winTemp)>int(dim): #if window slide became bigger than image dimention
            winTemp = win #reset the slide
            step += 0.01 #decrease overlap by 1%
    #print("Dim is {}".format(dims[0]))
    #print("Window is {}".format(window[0]))
    #print("Step is {}".format(step))
    return step

def newRect(x,y,w,h):
    #find center
    newW = min(w, h)
    center = (int((x+w)/2), int((y+h)/2))
    newx = center[0]-int(newW/2)
    newy = center[1]-int(newW/2)
    return (newx, newy, newW, newH)

def sliding_window(image, stepSize, windowSize): #sliding window function
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def netReshape(window): #Reshaping window to feed the network
    netInput = cv2.resize(window, (200, 200), interpolation = cv2.INTER_AREA) #resizing
    netInput = img_to_array(netInput) #numpy array of shape 3 200 200
    netInput = np.resize(netInput, (200, 200, 3)) #resize
    netInput = netInput.reshape((1,)+netInput.shape) #adding extra dimention
    return netInput

def getContours(heatmap):
    heatmap = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0,1)) #normalizing heatmap
    threshHeatMap = cv2.threshold(heatmap, 0.5, 1.0, cv2.THRESH_BINARY)[1] #thresholding
    threshHeatMap = cv2.erode(threshHeatMap, (5,5), iterations=4)

    #detecting multiple bright objects
    labels = measure.label(threshHeatMap, neighbors=8, background=0)
    mask = np.zeros(threshHeatMap.shape, dtype="uint8") #creating mask

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(threshHeatMap.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts #returning contours


def heatLayer(model, image, scale):
    print("Starting heatLayer function")
    #heatmap = cv2.resize(np.zeros((5,5)), (image.shape[:2]), interpolation = cv2.INTER_AREA) #creating new heatmap of input image size
    heatmap = np.zeros(image.shape, dtype="uint8")
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    windowSize = int(min(image.shape[:2]))
    sldWnd = (int(windowSize/2), int(windowSize/2))
    #step = calcStep(image, sldWnd)
    step = 0.1

    #for (x, y, window) in sliding_window(image, stepSize=int(sldWnd[0]/4), windowSize=sldWnd):
    for (x, y, window) in sliding_window(image, stepSize = int(sldWnd[0]*step), windowSize=sldWnd):
        print("Running window")
        if window.shape[0] != sldWnd[1] or window.shape[1] != sldWnd[0]: #rejecting inconsistent sliding windows
            continue

        netInput = netReshape(window) #reshaping input for network
        predictionResult = model.predict(netInput)

        if predictionResult[0] == 1: #if prediction positive
            heatmap[y:y+sldWnd[1], x:x+sldWnd[0]] = 255 #set heatmap to 255

    print("Done running")
    cnts = getContours(heatmap) #getting contours

    result = image

    for (i,c) in enumerate(cnts):
        (x,y,w,h) = cv2.boundingRect(c) #calculating bounding rectangle
        croppedImage = image[y:y+h, x:x+w] #cropping image
        cv2.imshow("Image", croppedImage) #showing
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(0)
        if croppedImage.shape == image.shape: #if no cropping happened
            croppedImage = cv2.resize(croppedImage,None,fx=0.8,fy=0.8)

        if scale < 0.4: #if the current sliding window is below 100 pixels should be fixed with scaler
            result = croppedImage #result is current cropped image
            cv2.imshow("Result", result) #show result
            cv2.waitKey(0)
            continue
        else:
            #newWindow = (int(sldWnd[0]*0.5), int(sldWnd[1]*0.5)) #if current sliding window is not under 100 pixels
            scale = scale*0.8
            if scale > 0.4: #and if new window will be over 80
                heatLayer(model, croppedImage, scale) #call recursivly
            else:
                continue
    print("Done")


def heatLayerSimple(model, image, decay, origin):
    decay -= 1 #subtract decay
    print("Starting simple layer function")
    heatmap = np.zeros(image.shape, dtype="uint8") #create heatmap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY) #convert heatmap to grayscale
    windowSize = int(min(image.shape[:2])) #calculating windowsize
    sldWnd = (int(windowSize/4), int(windowSize/4)) #creating sliding window
    step = 0.05 #step variable
    boxList = []

    print("Running window sliding")
    for (x,y,window) in sliding_window(image, stepSize=int(sldWnd[0]*step), windowSize=sldWnd): #sliding window
        if window.shape[0] != sldWnd[1] or window.shape[1] != sldWnd[0]: #rejecting inconsistant windows
            continue
        
        netInput = netReshape(window) #reshaping input for net
        #predictionResult = model.predict(netInput) #predicting result
        predictionResult = model.predict_proba(netInput)
        #print(predictionResult)
        heatmap[y:y+sldWnd[1], x:x+sldWnd[0]] += predictionResult[0].astype("uint8") #imprint predictions on the heatmap

    print("Done running")
    cnts = getContours(heatmap) #getting contours

    for (i, c) in enumerate(cnts): #for each contour
        (x,y,w,h) = cv2.boundingRect(c) #bounding rectangle
        croppedImage = image[y:y+h, x:x+w] #cropping image
        cv2.imshow("Image", croppedImage)
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(0)
        
        bbox = [] #bounding box list
        if croppedImage.shape == image.shape or decay == 0: #if image is not cropped or recursion decayed
            #cv2.imshow("Result", croppedImage)
            bbox.append((x+origin[0],y+origin[1],w,h)) #create bounding box list
            boxList += bbox #add bounding box to the main bbox list
            #print(bbox)
        else:
            origin = (origin[0]+x, origin[1]+y) #update origin
            boxList = heatLayerSimple(model, croppedImage, decay, origin) #call function

    return boxList, heatmap #return main bounding box list and heatmap

def heatLayerLite(model, image): #lite version of the heatLayer. Returns heatmap for one layer given image
    heatmap = np.zeros(image.shape, dtype="uint8")
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    windowSize = int(min(image.shape[:2]))
    sldWnd = (int(windowSize/2),int(windowSize/2))
    step = 0.1
    
    print("Running window sliding")
    for (x,y,window) in sliding_window(image, stepSize=int(sldWnd[0]*step), windowSize=sldWnd):
        if window.shape[0] != sldWnd[1] or window.shape[1] != sldWnd[0]:
            continue

        netInput = netReshape(window)
        predictionResult = model.predict_proba(netInput)
        #heatmap[y:y+sldWnd[1], x:x+sldWnd[0]] += predictionResult[0].astype("uint8")
        if predictionResult[0] == 1:
            heatmap[y:y+sldWnd[1], x:x+sldWnd[0]] += 1
    print("Done sliding")
    return heatmap

def layerStack(models, image, decay, origin): #layer stack function, gets models, image and sliding parameters
    print("Starting layer stack")
    decay -= 1 #decay update
    boxList = [] #empty list of boxes
    heatLayer = np.zeros(image.shape, dtype="uint8") #empty heatmap
    heatLayer = cv2.cvtColor(heatLayer, cv2.COLOR_BGR2GRAY) #convert heatmap

    #for i in models:
    for key, value in models.items(): #for each model in list
        print(key) #print model name 
        #heatLayer += heatLayerLite(i, image)
        #heatLayer = heatLayerLite(i, image)

        #heatLayer = heatLayerLite(value, image) #call heatLayerLite (used for each separate network#)
        heatLayer += heatLayerLite(value, image)

        print(np.max(heatLayer)) #print max value on received heatmap
        #cv2.imshow(key, heatLayer) #show heatmap layer
        #cv2.waitKey(0)
        
        

        #USE WHEN NEED FOR EACH SEPARATE NETWORK
        
        if np.max(heatLayer)>0: #if heatmap is not empty
            cnts = getContours(heatLayer)
            for (i, c) in enumerate(cnts): #for each contour
                (x,y,w,h) = cv2.boundingRect(c) #bounding rectangle
                croppedImage = image[y:y+h, x:x+w] #cropping image
                bbox = []
                if decay == 0:
                    bbox.append((x+origin[0],y+origin[1],w,h)) #create bounding box list
                    boxList += bbox #add bounding box to the main bbox list
                else:
                    origin = (origin[0]+x, origin[1]+y) #update origin
                    boxList = layerStack(models, croppedImage, decay, origin)
        

    '''
    if np.max(heatLayer)>0: #if heatmap is not empty
        cnts = getContours(heatLayer)
        for (i, c) in enumerate(cnts): #for each contour
            (x,y,w,h) = cv2.boundingRect(c) #bounding rectangle
            croppedImage = image[y:y+h, x:x+w] #cropping image
            bbox = []
            if decay == 0:
                bbox.append((x+origin[0],y+origin[1],w,h)) #create bounding box list
                boxList += bbox #add bounding box to the main bbox list
            else:
                origin = (origin[0]+x, origin[1]+y) #update origin
                boxList = layerStack(models, croppedImage, decay, origin)    

    '''
    return boxList

def baseMap(image):
    heatmap = np.zeros(image.shape, dtype="uint8")
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    windowSize = int(min(image.shape[:2]))
    sldWnd = (int(windowSize/2),int(windowSize/2))
    step = 0.01
    
    print("Running window sliding")
    for (x,y,window) in sliding_window(image, stepSize=int(sldWnd[0]*step), windowSize=sldWnd):
        if window.shape[0] != sldWnd[1] or window.shape[1] != sldWnd[0]:
            continue
        heatmap[y:y+sldWnd[1], x:x+sldWnd[0]] += 1
    return heatmap

#IMPORTANT - FIND CORRECT WAY TO CALCULATE RATIO FOR NEW WINDOW AND STEP POSSIBLY WITH SCALE

#THIS RUNS THE COLLECTIVE TEST ON EACH INDIVIDUAL NETWORK
def main():
    print("Start")
    print("Loading models")
    modelStock = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-stocks-big-set-small-model.h5')
    modelMag = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-mags-big-set-small-model.h5')
    modelReceiver = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-receivers-big-set-small-model.h5')
    modelBarrel = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-barrels-big-set-small-model.h5')

    '''
    models = []
    models.append(modelStock)
    models.append(modelMag)
    models.append(modelReceiver)
    models.append(modelBarrel)
    '''

    models = {}
    models['Stock'] = modelStock
    models['Mag'] = modelMag
    models['Receiver'] = modelReceiver
    models['Barrel'] = modelBarrel

    origin = (0, 0)
    decay = 1

    
    #testImage = cv2.imread('test/test3/pic16.jpg')
    #testImage = cv2.imread('test/testNoise/picStocksReceivers.jpg')
    #testImage = cv2.resize(testImage,None,fx=0.8,fy=0.8)

    #cv2.imshow("Image original", testImage)
    #cv2.waitKey(0)
    

    for i in glob.glob("test/testNoise/*.jpg"):

        testImage = cv2.imread(i)

        bboxStack = layerStack(models, testImage, decay, origin)

        for n in range(len(bboxStack)):
            i = bboxStack[n]
            cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0,0,255), 1)
    
        cv2.imshow("Image Detected", testImage)
        cv2.waitKey(0)

    #testImage = cv2.imread('test/test4/Man1.png')
   
    '''
    for j in glob.glob("CroppedPictures/*.jpg"):
        print(j)
        testImage = cv2.imread(j)
        bboxStack = layerStack(models, testImage, decay, origin)

        for i in bboxStack:
            cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 1)
        
        print(j)
        newname = j.replace('CroppedPictures/', 'detected')
        newname = 'DetectedPictures/'+newname
        print(newname)
        cv2.imwrite(newname, testImage)
        print("Done")
    '''
    #heatLayer(modelStock, testImage, (int(windowSize/2), int(windowSize/2)), 1)
    #heatLayer(modelStock, testImage, 1)
    #boxesStocks, heatMapStocks = heatLayerSimple(modelStock, testImage, decay, origin)
    #boxesMags, heatMapMags = heatLayerSimple(modelMag, testImage, decay, origin)
    #boxesBarrels = heatLayerSimple(modelBarrel, testImage, decay, origin)
    #print("Boxes Mags")
    #print(boxesMags)
    #print("Boxes Stocks")
    #print(boxesStocks)
    #print("Boxes Barrels")
    #print(boxesBarrels)
    
    #for i in boxesMags:
        #cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)
    #for i in boxesStocks:
        #cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (255, 0, 0), 2)
    #for i in boxesBarrels:
        #cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 0, 255), 2)

    #for i in bboxStack:
        #cv2.rectangle(testImage, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)

    #cv2.imshow("Stocks HeatMap", heatmap)
    #cv2.imshow("Result", testImage)
    #cv2.waitKey(0)

if __name__ == "__main__":
    main()