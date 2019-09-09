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

heatmap = np.zeros((5,5))

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

'''Non maximum suppression to sum overlapping bounding boxes'''
def nms(boxes, overlapThresh): #non maximum suppression
    #if there are no boxes, return empty
    if len(boxes) == 0:
        return[]

    #if bounding boxes are integers convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    #initialize the list of picked indexes
    pick = []

    #grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #compute the area of the bounding boxes and sort them
    #by the bottom-right y coordinate of the bounding box
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(y2)

    #keep looping while some indexes still remain in hte index list
    while len(idxs) > 0:
        #grab the last index in the indexes list and add
        #index value to the list of picked indexes
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)

        #find the larges (x,y) coordinates for the start
        #of the bounding box and the smalles (x,y) coordinates
        #for the end of the bounding box

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        #compute the width and height of the bounding box
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)

        #compute the ration of the overlap
        overlap = (w*h)/area[idxs[:last]]
        
        #delete all the indexes from the index list that left
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>overlapThresh)[0])))
    
    #return only the bounding boxes that were picked using integer data type
    return boxes[pick].astype("int")

def heatmapDetectionLayered(model, image, counter, origin, winW, winH):
    heatmap = np.zeros((5,5)) #creating new heatmap
    heatmap = cv2.resize(heatmap,(image.shape[1], image.shape[0]),interpolation = cv2.INTER_AREA) #resizing new heatmap

    print(origin)
    print(winW)
    if winW < (origin/4): # if window is less than 1/4 of original window 
        cv2.imwrite("result/ResultAt{}.jpg".format(counter), image) #save image
        print("Result Saved")
    else:
        for (x, y, window) in sliding_window(image, stepSize=int(winW/4), windowSize=(winW, winH)): #slide over image
            if window.shape[0] != winH or window.shape[1] != winW: #rejecting nonconsistant windows"
                continue #continue recursion from previous step
            
            netInput = netReshape(window) #reshaping window to feed the network
            predictionResult = model.predict(netInput) #predicting
            #print(predictionResult) #for debugging

            if predictionResult[0] == 1: #if predictions are not negative, aka something is there, but not 100% sure
                #image = cv2.rectangle(image, (x,y), (x+winW, y+winH), (0,255,0), 2)
                heatmap[y:y+winH, x:x+winW] = 1


        heatmap = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1)) #normalizing the values of HeatMap
        #heatMap[heatMap < 1] = 0
        cv2.imshow("Heat Map normalized", heatmap)
        cv2.waitKey(0)
        threshHeatMap = cv2.threshold(heatmap, 0.5, 1.0, cv2.THRESH_BINARY)[1] #setting threshold

        #Detecting multiple bright objects
        labels = measure.label(threshHeatMap, neighbors=8, background=0)
        mask = np.zeros(threshHeatMap.shape, dtype="uint8")
        
        for label in np.unique(labels): #looping over unique components
            if label == 0: #if background, continue
                continue

            #otherwise
            labelMask = np.zeros(threshHeatMap.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
    
            if numPixels > 300: #if area big enough, add it to mask
                mask = cv2.add(mask, labelMask)
        
        #Here we try to find contours that encapsulates the areas of interest
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for (i,c) in enumerate(cnts): #for each region
            (x,y,w,h) = cv2.boundingRect(c) #bounding rectangle
            croppedImage = image[y:y+h, x:x+w] #crop image
            if croppedImage.shape == image.shape: #if no cropping took place
                '''
                print("Same!")
                counter += 1
                heatmapDetectionLayered(model, croppedImage, counter, origin, int(winW/2.5), int(winH/2.5)) #half the window
                '''
                cv2.imwrite("result/ResultAt{}.jpg".format(counter), image) #save image
                print("Result Saved")
            else:
                cv2.imshow("Image", croppedImage)
                cv2.waitKey(0)
                counter += 1
                sh = min(croppedImage.shape[:2])
                (winW, winH) = (int(sh/2), int(sh/2)) 
                heatmapDetectionLayered(model, croppedImage, counter, origin, winW, winH)
            

def boundingBoxDetection(model, image):
    sh = min(image.shape[:2]) #find smallest edge to use as a window
    (winW, winH) = (int(sh/2), int(sh/2))
    boxes = []
    for (x, y, window) in sliding_window(image, stepSize=int(winW/4), windowSize=(winW, winH)): #slide over image
        if window.shape[0] != winH or window.shape[1] != winW: #rejecting nonconsistant windows"
            continue #continue recursion from previous step
        
        netInput = netReshape(window) #reshaping window to feed the network
        predictionResult = model.predict(netInput) #predicting
        #print(predictionResult) #for debugging

        if predictionResult[0] == 1: #if predictions are not negative, aka something is there, but not 100% sure
            #image = cv2.rectangle(image, (x,y), (x+winW, y+winH), (0,255,0), 2)
            boxes.append((x, y, x+winW, y+winH))

    print(boxes)
    boxes = nms(np.array(boxes), 0.4)
    print(boxes)

    for i in boxes:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        image = cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    
    return image

def heatMapRoiExtraction(heatMap, image):
    
    heatMap = np.interp(heatMap, (heatMap.min(), heatMap.max()), (0, 1)) #normalizing the values of HeatMap
    #heatMap[heatMap < 1] = 0
    cv2.imshow("Heat Map normalized", heatMap)
    blurredHeatMap = cv2.GaussianBlur(heatMap, (11, 11), 0) #blurring the heatmap
    threshHeatMap = cv2.threshold(blurredHeatMap, 0.5, 1.0, cv2.THRESH_BINARY)[1] #setting threshold

    threshHeatMap = cv2.erode(threshHeatMap, None, iterations=2)
    threshHeatMap = cv2.dilate(threshHeatMap, None, iterations=4)
    
    cv2.imshow("Heatmap thresh", threshHeatMap)
    cv2.waitKey(0)
    #Detecting multiple bright objects
    labels = measure.label(threshHeatMap, neighbors=8, background=0)
    mask = np.zeros(threshHeatMap.shape, dtype="uint8")
    
    for label in np.unique(labels): #looping over unique components
        if label == 0: #if background, continue
            continue

        #otherwise
        labelMask = np.zeros(threshHeatMap.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
 
        if numPixels > 300: #if area big enough, add it to mask
            mask = cv2.add(mask, labelMask)

    #Here we try to find contours that encapsulates the areas of interest
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    dataReturn = []
    #loop over contours
    for (i,c) in enumerate(cnts):
        (x,y,w,h) = cv2.boundingRect(c)
        if w > h: #choosing the biggest side
            h = w
        elif h > w:
            w = h
            
        dataReturn.append(image[y:y+h, x:x+w])
        
    return dataReturn

'''This is a sliding window that accepts an image and slides a rectangle across'''
def sliding_window(image, stepSize, windowSize): #sliding window function
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

'''Updates heatmap when detected an object'''
def updHeatMap(heatMap, x, y, windowSize):
    heatMap[y:y+windowSize[1], x:x+windowSize[0]] += 0.1
    return heatMap

'''Reshapes input for appropriate size for feeding to network'''
def netReshape(window): #Reshaping window to feed the network
    netInput = cv2.resize(window, (200, 200), interpolation = cv2.INTER_AREA) #resizing
    netInput = img_to_array(netInput) #numpy array of shape 3 200 200
    netInput = np.resize(netInput, (200, 200, 3)) #resize
    netInput = netInput.reshape((1,)+netInput.shape) #adding extra dimention
    return netInput

'''Big test'''
def bigTest(model, image):

    (winW, winH) = (150, 150) #sliding window dimentions

    '''
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=1.5)):
        if resized.shape[0] <30 or resized.shape[1]<30: #stops resizing when image too small
            break
    '''
    for (x, y, window) in sliding_window(image, stepSize=24, windowSize=(winW, winH)): #sliding
        if window.shape[0] != winH or window.shape[1] != winW: #rejecting nonconsistant windows
            continue

        netInput = netReshape(window) #reshaping window to feed the network

        predictionResult = model.predict(netInput) #predicting

        print(predictionResult)

        if predictionResult[0] == 1:
            print("Prediction result is :")
            print(predictionResult[0])
            cv2.imshow("Positive window", window)
            cv2.waitKey(0)


'''Simple function to write out on which depth level we are
'''
def writeDepth(number, window):
    depth = '-'
    for i in range(0, number):
        depth = depth+'---'

    sys.stdout.write("\033[K")
    #print("|| Window size is " , window )
    print("Currently on ", number, " || Depth is: [", depth, end='\r'),


'''RECURSIVE FUNCTION
## Searches for the object on the image

### inuputs:
model - pretrained model
image - image to analyze
winW - window parameter for width
winH - window parameter for height'
number - depth number, how deep function is into recursion
original - original window parameter, used to calculate where to stop based on final size of the window 
cX - x coordinate of the top left corner of the current sliding window used to update heatmap
CY - y coordinate of the top left corner of the current sliding window

### What it does
Function takes in image, slides window with given initial parameters and checks if model reacts on the current window
If model thinks that something is present in the window, function calls intself with that smaller window as input.
That creates a "focus" effect where function recursivly "zooms" in into the area of interest.
This approach helps algorithm to stop paying attention to detailes and look at the image as a whole
'''

def recursive(model, image, winW, winH, number, original, cX, cY):
    
    writeDepth(number, (winW, winH)) #depth write out

    if winW < int(original/4): #if current sliding window is smaller than 5 of the original one
        netInput = netReshape(image) #send current image to reshape function
        predictiopnResult = model.predict(netInput) #predict if image contains something we are looking for
        if predictiopnResult[0] == 1: #if it does contain
            global heatmap #take global heatmap
            heatmap = updHeatMap(heatmap, cX, cY, (int(winW*1.2), int(winH*1.2))) #update heatmap with positive result
    else: #if window is still big enough
        for (x, y, window) in sliding_window(image, stepSize=int(winW/4), windowSize=(winW, winH)): #slide over image
            if window.shape[0] != winH or window.shape[1] != winW: #rejecting nonconsistant windows"
                continue #continue recursion from previous step
     
            netInput = netReshape(window) #reshaping window to feed the network
            predictionResult = model.predict(netInput) #predicting
            #print(predictionResult) #for debugging

            if predictionResult[0] > 0: #if predictions are not negative, aka something is there, but not 100% sure
                recursive(model, window, int(winW*0.8), int(winH*0.8), number+1, original, cX+x, cY+y) #enable recursion on that area
            
        if number == 0: #if we are on the root recursion and none of the areas gave us positive result, make window smaller and try again
            recursive(model, image, int(winW*0.8), int(winH*0.8), number, original, cX, cY)

def newTest(model, image):
    sh = min(image.shape[:2]) #find smallest edge to use as a window
    (winW, winH) = (sh, sh) #initial sliding window
    recursive(model, image, winW, winH, 0, sh, 0, 0) #call recursive

#the deeper you are the more it should give
def main():
    #model
    model1 = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-stocks-big-set-small-model.h5')
    model2 = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-mags-big-set-small-model.h5')
    model3 = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-receivers-big-set-small-model.h5')
    model4 = load_model('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-barrels-big-set-small-model.h5')
    image = cv2.imread('test/test3/pic14.jpg')
    global heatmap
    heatmap = cv2.resize(heatmap,(image.shape[1], image.shape[0]),interpolation = cv2.INTER_AREA)
    winSize = min(image.shape[:2])
    heatmapDetectionLayered(model4, image, 0, winSize, int(winSize/4), int(winSize/4))
    #stocks = boundingBoxDetection(model1, image)
    #cv2.imshow("Final result", layerHM)
    #cv2.waitKey(0)

    #stocks = boundingBoxDetection(model1, image)
    #mags = boundingBoxDetection(model2, image)
    #recievers = boundingBoxDetection(model3, image)
    #barrels = boundingBoxDetection(model4, image)

    #cv2.imshow("Stocks", stocks)
    
    #cv2.imshow("Mags", mags)

    #cv2.imshow("Recievers", recievers)

    #cv2.imshow("Barrels", barrels)


    


    '''
    #heatmap = cv2.resize(heatmap,(2000, 1000), interpolation= cv2.INTER_AREA)
    #newTest(model1, image)
    #newTest(model2, image)
    #newTest(model3, image)
    #newTest(model4, image)
    #heat = heatMapRoiExtraction(heatmap, image)

    
    #cv2.imshow("Heatmap", heatmap)
    
    for i in heat:
        cv2.imshow("Heatmap focus", i)
        cv2.waitKey(0)
    '''


if __name__ == "__main__":
    main()