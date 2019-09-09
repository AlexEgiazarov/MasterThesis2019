import cv2
import numpy as np
import glob
import tqdm

def sizeCheck(image): #sizeChecker to ensure that sliding windows works ok
    standartWidth = 1600
    height, width, channels = image.shape
    while width > standartWidth: #placeholder - find a way to calculate the scale factor
        image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)
        height, width, channels = image.shape
    return image #returns a scaled down version

def main(): #Main function
    name = 0
    counter = 0
    saveCounter = 0
    start = 0 #starting position
    sourceFolder = 'CustomPics/'
    for filename in glob.glob('CustomPics/*'): #for loop        

        print(counter,filename)
        #print(filename[-3:])

        if counter<start:
            counter+=1
            continue
            

        if '-jpg' in filename:
            print("Changed")
            filename = filename.replace('-jpg', '.jpg')
        if '-png' in filename:
            print("Changed")
            filename = filename.replace('-png', '.png')
        if '-jpeg' in filename:
            print("Changed")
            filename = filename.replace('-jpeg', '.jpeg')
    
        if filename[-4:] == '.jpg':
            print("ok 1")
        elif filename[-4:] == '.png':
            print("ok 2")
        elif filename[-4:] == 'jpeg':
            print("ok 3")
        else:
            print("not ok")
            continue

        im = cv2.imread(filename) #reading in the image
        im = sizeCheck(im)
        cv2.imshow("Image", im)
        
        key = cv2.waitKey(0)
        if key != 115: #key s
            cv2.destroyAllWindows()
            flag = True
            picCounter = 0
            while(flag):
                r = cv2.selectROI(im, False, False) #selecting ROI
                imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #cropping initial image
                size = min(imCrop.shape[:2])-1 #getting shape
                imCrop = imCrop[0:size, 0:size] #resising to rectangular

                resized = imCrop
                #resized = cv2.resize(imCrop, (200,200), interpolation = cv2.INTER_AREA)
            
                newname = filename.replace('CustomPics/', 'custom_res_')
                cv2.imwrite('CroppedImages/{}{}'.format(picCounter, newname), resized)
                saveCounter += 1
                print("Pictures saved: {}".format(saveCounter))
                picCounter += 1
                name += 1
                nextkey = cv2.waitKey(0)
                if nextkey == 110: #n?
                    flag = False
            cv2.destroyAllWindows()
            #key = cv2.waitKey(0)

        
        counter += 1

if __name__ == '__main__':
    main()
