import cv2
import numpy as np
import sys
import os

def GntData():
    MIN_Cnt_area = 20
    RESIZED_IMG_WIDTH = 20
    RESIZED_IMG_HEIGHT = 30
    imgTrainingNums = cv2.imread("nums.jpg")
    
    if imgTrainingNums is None:
        print("error:image not read \n")
        os.system("pause")
        return
    
    imgGray = cv2.cvtColor(imgTrainingNums,cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray,(5,5),0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,2)
    
    cv2.imshow("imgThresh",imgThresh)
    imgThreshCopy = imgThresh.copy()
    
    imgContours,npaContours,npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    npaFlattenedImages =  np.empty((0, RESIZED_IMG_WIDTH * RESIZED_IMG_HEIGHT))
    intClassifications = []
    
    intValidChars = [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9')]
    
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_Cnt_area:
            [X,Y,W,H] = cv2.boundingRect(npaContour)
            cv2.rectangle(imgTrainingNums,(X,Y),(X+W,Y+H),(0,0,255),2)
            imgROI = imgThresh[Y:Y+H,X:X+W]
            imgROIResized = cv2.resize(imgROI,(RESIZED_IMG_WIDTH,RESIZED_IMG_HEIGHT))
            cv2.imshow("imgROI",imgROI)
            cv2.imshow("imgROIResized",imgROIResized)
            cv2.imshow("",imgTrainingNums)
            intChar = cv2.waitKey(0)
            
            if intChar == 27: #if esc button is pressed
                sys.exit()
            elif intChar in intValidChars:
                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMG_WIDTH * RESIZED_IMG_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
    
    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
    print ("\n\ntraining complete !!\n")
    
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    cv2.destroyAllWindows()
    return
