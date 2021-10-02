import cv2
import imutils as imutils
from matplotlib import pyplot as plt

img = cv2.imread("./edge-test-imgs/test1.tif")
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("greyscale", img_grey)
cv2.waitKey(0)

edges = cv2.Canny(image=img_grey, threshold1=100, threshold2=200)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

#https://stackoverflow.com/questions/34389384/improve-contour-detection-with-opencv-python

# this only shows the outside contour
blur = cv2.GaussianBlur(img_grey,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True)
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
numcards=len(listindex)

imgcont = img.copy()
[cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 5) for i in listindex]
cv2.imshow("Countours", imgcont)
cv2.waitKey(0)





