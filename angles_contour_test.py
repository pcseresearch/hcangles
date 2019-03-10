import numpy as np
import cv2
import imutils

im = cv2.imread('SHMS_angle_7562.jpg')
cv2.imshow("Orig",im)
cv2.waitKey(0)

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",imgray)
cv2.waitKey(0)

ret,thresh = cv2.threshold(imgray,127,255,0)
cv2.imshow("Thresh",thresh)
cv2.waitKey(0)

if imutils.is_cv4():
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
else:
	contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if imutils.is_cv2():
		contours = contours[0]
	else:
		contours = contours[1]

cv2.drawContours(im,contours,-1, (0,255,0), 3)
cv2.imshow("Contours",im)
cv2.waitKey(0)

    
