#!/apps/python/bin/python3
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt

run = input("Run Number:")

file = open("SHMS_encoder_" + run + ".dat")
lang = []

with open("SHMS_encoder_" + run + ".dat") as f:
	for line in f:
		data = line.split()
		lang.append(float(data[0]))

file.close()

'''ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
args = vars(ap.parse_args())'''

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
sqreKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
squrKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))


image = cv2.imread("SHMS_angle_" + run + ".jpg")
#image = imutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
cv2.waitKey(0)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)


gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")


gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imshow("GradX",gradX)
cv2.waitKey(0)


thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh",thresh)
cv2.waitKey(0)


thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("Thresh Again",thresh)
cv2.waitKey(0)

if imutils.is_cv4():
	cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if imutils.is_cv2():
		cnts = cnts[0]
	else:
		cnts = cnts[1]

cv2.drawContours(image,cnts,-1, (0,255,0), 3)
cv2.imshow("With Contours",image)
cv2.waitKey(0)

middlesum = 0.0
middleval = [0.0 for i in range(0,21)]
count = [0 for i in range(0,21)]
index = -10

for (i, c) in enumerate(cnts):
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	#if (ar>0.0001):
	#	print (i,x, y, w, h, ar)
	if w > 5 and h > 25 and y > 260 and y < 270 and x > 220 and x < 560:
		print("Vernier Scale ROI Values:")
		print(x,y,w,h,ar)
		print("Veriner Scale Center Value:")
		count[index+10] = index
		middleval[index+10] = (w/2) + x
		index = index + 1
		middlesum = middlesum + (w/2) + x
	if x > 370 and x < 400 and y > 120 and y < 260 and w > 3 and h > 10:
		print("Center Marker Value:")
		print(x,y,w,h,ar)
		xvalue = (w/2) + x
		print(xvalue)

middleval.sort()
middle = middlesum/21.0

print (middleval)
print (count)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.set_title("Vernier Scale Positions")
ax1.set_xlabel('Index of Vernier Marker')
ax1.set_ylabel('X-position (pixels)')
ax1.set_yscale("linear",nonposy='clip')
ax1.grid(True)
ax1.plot(count,middleval)

plt.show()

print("Hundredth Angle Value:")
result = (xvalue - middle)/(50)
sang = (result)/(10)
central_angle = float(int(10*lang[0])/10.0)
final = central_angle+sang
print(sang)
print("Final Angle Value:")
print(final)

cv2.waitKey(0)
exit()