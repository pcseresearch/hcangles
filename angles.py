from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

run = input("Run Number:")
file = open("run_" + run + ".dat")
lang = []

with open("run_" + run + ".dat") as f:
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


image = cv2.imread("SHMS_angle_0" + run + ".jpg")
image = imutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)


gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")


gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)


thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
#cv2.imshow("Image Thresh Again",thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for (i, c) in enumerate(cnts):
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	if ar > 5 and y > 100 and y < 125 and x > 75 and x < 210:
		print("Vernier Scale ROI Values:")
		print(x,y,w,h,ar)
		print("Veriner Scale Center Value:")
		middle = (w/2) + x
		print(middle)
	if x > 20 and x < 60 and ar < 1:
		xvalue = x + (56*2)
		print("Center Marker Value:")
		print(xvalue)

print("Hundredth Angle Value:")
result = (xvalue - middle)/(50)
sang = (result)/(10)
final = sang + lang[0]
print(sang)
print("Final Angle Value:")
print(final)

cv2.waitKey(0)
exit()
