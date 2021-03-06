#!/apps/python/bin/python3
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import subprocess
from pylab import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def fitfunction(x,a,b):
    xp = np.array(x)
    return a+b*xp

#run = input("Run Number:")

runlist = []
angle_final = []
angle_encoder = []
angle_diff = []

with open("SHMS_runlist.dat") as runfile:
    for line in runfile:
        data = line.split()
        runlist.append(int(data[0]))
runfile.close()

for runs in runlist:
    print ("Analyzing Run = ",runs)
    run = str(runs)
    file = open("SHMS_encoder_" + run + ".dat")
    lang = []

    with open("SHMS_encoder_" + run + ".dat") as f:
        for line in f:
            data = line.split()
            lang.append(float(data[0]))
            angle_encoder.append(float(data[0]))

    file.close()

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    sqreKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    squrKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))

    image = cv2.imread("SHMS_angle_" + run + ".jpg")
    #image = imutils.resize(image,width=300)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png",gray)
    #cv2.imshow("Gray",gray)
    #cv2.waitKey(0)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, sqreKernel)
    cv2.imwrite("tophat.png",tophat)
    #cv2.imshow("Tophat",tophat)
    #cv2.waitKey(0)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqreKernel)
    #cv2.imshow("GradX",gradX)
    #cv2.waitKey(0)
    cv2.imwrite("gradX.png",gradX)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Thresh",thresh)
    #cv2.waitKey(0)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    #cv2.imshow("Thresh Again",thresh)
    #cv2.waitKey(0)
    cv2.imwrite("thresh.png",thresh)

    if imutils.is_cv4():
        cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if imutils.is_cv2():
            cnts = cnts[0]
        else:
            cnts = cnts[1]

    cv2.drawContours(image,cnts,-1, (0,255,0), 3)
    #cv2.imshow("With Contours",image)
    #cv2.waitKey(0)
    cv2.drawContours(gray,cnts,-1, (0,255,0), 3)
    cv2.imwrite("contours.png",gray)

    middleval = [0.0 for i in range(0,21)]
    count = [0 for i in range(0,21)]
    index = -10

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        #if (ar>0.0001):
        #	print (i,x, y, w, h, ar)
        if w > 5 and h > 22 and y > 260 and y < 270 and x > 150 and x < 660:
            #print("Vernier Scale ROI Values:")
            #print(x,y,w,h,ar)
            #print("Vernier Scale Center Value:")
            if w > 10 and w < 30:
                #print("Vernier Scale Double:")
                count[index+10] = index
                count[index+11] = index+1
                middleval[index+10] = (w/4) + x
                middleval[index+11] = (3*w/4) + x
                index = index + 2
            elif w >= 30:
                print("Vernier Scale Treble:")
                count[index+10] = index
                count[index+11] = index+1
                count[index+12] = index+2
                middleval[index+10] = (w/4) + x
                middleval[index+11] = (w/2) + x
                middleval[index+12] = (3*w/4) + x
                index = index + 3
            else:
                #print("Vernier Scale Single:")
                count[index+10] = index
                middleval[index+10] = (w/2) + x
                index = index + 1

    middleval.sort()

    #print (middleval)
    #print (count)

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)

    #ax1.set_title("Vernier Scale Positions")
    #ax1.set_xlabel('Index of Vernier Marker')
    #ax1.set_ylabel('X-position (pixels)')
    #ax1.set_yscale("linear",nonposy='clip')
    #ax1.grid(True)
    #ax1.scatter(count,middleval)

    init_vals = [400.0,50.0]
    popt, pcov = curve_fit(fitfunction, count, middleval, p0=init_vals)

    #print (popt[0],popt[1])
    #print (pcov)

    #ax1.plot(count,fitfunction(count, *popt), 'r-', label = "Linear Fit")
    #leg = ax1.legend()
    #plt.show()

    middle = popt[0]
    xmin = middle - 5*popt[1]
    xmax = middle + 5*popt[1]    
    ycentral_max = 0

    xvalue = -1.0

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if x > xmin and x < xmax and y > 120 and y < 260 and w > 3 and h > 3 and h > ycentral_max:
            #print("Center Marker Value:")
            #print(x,y,w,h,ar)
            ycentral_max = h
            xvalue = (w/2) + x
            #print(xvalue)

    xvaluel = -1.0
    xvalueh = -1.0

    if xvalue == -1.0:
        xmin = middle - 12*popt[1]
        xmax = middle - 8*popt[1]
        ycentral_max = 0

        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if x > xmin and x < xmax and y > 120 and y < 260 and w > 3 and h > 3 and h > ycentral_max:
                #print("Lower Center Marker Value:")
                #print(x,y,w,h,ar)
                ycentral_max = h
                xvaluel = (w/2) + x
                #print(xvaluel)

        xmin = middle +8*popt[1]
        xmax = middle +12*popt[1]
        ycentral_max = 0
        
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if x > xmin and x < xmax and y > 120 and y < 260 and w > 3 and h > 3 and h > ycentral_max:
                #print("Lower Center Marker Value:")
                #print(x,y,w,h,ar)
                ycentral_max = h
                xvalueh = (w/2) + x
                #print(xvalueh)

    if (xvaluel > 0 and xvalueh > 0):
        xvalue = (xvaluel + xvalueh)/2.0
    
    if (xvalue > 0):    
        #print("Hundredth Angle Value:")
        result = (xvalue - middle)/popt[1]
        sang = (result)/(100)
        central_angle = float(round(10*lang[0])/10.0)
        final = central_angle+sang
        #print(sang)
    else:
        print("Could not determine angle - using encoder value")
        final = lang[0]

    print("Run = ",run," Image Angle = ",final," Encoder Angle = ",lang[0])
    angle_final.append(float(final))
    angle_diff.append(float(final-lang[0]))

#print (angle_final)
#print (angle_encoder)
#print (angle_diff)

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Comparison of Angle Readings")
ax1.set_xlabel('SHMS Encoder Angle (Degrees)')
ax1.set_ylabel('SHMS Final Angle (Degrees)')
ax1.grid(True)
ax1.scatter(angle_encoder,angle_final)

ax2 = fig.add_subplot(122)

ax2.set_title("Final vs. Encoder Difference")
ax2.set_xlabel('Angle Difference (Degrees)')
ax2.grid(True)
ax2.hist(angle_diff,bins='auto')

plt.show()

