#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:34:15 2021

@author: jayesh
"""

import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt

#Reading the video file
cap= cv2.VideoCapture('night_drive.mp4') 

#Creating a video to write the ouput histogram equalized frame
#size = (540, 480)         
#fourcc = cv2.VideoWriter_fourcc(*'XVID')           
#out = cv2.VideoWriter('histeq3.avi',fourcc, 15, size,0)


while(cap.isOpened):
    
    ret, frame = cap.read()     #Reading the frames in the video
    
    if ret == False:
        break
    
    #Scaling the frames to a smaller size
    dx,dy,chn =frame.shape
    frame = cv2.resize(frame,(int(dx/2),int(dy/4)))
    #print(frame.shape[0],frame.shape[1])
    
    cv2.imshow('Original Frame',frame)                  #Displaying the original frame
    
    #Converting to grayscale and blurring the image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
   
    #Creating an array of zeros to generate a histogram of the different intensities
    n=np.zeros(shape=(256,1))

    N=frame.shape
    
    #Generating the histogram of the intensities
    for i in range(N[0]):
        for j in range(N[1]):
            intensity=frame[i,j]
            n[intensity]=n[intensity]+1


    n=n.reshape(1,256)
    
    #Array to store the corresponsing CDF values
    cdf=np.array([])
    cdf=np.append(cdf,n[0,0])

    for i in range(255):
        pdf=n[0,i+1]+cdf[i]
        cdf=np.append(cdf,pdf)
    
    #Multiplying the obtained CDF with 255 to obtain the new intensity value
    cdf=np.round((cdf/(N[0]*N[1]))*255)

    #Updating the original image with the new intensities
    for i in range(N[0]):
        for j in range(N[1]):
            intensity=frame[i,j]
            #print(k)
            frame[i,j]=cdf[intensity]
    
    cv2.imshow('Histogram Equalization',frame)
    
    #out.write(frame)

    
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()
