#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 00:42:55 2021

@author: jayesh
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy
import matplotlib.patches as patches

#Reading the video file
cap = cv2.VideoCapture('Tag1.mp4')
count=1
while(cap.isOpened() and count!=2):  #Performing the FFT for the first frame alone
    
    count+=1
    ret, img = cap.read()
    
    if ret == False:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert to grayscale
    #cv2.imshow('grayscale',gray_img)

    _,thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)    #Getting the binary image
    #cv2.imshow('thresh',thresh)
    
    #Finding contours on the thresholded image
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    for i in img_contours:
        if cv2.contourArea(i) > 100:
            break
    
    #Creating a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    print('i',[i])
    cv2.drawContours(mask, [i],-1, (255,55,255), -1)
    
    #Doing the bitwise operation to eliminate the background and obtain the AR tag alone.
    new_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Image with background removed", new_img) #Display of the image

    #Performing DFT followed by FFT to obtain the masgnitude spectrum
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    dft=cv2.dft(np.float32(gray_img),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift=np.fft.fftshift(dft)
    magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    #Creating a hpf mask to elimiate the lower frequency components from the image
    rows, cols= gray_img.shape
    rows1, cols1= int(rows/2), int(cols/2)
    mask=np.ones((rows,cols,2),np.uint8)
    r=150
    center=[rows1,cols1]
    x,y=np.ogrid[:rows,:cols]
    mask_area=(x-center[0])**2+(y-center[1])**2<=r*r
    mask[mask_area]=0

    fshift=dft_shift*mask

    try:
        fshift_mask_mag=2000*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        print('fshift mask',fshift_mask_mag,len(fshift_mask_mag))
    except:
        print('an error occured')

    #Performing the inverse fourier transform to obtain just the edges of the tag.
    f_ishift=np.fft.ifftshift(fshift)
    img_back=cv2.idft(f_ishift)
    img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


    #Plots
    fig=plt.figure(figsize=(12,12))
    ax1=fig.add_subplot(2,2,1)
    ax1.imshow(new_img,cmap='gray')
    ax1.title.set_text('Background removed Binary Image')
    plt.xlim(980,1030)
    plt.ylim(486,440)
    ax2=fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum,cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3=fig.add_subplot(2,2,3)
    ax3.imshow(fshift_mask_mag,cmap='gray')
    ax3.title.set_text('FFT + HPF Mask')
    ax4=fig.add_subplot(2,2,4)
    ax4.imshow(img_back,cmap='gray')
    ax4.title.set_text('Inverse FFT')
    #Bounding the AR tag
    rect = patches.Rectangle((980, 442), 50, 40, linewidth=1, edgecolor='r', facecolor='none')
    ax4.add_patch(rect)
    ax4.set_xlim([970,1040])
    ax4.set_ylim([486,440])

    plt.show()

    cv2.waitKey(0)
    
cap.release()
cv2.destroyAllWindows()

