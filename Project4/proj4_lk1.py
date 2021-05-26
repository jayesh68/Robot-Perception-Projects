#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:25:04 2021

@author: jayesh
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('Cars On Highway.mp4')
size = (1920, 1080)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')                    
#out = cv2.VideoWriter('lhighwaycar.avi',fourcc, 15, size)  

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*'XVID')                    
out = cv2.VideoWriter('lhighwaycar.avi',fourcc, 15, size)    

while(cap.isOpened() ):
    ret, frame = cap.read()
    
    fgmask = fgbg.apply(frame)
    
    backtorgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB) 
    cv2.imshow('frame',fgmask)
    cv2.imshow('frame1',backtorgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    #out.write(backtorgb)
cap.release()
cv2.destroyAllWindows()