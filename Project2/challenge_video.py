#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:20:30 2021

@author: jayesh
"""

import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt


#Using the histogram and the image after sobel operation to identify the pixels corresponding to the left and right lanes
def sliding_window(hist, img):
    
        # Image to visualize the sliding window operation
        out_img = np.dstack((img, img, img))*255
        
        # To calculate the peak values of the histogram in the x direction to obtain the starting points of the lanes
        mid = np.int(hist.shape[0]/2)

        left_start = np.argmax(hist[:mid])

        right_start = np.argmax(hist[mid:]) + mid

        
        # Creating windows to obtain the white pixel coordinates in different parts of the image
        image_split = 10
        window_height = np.int(img.shape[0]/image_split)

        
        #Extracing the nonzero pixels in the image in both the x and y directions
        lanes = img.nonzero()

        lanesy = np.array(lanes[0])
        lanesx = np.array(lanes[1])

        # Current position of the window
        left_curr = left_start
        right_curr = right_start
        
        # Set the width of the windows
        margin = 60
        
        #Minimum number of pixels found to recenter window
        minpix = 60
        
        # Create empty lists to store left and right lane pixel indices
        left_loc = []
        right_loc = []
        
        #Performing the sliding window operation to obtain the non black pixels and their respective indices in the 2 lanes
        for window in range(image_split):
            
            #Setting the boundaries of the window
            window_y_low = img.shape[0] - (window+1)*window_height
            window_y_high = img.shape[0] - window*window_height
            window_xleft_low = left_curr - margin
            window_xleft_high = left_curr + margin
            window_xright_low = right_curr - margin
            window_xright_high = right_curr + margin
            
            # Drawing the windows on the image
            cv2.rectangle(out_img,(window_xleft_low,window_y_low),(window_xleft_high,window_y_high),(255,255,255), 2) 
            cv2.rectangle(out_img,(window_xright_low,window_y_low),(window_xright_high,window_y_high),(255,255,255), 2) 
            
            cv2.imshow('Out hist',out_img)
            
            # Identifying the nonzero pixels within the window for the lanes
            left_window = ((lanesy >= window_y_low) & (lanesy < window_y_high) & (lanesx >= window_xleft_low) & (lanesx < window_xleft_high)).nonzero()[0]
            right_window = ((lanesy >= window_y_low) & (lanesy < window_y_high) & (lanesx >= window_xright_low) & (lanesx < window_xright_high)).nonzero()[0]
                          
            # Append the coordinates
            left_loc.append(left_window)
            right_loc.append(right_window)
            
            if len(left_window) > minpix:
                left_curr = np.int(np.mean(lanesx[left_window]))
            if len(right_window) > minpix:  
                right_curr = np.int(np.mean(lanesx[right_window]))
                               
        # Concatenate the arrays of indices
        left_loc = np.concatenate(left_loc)
        right_loc = np.concatenate(right_loc)
        
        # Extract left and right line pixel positions
        lx = lanesx[left_loc]
        ly = lanesy[left_loc] 
        rx = lanesx[right_loc]
        ry = lanesy[right_loc] 
                    
        #Setting the identified white and yellow pixels in the histogram as red and blue 
        out_img[lanesy[left_loc], lanesx[left_loc]] = [255, 0, 0]
        out_img[lanesy[right_loc], lanesx[right_loc]] = [0, 0, 255]
        cv2.imshow("ok",out_img)
        
        return lx, ly, rx, ry, left_loc, right_loc, out_img
        
    
def poly_fit(lx, ly, rx, ry, left_loc, right_loc, out_img, hinv,orig_img):  

        # Fit a second order polynomial to obtain the coefficients on the lane candidates
        poly_left = np.polyfit(ly, lx, 2)
        poly_right = np.polyfit(ry, rx, 2)
        
        # Generating x and y values for plotting
        ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )

        #Obtaining the new points corresponding to the polynomial
        new_poly_left = poly_left[0]*ploty**2 + poly_left[1]*ploty + poly_left[2]
        new_poly_right = poly_right[0]*ploty**2 + poly_right[1]*ploty + poly_right[2]
        #print('plyright',polyfit_rightx)

        #Creating a blank image similar to the image obtained after identifying the left and right pixel coordinates 
        blank_mesh = np.zeros_like(out_img).astype(np.uint8)
    
        #Putting the left and right into a format fir for cv2.fillPoly()
        leftpoints = np.array([np.transpose(np.vstack([new_poly_left, ploty]))])
        rightpoints = np.array([np.flipud(np.transpose(np.vstack([new_poly_right, ploty])))])
        pts = np.hstack((leftpoints, rightpoints))

        
        # Draw the lanes onto the blank image
        cv2.fillPoly(blank_mesh, np.int_([pts]), (255,255,0))
        cv2.imshow('Polynomial fit of the lanes',blank_mesh)
        
        # Warping the polynomial fir of the lanes in the original frame's image space
        new_warp = cv2.warpPerspective(blank_mesh, hinv, (orig_img.shape[1], orig_img.shape[0]))
        cv2.imshow('warp col',new_warp)

        # Combining the result with the original image by overlaying the warped image on the lane
        final_image = cv2.addWeighted(orig_img, 1, new_warp, 0.3, 0)
        
        return new_poly_left, new_poly_right,final_image
    
    
def turn_predict(new_poly_left, new_poly_right, img):
        
        #Finding the middle point of the image from the bird's eye view
        center = img.shape[1] / 2
        
        #Middle point of the first points of the lanes
        lane_mid = (new_poly_left[0] + new_poly_right[0]) / 2
        
        #Idenitfying the gradient and setting a threshold to output a direction as the vehicle travels along the lane
        turn = (lane_mid-center)
                       
        if turn<=-25:
            print_turn ="Left Turn"
        elif turn>=3:
            print_turn ="Right Turn"                                        
        else:
            print_turn ="Going Straight"
        return print_turn  
'''
def gamma_correction(img,gamma):
    #Output = Input^(1/gamma)
    #Scale the input from (0 to 256) to (0 to 1)
    #Apply gamma correction
    #Scale back to original values
    gamma = 1/gamma
    lT =[]
    for i in np.arange(0,256).astype(np.uint8):
        lT.append(np.uint8(((i/255)**gamma)*255))
    lookup = np.array(lT)
    #Creating the lookup table, cv can find the gamma corrected value of each pixel value
    corrected = cv2.LUT(img,lookup)
    return corrected
'''

#Function to compute the homography from the corner points
def find_homography(img1, img2):
    ind = 0
    A_matrix = np.empty((8, 9))

    for pixel in range(0, len(img1)):
        
        x_1 = img1[pixel][0]  
        y_1 = img1[pixel][1]

        x_2 = img2[pixel][0]  
        y_2 = img2[pixel][1]

        A_matrix[ind] = np.array([x_1, y_1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2])
        A_matrix[ind + 1] = np.array([0, 0, 0, x_1, y_1, 1, -y_2*x_1, -y_2*y_1, -y_2])

        ind = ind + 2

    U, s, V = np.linalg.svd(A_matrix, full_matrices=True)
    V = (V) / (V[8][8]) 
    H = V[8,:].reshape(3, 3)
    return H



k = np.array([[1.15422732e+03, 0.000000e+00, 6.71627794e+02] ,
                    [0.000000e+00, 1.14818221e+03, 3.86046312e+02], 
                    [0.000000e+00, 0.000000e+00 ,1.000000e+00]],dtype=np.int32)
d = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02],dtype=np.int32)

cap= cv2.VideoCapture('challenge_video.avi')                                                            
                      
size = (720, 640)     

#fourcc = cv2.VideoWriter_fourcc(*'XVID')                    
#out = cv2.VideoWriter('lanedata2final7.avi',fourcc, 15, size)    

while(cap.isOpened() ):    
    ret, frame = cap.read()     #Reading the frames in the video
    
    if ret == False:
        break
                            
    dx,dy,chn =frame.shape
    frame = cv2.resize(frame,(int(dx),int(dy/2)))
    #print('dx',frame.shape[0],frame.shape[1])
    #cv2.imshow('undis',frame)
                   
    img1=cv2.undistort(frame,k,d, None, k)
    cv2.imshow('undistorted image',img1)
    
    #img1 = cv2.fastNlMeansDenoisingColored(img1,None,10,10,7,21)
    #img1 = gamma_correction(img1,1.7)                   

    img1=cv2.GaussianBlur(img1, (7,7), 0)
    ####frame_points=np.array([[616,472],[746,472],[294,690],[1155,690]])
    #####frame_points=np.array([[622,467],[733,467],[294,690],[1155,690]])
    #frame_points=np.array([[333,428],[766,473],[162,600],[433,428]])
    frame_points=np.array([[331,437],[432,437],[169,606],[617,606]])
    #frame_points=np.array([[560,450], [740, 450], [95,710],[1260, 710]])
    H = find_homography(frame_points, [[0,0],[254,0],[0,254],[254,254]])               #Computing Homography
    
    Hinv=np.linalg.inv(H)
    im_out = cv2.warpPerspective(img1, H, (255,255))
    cv2.imshow('Warped',im_out)
    
    '''
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([255, 40, 255], dtype=np.uint8)
    
    hsv = cv2.cvtColor(im_out,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv) 
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    '''
    
    img_hls = cv2.cvtColor(im_out, cv2.COLOR_BGR2HLS)
    
    #Yellow lines
    #lower_yellow = np.array([20, 20, 20], dtype='uint8')
    #lower_yellow = np.array([20, 50, 30], dtype='uint8')
    #upper_yellow = np.array([45, 245, 255], dtype='uint8')
    #upper_yellow = np.array([125, 245, 255], dtype='uint8')
    lower_yellow = np.array([20, 50, 30], dtype='uint8')
    #upper_yellow = np.array([45, 245, 255], dtype='uint8')
    upper_yellow = np.array([125, 245, 255], dtype='uint8')
    mask_yellow = cv2.inRange(img_hls, lower_yellow, upper_yellow)

    #White lines
    lower_white = np.array([0, 163, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(img_hls, lower_white, upper_white)
    
    # Bitwise OR
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    #cv2.imshow('combine',combined_mask)

    hist = np.sum(combined_mask[0:,:], axis=0)
                      
    #print('hist',histogram,len(histogram))
    
    #Extracting the pixels corresponding to the left and right lanes in the image using the histogram and the image after sobel operation
    lx, ly, rx, ry, left_loc, right_loc, out_img=sliding_window(hist, combined_mask)


    #Fitting a polynomial to the lane pixels and overlaying the image on top of the original image
    new_poly_left, new_poly_right,new_frame=poly_fit(lx, ly, rx, ry,left_loc,right_loc,out_img,Hinv,img1)
    cv2.imshow('Overlayed Image',new_frame)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    #Predicting the turn of the vehicle as the lane curves in different directions in the video
    value=turn_predict(new_poly_left, new_poly_right, out_img)
    cv2.putText(new_frame,str(value),(440, 50), font, 0.5,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Lane Detection',new_frame)
    #out.write(new_frame)
    
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()

