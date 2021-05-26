import numpy as np
import cv2
import copy
import math

#Function dividing the image into 8x8 and assigning each grid of the image as white or black
def TagGrid(img):
    tag_img = img.shape        
    height_img = tag_img[0]
    width_img = tag_img[1]
    grid_height = int((height_img/8))  #Grid height
    grid_width = int(width_img/8)      #Grid width
    countblack = 0
    countwhite = 0
    a=0
    ar_mat = np.empty((8,8))           #Initialising the 8X8 matrix
    
    #Checking the count of black and white pixels in a grid
    for i in range(0,height_img,grid_height):
        b=0
        for j in range(0,width_img,grid_width):
            countblack=0
            countwhite=0
            for x in range(0,grid_height-1):
                for y in range(0,grid_width-1):
                    if(img[i+x][j+y]==0):
                        countblack = countblack + 1
                    else:
                        countwhite = countwhite + 1
            
            if(countwhite >= countblack):    # Setting the matrix value as 1 or 0 based on the count of the corresponding pixels
                ar_mat[a][b]=1
            else:
                ar_mat[a][b]=0
            b=b+1
        a=a+1
    return ar_mat

#Obtaining the orientation of the inner 4x4 grid based on where the white pixel is present
def TagAngle(artag):
    print('AR_TAG',artag)
    if(artag[2][2] == 0 and artag[2][5] == 0 and artag[5][2] == 0 and artag[5][5] == 1):
        orientation = 0
    elif(artag[2][2] == 1 and artag[2][5] == 0 and artag[5][2] == 0 and artag[5][5] == 0):
        orientation = 180
    elif(artag[2][2] == 0 and artag[2][5] == 0 and artag[5][2] == 1 and artag[5][5] == 0):
        orientation = -90
    elif(artag[2][2] == 0 and artag[2][5] == 1 and artag[5][2] == 0 and artag[5][5] == 0):
        orientation = 90
    else:
        orientation = None
       
    if (orientation == None):
        return orientation, False
    else:
        return orientation, True
        
#Obtaining the binary and decimal value of the tag in the iinermost 2x2 matrix
def TagId(image):

    tag_matrix = TagGrid(image)

    angle_value , flag = TagAngle(tag_matrix)
     
    if (flag == False):         #Checking the tag is detected or not.
        return flag , angle_value , None
        
    if(flag == True):      
        #Assigning tag value from the top left Most significant bit to the bottom left most significant bit based on tag orientation
        #Returning the angle value and decimal value of the tag and a flag if an angle was obtained or not
        if (angle_value == 0):
            Id = tag_matrix[3][3]*1 + tag_matrix[3][4]*2 +tag_matrix[4][4]*4 +tag_matrix[4][3]*8  
        elif(angle_value == -90):
            Id = tag_matrix[3][3]*8 + tag_matrix[3][4]*1 + tag_matrix[4][4]*2 +tag_matrix[4][3]*4
        elif(angle_value == 180):
            Id = tag_matrix[3][3]*4 + tag_matrix[4][3]*2 + tag_matrix[4][4] + tag_matrix[3][4]*8
        elif(angle_value == 90):
            Id= tag_matrix[3][3]*2 + tag_matrix[3][4]*4 + tag_matrix[4][4]*8 +tag_matrix[4][3]*1
        return flag, angle_value, Id

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

image1=cv2.imread('testudo.png')    #Reading the testudo.png image to superimpose on the tags
cap= cv2.VideoCapture('Tag2.mp4') 
i=0
#count=0
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
  
size = (frame_width, frame_height)

out = cv2.VideoWriter('supTag2det.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

while(cap.isOpened()):# and count!=1
    ret, frame = cap.read()     #Reading the frames in the video

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Converting to grayscale
    blur = cv2.GaussianBlur(gray, (7,7), 0)         #Using a Gaussian Blur
    _,thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)    #THresholding the image to obtain a bianry image
    
    #Obtaining the different contours and the hierarchy in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    hierarchy = hierarchy[0]
    corner_points = []

    
    for j,cnt in zip(hierarchy,contours):
        currentContour = cnt
        cnt_len = cv2.arcLength(cnt, True)

        #Using the contour approximation to reduce the number of vertices of the contour and obtain 4 vertices in the contour to obtain corner points 
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):

            cnt = cnt.reshape(-1, 2)

            
            #FInding the contours of the inner AR tag. Since this will be the only contour with a parent the contour would be drawn on this AR tag alone
            if j[3] != -1:
                cv2.drawContours(frame, [currentContour], -1, (0, 255, 0), 3)
                corner_points.append(cnt)
                
    for corn in corner_points:
        #Calling the function to find the homography from the corner points to the new image created (200x200) consisting of the inner AR tag
        H = find_homography(corn,[[0,0],[0,200],[200,200],[200,0]])
        #im_out = cv2.warpPerspective(thresh, H, (200,200))
        
        #Obtaining the warp perspective of the homography matrix
        H_inv=np.linalg.inv(H)
        im_out=np.zeros((200,200))        
        for a in range(0,200):
            for b in range(0,200):
                x, y, z = np.matmul(H_inv,[a,b,1])
                rows,cols=thresh.shape
                if (int(y/z) < rows and int(y/z) > 0) and (int(x/z) < cols and int(x/z) > 0):
                    im_out[a][b] = thresh[int(y/z)][int(x/z)]   
        
        #Finding the TAG ID using the new image consisting of the AR tag
        l,m,n = TagId(im_out)
        cv2.imshow('imout',im_out)
        
        #If an orientation angle is obtained setting the corresponding corner points to superimpose the image
        if l:    
            if m == 0:
                print('angle0')
                corner_actual = corn
            elif m == 90:
                print('90')
                corner_actual = [corn[3], corn[0], corn[1], corn[2]]
            elif m == -90:
                print('-90')
                corner_actual = [corn[1], corn[2], corn[3], corn[0]]
            elif m == 180:
                print('180')
                corner_actual = [corn[2], corn[3], corn[0], corn[1]]
        
        image1 = cv2.resize(image1, (200,200))
        
        #Computing the homography matrix of the corner points obtained after finding orientation angle 
        H_new= find_homography ( corner_actual,[[0,0],[0,200],[200,200],[200,0]])
        
        #Perfomring warp perspective
        H_inv = np.linalg.inv(H_new)  
                
        for j in range(0,200):
            for k in range(0,200):
                x_test, y_test, z_test = np.matmul(H_inv,[j,k,1])
                rows,cols,chn=frame.shape
                if (int(y_test/z_test) < rows and int(y_test/z_test) > 0) and (int(x_test/z_test) < cols and int(x_test/z_test) > 0):
                    #Superimposing the corresponding pixels from testudo.png in the frames of the video
                    frame[int(y_test/z_test)][int(x_test/z_test)] = image1[j][k]
                    
        cv2.imshow('Display', frame)   #Displaying the image output in a window named "Display"
        out.write(frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
