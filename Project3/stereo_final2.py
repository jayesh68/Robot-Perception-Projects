#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:06:47 2021

@author: jayesh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import matrix_rank
from scipy.optimize import least_squares

k1= np.array([[4396.869, 0, 1353.072],[0, 4396.869, 989.702], [0, 0, 1]])
k2= np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
fx=4396.869
baseline = 144.049   # distance in mm between the two cameras

def fundamentalMatrix(feat1, feat2):
    A = np.empty((8, 9))
    for i in range(len(feat1)):
        x1 = feat1[i][0] 
        y1 = feat1[i][1]
        x2 = feat2[i][0] 
        y2 = feat2[i][1] 
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)  
    F = v[-1].reshape(3,3) 
    
    u1,s1,v1 = np.linalg.svd(F) 
    F_S = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    F = u1 @ F_S @ v1  
    
    return F 

def ransac(feat1,feat2):
    inliers = 0
    F= np.zeros((3,3))
    
    inlier1 = []
    inlier2 = [] 

    for i in range(0, 10000): # 10000 iterations for RANSAC 
        count = 0
        eightpoint = [] 
        infeat1 = [] 
        infeat2 = []
        tempfeat1 = [] 
        tempfeat2 = []
            
        while(True): # Loop runs while we do not get eight distinct random points
            num = random.randint(0, len(features1)-1)
            if num not in eightpoint:
                eightpoint.append(num)
            if len(eightpoint) == 8:
                break
            
        for point in eightpoint: # Looping over eight random points
            infeat1.append([feat1[point][0], feat1[point][1]]) 
            infeat2.append([feat2[point][0], feat2[point][1]])
        
        # Computing Fundamentals Matrix from current frame to next frame
        F_Matrix = fundamentalMatrix(infeat1, infeat2)
        
        for number in range(0, len(features1)):
                
            # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
            if checkFmatrix(feat1[number], feat2[number], F_Matrix) < 0.01:
                count = count + 1 
                tempfeat1.append(feat1[number])
                tempfeat2.append(feat2[number])
    
            if count > inliers: 
                inliers = count
                F = F_Matrix
                inlier1 = tempfeat1
                inlier2 = tempfeat2  
                
    return F, inlier1, inlier2

def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    E = K2.T.dot(F).dot(K1)
    u1,s1,v1 = np.linalg.svd(E) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    E = u1 @ s2 @ v1
    return E

def cameraposestimation(E):
    U_est, S_est, V_est = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    R1 = np.matmul(np.matmul(U_est, W), V_est)
    R2 = np.matmul(np.matmul(U_est, W), V_est)
    R3 = np.matmul(np.matmul(U_est, W.T), V_est)
    R4 = np.matmul(np.matmul(U_est, W.T), V_est)

    C1 = U_est[:, 2]
    C2 = -U_est[:, 2]
    C3 = U_est[:, 2]
    C4 = -U_est[:, 2]

    if (np.linalg.det(R1) < 0):
        R1 = -R1
        C1 = -C1
    if (np.linalg.det(R2) < 0):
        R2 = -R2
        C2 = -C2
    if (np.linalg.det(R3) < 0):
        R3 = -R3
        C3 = -C3
    if (np.linalg.det(R4) < 0):
        R4 = -R4
        C4 = -C4
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    C3 = C3.reshape((3,1))
    C4 = C4.reshape((3,1))

    return [R1, R2, R3, R4], [C1, C2, C3, C4]

def linear_trainagulation(pt1,pt2,R2,C2,k1,k2):
    
    C1 = [[0],[0],[0]]
    R1 = np.identity(3)
    P1 = k1 @ np.hstack((R1, -R1 @ C1))
    P2 = k2 @ np.hstack((R2, -R2 @ C2))	
    X = []
    for i in range(len(pt1)):
        x1 = pt1[i]
        x2 = pt2[i]
        A1 = x1[0]*P1[2,:]-P1[0,:]
        A2 = x1[1]*P1[2,:]-P1[1,:]
        A3 = x2[0]*P2[2,:]-P2[0,:]
        A4 = x2[1]*P2[2,:]-P2[1,:]		
        A = [A1, A2, A3, A4]

        U,S,V = np.linalg.svd(A)
        V = V[3]

        V = V/V[-1]
        X.append(V)
    return X

#Non linear triangulation not carried out
'''
def nlt_error(point_3D, P1, P2, points_left, points_right):
    #print('3d',point_3D)
    X, Y, Z = point_3D
    point_3D = np.array([X, Y, Z, 1])

    p1 = P1 @ point_3D
    p2 = P2 @ point_3D

    projected_x1, projected_y1 = p1[0]/p1[2], p1[1]/p1[2]
    projected_x2, projected_y2 = p2[0]/p2[2], p2[1]/p2[2]

    dist1 = (points_left[0] - projected_x1)**2 + (points_left[1] - projected_y1)**2
    dist2 = (points_right[0] - projected_x2)**2 + (points_right[1] - projected_y2)**2

    error = dist1 + dist2
    return error

def nonlinear_triangulation(points_3D, pose, point_list1, point_list2):
    P = np.eye(3,4)
    P_dash = pose
    tot_points = len(points_3D)
    approx_points = []
    for p1, p2, point_3D in zip(point_list1, point_list2, points_3D):      
        x, y, z, w = point_3D
        est_point = [x, y, z]
        point_3D_approx = least_squares(nlt_error, est_point, args = (P, P_dash, point_list1, point_list2))
        approx_points.append(point_3D_approx)
    print('3d approx',point_3D_approx)
    return approx_points
'''

#Function to extract the unique camera pose using linear triangulation
def extract_Rot_and_Trans(pt1,pt2, R,C,k1,k2):
    X1 = linear_trainagulation(pt1,pt2,R,C,k1,k2)
    X1 = np.array(X1)	
    count = 0
    for i in range(X1.shape[0]):
        x = X1[i,:].reshape(-1,1)
        if R[2]@np.subtract(x[0:3],C) > 0: 
            count += 1
    return count

#Function to rectify the two images
def rectify_pair(feat1, feat2, f,imsize):
    feat1=np.array(feat1)
    feat2=np.array(feat2)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(feat1),np.float32(feat2),f, imsize)

    return H1, H2

#Function to check if the F matrix satisfies the epipolar constraint
def checkFmatrix(x1,x2,F): 
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

#Function to draw epipolar line on one image using the features of the other
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    r, c, chn = img1src.shape
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        a=int(pt1[0])
        b=int(pt1[1])
        c=int(pt2[0])
        d=int(pt2[1])
        e=[a,b]
        f=[c,d]
        img1src = cv2.line(img1src, (x0, y0), (x1, y1), color, 1)
        img1src = cv2.circle(img1src, tuple(e), 5, color, -1)
        img2src= cv2.circle(img2src, tuple(f), 5, color, -1)
    return img1src, img2src

                         
img1=cv2.imread('/home/jayesh/ENPM673_CODE/Dataset 2-20210409T165827Z-001/Dataset 2/im0.png')
img2=cv2.imread('/home/jayesh/ENPM673_CODE/Dataset 2-20210409T165827Z-001/Dataset 2/im1.png')

dx,dy,chn =img1.shape
img1 = cv2.resize(img1,(int(dx/3),int(dy/5)))
dx,dy,chn =img2.shape
img2 = cv2.resize(img2,(int(dx/3),int(dy/5)))
imsize=(img2.shape[0],img2.shape[1])


#Using ORB instead of SIFT to obtain the matching features in the 2 images
orb=cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

imgkey1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
imgkey2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
cv2.imshow('imgkey1',imgkey1)
cv2.imshow('imgkey2',imgkey2)

# create BFMatcher object
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

features1 = []
features2 = []
good=[]
for m,n in matches:
    #Obtaining matches by comparing distance between the two images
    if m.distance < 0.5*n.distance:
        good.append(m)
        features1.append(kp1[m.queryIdx].pt)
        features2.append(kp2[m.trainIdx].pt)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matching features',img3)

#Calculation of fundamental matrix and best feature matches using RANSAC    
F,inlier1,inlier2=ransac(features1,features2)
print('Fundamental Matrix',F)
print(np.linalg.matrix_rank(F))

inlier1=np.array(inlier1)
inlier2=np.array(inlier2)

# Find epilines corresponding to points in second image and drawing its lines on first image
lines1 = cv2.computeCorrespondEpilines(inlier2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img3=imgkey1.copy()
img4=imgkey2.copy()
imgepi1,imgepi12 = drawlines(img3,img4,lines1,inlier1,inlier2)

# Find epilines corresponding to points in first image and drawing its lines on second image
lines2 = cv2.computeCorrespondEpilines(inlier1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
imgepi2,imgepi21 = drawlines(img4,img3,lines2,inlier2,inlier1)
        
cv2.imshow('img1 lines',imgepi1)
cv2.imshow('img2 lines',imgepi2)

e = essentialMatrix(k1, k2, F)

print('Essential Matrix',e)

#CAlling function to obtain the possible camera configurations
r,c = cameraposestimation(e)
#print('row col',r,c)
count_depth = 0
for p in range(4):
    Z = extract_Rot_and_Trans(inlier1,inlier2,r[p], c[p],k1,k2)
    print('z',Z)
    if count_depth < Z : 
        count_depth, reg = Z, p  #FInding the camera pose which has the maximum number of positive depth points
        
R = r[reg]
t = c[reg]

print('Rotation Matrix Final',R)
print('Translational Matrix Final',t)

#Function call to rectify images and obtain homography matrices and doing a warp perspective
h1,h2=rectify_pair(inlier1,inlier2,F,imsize)
print('Homography',h1,h2)
img1_rectified = cv2.warpPerspective(imgepi1, h1, imsize)
img2_rectified = cv2.warpPerspective(imgepi2, h2, imsize)

cv2.imshow('img1rec',img1_rectified)
cv2.imshow('img2rec',img2_rectified)

#Computing the Disparity Map by comparing matches along epipolar lines of the two images using SSD
gray1 = cv2.cvtColor(img1_rectified,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_rectified,cv2.COLOR_BGR2GRAY)

disparities = 16 # num of disparities to consider
block = 7       # block size to match

height, width = gray1.shape
disparity_img = np.zeros(shape = (height,width))

for i in range(block, gray1.shape[0] - block - 1):
    for j in range(block + disparities, gray1.shape[1] - block - 1):
        ssd = np.empty([disparities, 1])
        l = gray1[(i - block):(i + block), (j - block):(j + block)]
        height, width = l.shape
        for d in range(0, disparities):
            r = gray2[(i - block):(i + block), (j - d - block):(j - d + block)]
            ssd[d] = np.sum((l[:,:]-r[:,:])**2)
        disparity_img[i, j] = np.argmin(ssd)


final_img = ((disparity_img/disparity_img.max())*255).astype(np.uint8)
cv2.imshow('Disparity grayscale',final_img)
colormap = plt.get_cmap('inferno')
heatmap = (colormap(final_img) * 2**16).astype(np.uint16)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
cv2.imshow('Disparity heatmap', heatmap)

#Computing depth based on disparity
depth = np.zeros(shape=gray1.shape).astype(float)
depth[final_img > 0] = (fx * baseline) / (final_img[final_img > 0])
final_img1 = ((depth/depth.max())*255).astype(np.uint8)
colormap = plt.get_cmap('inferno')
heatmap1 = (colormap(final_img1) * 2**16).astype(np.uint16)[:,:,:3]
heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_RGB2BGR)

cv2.imshow('Depth grayscale',final_img1)
cv2.imshow('Depth heatmap',heatmap1)

cv2.waitKey(0)  

