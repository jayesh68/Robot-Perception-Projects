import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import svd

#input values
x1,x2,x3,x4,y1,y2,y3,y4,xp1,xp2,xp3,xp4,yp1,yp2,yp3,yp4 = 5,150,150,5,5,5,150,150,100,200,220,100,100,80,80,200 

#Input matrix
A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],[0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],[-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],[0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],[-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],[0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],[-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],[0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])



def compute_svd(A):   
    #Computing eigen values and vectors to find Vtranspose                          
    AT=A.T                                      
    ATA=AT.dot(A)                  
    eigval_V,eigvec_V=LA.eig(ATA)    
    sort_eig = eigval_V.argsort()[::-1]
    new_eigval_V = eigval_V[sort_eig]	
    new_eigvec_V = eigvec_V[:,sort_eig]
    new_eigvec_VT = new_eigvec_V.T
	
    #Computing eigen values and vectors to find U transpose
    AAT=A.dot(AT)
    eigval_U,eigvec_U=LA.eig(AAT)
    sort_eig1 = eigval_U.argsort()[::-1]
    new_eigval_U = eigval_U[sort_eig1]
    new_eigvec_U = eigvec_U[:,sort_eig1]
    diag = np.diag((np.sqrt(new_eigval_U))) 

    #TO find the sigma matrix which would be a diagonal matrix consisting of singular elements at its diagonal 
    sigma = np.zeros_like(A).astype(np.float64)
    sigma[:diag.shape[0],:diag.shape[1]]=diag

    #The homography matrix would be the last column of V
    H = new_eigvec_V[:,8]                
    H = np.reshape(H,(3,3))
    return new_eigvec_VT,new_eigvec_U,sigma,H



VT,U,S,H = compute_svd(A)
U,S,VT = svd(A)
print('U',U,'S',S,'VT',VT,svd(A))
print("Homography matrix=",H)


