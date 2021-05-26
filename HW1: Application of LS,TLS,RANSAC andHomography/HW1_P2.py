import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
from numpy import linalg as LA

def calc_ls(A):                #Function to calculate the least squares fit
	x_values = A[:,0]      
	y_values = A[:,1]
	
	x_square = np.power(x_values, 2)
	
	A = np.stack((x_square, x_values, np.ones((len(x_values)), dtype = int)), axis = 1) #creating the A matrix quadratic equation (y=ax^2 + bx +c) to obtain the best fit in the form of a parabola
    
	ls_estimate = ls_fit(A, y_values) #calling a function to compute the a,b,c values
	ls_model = A.dot(ls_estimate) 	  #the new Y values are obtained by performing AX=Y

	return ls_model

def ls_fit(A, Y):
        A_transpose = A.transpose()
        ATA = A_transpose.dot(A)
        ATY = A_transpose.dot(Y)
        estimate = (np.linalg.inv(ATA)).dot(ATY) #a,b,c values are obtained
        return estimate

def calc_tls(x_1,y_1):   #Function to compute the best fit to the data points using TLS 
	#To obtain the sum of the values of the elemnts in the A matric and B matrix by computing x^2,x^3,x^4,xy,x^2y,xy^2
	x2 = []
	xy = []
	x2y = []
	x3 = []
	x4 = []
	
	for i in range(len(x_1)):
		x2.append(float(x_1[i]**2))
		xy.append(float(x_1[i]*y_1[i]))
		x2y.append(float((x_1[i]**2)*y_1[i]))
		x3.append(float(x_1[i]**3))
		x4.append(float(x_1[i]**4))
	
	sum_n = 0.0
	sum_x1=0.0
	sum_y1=0.0
	sum_x2 =0.0
	sum_xy =0.0
	sum_x2y =0.0
	sum_x3 = 0.0
	sum_x4 =0.0
	
	for i in range(len(x_1)):
		sum_n = sum_n + i
		sum_x1=sum_x1+x_1[i]
		sum_y1=sum_y1+y_1[i]
		sum_x2 =sum_x2 + x2[i]
		sum_xy =sum_xy+xy[i]
		sum_x2y =sum_x2y+x2y[i]
		sum_x3 = sum_x3+x3[i]
		sum_x4 = sum_x4 + x4[i]
	
	#A matrix consists of the x and x power elements alone and B matrix consists of the y,xy,x^2y elements
	A= np.array([[sum_x2,sum_x1,sum_n],[sum_x3,sum_x2,sum_x1],[sum_x4,sum_x3,sum_x2]])
	B= np.array([sum_y1,sum_xy,sum_x2y])
	
	#To compute the SVD elements 
	U,Sigma_inv,VT = Compute_Svd(A)

	#In SVD, A= (U.SIGMA.Vtranspose). To compute a,b,c we need to find INVERSE(A)*B
	#The a,b,c values which can be considered as a matrix X is obtained by computing (V.sigma-inverse.Utranspose).B
	quad_soln = VT.T.dot(Sigma_inv.dot(U.T.dot(B)))
	a = quad_soln[0]
	b = quad_soln[1]
	c = quad_soln[2]

	y_tls=[]#empty list 
	
	#Obtaining the new y values to plot TLS
	for i in range(0,len(x_1)):
    		y = (a*(x_1[i]**2))+(b*x_1[i]) + c
    		y_tls.append(y)

	return y_tls
	
def Compute_Svd(A): 
	#To obtain eigen values and vectors of U                      
	AT=A.T                                     
	AAT=A.dot(AT)
	eigval_U,eigvect_U=LA.eig(AAT)
	sort = eigval_U.argsort()[::-1]
	eigval_sorted = eigval_U[sort]

	#Incase of negative eigen values
	for i in range(len(eigval_sorted)):
		if eigval_sorted[i] <= 0:
			eigenval_sorted[i]*=-1

	eigvect_sorted = eigvect_U[:,sort]

	#To compute the sigma matrix consisting of the singularties in the diagonal elements
	diag = np.diag((np.sqrt(eigval_sorted)))  

	#Finding the inverse of sigma matrix
	diag = np.linalg.inv(diag)
	sigma_inv = np.zeros_like(A).astype(np.float64)	
	sigma_inv[:diag.shape[0],:diag.shape[1]]=diag

	#To obtain V transpose which is given by (U-TRANSPOSE).(Sigma inverse). (A)
	V_trans= sigma_inv.dot(eigvect_sorted.T)
	V_trans = V_trans.dot(A)
    
	return eigvect_sorted,sigma_inv,V_trans

def calc_ransac(A):      #Function to compute the best fit to all the points using RANSAC
	x_values = A[:,0] 
	y_values = A[:,1]
	
	x_square = np.power(x_values, 2)

	#Creating the polynomial array similar to that of a parabola
	A = np.stack((x_square, x_values, np.ones((len(x_values)), dtype = int)), axis = 1)

	# Setting a threshold value based on which we can differentiate between inliers and outliers
	threshold = np.std(y_values)/3

	ransac_model_estimate = ransac_fit(A, y_values, 3, threshold)
	ransac_model_y = A.dot(ransac_model_estimate)

	return ransac_model_y

def ransac_fit(A, Y, num_sample, threshold):
	#setting initial values
	number_iterations = math.inf
	iterations_completed = 0
	max_inlier_count = 0
	best_fit = None

	prob_outlier= 0
	
	prob_desired = 0.95  #Setting a desired probability
	
	combined_data = np.column_stack((A, Y)) 
	data_length = len(combined_data)
	
	# Finding the number of iterations
	while number_iterations > iterations_completed:
		
		# shuffling the rows and takeing the first 'num_sample' rows
		np.random.shuffle(combined_data)
		sample_data = combined_data[:num_sample, :]
		
		#Estimating the corresponding Y values based on LS method and using this model to count the number of inliers in the threshold
		estimated_model = ls_fit(sample_data[:,:-1], sample_data[:, -1:])
		
		# count the inliers within the threshold
		y_inliers = A.dot(estimated_model)
		err = np.abs(Y - y_inliers.T)
		
		#if err is less than the threshold value, then the point is an inlier 
		inlier_count = np.count_nonzero(err < threshold)
		print('Inlier COunt',inlier_count)
		# The best fit would be chosen for the y values which have the maximum inlier count 
		if inlier_count > max_inlier_count:
			max_inlier_count = inlier_count
			best_fit = estimated_model
		
		#Calculating the outlier probaility. This value is zero for thresholds which have a standard deviation of np.std(y_values) and for certain runs of np.std(y_values)/2 
		prob_outlier = 1 - inlier_count/data_length
	
		#Computing the number of iterations based on the number of samples, the outlier probability and the desired probability.
		number_iterations = math.log(1 - prob_desired)/math.log(1 - (1 - prob_outlier)**num_sample)
		iterations_completed = iterations_completed + 1

	return best_fit

if __name__ == '__main__':
	#reading the video files
	cap1 = cv2.VideoCapture('Ball_travel_10fps.mp4')
	cap2 = cv2.VideoCapture('Ball_travel_2_updated.mp4')

	i = 0
	
	a=np.empty((0,2),int)
	b=np.empty((0,2),int)
	c=np.empty((0,2),int)
	d=np.empty((0,2),int)
	e=np.empty((0,2),int)
	f=np.empty((0,2),int)
	x1=np.empty((0,1),int)
	y1=np.empty((0,1),int)
	x2=np.empty((0,1),int)
	y2=np.empty((0,1),int)

	while(cap1.isOpened()):
		
		#reading the first video file frame by frame
		ret1, frame1 = cap1.read()

		count=0
		count1=0
		count2=0
		count3=0

		if ret1 == False:
			break
		
		#converting into HSV workspace
		hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
		
		# Obtaining the ranges of red values
		# Range for lower red
		lower_red = np.array([0,120,70])
		upper_red = np.array([10,255,255])

		mask1 = cv2.inRange(hsv1, lower_red, upper_red)
		

		# Range for upper red
		lower_red = np.array([170,120,70])
		upper_red = np.array([180,255,255])
		mask2 = cv2.inRange(hsv1,lower_red,upper_red)
		
	
		# Generating the final mask to detect red color
		mask = mask1+mask2

		# The black region in the mask has the value of 0,
		# so when multiplied with original image removes all non-red regions
		result1 = cv2.bitwise_and(frame1, frame1, mask = mask)

		dx,dy,chn =result1.shape

		result1 = cv2.resize(result1,(int(dx/5),int(dy/5)))

		dx,dy,chn =result1.shape
		
		#Obtaining the top and bottom pixels of the red values
		for i in range(dx):
			for j in range(dy):
				if result1[i,j,0]!=0:
					count+=1
					if count == 1:
						i=dx-i
						a=np.append(a,np.array([[i,j]]),axis=0)

		for l in reversed(range(dx)):
			for k in reversed(range(dy)):
				if result1[l,k,0]!=0:
					count1+=1
					if count1 == 1:
						l=dx-l
						b=np.append(b,np.array([[l,k]]),axis=0)

		c=np.append(a,b,axis=0)
		i+=1
		cv2.waitKey(1)

	while(cap2.isOpened()):
		
		# Reading the second video file frame by frame
		ret2, frame2 = cap2.read()

		count2=0
		count3=0

		if ret2 == False:
			break

		hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

		# Range for lower red
		lower_red = np.array([0,120,70])
		upper_red = np.array([10,255,255])
		mask3 = cv2.inRange(hsv2, lower_red, upper_red)
		# Range for upper red
		lower_red = np.array([170,120,70])
		upper_red = np.array([180,255,255])
		mask4 = cv2.inRange(hsv2,lower_red,upper_red)
	
		# Generating the final mask to detect red color
		mask5 = mask3+mask4

		# The black region in the mask has the value of 0,
		# so when multiplied with original image removes all non-red regions
		result2 = cv2.bitwise_and(frame2, frame2, mask = mask5)

		dx1,dy1,chn1 =result2.shape
		
		result2 = cv2.resize(result2,(int(dx1/5),int(dy1/5)))

		dx1,dy1,chn =result2.shape

		for i in range(dx1):
			for j in range(dy1):
				if result2[i,j,0]!=0:
					count2+=1
					if count2 == 1:
						i=dx1-i
						d=np.append(d,np.array([[i,j]]),axis=0)

		for k in reversed(range(dx1)):
			for l in reversed(range(dy1)):
				if result2[k,l,0]!=0:
					count3+=1
					if count3 == 1:
						k=dx1-k
						e=np.append(e,np.array([[k,l]]),axis=0)

		f=np.append(e,d,axis=0)
		i+=1
		cv2.waitKey(1)
	
	# Obtaining the averages or center of the top and bottom pixel values of the 2 videos in eaxh frame
	for i in range(len(a)):
		x1=np.append(x1,((a[i][1]+b[i][1])/2))

	for i in range(len(a)):
		y1=np.append(y1,((a[i][0]+b[i][0])/2))

	for i in range(len(d)):
		x2=np.append(x2,((d[i][1]+e[i][1])/2))

	for i in range(len(d)):
		y2=np.append(y2,((d[i][0]+d[i][0])/2))

	c = np.vstack((x1, y1)).T
	f = np.vstack((x2, y2)).T

	#Calling the function to find the best fit using least squares
	y1_ls = calc_ls(c)
	y2_ls = calc_ls(f)
	
	#Calling the function t find the best fit using TLS
	y1_tls = calc_tls(x1,y1)
	y2_tls = calc_tls(x2,y2)

	#Calling the function to find the best fit using RANSAC
	y1_ransac = calc_ransac(c)
	y2_ransac = calc_ransac(f)
	
	#Creating subplots to view the different methods for both the videos
	fig, (ax1, ax2) = plt.subplots(1, 2)
	
	# Plot depicting LS, TLS & RANSAC for the first video
	ax1.set_title('Video 1')
	ax1.scatter(x1, y1, marker='o', color = (0,1,0), label='data points')
	ax1.plot(x1, y1_ls, color = 'red', label='Least sqaure model')
	ax1.plot(x1, y1_tls, color = 'blue', label='TLS model')
	ax1.plot(x1, y1_ransac, color = 'yellow', label='Ransac model')
	ax1.set(xlabel='x-axis', ylabel='y-axis')
	ax1.legend()

	# Plot depicting LS, TLS & RANSAC for the second video
	ax2.set_title('Video 2')
	ax2.scatter(f[:,0], f[:,1], marker='o', color = (0,1,0), label='data points')
	ax2.plot(f[:,0], y2_ls, color = 'red', label='Least sqaure model')
	ax2.plot(f[:,0], y2_tls, color = 'blue', label='TLS model')
	ax2.plot(f[:,0], y2_ransac, color = 'yellow', label='Ransac model')
	ax2.set(xlabel='x-axis', ylabel='y-axis')
	ax2.legend()

	plt.show()
	
	cap1.release()
	cap2.release()
	cv2.destroyAllWindows()
