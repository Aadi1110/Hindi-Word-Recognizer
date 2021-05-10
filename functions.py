import cv2
import numpy as np

def cropping(threshold):
	'''
	This function is used for cropping the text out of the total area of the image
	'''
	#top
	tot = 0
	val1 = 720
	for i in range(threshold.shape[0]):
		temp = 720
		for j in range(threshold.shape[1]):
			tot = tot+255-threshold[i,j]
			if(tot>0):
				temp = i
				break
		val1 = min(temp,val1)

	#bottom
	tot = 0
	val2 = 0
	for i in range(threshold.shape[0]-1,0,-1):
		temp = 0
		for j in range(threshold.shape[1]):
			tot = tot+255-threshold[i,j]
			if(tot>0):
				temp = i
				break
		val2 = max(temp,val2)

	#left
	tot = 0
	val3 = 720
	for i in range(threshold.shape[1]):
		temp = 720
		for j in range(threshold.shape[0]):
			tot = tot+255-threshold[j,i]
			if(tot>0):
				temp = i
				break
		val3 = min(temp,val3)

	#right
	tot = 0
	val4 = 0
	for i in range(threshold.shape[1]-1,0,-1):
		temp = 0
		for j in range(threshold.shape[0]):
			tot = tot+255-threshold[j,i]
			if(tot>0):
				temp = i
				break
		val4 = max(temp,val4)

	crop = threshold[max(0,val1-20):min(720,val2+20),max(0,val3-20):min(720,val4+20)]
	return crop

def resz(crop):
	'''
	This function is used for resizing the image with its longer dimension becoming 720, while the shorter dimension is adjusted accordingly so that the
	aspect ratio of the image is maintained. This resizing the performed to improve the detection of text
	'''
	if(crop.shape[0]<crop.shape[1]):
		crop = cv2.resize(crop,(720,int(crop.shape[0]*720/crop.shape[1])))
	else:
		crop = cv2.resize(crop,(int(crop.shape[1]*720/crop.shape[0]),720))

	crop = cv2.copyMakeBorder(crop, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, 255)
	return(crop)

def segment(img):
	'''
	This function is used for extraction of text out the image and segmentation of different characters present in the image.
	'''
	h=0
	w=0

	if(img.shape[0]>img.shape[1]):
		h = 720
		w = int(img.shape[1]*h/img.shape[0])
	else:
		w = 720
		h = int(img.shape[0]*w/img.shape[1])

	dsize = (w,h)
	img = img[10:img.shape[0]-10,10:img.shape[1]-10]
	img = cv2.resize(img,dsize)
	img_area=img.shape[0]*img.shape[1]
	gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	dn = cv2.fastNlMeansDenoising(gray_img,1.0,9,15)
	threshold=cv2.adaptiveThreshold(dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 30)
	threshold = cropping(threshold)
	threshold = resz(threshold)
	area = threshold.shape[0]*threshold.shape[1]
	bp = sum(list(map(sum,threshold)))/(area*255)

	kernel=np.ones((2,2),'uint8')
	it = 0
	if(bp>=0.85 and bp<=0.88):
		it = 4
	elif(bp>=0.88):
		it = 6
	thresholdn=cv2.erode(threshold,kernel,iterations=it)

	contours,hierarchy=cv2.findContours(thresholdn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(thresholdn, contours, -1, (0,255,0), 3)

	words = []
	chars = []
	maxar = 0
	maxcnt = contours[0]
	maxrect = tuple()
	for i in range(len(contours)):
		cnt = contours[i]
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		if cv2.contourArea(cnt)>0.01*area and cv2.contourArea(cnt)<0.98*area :
			if(cv2.contourArea(cnt)>maxar):
				maxar = cv2.contourArea(cnt)
				maxcnt = box
				maxrect = rect

	h_t = int(maxrect[1][1])
	w_t = int(maxrect[1][0])
	srcp = maxcnt.astype('float32')
	dstp = np.array([[0,h_t-1],[0,0],[w_t-1,0],[w_t-1,h_t-1]], dtype='float32')

	#The part below is used for handling rotated images.
	M = cv2.getPerspectiveTransform(srcp,dstp)
	M2 = cv2.getPerspectiveTransform(srcp,srcp)
	warp = cv2.warpPerspective(thresholdn, M, (w_t,h_t))
	warp2 = cv2.warpPerspective(thresholdn[3:threshold.shape[0]-3,3:threshold.shape[1]-3], M2, (threshold.shape[1]-6,threshold.shape[0]-6))

	if(warp.shape[0]>1.25*warp.shape[1]):
		warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

	ubp = 0
	lbp = 0
	for i in range(int(warp.shape[0]/2)):
		for j in range(warp.shape[1]):
			ubp = ubp+warp[i,j]

	for i in range(int(warp.shape[0]/2),warp.shape[0],1):
		for j in range(warp.shape[1]):
			lbp = lbp+warp[i,j]


	if(ubp>lbp):
		warp = cv2.rotate(warp, cv2.ROTATE_180)

	if(warp.shape[0]/warp.shape[1]>0.75 and warp.shape[0]/warp.shape[1]<1.5):
		warp = warp2

	kernel_w=np.ones((2,2),'uint8')
	if(warp.shape[1]/warp.shape[0]>6):
		warp = cv2.dilate(warp,kernel_w, iterations=2)

	chars = [] #This list stores different characters in the word
	cpix = [] #This list stores the sum of pixels in each column of the image
	
	for i in range(warp.shape[1]):
		ctr=0;
		for j in range(warp.shape[0]):
			ctr+= 255-warp[j,i]
		cpix.append(ctr)

	'''
	The part below segments the characters from the words using the cpix list. If the sum of pixel values in any column of
	the image is below a threshold value (it indicates the presence of gap between two characters), the image is sliced 
	vertically at that position, separating the characters.
	'''

	h_w = warp.shape[0]
	st = 0
	end = st + int(0.5*h_w)
	while(end<warp.shape[1]):
		if(cpix[end]<warp.shape[1]*255*0.045):
			chars.append(warp[:,st:end])
			while(end<warp.shape[1] and cpix[end]<warp.shape[1]*255*0.045):
				end = end+1
			st=end
			end = int(st+0.5*h_w)
		else:
			end = end+1

	return chars