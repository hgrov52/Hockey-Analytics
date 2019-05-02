import cv2, imutils, os
import numpy as np
import matplotlib.pyplot as plt

wait = 1

def stitch(prev, im, result, posx,posy, ratio=0.75, reprojThresh=4.0,
	show=False):
	
	offsetx,offsety,H = generate_homography(prev,im)

	posx+=offsetx
	posy-=offsety
	print(posx,posy)
	if(offsetx>0):
		print('right')
	elif(offsetx<0):
		print('left')
	if(offsety>0):
		print('down')
	elif(offsety<0):
		print('up')
	print()
	
	#prev[offsety:im.shape[0]+offsety, offsetx:im.shape[1]+offsetx] = im

	result = append_images(result,im,-offsetx,-offsety)


	if show:
		global wait
		cv2.imshow("result",result)
		cv2.imshow("current",im)
		k = cv2.waitKey(wait)
		if(k==27):
			cv2.destroyAllWindows()
			exit(1)
		if(k==32):
			wait = not wait


	return result,posx,posy

def analyze(imageA, imageB, ratio = 0.75,reprojThresh=4.0):
	(kpsA, featuresA) = detectAndDescribe(imageA)
	(kpsB, featuresB) = detectAndDescribe(imageB)
	M = matchKeypoints(kpsA, kpsB,
		featuresA, featuresB, ratio, reprojThresh)
	if M is None:
		return None
	(matches, H, status) = M
	return find_shift(kpsA, kpsB, matches, status)
		

def append_images(imageA,imageB,shift_x,shift_y):
	right,shift_x = (abs(shift_x),0) if shift_x<0 else (0,shift_x) 
	down,shift_y = (abs(shift_y),0) if shift_y<0 else (0,shift_y) 
	result = np.zeros((max(imageA.shape[0]+shift_y,imageB.shape[0]+down),max(imageA.shape[1]+shift_x,imageB.shape[1]+right),3))
	result[:,:]=-np.inf
	#print(imageA.shape,result.shape)
	result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:]=\
	np.where(result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:]==-np.inf,\
	imageA,result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:])
	result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:]= \
	np.where(result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:]==-np.inf, \
	imageB,result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:])
	result = result.astype(np.uint8)
	return result

def generate_homography(imageA, imageB, ratio=0.75, reprojThresh=4.0):
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kpsA, featuresA) = descriptor.detectAndCompute(imageA, None)
	(kpsB, featuresB) = descriptor.detectAndCompute(imageB, None)
	kpsA = np.float32([kp.pt for kp in kpsA])
	kpsB = np.float32([kp.pt for kp in kpsB])

	M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
	if M is None:
		return None
	(matches, H, status) = M
	H_inv = np.linalg.inv(H)
	ds = np.dot(H_inv, np.array([imageA.shape[1], imageA.shape[0], 1]))
	ds = ds/ds[-1]
	f1 = np.dot(H_inv, np.array([0,0,1]))
	f1 = f1/f1[-1]
	H_inv[0][-1] += abs(f1[0])
	H_inv[1][-1] += abs(f1[1])
	ds = np.dot(H_inv, np.array([imageA.shape[1], imageA.shape[0], 1]))
	offsety = int(f1[1])
	offsetx = int(f1[0])
	return offsetx, offsety, H
	


def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
	ratio, reprojThresh):
	# compute the raw matches and initialize the list of actual
	# matches
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []

	# loop over the raw matches
	for m in rawMatches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))

	# computing a homography requires at least 4 matches
	if len(matches) > 4:
		# construct the two sets of points
		ptsA = np.float32([kpsA[i] for (_, i) in matches])
		ptsB = np.float32([kpsB[i] for (i, _) in matches])

		# compute the homography between the two sets of points
		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
			reprojThresh)

		# return the matches along with the homograpy matrix
		# and status of each matched point
		return (matches, H, status)

	# otherwise, no homograpy could be computed
	return None

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
	# initialize the output visualization image
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB

	# loop over the matches
	for ((trainIdx, queryIdx), s) in zip(matches, status):
		# only process the match if the keypoint was successfully
		# matched
		if s == 1:
			# draw the match
			ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
			ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
			cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

	# return the visualization
	return vis

def find_shift(kpsA, kpsB, matches, status):
	avgA=[0,0]
	avgB=[0,0]

	maxxA = -np.inf
	maxyA = -np.inf
	minxA = np.inf
	minyA = np.inf

	maxxB = -np.inf
	maxyB = -np.inf
	minxB = np.inf
	minyB = np.inf
	# loop over the matches
	for ((trainIdx, queryIdx), s) in zip(matches, status):
		# only process the match if the keypoint was successfully
		# matched
		if s == 1:
			ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
			ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
			
			if(ptA[0]>maxxA):
				maxxA = ptA[0]
			if(ptA[0]<minxA):
				minxA = ptA[0]
			if(ptA[1]>maxyA):
				maxyA = ptA[1]
			if(ptA[1]<minyA):
				minyA = ptA[1]

			if(ptB[0]>maxxB):
				maxxB = ptB[0]
			if(ptB[0]<minxB):
				minxB = ptB[0]
			if(ptB[1]>maxyB):
				maxyB = ptB[1]
			if(ptB[1]<minyB):
				minyB = ptB[1]

			avgA[0]+=ptA[0]/len(kpsA)
			avgA[1]+=ptA[1]/len(kpsA)
			avgB[0]+=ptB[0]/len(kpsB)
			avgB[1]+=ptB[1]/len(kpsB)
			
	return (int(avgA[0]),int(avgA[1])),(int(avgB[0]),int(avgB[1])), ((maxxA - minxA),(maxyA - minyA)), ((maxxB - minxB),(maxyB - minyB))

if __name__ == '__main__':
	prev = None
	result = None
	posx = 0
	posy = 0
	count=-1
	lst = sorted(os.listdir(os.fsencode('../../../data/frames/continuous/ACHA UNH/')))
	while count < len(lst):
		file = lst[count]
		filename = os.fsdecode(file)
		if(filename.endswith('.jpg') == False):
			continue
		
		count+=1
		if(count%1==0):
			#print(str(float(count)/float(len(lst))*100)+"%")
			im = cv2.imread('../../../data/frames/continuous/ACHA UNH/'+filename)
			im = cv2.resize(im,(480, 320))
			if(prev is None):
				result = im
				prev = im
				continue


			result,posx,posy = stitch(prev, im, result, posx,posy, show=True)
			prev = im
			