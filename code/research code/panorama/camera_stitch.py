import cv2, imutils
import numpy as np
import matplotlib.pyplot as plt

def stitch(images, ratio=0.75, reprojThresh=4.0,
	showMatches=False):
	# unpack the images, then detect keypoints and extract
	# local invariant descriptors from them
	(imageA, imageB) = images
	(kpsA, featuresA) = detectAndDescribe(imageA)
	(kpsB, featuresB) = detectAndDescribe(imageB)

	# match features between the two images
	M = matchKeypoints(kpsA, kpsB,
		featuresA, featuresB, ratio, reprojThresh)

	# if the match is None, then there aren't enough matched
	# keypoints to create a panorama
	if M is None:
		return None
	

	# otherwise, apply a perspective warp to stitch the images
	# together
	(matches, H, status) = M
	result = cv2.warpPerspective(imageA, H,
		(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
	#cv2.imshow('r',result)
	#cv2.waitKey(0)
	#result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB



	avgA, avgB = find_shift(kpsA, kpsB, matches, status)
	print(avgA,avgB)
	cv2.circle(imageA, avgA, 3, (0, 255, 0), -1)
	cv2.circle(imageB, avgB, 3, (0, 255, 0), -1)
	#cv2.imshow('imageA',imageA)
	#cv2.imshow('imageB',imageB)

	
	result = append_images(imageA,imageB,avgA,avgB)
	
	

	#cv2.imshow('result',result)
	#cv2.waitKey(0)
	#result = np.zeros((imageA.shape[0]+abs(avgA[0]-avgB[0]),imageA.shape[1]+abs(avgA[0]-avgB[0]),imageA.shape[2]))
	#cv2.destroyAllWindows()

	if showMatches:
		vis = drawMatches(imageA, imageB, kpsA, kpsB, matches,
			status)
		return (result, vis)

	return result

def append_images(imageA,imageB,avgA,avgB):
	shift_x = avgA[0]-avgB[0]
	shift_y = avgA[1]-avgB[1]
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

def detectAndDescribe(image):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect and extract features from the image
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)

	# convert the keypoints from KeyPoint objects to NumPy
	# arrays
	kps = np.float32([kp.pt for kp in kps])
	#print(kps)

	# return a tuple of keypoints and features
	return (kps, features)


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

	# loop over the matches
	for ((trainIdx, queryIdx), s) in zip(matches, status):
		# only process the match if the keypoint was successfully
		# matched
		if s == 1:
			ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
			ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
			
			avgA[0]+=ptA[0]/len(kpsA)
			avgA[1]+=ptA[1]/len(kpsA)
			avgB[0]+=ptB[0]/len(kpsB)
			avgB[1]+=ptB[1]/len(kpsB)
			
	return (int(avgA[0]),int(avgA[1])),(int(avgB[0]),int(avgB[1]))