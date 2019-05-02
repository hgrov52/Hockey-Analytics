from __future__ import print_function
import numpy as np
import cv2,os
import camera_stitch
import matplotlib.pyplot as plt

def build_panorama():



	vidcap = cv2.VideoCapture('../../../data/video/ACHA_vid.mp4')
	result = None
	success = True
	outer = False
	while success:
		success,image = vidcap.read()
		cv2.imshow('image',image)
		k = cv2.waitKey(1)
		if(k == 27):
			cv2.destroyAllWindows()
			exit(1)
		if(k==32):
			while(1):
				success,image = vidcap.read()
				cv2.imshow('image',image)
				k = cv2.waitKey(0)
				if(result is None):
					result = image
					continue
				
				result= camera_stitch.stitch([result,image],showMatches=False)
				cv2.imshow('result',result)
				k = cv2.waitKey(0)
				if(k==27):
					outer = True
					break
		if(outer):
			break
	cv2.destroyAllWindows()










def build_panorama3():


	im1 = cv2.imread('frame907.jpg')
	im2 = cv2.imread('frame1039.jpg')
	
	MAX_FEATURES = 500
	GOOD_MATCH_PERCENT = 0.15

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# Draw top matches
	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
	cv2.imwrite("matches.jpg", imMatches)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))

	cv2.imshow('imreg', im1Reg)
	cv2.waitKey(0)

	# Print estimated homography
	print("Estimated homography : \n",  h)

	cv2.destroyAllWindows()




def remove_differences():

	vidcap = cv2.VideoCapture('../Frame_Images/ACHA UNH/ACHA_vid.mp4')
	prev = None
	success = True
	while success:
		success,image = vidcap.read()
		#cv2.imshow('image',image)
		#k = cv2.waitKey(0)

		if(prev is None):
			prev = image
			continue

		cv2.imshow('sub',cv2.subtract(image,prev))
		k=cv2.waitKey(0)
		if(k==27):
			break

		prev = image

		cv2.destroyAllWindows()

if __name__ == '__main__':
	build_panorama2()