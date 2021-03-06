import numpy as np
import cv2,os,sys,time
from matchers import matchers

class Stitch:
	def __init__(self, images):
		self.images = [cv2.resize(each,(480, 320)) for each in images]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()

	def prepare_lists(self):
		print ("Number of images : %d"%self.count)
		self.centerIndex = self.count/2 
		print ("Center index image : %d"%self.centerIndex)
		self.center_im = self.images[int(self.centerIndex)]
		for i in range(self.count):
			if(i<=self.centerIndex):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print ("Image lists prepared")

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		cv2.imshow('a',a)
		for b in self.left_list[1:]:

			cv2.imshow('b',b)
			cv2.waitKey(0)
			
			print("\n",len(self.left_list),"\n")
			# calculate homography of two images using a as reference
			H = self.matcher_obj.match(a, b, 'left')
			# calculate inverse of homagraphy matrix
			xh = np.linalg.inv(H)

			# dot product of inv homography and the shape of a 
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]

			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]

			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])

			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print ("image dsize =>", dsize)
			a = cv2.warpPerspective(a, xh, dsize)
			# cv2.imshow("warped", tmp)
			# cv2.waitKey()
			a[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			

			self.leftImage = a

			cv2.imshow('r', self.leftImage)
			cv2.waitKey(0)

		
	def rightshift(self):
		for each in self.right_list:
			print("here")
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			print ("Homography :", H)
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			#cv2.imshow("tp", tmp)
			#cv2.waitKey()
			# tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
			tmp = self.mix_and_match(self.leftImage, tmp)
			print ("tmp shape",tmp.shape)
			print ("self.leftimage shape=", self.leftImage.shape)
			self.leftImage = tmp
		# self.showImage('left')



	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print (leftImage[-1,-1])

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print (time.time() - t)
		print (black_l[-1])
		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
		return warpedImage




	def trim_left(self):
		pass

	def showImage(self, string=None):
		if string == 'left':
			cv2.imshow("left image", self.leftImage)
			# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
		elif string == "right":
			cv2.imshow("right Image", self.rightImage)
		cv2.waitKey()


if __name__ == '__main__':
	images = []
	"""
	count=0
	lst = sorted(os.listdir(os.fsencode('../../Frame_Images/ACHA UNH')))
	file_num = 0
	l = len(lst)
	while file_num < l:
		file = lst[file_num]
		filename = os.fsdecode(file)
		file_num+=1
		if(filename.endswith('.jpg') == False):
			continue
		count+=1
		print(str(float(count)/float(l)*100)+"%")
		images.append(cv2.imread('../../Frame_Images/ACHA UNH/'+filename))
	"""
	images.append(cv2.imread('frame1094.jpg'))
	images.append(cv2.imread('frame1039.jpg'))
	images.append(cv2.imread('frame907.jpg'))
	s = Stitch(images)
	s.leftshift()
	# s.showImage('left')
	s.rightshift()
	cv2.imshow('r', s.leftImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	