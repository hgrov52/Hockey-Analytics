import json,pprint,cv2
import numpy as np

ground_truths = {'Red Line':(100,0),
				 'Blue Line':{'right': (125,0),'left': (75,0)},
				 'Goal Line':{'right': (189,0),'left': (11,0)},
				 'Center ice':(100,42.5),
				 'Neutral Faceoff Dot':{"['right', 'top']":(120,20.5),
				 						"['bottom', 'right']":(120,64.5),
				 						"['left', 'top']":(80,20.5),
				 						"['bottom', 'left']":(80,64.5)},
				 'Zone Faceoff Dot':{"['right', 'top']":(169,20.5),
			 						 "['bottom', 'right']":(169,64.5),
			 						 "['left', 'top']":(31,20.5),
			 						 "['bottom', 'left']":(31,64.5)},
			 	 'Goal Post':{"['right', 'top']":(189,39.5),
	 						  "['bottom', 'right']":(189,45.5),
	 						  "['left', 'top']":(11,39.5),
	 						  "['bottom', 'left']":(11,45.5)},
	 			 'Inner Hash Mark':{"['hash-left', 'left', 'top']":(29.5,35.5),
	 			 					"['hash-right''left', 'top']":(32.5,35.5),
	 			 					"['bottom', 'hash-left', 'left']":(29.5,49.5),
	 			 					"['bottom', 'hash-right', 'left']":(32.5,49.5),
	 			 					"['hash-left', 'right', 'top']":(167.5,35.5),
	 			 					"['hash-right', 'right', 'top']":(170.5,35.5),
	 			 					"['bottom', 'hash-left', 'right']":(167.5,49.5),
	 			 					"['bottom', 'hash-right', 'right']":(170.5,49.5),





	 			 },
	 			 'Center Blue Line':{'right': (125,42.5),'left': (75,42.5)},
	 			 'Center Goal Line':{'right': (189,42.5),'left': (11,42.5)},
}

def count(lst,x,index):
	num=0
	for element in lst:
		num+=(element[index]==x)
	return num

def trim_warp_points(pts1,pts2):
	redo = True
	while(redo):
		redo = False
		for i in range(len(pts2)):
			x,y = pts2[i]
			if(count(pts2,y,1)>2 or count(pts2,x,0)>2):
				pts2.remove(pts2[i])
				pts1.remove(pts1[i])
				redo = True
				break
	


def init():
	file = open('data.json','r')
	dataset = []
	supplemental = {}
	for line in file:
		s = json.loads(line)
		for image in s:
			features = {}
			features['filename']=image['External ID']
			index = 0


			# ===========
			# Need to load image to get height bc labelbox treats y like a graph not like an image (bottom to top rather than top to bottom)
			im = cv2.imread('../../../data/frames/continuous/chances/'+image['External ID'])
			height = len(im)



			# ===========
			primary = True
			for label in image['Label']:
				if(label in ['Inner Hash Mark']):
					primary = False
					supplemental[image['External ID']] = []

					# =========================
					# need to change it for lookup by frame number
					for feature in image['Label'][label]:
						desc = str(feature['side'][1] + " " + feature['side'][0] if len(feature['side'])==2 else feature['side'])
						#print(label,desc,"|",ground_truths[label][str(feature['side'])],"|",(feature['geometry']['x'],feature['geometry']['y']))
						supplemental[image['External ID']] += {'truth':ground_truths[label][str(sorted(feature['side']) if len(feature['side'])<4 else feature['side'])],'image':(feature['geometry']['x'],height-feature['geometry']['y']),'description':label+" "+desc}
						index+=1
					# ==========================

				elif(image['Label'][label][0] not in ['right','left']):
					for feature in image['Label'][label]:
						if(label == 'side of ice'):
							#print(label,"|",image['Label'][label])
							continue
						desc = str(feature['side'][1] + " " + feature['side'][0] if len(feature['side'])==2 else feature['side'])
						#print(label,desc,"|",ground_truths[label][str(feature['side'])],"|",(feature['geometry']['x'],feature['geometry']['y']))
						features[index] = {'truth':ground_truths[label][str(sorted(feature['side']) if len(feature['side'])<4 else feature['side'])],'image':(feature['geometry']['x'],height-feature['geometry']['y']),'description':label+" "+desc}
						index+=1
			if(primary):
				dataset.append(features)
			
	return dataset,supplemental


			
			

if __name__ == '__main__':
	dataset,supplemental = init()
	pp = pprint.PrettyPrinter(indent=4)
	#pp.pprint(dataset)

	for frame in dataset:
		print(frame['filename'])
		im = cv2.imread('../../../data/frames/continuous/chances/'+frame['filename'])
		pts1 = []
		pts2 = []
		for index in frame:
			if(index == 'filename'):
				continue
			feature = frame[index]

			pts1.append(feature['image'])
			pts2.append(feature['truth'])
			
			cv2.circle(im, feature['image'], 3, (0, 0, 255), -1)
			cv2.putText(im, feature['description'], (feature['image'][0] - 20, feature['image'][1] - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)




		cv2.imshow('im',im)


		trim_warp_points(pts1,pts2)
		if(len(pts2)<4):
			print(supplemental)
			continue
				
			


		pts1 = np.float32(pts1)
		pts2 = np.float32(pts2)

		
		pts2[:,0] = pts2[:,0]*6-200
		pts2[:,1] = pts2[:,1]*6+400



		try:
			M = cv2.getPerspectiveTransform(np.array(pts1[:4]),np.array(pts2[:4]))
			warp = cv2.warpPerspective(im,M,(2000,1000))
			cv2.imshow('final',warp)
		except:
			print("FAILED")
			print(pts2)


		cv2.waitKey(0)
	cv2.destroyAllWindows()

	"""
	mislabelled:
	frame4460.jpg
	frame3530.jpg

	poor warp:
	frame3560.jpg

	






	"""

