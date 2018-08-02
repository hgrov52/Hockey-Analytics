import cv2,sys
import numpy as np

def nothing(x):
    pass

vidcap = cv2.VideoCapture('../Frame_Images/ACHA UNH/ACHA_vid.mp4')
success = True
cv2.namedWindow('image')

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('range','image',0,255,nothing)

cv2.setTrackbarPos('R','image',112)
cv2.setTrackbarPos('G','image',40)
cv2.setTrackbarPos('B','image',246)
cv2.setTrackbarPos('range','image',1)
while success:
  success,image = vidcap.read()
  cv2.imshow('separate image',image)

  r = cv2.getTrackbarPos('R','image')
  g = cv2.getTrackbarPos('G','image')
  b = cv2.getTrackbarPos('B','image')
  range_ = cv2.getTrackbarPos('range','image')

  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower_range = np.array([r-range_, g-range_, b-range_], dtype=np.uint8)
  upper_range = np.array([r+range_, g+range_, b+range_], dtype=np.uint8)
  mask = cv2.inRange(hsv, lower_range, upper_range)
  cv2.imshow('image',mask)

  k = cv2.waitKey(1)
  if(k==27):
    break
  if(k==32):
    while(1):
      r = cv2.getTrackbarPos('R','image')
      g = cv2.getTrackbarPos('G','image')
      b = cv2.getTrackbarPos('B','image')
      range_ = cv2.getTrackbarPos('range','image')
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      lower_range = np.array([r-range_, g-range_, b-range_], dtype=np.uint8)
      upper_range = np.array([r+range_, g+range_, b+range_], dtype=np.uint8)
      mask = cv2.inRange(hsv, lower_range, upper_range)
      cv2.imshow('image',mask)

      q = cv2.waitKey(1)
      if(q == 32):
        break

"""

red line 158, 026, 177 +- 24
cv2.setTrackbarPos('R','image',158)
cv2.setTrackbarPos('G','image',26)
cv2.setTrackbarPos('B','image',177)
cv2.setTrackbarPos('range','image',24)
(128, 70, 148, 20) v2

blue line (117,41,242,13)




neutral zone face off circles:
cv2.setTrackbarPos('R','image',120)
cv2.setTrackbarPos('G','image',40)
cv2.setTrackbarPos('B','image',200)
cv2.setTrackbarPos('range','image',40)

center ice face off circle:
cv2.setTrackbarPos('R','image',66)
cv2.setTrackbarPos('G','image',58)
cv2.setTrackbarPos('B','image',187)
cv2.setTrackbarPos('range','image',41)

cv2.setTrackbarPos('R','image',60)
cv2.setTrackbarPos('G','image',60)
cv2.setTrackbarPos('B','image',180)
cv2.setTrackbarPos('range','image',50)


all ice no crowd:
(120, 40, 200, 40)


"""


  
 
cv2.destroyAllWindows()

