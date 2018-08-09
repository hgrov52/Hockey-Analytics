import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
#from cvutil import url_to_array, #color_flip
from pylab import rcParams

#%matplotlib inline

# set the figsize and dpi globally for all images
rcParams['figure.figsize'] = (16, 16)
rcParams['figure.dpi'] = 300

# color_flip converts cv2 BGR image to numpy RGB image
# test_frame = color_flip(cv2.imread('C:/Users/tmorgan/Desktop/Data Analytics/Sports/Hockey/test_frame_rpi.jpg'))
test_frame = cv2.imread('../../../data/frames/continuous/ACHA UNH/frame1370.jpg')
#plt.imshow(test_frame)
#plt.show()

#cv2.imshow('test',test_frame)
#v2.waitKey(0)
#cv2.destroyAllWindows()

# [535,233],[335,384],[780,423],[873,249]


# Make before and after corner picks. Coordinate order is horizontal, vertical.
#orig_pts = np.float32([[430.0, 120.0], [1020.0, 120.0], [0.0, 970.0], [1250.0, 950.0]])
constant = 20
scale = 10,17
X_SHIFT = 600
Y_SHIFT = 600

orig_pts = np.float32([[535,233],[873,249],[335,384],[780,423]])
dest_pts = np.float32([[X_SHIFT, Y_SHIFT], [scale[0]*constant+X_SHIFT, Y_SHIFT], [X_SHIFT, scale[1]*constant+Y_SHIFT], [scale[0]*constant+X_SHIFT, scale[1]*constant+Y_SHIFT]])

# verify corner picks
test_frame_lines = test_frame.copy()

cv2.line(test_frame_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
cv2.line(test_frame_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)

plt.imshow(test_frame_lines)
plt.show()

# Get perspective transform M
M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
# warp image with M
#drawing = cv2.warpPerspective(test_frame, M, (1920, 1080))
persp_frame = cv2.warpPerspective(test_frame, M, (1500, 1500))
# show the image
plt.imshow(persp_frame)
plt.show()

