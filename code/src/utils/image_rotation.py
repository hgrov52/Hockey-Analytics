from skimage.transform import warp
import matplotlib.pyplot as plt
import numpy as np
import cv2,math

from skimage.data import text
from skimage.transform import ProjectiveTransform

def rotate_along_axis(im, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):        
    rtheta, rphi, rgamma = theta*math.pi/180.0, phi*math.pi/180.0, gamma*math.pi/180.0
    w = len(im[0])
    h = len(im)
    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(h**2 + w**2)
    dz = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])
    # rotation around x, y, z
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R = np.dot(np.dot(RX, RY), RZ)
    # translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])
    A2 = np.array([ [dz, 0, w/2, 0],
                    [0, dz, h/2, 0],
                    [0, 0, 1, 0]])
    M = np.dot(A2, np.dot(T, np.dot(R, A1)))
    return cv2.warpPerspective(im, M, (w, h))


im = cv2.imread('acha2.jpg')
source = np.array(((370, 190), (0,520),
                   			(1255,620), (1220,230),
                  			(370,190)))
mapping = ProjectiveTransform()

MAX_SHAPE = 256*2
X_SHAPE = MAX_SHAPE/2
Y_SHAPE = MAX_SHAPE/2

target = np.array(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
					(MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE),
    				(MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE)))
mapping.estimate(target,source)
print("H:")
print(mapping.params)
plt.figure(); 
plt.subplot(121); 
plt.imshow(im); 
plt.gray(); 

plt.plot(source[:,0], source[:,1],'-', lw=1, color='red'); 

plt.subplot(122); 

warped = warp(im, mapping,output_shape=(MAX_SHAPE*2,MAX_SHAPE*2))
plt.imshow(warped); 
plt.plot(target[:,0], target[:,1],'-', lw=1, color='red'); 
 
#plt.show()



H = np.array([[ 1.76677745e+00, -3.42868808e+00,  1.52262931e+03],
	[ 5.45095835e-02,  1.51260519e+00, -4.66234836e+02],
	[-1.65902878e-04, -2.97194983e-03,  2.56004540e+00]])


pts1 = np.float32(((370, 190), (0,520),
                   	(1255,620), (1220,230)))
pts2 = np.float32(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
					(MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE)))

M = cv2.getPerspectiveTransform(pts1,pts2)
print("M:")
print(M)

warp = cv2.warpPerspective(im,M,(1024,1024))
cv2.imshow('final',warp)
cv2.waitKey(0)








cv2.destroyAllWindows()




