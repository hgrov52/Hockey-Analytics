import cv2
import numpy as np

def test():
  vidcap = cv2.VideoCapture('../../../data/video/ACHA_vid.mp4')
  success = True
  while success:
    success,image = vidcap.read()
    k = define_line(image)
    if(k == 27):
      cv2.destroyAllWindows()
      break










def define_line(im=None,draw = False):
  if(im is None):
    im = cv2.imread('../../../data/frames/continuous/ACHA UNH/frame1121.jpg')

  # preprocessing
  im = cv2.resize(im, None, fx=.5, fy=.5)
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150)
  dilated = cv2.dilate(edges, np.ones((3,3), dtype=np.uint8))

  

  lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=10, maxLineGap=20, minLineLength=400)
  if(lines is not None):
    for line in lines:
      for x1, y1, x2, y2 in line:
        color = [255,0,0] # color vert lines blue
        cv2.line(im, (x1, y1), (x2, y2), color=color, thickness=1)

  cv2.imshow('boards',im)

  k = cv2.waitKey(0)
  if(k==27):
    return k

if __name__ == '__main__':
  #define_line(None,True)
  test()