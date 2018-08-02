import cv2
import numpy as np

def define_lines(im,draw = False):
  blue_lines = [None,None]
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  r,g,b,rng = (117,41,242,13)
  lower_range = np.array([r-rng, g-rng, b-rng], dtype=np.uint8)
  upper_range = np.array([r+rng, g+rng, b+rng], dtype=np.uint8)
  mask = cv2.inRange(hsv, lower_range, upper_range)
  #image, cnts, hier = cv2.findContours(mask.copy(), 1, 2)
  #edges = cv2.Canny(image,0,0,apertureSize = 5)
  edges = cv2.Canny(mask,0,0,apertureSize = 5)
  lines = cv2.HoughLines(edges,1,np.pi/180,140)
  radial = {}
  if(lines is not None):
    for line in lines:
      for rho,theta in line:
        radial.setdefault(theta,[]).append(rho)
  
  if(len(radial.keys())>0):
    BLUE_RADIAN_THRESHOLD = 0.3
    
    left_blue = {theta: rhos for theta,rhos in radial.items() if (theta>(1 - BLUE_RADIAN_THRESHOLD/2) and theta<(1 + BLUE_RADIAN_THRESHOLD/2))} 
    right_blue = {theta: rhos for theta,rhos in radial.items() if (theta>(2.77 - BLUE_RADIAN_THRESHOLD/2) and theta<(2.75 + BLUE_RADIAN_THRESHOLD/2))}
    left_duplicates = {theta: rhos for theta, rhos in left_blue.items() if len(rhos)==2}
    right_duplicates = {theta: rhos for theta, rhos in right_blue.items() if len(rhos)==2}

    if(len(left_duplicates.keys())==1):
      theta = list(left_duplicates.keys())[0] # bc its the only one
      blue_lines[0] = [theta,max(left_duplicates[theta])]
    if(len(right_duplicates.keys())==1):
      theta = list(right_duplicates.keys())[0] # bc its the only one
      blue_lines[1] = [theta,max(right_duplicates[theta])]

    if(blue_lines[0] == None and len(left_blue.keys())==1):
      theta = list(left_blue.keys())[0]
      blue_lines[0] = [theta,max(left_blue[theta])]
    if(blue_lines[1] == None and len(right_blue.keys())==1):
      theta = list(right_blue.keys())[0]
      blue_lines[1] = [theta,max(right_blue[theta])]

    if(blue_lines[0] == None and len(left_blue.keys())>0):
      theta = max(left_blue.keys())
      blue_lines[0] = [theta,max(left_blue[theta])]
    if(blue_lines[1] == None and len(right_blue.keys())>0):
      theta = max(right_blue.keys())
      blue_lines[1] = [theta,max(right_blue[theta])]
    
    # both blue lines are in view - need to check yellow line
    if(draw):
      for line in blue_lines:
        if(line!=None):
          theta,rho = line
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1500*(-b))
          y1 = int(y0 + 1500*(a))
          x2 = int(x0 - 1500*(-b))
          y2 = int(y0 - 1500*(a))
          cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

  return blue_lines