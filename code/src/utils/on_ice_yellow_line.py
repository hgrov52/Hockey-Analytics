import cv2
import numpy as np

def define_lines(im,draw = False):

  corner_lines = [None,None] 
  yellow_contours = []
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  r, g, b, rng = (62, 90, 206, 42) # yellow (use 1000 bottom threshold)
  lower_range = np.array([r-rng, g-rng, b-rng], dtype=np.uint8)
  upper_range = np.array([r+rng, g+rng, b+rng], dtype=np.uint8)
  mask = cv2.inRange(hsv, lower_range, upper_range)
  edges = cv2.Canny(mask,0,0,apertureSize = 5)
  lines = cv2.HoughLinesP(edges,1,np.pi/180,100,20)
  lines2 = cv2.HoughLines(edges,1,np.pi/180,150)

  radial = {}
  if(lines2 is not None):
    for line in lines2:
      for rho,theta in line:
        radial.setdefault(theta,[]).append(rho)
  if(len(radial.keys())>0):
    rad_min = min(radial.keys())
    rad_max = max(radial.keys())
    list_range = rad_max-rad_min
    HORIZONTAL_RADIANS = 1.5708
    PERPENTICULAR_RANGE = 0.2
    
    # if perpendicular enough lines marking hard corner of ice
    if(list_range>PERPENTICULAR_RANGE):
      positive_slope = {theta: rhos for theta,rhos in radial.items() if (theta<HORIZONTAL_RADIANS and abs(theta - HORIZONTAL_RADIANS)>PERPENTICULAR_RANGE/2.0 )} 
      negative_slope = {theta: rhos for theta,rhos in radial.items() if (theta>HORIZONTAL_RADIANS and abs(theta - HORIZONTAL_RADIANS)>PERPENTICULAR_RANGE/2.0 )}
      pos_duplicates = {theta: rhos for theta, rhos in positive_slope.items() if len(rhos)==2}
      neg_duplicates = {theta: rhos for theta, rhos in negative_slope.items() if len(rhos)==2}      

      if(len(pos_duplicates.keys())==1):
        theta = list(pos_duplicates.keys())[0] # bc its the only one
        corner_lines[0] = [theta,max(pos_duplicates[theta])]
      if(len(neg_duplicates.keys())==1):
        theta = list(neg_duplicates.keys())[0] # bc its the only one
        corner_lines[1] = [theta,max(neg_duplicates[theta])]

      if(corner_lines[0] == None and len(positive_slope.keys())==1):
        theta = list(positive_slope.keys())[0]
        corner_lines[0] = [theta,max(positive_slope[theta])]
      if(corner_lines[1] == None and len(negative_slope.keys())==1):
        theta = list(negative_slope.keys())[0]
        corner_lines[1] = [theta,max(negative_slope[theta])]

      if(corner_lines[0] == None and len(positive_slope.keys())>0):
        theta = max(positive_slope.keys())
        corner_lines[0] = [theta,max(positive_slope[theta])]
      if(corner_lines[1] == None and len(negative_slope.keys())>0):
        theta = max(negative_slope.keys())
        corner_lines[1] = [theta,max(negative_slope[theta])]

  if(draw):
    for theta,rhos in radial.items():
      for rho in rhos:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1500*(-b))
        y1 = int(y0 + 1500*(a))
        x2 = int(x0 - 1500*(-b))
        y2 = int(y0 - 1500*(a))
        cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)
  
  # for blue board line use
  if(lines is not None):
    best = None
    for line in lines:
      for x1,y1,x2,y2 in line:
        yellow_contours.append((x1,y1,x2,y2))
  return yellow_contours,corner_lines
