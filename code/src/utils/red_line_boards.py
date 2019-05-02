import cv2
import numpy as np 

def in_line_with_yellow_contours(im,cX,cY,yellow_contours,y_threshold=50):
  closest = []
  for pt in yellow_contours:

    dist1 = abs(cX-pt[0])
    dist2 = abs(cX-pt[2])
    if(dist1<dist2):
      dist,pt = dist1,(pt[0],pt[1])
    else:
      dist,pt = dist2,(pt[2],pt[3])
    if(len(closest)<4):
      closest.append((dist,pt))
      continue
    for i,x in enumerate(closest):
      if(dist<x[0]):
        closest[i] = (dist,pt)
        break
  best = None,None
  for x in closest:
    #cv2.circle(im, (x[1][0], x[1][1]), 5, (0, 255, 0), -1)
    if(best == (None,None) or x[1][1]>best[1] and x[1][0] in range(best[0]-40,best[0]+40)):
      best = x[1]
  if(cY in range(best[1]-y_threshold,best[1]+y_threshold)):# maybe not +y_threshold bc yellow wont be below red/blue line on image
    #cv2.circle(im, (best[0], best[1]), 5, (100, 100, 0), -1)
    return best[1]
  return False

def define_point(im, yellow_contours, red_x, red_y_recent_avg, RECENT_AVG_LEN, draw = False):
    hsv_vals = [(150, 94, 118, 16),(154, 42, 164, 14)]
    mask = None
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    for r,g,b,rng in hsv_vals:
        lower_range = np.array([r-rng, g-rng, b-rng], dtype=np.uint8)
        upper_range = np.array([r+rng, g+rng, b+rng], dtype=np.uint8)
        if(mask is None):
            mask = cv2.inRange(hsv, lower_range, upper_range)
        else:
            mask = cv2.add(mask,cv2.inRange(hsv, lower_range, upper_range))
      
    image, cnts, hier = cv2.findContours(mask.copy(), 1, 2)
    best = None
    for c in cnts:
      M = cv2.moments(c,True)
      if(M['m00'] >20):
        found = True
        cX = int(M["m10"] / M["m00"]) 
        cY = int(M["m01"] / M["m00"])
        if(cY>len(image)/2):
          continue    

        # get yellow ling immediately to right and left
        pt = in_line_with_yellow_contours(im,cX,cY,yellow_contours,75)
        if(pt==False):
          continue

        cY = pt

        if(best is None or cY>best[2]):
          best = [c,cX,cY,M['m00']]

    
    if(best is not None):
 
      if(len(red_y_recent_avg)<RECENT_AVG_LEN): 
        red_y_recent_avg.append(best[2])
      else:
        tmp = best[2]
        best[2] = int((float(sum(red_y_recent_avg))/(RECENT_AVG_LEN+1))+(float(best[2])/(RECENT_AVG_LEN+1)))

        red_y_recent_avg.append(tmp)
        red_y_recent_avg.remove(red_y_recent_avg[0])
    

    if(best is not None):
      red_x = best[1]
      if(draw):
        cv2.circle(im, (best[1], best[2]), 3, (0, 255, 0), -1)
        cv2.putText(im, "RED: "+str(best[3]), (best[1] - 20, best[2] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)