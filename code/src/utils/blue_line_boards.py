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

def define_points(im, yellow_contours, blue_y_recent_avg_L, blue_y_recent_avg_R, RECENT_AVG_LEN,draw = False):
    r,g,b,rng = (136, 130, 150, 25) # blue (use 15 bottom threshold)
    lower_range = np.array([r-rng, g-rng, b-rng], dtype=np.uint8)
    upper_range = np.array([r+rng, g+rng, b+rng], dtype=np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    image, cnts, hier = cv2.findContours(mask.copy(), 1, 2)
    best = [None,None]
    for c in cnts:
      M = cv2.moments(c,True)
      if(M['m00'] >15):
        cX = int(M["m10"] / M["m00"]) 
        cY = int(M["m01"] / M["m00"])
        if(cY>len(image)/2):
          continue

        # get yellow ling immediately to right and left
        pt = in_line_with_yellow_contours(im,cX,cY,yellow_contours)

        if(pt==False):
          continue

        cY = pt

        

        if(best == [None,None]):
          best[0] = [c,cX,cY,M['m00']]
          continue

        

        # if have best candidate for left and right already
        if(best[0]!=None and best[1]!=None):
          # if close to left candidate on x axis and is lower on screen
          if(cX in range(best[0][1]-5,best[0][1]+5) and cY>best[0][2]):
            # we found a better left candidate
            best[0] = [c,cX,cY,M['m00']]
          # if close to right candidate on x axis and is lower on screen
          elif(cX in range(best[1][1]-5,best[1][1]+5) and cY>best[1][2]):
            # we found a better right candidate
            best[1] = [c,cX,cY,M['m00']]

        # if we have left candidate and we have new candidate that is far left, we switch candidates
        elif(best[0]!=None and cX>best[0][1] and cX not in range(best[0][1],best[0][1]+10)):
          # we switch prev left candidate to right 
          best[1] = best[0]
          # and we found an actual left candidate
          best[1] = [c,cX,cY,M['m00']]

        # if only left candidate has been selected and far enough right of left candidate
        elif(best[0]!=None and cX<best[0][1] and cX not in range(best[0][1]-10,best[0][1])):
          # we found a right candidate
          best[1] = [c,cX,cY,M['m00']]

    # if we only have a left candidate, make sure it should be the right candidate
    #if(best[0]!=None and best[1] == None and best[0][1]<len):
      
    # avg for stabilization
    if(best[0]!=None):

      if(len(blue_y_recent_avg_L)<RECENT_AVG_LEN): 
        blue_y_recent_avg_L.append(best[0][2])
      else:
        tmp = best[0][2]
        best[0][2] = int((float(sum(blue_y_recent_avg_L))/(RECENT_AVG_LEN+1))+(float(best[0][2])/(RECENT_AVG_LEN+1)))
        blue_y_recent_avg_L.append(tmp)
        blue_y_recent_avg_L.remove(blue_y_recent_avg_L[0])
    if(best[1]!=None):

      if(len(blue_y_recent_avg_R)<RECENT_AVG_LEN): 
        blue_y_recent_avg_R.append(best[1][2])
      else:
        tmp = best[1][2]
        best[1][2] = int((float(sum(blue_y_recent_avg_R))/(RECENT_AVG_LEN+1))+(float(best[1][2])/(RECENT_AVG_LEN+1)))
        blue_y_recent_avg_R.append(tmp)
        blue_y_recent_avg_R.remove(blue_y_recent_avg_R[0])


    if(best != [None,None]):
      if(best[0]!=None):
        cv2.putText(im, "Right: " + str(int(best[0][3])), (best[0][1],best[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        #cv2.drawContours(im, [best[0][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[0][1], best[0][2]), 3, (0, 255, 0), -1)
      if(best[1]!=None):
        cv2.putText(im, "Left: " + str(int(best[1][3])), (best[1][1],best[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        #cv2.drawContours(im, [best[1][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[1][1], best[1][2]), 3, (0, 255, 0), -1)
    else:
      blue_y_recent_avg_L = []
      blue_y_recent_avg_R = []