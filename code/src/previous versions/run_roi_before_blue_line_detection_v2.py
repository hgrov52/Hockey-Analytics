import cv2,os,collections,sys
import numpy as np
sys.path.append('../utils/')
import find_color_filter

"""
REVISE:
x,y,w,h = cv2.boundingRect(cnt)


TODO:
take out random red line detection by getting past 
positions and if it hasnt shown up in some range of 
that area in N frames, then it is an inconsistency


"""

def init_values(im,file_num):
  
  cv2.imshow('im',im)
  k = cv2.waitKey(0)
  cv2.destroyWindow('im')
  if(k == 32): # space bar iterate by 1
    return None,None,None,None,False,file_num+1
  if(k == 120): # x iterate by 50
    return None,None,None,None,False,file_num+50
  if(k == 98): # b backwards iterate by 1
    return None,None,None,None,False,file_num+-1
  if(k == 13):
    return find_color_filter.find_color_values(im),True
    

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


vidcap = cv2.VideoCapture('../../../data/video/ACHA_vid.mp4')

Rr2 = None
blue,yellow,red,goal_line,blue_line = True,True,True,False,True
success = True
outer = False
detect_colors = False
if(detect_colors):
  # Loop to get color vals
  # ========================================
  while success:
    success,image = vidcap.read()
    cv2.imshow('image',image)
    k = cv2.waitKey(1)
    if(k == 27):
      cv2.destroyAllWindows()
      exit(1)
    if(k == 32):
      while(1):
        success,image = vidcap.read()
        cv2.imshow('image',image)
        k = cv2.waitKey(0)
        if(k == 27):
          break
        if(k == 32):
          continue
        if(k == 13):
          r,g,b,range_ = find_color_filter.find_color_values(image,operation = 'd')
          outer=True
          break
      if(outer):
        break
else:
  Br,Bg,Bb,Brange_ = (136, 130, 150, 25) # blue (use 15 bottom threshold)
  Yr, Yg, Yb, Yrange_ = (62, 90, 206, 42) # yellow (use 1000 bottom threshold)
  Rr, Rg, Rb, Rrange_ = (150, 94, 118, 16) # red line |clear|
  Rr2,Rg2,Rb2,Rrange2 = (154, 42, 164, 14) # red line |blur|
  goal_line_hsv_values = [(146, 44, 118, 10), # clear/highly zoomed:
                           # blurry/from distance:
                          ]

# Loop to test color vals
# =========================================
lst = sorted(os.listdir(os.fsencode('../../../data/frames/continuous/ACHA UNH/')))
file_num = 0
prev_thetas = []
RECENT_AVG_LEN = 2
red_y_recent_avg = []
goal_y_recent_avg = []
blue_y_recent_avg_L = []
blue_y_recent_avg_R = []

TRANSFORM_X = 200
TRANSFORM_Y = 600
TRANSFORM_X_SHAPE = 20
TRANSFORM_Y_SHAPE = 5
MAX_SHAPE = 400

while file_num < len(lst):
  file = lst[file_num]
  filename = os.fsdecode(file)
  if(filename.endswith('.jpg') == False):
    file_num+=1
    continue

  im = cv2.imread('../../../data/frames/continuous/ACHA UNH/'+filename)
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)



  # =================================
  # generate yellow mask
  yellow_contours = []
  if(yellow):
    lower_range = np.array([Yr-Yrange_, Yg-Yrange_, Yb-Yrange_], dtype=np.uint8)
    upper_range = np.array([Yr+Yrange_, Yg+Yrange_, Yb+Yrange_], dtype=np.uint8)
    Ymask = cv2.inRange(hsv, lower_range, upper_range)
    image, cnts, hier = cv2.findContours(Ymask.copy(), 1, 2)
    for c in cnts:
      M = cv2.moments(c,True)
      if(M['m00'] >500):
        cX = int(M["m10"] / M["m00"]) 
        cY = int(M["m01"] / M["m00"])
        #cv2.drawContours(im, [c], -1, (255, 0, 0), 2)
        #cv2.circle(im, (cX, cY), 1, (0, 255, 0), -1)
        #v2.putText(im, str(M['m00']), (cX - 20, cY - 20),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #gray = cv2.cvtColor(Ymask,cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(image,0,0,apertureSize = 5)
    #cv2.imshow('Ymask',edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,20)
    lines2 = cv2.HoughLines(edges,1,np.pi/180,150)

    radial = {}
    if(lines2 is not None):
      for line in lines2:
        for rho,theta in line:
          radial.setdefault(theta,[]).append(rho)
      
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
      
      #               pos  neg
      lines_to_draw = [None,None] 

      if(len(pos_duplicates.keys())==1):
        theta = list(pos_duplicates.keys())[0] # bc its the only one
        lines_to_draw[0] = [theta,max(pos_duplicates[theta])]
      if(len(neg_duplicates.keys())==1):
        theta = list(neg_duplicates.keys())[0] # bc its the only one
        lines_to_draw[1] = [theta,max(neg_duplicates[theta])]

      if(lines_to_draw[0] == None and len(positive_slope.keys())==1):
        theta = list(positive_slope.keys())[0]
        lines_to_draw[0] = [theta,max(positive_slope[theta])]
      if(lines_to_draw[1] == None and len(negative_slope.keys())==1):
        theta = list(negative_slope.keys())[0]
        lines_to_draw[1] = [theta,max(negative_slope[theta])]

      if(lines_to_draw[0] == None and len(positive_slope.keys())>0):
        theta = max(positive_slope.keys())
        lines_to_draw[0] = [theta,max(positive_slope[theta])]
      if(lines_to_draw[1] == None and len(negative_slope.keys())>0):
        theta = max(negative_slope.keys())
        lines_to_draw[1] = [theta,max(negative_slope[theta])]

      if(lines_to_draw[0]!=None and lines_to_draw[1]!=None):
        A = []
        B = []
        for theta,rho in lines_to_draw:
          #print(theta*180/np.pi,rho)
          A.append([np.cos(theta),np.sin(theta)])
          B.append([rho])
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1500*(-b))
          y1 = int(y0 + 1500*(a))
          x2 = int(x0 - 1500*(-b))
          y2 = int(y0 - 1500*(a))
          cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

        # solve for intersection point
        A = np.array(A)
        B = np.array(B)
        x00, y00 = np.linalg.solve(A, B)
        x00, y00 = int(np.round(x00)), int(np.round(y00))
        cv2.circle(im,(x00,y00),5,(0,255,0),-1)

        # 1.414 theta threshold found by averaging 
        # the averages between the two measured ranges 
        # of positive sloped thetas
        # 1.3788 - 1.4661
        # 1.3614 - 1.4486
        #        ->        1.42245 & 1.405
        # so if theta is greater, rotate warping box cc once
        if(lines_to_draw[0][0]>1.414):
          #TRANSFORM_X = 200
          #TRANSFORM_Y = 200
          a_pos = np.cos(lines_to_draw[0][0])
          b_pos = np.sin(lines_to_draw[0][0])
          a_neg = np.cos(lines_to_draw[1][0])
          b_neg = np.sin(lines_to_draw[1][0])

          x01 = int(x00 + TRANSFORM_X*(-b_pos))
          y01 = int(y00 + TRANSFORM_X*(a_pos))
          x10 = int(x00 - TRANSFORM_Y*(-b_neg))
          y10 = int(y00 - TRANSFORM_Y*(a_neg))
          cv2.circle(im,(x01,y01),5,(0,255,0),-1)
          cv2.circle(im,(x10,y10),5,(0,255,0),-1)

          x11 = int(x01 - TRANSFORM_Y*(-b_neg))
          y11 = int(y01 - TRANSFORM_Y*(a_neg))
          cv2.circle(im,(x11,y11),5,(0,255,0),-1)

          tmp_x00,tmp_y00 = x00,y00
          x00,y00 = x01,y01
          x01,y01 = x11,y11
          x11,y11 = x10,y10
          x10,y10 = tmp_x00,tmp_y00
        else:
          #TRANSFORM_X = 200
          #TRANSFORM_Y = 600
          a_pos = np.cos(lines_to_draw[0][0])
          b_pos = np.sin(lines_to_draw[0][0])
          a_neg = np.cos(lines_to_draw[1][0])
          b_neg = np.sin(lines_to_draw[1][0])

          x01 = int(x00 + TRANSFORM_Y*(-b_pos))
          y01 = int(y00 + TRANSFORM_Y*(a_pos))
          x10 = int(x00 - TRANSFORM_X*(-b_neg))
          y10 = int(y00 - TRANSFORM_X*(a_neg))
          cv2.circle(im,(x01,y01),5,(0,255,0),-1)
          cv2.circle(im,(x10,y10),5,(0,255,0),-1)

          x11 = int(x01 - TRANSFORM_X*(-b_neg))
          y11 = int(y01 - TRANSFORM_X*(a_neg))
          cv2.circle(im,(x11,y11),5,(0,255,0),-1)

        # warp perspective 
        #MAX_SHAPE = 400
        X_SHAPE = MAX_SHAPE/TRANSFORM_X_SHAPE
        Y_SHAPE = MAX_SHAPE/TRANSFORM_Y_SHAPE
        pts1 = np.float32(((x00,y00), (x01,y01),
                  (x11,y11), (x10,y10)))
        pts2 = np.float32(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
                  (MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE)))
        M = cv2.getPerspectiveTransform(pts1,pts2)
        warp = cv2.warpPerspective(im,M,(1000,1000))
        cv2.imshow('final',warp)

      else:
        if(lines_to_draw==[None,None]):
          print("missing both")
        elif(lines_to_draw[0]==None):
          print("missing positive slope")
        elif(lines_to_draw[1]==None):
          print("missing negative slope")
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

  # =================================
  # blue line

  # =================================
  # generate red mask
  red_x = None
  if(red):
    lower_range = np.array([Rr-Rrange_, Rg-Rrange_, Rb-Rrange_], dtype=np.uint8)
    upper_range = np.array([Rr+Rrange_, Rg+Rrange_, Rb+Rrange_], dtype=np.uint8)
    Rmask = cv2.inRange(hsv, lower_range, upper_range)
    if(Rr2 is not None):
      lower_range = np.array([Rr2-Rrange2, Rg2-Rrange2, Rb2-Rrange2], dtype=np.uint8)
      upper_range = np.array([Rr2+Rrange2, Rg2+Rrange2, Rb2+Rrange2], dtype=np.uint8)
      Rmask = cv2.add(Rmask,cv2.inRange(hsv, lower_range, upper_range))
      
    image, cnts, hier = cv2.findContours(Rmask.copy(), 1, 2)
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
      #cv2.drawContours(im, [best[0]], -1, (255, 0, 0), 2)
      cv2.circle(im, (best[1], best[2]), 3, (0, 255, 0), -1)
      cv2.putText(im, "RED: "+str(best[3]), (best[1] - 20, best[2] - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          


  # =================================
  # generate blue mask
  if(blue):
    lower_range = np.array([Br-Brange_, Bg-Brange_, Bb-Brange_], dtype=np.uint8)
    upper_range = np.array([Br+Brange_, Bg+Brange_, Bb+Brange_], dtype=np.uint8)
    Bmask = cv2.inRange(hsv, lower_range, upper_range)
    image, cnts, hier = cv2.findContours(Bmask.copy(), 1, 2)
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
        cv2.putText(im, "LEFT: " + str(int(best[0][3])), (best[0][1],best[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        #cv2.drawContours(im, [best[0][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[0][1], best[0][2]), 3, (0, 255, 0), -1)
      if(best[1]!=None):
        cv2.putText(im, "RIGHT: " + str(int(best[1][3])), (best[1][1],best[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        #cv2.drawContours(im, [best[1][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[1][1], best[1][2]), 1, (0, 255, 0), -1)
    else:
      blue_y_recent_avg_L = []
      blue_y_recent_avg_R = []

  # =================================
  # generate goal line mask
  goal_line_x = None
  if(goal_line):
    goal_line_masks = []
    Gmask = None
    for r,g,b,rng in goal_line_hsv_values:
      lower_range = np.array([r-rng, g-rng, b-rng], dtype=np.uint8)
      upper_range = np.array([r+rng, g+rng, b+rng], dtype=np.uint8)
      if(Gmask is None):
        Gmask = cv2.inRange(hsv, lower_range, upper_range)
      else:
        Gmask = cv2.add(Gmask,cv2.inRange(hsv, lower_range, upper_range))

    cv2.imshow('Gmask',Gmask)

    image, cnts, hier = cv2.findContours(Gmask.copy(), 1, 2)
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
        pt = in_line_with_yellow_contours(im,cX,cY,yellow_contours,100)
        if(pt==False):
          continue

        cY = pt

        if(best is None or cY>best[2]):
          best = [c,cX,cY,M['m00']]

    if(best is not None):
 
      if(len(goal_y_recent_avg)<RECENT_AVG_LEN): 
        goal_y_recent_avg.append(best[2])
      else:
        tmp = best[2]
        best[2] = int((float(sum(goal_y_recent_avg))/(RECENT_AVG_LEN+1))+(float(best[2])/(RECENT_AVG_LEN+1)))

        goal_y_recent_avg.append(tmp)
        goal_y_recent_avg.remove(goal_y_recent_avg[0])
    

    if(best is not None):
      red_x = best[1]
      #cv2.drawContours(im, [best[0]], -1, (255, 0, 0), 2)
      cv2.circle(im, (best[1], best[2]), 3, (0, 255, 0), -1)
      cv2.putText(im, "Goal: "+str(best[3]), (best[1] - 20, best[2] - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          

  # =================================================
  cv2.imshow('img', im)
  cv2.moveWindow('img',0,0)
  k = cv2.waitKey(0)

  if(k==98):
    file_num -=2


  if(k==27):
    break

  if(k==61):
    TRANSFORM_X+=5 # =
    file_num-=1
  if(k==45):
    TRANSFORM_X-=5 # -
    file_num-=1
  if(k==93):
    TRANSFORM_Y+=5 # ]
    file_num-=1
  if(k==91):
    TRANSFORM_Y-=5 # [
    file_num-=1
  if(k==39):
    TRANSFORM_X_SHAPE+=1 # '
    file_num-=1
  if(k==59):
    TRANSFORM_X_SHAPE-=1 # ;
    file_num-=1
  if(k==47):
    TRANSFORM_Y_SHAPE+=1 # /
    file_num-=1
  if(k==46):
    TRANSFORM_Y_SHAPE-=1 # .
    file_num-=1
  if(k==44):
    MAX_SHAPE+=5 # ,
    file_num-=1
  if(k==109):
    MAX_SHAPE-=5 # m
    file_num-=1
  
  
  
  if(k==112): # p
    print("TRANSFORM_X:",TRANSFORM_X)
    print("TRANSFORM_Y:",TRANSFORM_Y)
    print("TRANSFORM_X_SHAPE:",TRANSFORM_X_SHAPE)
    print("TRANSFORM_Y_SHAPE:",TRANSFORM_Y_SHAPE)
    print("MAX_SHAPE:",MAX_SHAPE)
    file_num-=1
  
  

  file_num+=1
    
cv2.destroyAllWindows()

