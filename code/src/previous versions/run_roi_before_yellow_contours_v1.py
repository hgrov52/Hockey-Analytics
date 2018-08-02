import cv2,os,sys
import numpy as np
sys.path.append('../../research code/color filtering/')
import find_color_filter

"""

find yellow line and compare heights on the image

x,y,w,h = cv2.boundingRect(cnt)

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
    
vidcap = cv2.VideoCapture('../../../data/video/ACHA_vid.mp4')

Rr2 = None
blue,yellow,red = True,True,True
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
          r,g,b,range_ = find_color_filter.find_color_values(image,operation = 'sub')
          outer=True
          break
      if(outer):
        break
else:
  Br,Bg,Bb,Brange_ = (136, 130, 150, 25) # blue (use 15 bottom threshold)
  Yr, Yg, Yb, Yrange_ = (62, 90, 206, 42) # yellow (use 1000 bottom threshold)
  Rr, Rg, Rb, Rrange_ = (150, 94, 118, 16) # red line |clear|
  Rr2,Rg2,Rb2,Rrange2 = (154, 42, 164, 14) # red line |blur|
  


# Loop to test color vals
# =========================================
lst = sorted(os.listdir(os.fsencode('../../../data/frames/continuous/ACHA UNH/')))
file_num = 0
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
        cv2.drawContours(im, [c], -1, (255, 0, 0), 2)
        cv2.circle(im, (cX, cY), 1, (0, 255, 0), -1)
        cv2.putText(im, str(M['m00']), (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #gray = cv2.cvtColor(Ymask,cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(image,0,0,apertureSize = 5)
    cv2.imshow('Ymask',edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,20)
    if(lines is not None):
      best = None
      print(len(lines)) 
      for line in lines:
        """
        for rho,theta in line:
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          cv2.circle(im,(x2,y2),5,(0,0,255),-1)
          print(y2)
        """
        for x1,y1,x2,y2 in line:
          cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)


          #if(best == None or y1>best[1]):
            #best = (x1,y1,x2,y2)
      #print("best:",best[3])
      #cv2.line(im,(best[0],best[1]),(best[2],best[3]),(0,0,255),2)
      print()


  # =================================
  # generate red mask
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

        if(best is None or cY>best[2]):
          best = (c,cX,cY,M['m00'])

    if(best is not None):
      cv2.drawContours(im, [best[0]], -1, (255, 0, 0), 2)
      cv2.circle(im, (best[1], best[2]), 1, (0, 255, 0), -1)
      cv2.putText(im, str(best[3]), (best[1] - 20, best[2] - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          


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

        if(best == [None,None]):
          best[0] = (c,cX,cY,M['m00'])
          continue

        # get yellow ling immediately to right and left
        

        # if have best candidate for left and right already
        if(best[0]!=None and best[1]!=None):
          # if close to left candidate on x axis and is lower on screen
          if(cX in range(best[0][1]-5,best[0][1]+5) and cY>best[0][2]):
            # we found a better left candidate
            best[0] = (c,cX,cY,M['m00'])
          # if close to right candidate on x axis and is lower on screen
          elif(cX in range(best[1][1]-5,best[1][1]+5) and cY>best[1][2]):
            # we found a better right candidate
            best[1] = (c,cX,cY,M['m00'])

        # if we have left candidate and we have new candidate that is far left, we switch candidates
        elif(best[0]!=None and cX>best[0][1] and cX not in range(best[0][1],best[0][1]+10)):
          # we switch prev left candidate to right 
          best[1] = best[0]
          # and we found an actual left candidate
          best[1] = (c,cX,cY,M['m00'])

        # if only left candidate has been selected and far enough right of left candidate
        elif(best[0]!=None and cX<best[0][1] and cX not in range(best[0][1]-10,best[0][1])):
          # we found a right candidate
          best[1] = (c,cX,cY,M['m00'])

    # if we only have a left candidate, make sure it should be the right candidate
    #if(best[0]!=None and best[1] == None and best[0][1]<len):
      


    if(best != [None,None]):
      if(best[0]!=None):
        cv2.putText(im, "LEFT: " + str(int(best[0][3])), (best[0][1],best[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        cv2.drawContours(im, [best[0][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[0][1], best[0][2]), 1, (0, 255, 0), -1)
      if(best[1]!=None):
        cv2.putText(im, "RIGHT: " + str(int(best[1][3])), (best[1][1],best[1][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
        cv2.drawContours(im, [best[1][0]], -1, (255, 0, 0), 2)
        cv2.circle(im, (best[1][1], best[1][2]), 1, (0, 255, 0), -1)

        


  
  
  cv2.imshow('img', im)
  cv2.moveWindow('img',0,0)
  k = cv2.waitKey(0)
  """
  if(k==112): # press p for info on what was drawn
    for m in moment_info:
      print(m)
      q = cv2.waitKey(0)

  if(k==97): # press a for info on all moments
    cp = im.copy()
    for c in cnts:
      M = cv2.moments(c,True)
      if(M['m00'] != 0):
        cX = int(M["m10"] / M["m00"]) 
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(cp, [c], -1, (255, 0, 0), 2)
        cv2.circle(cp, (cX, cY), 1, (0, 255, 0), -1)
        cv2.putText(cp, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print()
        print(cX)
        print(cY)
        for m in M:
          print(str(m)+":",M[m])
    cv2.imshow('cp',cp)
    q = cv2.waitKey(0)
    cv2.destroyWindow('cp')
  """

  if(k==98):
    file_num -=2


  if(k==27):
    break

  file_num+=1
    
cv2.destroyAllWindows()

