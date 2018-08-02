import cv2,sys,glob,os
import numpy as np

def find_blue_lines(im):
  im = im.copy()
  r,g,b,range_ = (80,160,160,50)
  
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  lower_range = np.array([r-range_, g-range_, b-range_], dtype=np.uint8)
  upper_range = np.array([r+range_, g+range_, b+range_], dtype=np.uint8)
  mask = cv2.inRange(hsv, lower_range, upper_range)

  
  #cv2.imshow('im',im)
  #cv2.moveWindow('im',400,200)
  cv2.imshow('mask',mask)
  cv2.moveWindow('mask',400,550)
  
  image, cnts, hier = cv2.findContours(mask.copy(), 1, 2)
  moment_info = []
  all_info = []
  cp = im.copy()
  for c in cnts:
    M = cv2.moments(c,True)
    if(M['m00'] >=1 and M['m00']<35):
      cX = int(M["m10"] / M["m00"]) 
      cY = int(M["m01"] / M["m00"])
      moment_info.append((cX,cY,M['m00']))
      cv2.drawContours(im, [c], -1, (255, 0, 0), 2)
      cv2.circle(im, (cX, cY), 1, (0, 255, 0), -1)
      cv2.putText(im, "center", (cX - 20, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if(M['m00'] != 0):
      cX = int(M["m10"] / M["m00"]) 
      cY = int(M["m01"] / M["m00"])
      all_info.append(M)
      cv2.drawContours(cp, [c], -1, (255, 0, 0), 2)
      cv2.circle(cp, (cX, cY), 1, (0, 255, 0), -1)
      cv2.putText(cp, "center", (cX - 20, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  return (im,cp,moment_info,all_info,(len(im),len(im[0])))


def find_red_line(im):
  return im

def find_goal_lines(im):
  return im

def draw_blue_lines(im,moment_info):
  for x,y,area in moment_info:
    cv2.circle(im,(x,y),3,(0,255,0),-1)
    cv2.putText(im, "blue line", (x - 20, y - 20),
    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

  return im

def draw_red_line(im,moment_info):
  for x,y,area in moment_info:
    cv2.circle(im,(x,y),3,(0,0,255),-1)
  return im

def draw_goal_lines(im,moment_info):
  for x,y,area in moment_info:
    cv2.circle(im,(x,y),3,(0,0,255),-1)
  return im


    

lst = os.listdir(os.fsencode('../Frame_Images/2018 cornell 0223'))
file_num=0
while file_num < len(lst):
  file = lst[file_num]
  filename = os.fsdecode(file)
  print(filename)
  im = cv2.imread('../Frame_Images/2018 cornell 0223/'+filename)
  im = cv2.resize(im,None,fx=.25, fy=.25)

  im_b,cp_b,moment_info_b,all_info_b,(h,w) = find_blue_lines(im)
  #im_r,cp_r,moment_info_r,all_info_r = find_red_line(im)
  #im_g,cp_g,moment_info_g,all_info_g = find_goal_lines(im)

  print(moment_info_b)
  #im = cv2.resize(im,None,fx=4, fy=4)
  im = draw_blue_lines(im,moment_info_b)


  cv2.imshow('im', im)
  #cv2.moveWindow('img',900,200)
  k = cv2.waitKey(0)
  if(k==112): # press p for info on what was drawn
    print(moment_info_b)
    q = cv2.waitKey(0)
    file_num-=1

  if(k==97): # press a for info on all moments
    for M in all_info_b:
      for m in M:
        print(str(m)+":",M[m])
        print()
    cv2.imshow('cp_b',cp_b)
    q = cv2.waitKey(0)
    cv2.destroyWindow('cp_b')
    q = cv2.waitKey(0)
    file_num-=1

  if(k==98):
    file_num -=2


  if(k==27):
    break

  file_num+=1
    
cv2.destroyAllWindows()

