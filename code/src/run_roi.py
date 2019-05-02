import cv2,os,sys
import numpy as np
from utils import find_color_filter
from utils import on_ice_blue_line
from utils import on_ice_yellow_line
from utils import warp_image
from utils import get_bottom_boards
from utils import blue_line_boards
from utils import red_line_boards
from utils import goal_line_boards

"""
REVISE:
x,y,w,h = cv2.boundingRect(cnt)


TODO:
take out random red line detection by getting past 
positions and if it hasnt shown up in some range of 
that area in N frames, then it is an inconsistency


"""

vidcap = cv2.VideoCapture('../../data/video/ACHA_vid.mp4')

Rr2 = None
blue_points,yellow,red,goal_line,blue_line,bottom = True,True,True,False,True,False
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

  

# Loop to test color vals
# =========================================
lst = sorted(os.listdir(os.fsencode('../../data/frames/continuous/ACHA UNH/')))
file_num = 0
prev_thetas = []
RECENT_AVG_LEN = 2
red_y_recent_avg = []
goal_y_recent_avg = []
blue_y_recent_avg_L = []
blue_y_recent_avg_R = []

TRANSFORM_X = 200
TRANSFORM_Y = 600
TRANSFORM_X_SHAPE = 5
TRANSFORM_Y_SHAPE = 5
MAX_SHAPE = 400

while file_num < len(lst):
  file = lst[file_num]
  filename = os.fsdecode(file)
  if(filename.endswith('.jpg') == False):
    file_num+=1
    continue

  im = cv2.imread('../../data/frames/continuous/ACHA UNH/'+filename)
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

  if(bottom):
    get_bottom_boards.define_line(im,draw=True)

  # =================================
  # blue line
  if(blue_line):
    blue_lines = on_ice_blue_line.define_lines(im,draw=False)

    while(None in blue_lines):
      blue_lines.remove(None)

    im_rev = im[:,:,::-1]

    blue_lines2 = on_ice_blue_line.define_lines(im_rev,draw=True)

    
      
  # =================================
  # yellow line
  if(yellow):
    info = [TRANSFORM_X,TRANSFORM_Y,TRANSFORM_X_SHAPE,TRANSFORM_Y_SHAPE,MAX_SHAPE]
    yellow_contours,warp_lines = on_ice_yellow_line.define_lines(im,draw=True)
    
    if(len(blue_lines)==2):
      warp_image.get_ratio(blue_lines,warp_lines)

    for line in blue_lines:
      if(line != None):
        warp_lines.append(line)

    warp_image.warp_image(im,warp_lines,draw=True, params = info)

  # =================================
  # generate red mask
  red_x = None
  if(red):
    red_line_boards.define_point(im, yellow_contours, red_x, red_y_recent_avg, RECENT_AVG_LEN, draw=True)
          


  # =================================
  # generate blue mask
  if(blue_points):
    blue_line_boards.define_points(im, yellow_contours, blue_y_recent_avg_L, blue_y_recent_avg_R, RECENT_AVG_LEN, draw=True)

  # =================================
  # generate goal line mask
  goal_line_x = None
  if(goal_line):
    goal_line_boards.define_points(im, yellow_contours
      ,draw=True)

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

