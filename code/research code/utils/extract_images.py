import cv2
#vidcap = cv2.VideoCapture('/mnt/20180223_hockey_cornell_cam1.mp4')
vidcap = cv2.VideoCapture('/home/groveh/Documents/Research/Frame_Images/ACHA UNH/ACHA_vid.mp4')

count = 0
success = True


while success:
  success,image = vidcap.read()

  if(count>0):
    #cv2.imwrite("/home/groveh/Documents/Research/Frame_Images/2018 cornell 0223/frame%d.jpg" % count, image)     # save frame as JPEG file      
    cv2.imwrite("/home/groveh/Documents/Research/Frame_Images/ACHA UNH/frame%d.jpg" % count, image)
  count += 1
