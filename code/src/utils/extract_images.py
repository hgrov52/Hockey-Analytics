import cv2
vidcap = cv2.VideoCapture('../../../data/video/Chances.mp4')

count = 0
success = True


while success:
  success,image = vidcap.read()

  if(count>0 and count%30==0):
    cv2.imwrite("../../../data/frames/continuous/chances/frame%d.jpg" % (5000-count), image)
  count += 1
  print(count)
