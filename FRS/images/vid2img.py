import cv2
import time


vidcap = cv2.VideoCapture('id - face turn.MOV')
success,image = vidcap.read()
count = 0
while success and count < 10:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  time.sleep(0.5)
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
