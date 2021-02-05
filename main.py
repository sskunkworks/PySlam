import cv2
import numpy as np

videoFile = 'video.mp4'
cap = cv2.VideoCapture(videoFile)

while cap.isOpened():
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
cap.release()
cv2.destroyAllWindows()
