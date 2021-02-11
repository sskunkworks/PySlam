import cv2
import numpy as np

videoFile = '../video.mp4'

cap = cv2.VideoCapture(videoFile)


while cap.isOpened():
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(100)
    # orb detect
    #kp = orb.detect(frame,None)
    # Use goodfeature
    pts = cv2.goodFeaturesToTrack(frame, 25,0.01, 10)
    kp = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in pts]
    kp,des = orb.compute(frame, kp)
    img2 = cv2.drawKeypoints(frame,kp,outImage=None,color=(0,255,0),flags=0)
    cv2.imshow('frame', img2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
cap.release()
cv2.destroyAllWindows()

