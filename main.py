import cv2
import numpy as np
import collections 
videoFile = '../video.mp4'

q = collections.deque()
cap = cv2.VideoCapture(videoFile)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
index = 1
while cap.isOpened():
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(100)
    # orb detect
    #kp = orb.detect(frame,None)
    # Use goodfeature
    pts = cv2.goodFeaturesToTrack(frame, 3000, 0.01, 10)
    kp = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in pts]
    kp2,des2 = orb.compute(frame, kp)
    q.append([kp2,des2])
    #if index >= 2:
    #  kp1, des1 = q.pop() 
    #  matches = bf.match(des1, des2)
    #  img3 = cv2.drawMatches()
    points = []
    for p in kp:
      points.append([p.pt[0], p.pt[1]])
      print(p.pt[0], p.pt[1])
      cv2.circle(frame, (int(p.pt[0]),int(p.pt[1])), color=(0,255,0), radius=3)
    #cv2.circle(frame, (u1,v1), color=(0,255,0), radius=3)
    #img2 = cv2.drawKeypoints(frame,kp,outImage=None,color=(0,255,0),flags=0)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    index += 1
  else:
    break
cap.release()
cv2.destroyAllWindows()

