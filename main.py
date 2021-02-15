import cv2
import numpy as np
import collections 
videoFile = '../video.mp4'

q = collections.deque()
qimg = collections.deque()
cap = cv2.VideoCapture(videoFile)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher()
index = 1
while cap.isOpened():
  ret, frame = cap.read()
  if ret == True:
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    # orb detect
    # Train Index
    kp2, des2 = orb.detectAndCompute(gframe,None)
    qimg.append(gframe)
    # Use goodfeature
    #pts = cv2.goodFeaturesToTrack(gframe, 3000, 0.01, 10)
    #kp = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in pts]
    #kp2,des2 = orb.compute(frame, kp)
    q.append([kp2,des2])
    if index >= 2:
      # Query Index
      kp1, des1 = q.pop() 
      img1 = qimg.pop()
      matches = bf.match(des1, des2)
      #matches = bf.knnMatch(des1,des2,k=2)
      matches = sorted(matches, key=lambda x:x.distance)
      #img3 = cv2.drawMatches(img1, kp1, gframe, kp2, matches[:10],None,flags=2) 
      #cv2.imshow('frame',img3)
      for idx in range(0,len(matches)):
        trainIdx =  matches[idx].trainIdx
        queryIdx = matches[idx].queryIdx 
        train_x = kp1[trainIdx].pt[0]
        train_y = kp1[trainIdx].pt[1]
        query_x = kp2[queryIdx].pt[0]
        query_y = kp2[queryIdx].pt[1]
        #print(train_x, query_x, train_y, query_y)
        cv2.circle(frame, (int(train_x),int(train_y)), color=(0,255,0), radius=3)
        cv2.circle(frame, (int(query_x),int(query_y)), color=(0,0,255), radius=3)
        cv2.line(frame, (int(train_x),int(train_y)), (int(query_x)+5,int(query_y)),color=(255,0,0))
      cv2.imshow('frame', frame)
    points = []
    #for p in kp:
    #  points.append([p.pt[0], p.pt[1]])
    #  #print(p.pt[0], p.pt[1])
    #  cv2.circle(frame, (int(p.pt[0]),int(p.pt[1])), color=(0,255,0), radius=3)
    #cv2.circle(frame, (u1,v1), color=(0,255,0), radius=3)
    #img2 = cv2.drawKeypoints(frame,kp,outImage=None,color=(0,255,0),flags=0)
    #cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    index += 1
  else:
    break
cap.release()
cv2.destroyAllWindows()

