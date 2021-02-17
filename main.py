import cv2
import numpy as np
import collections 
videoFile = '../video.mp4'

q = collections.deque()
qimg = collections.deque()
cap = cv2.VideoCapture(videoFile)
index = 1
while cap.isOpened():
  # Read current Image
  ret, frame = cap.read()
  if ret == True:
    # Color to Gray Image
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Need 2 frames more
    if index >= 2:
      # Query Index
      img_ref = qimg.pop()
      img_cur = gframe

      orb = cv2.ORB_create()
      pts = cv2.goodFeaturesToTrack(img_cur, 3000, qualityLevel=0.01, minDistance=3)
      kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1], _size=20) for f in pts]
      kps_cur,des_cur = orb.compute(gframe,kps)
      # matching 
      # Need 3 datas more
      if index >= 3:
        kps_ref, des_ref = q.pop() 
        pts_cur = pts
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des_ref, des_cur, k=2) # query, train
        p_cur,p_ref = [],[]
        for m,n in matches:
          #print(m.distance,n.distance)
          if m.distance < n.distance*0.3:
            query_idx = m.queryIdx
            train_idx = m.trainIdx
            train_x, train_y = kps_cur[train_idx].pt[0], kps_cur[train_idx].pt[1]
            query_x, query_y = kps_ref[query_idx].pt[0], kps_ref[query_idx].pt[1]
            #print(train_x,train_y)
            #cv2.circle(frame, (int(train_x),int(train_y)), color=(0,255,0), radius=3) # green, past feature
            cv2.circle(frame, (int(query_x),int(query_y)), color=(0,0,255), radius=3) # red, current feature
            cv2.line(frame, (int(train_x),int(train_y)), (int(query_x),int(query_y)),color=(255,0,0)) # green to red line
         
      # Past Keypoints, Descriptors
      q.append([kps_cur,des_cur])
      cv2.imshow('frame',frame)
    
    qimg.append(gframe)
 
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    index += 1
  else:
    break
cap.release()
cv2.destroyAllWindows()

