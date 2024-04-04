import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img1=cv.imread('./images/img1.jpg')
img2=cv.imread('./images/img2.jpg')
print(img1.shape)
grayImage1=cv.cvtColor(img1,cv.COLOR_RGBA2GRAY)
grayImage2=cv.cvtColor(img2,cv.COLOR_RGBA2GRAY)
#Key point detection with sift
sift=cv.SIFT.create()

kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

##Create feature matcher
#bf Matcher
bf=cv.BFMatcher(cv.NORM_L2, crossCheck=True)
bfmatches=bf.match(des1,des2)
bfmatches=sorted(bfmatches,key=lambda x:x.distance)

#Flann Matcher
index_params=dict(algorithm=1,trees=5)
search_params=dict(checks=50)
flann=cv.FlannBasedMatcher(index_params,search_params)

N=30
#Homography for bfmatch
src_pts=np.float32([kp1[m.queryIdx].pt for m in bfmatches[:N]]).reshape(-1,1,2)
dst_pts=np.float32([kp2[m.queryIdx].pt for m in bfmatches[:N]]).reshape(-1,1,2)
H,status=cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)
h,w,channels=img1.shape
im1reg=cv.warpPerspective(img1,H,(w,h))
cv.imshow('warpped', im1reg)
cv2.waitKey(0)



