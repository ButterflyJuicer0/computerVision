import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img1=cv.imread('./images/img1.jpg')
img2=cv.imread('./images/img2.jpg')
print(img1.shape)
grayImage1=cv.cvtColor(img1,cv.COLOR_RGBA2GRAY)
grayImage2=cv.cvtColor(img2,cv.COLOR_RGBA2GRAY)
#edge detection
sift=cv.SIFT.create()
orb=cv.ORB.create()
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)


#Create feature matcher
index_params=dict(algorithm=1,trees=5)
search_params=dict(checks=50)
flann=cv.FlannBasedMatcher(index_params,search_params)



bf=cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches=flann.match(des1,des2)


matches=sorted(matches,key=lambda x:x.distance)
#Match feature points
N=50
matched_image=cv2.drawMatches(img1,kp1,img2,kp2,matches[:N],None,flags=2)
src_pts=np.float32([kp1[m.queryIdx].pt for m in matches[:N]]).reshape(-1,1,2)
dst_pts=np.float32([kp2[m.trainIdx].pt for m in matches[:N]]).reshape(-1,1,2)

H,status=cv.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
height,width,channels=img2.shape
im1Reg=cv.warpPerspective(img1,H,(width,height))
cv2.imshow("Registerd",im1Reg)


cv.imshow('orb',matched_image)
cv.waitKey(0)

