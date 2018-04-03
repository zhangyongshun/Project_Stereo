import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import *


#inner corners on the chessboard
pattern_size = (9, 6)

#suoppose plane of the chessboard in world coordinates is "Z = 0", and encode the points on chessboard.
single_pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
single_pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

#store the corner points in chessboard and corresponding points in image plane.
pattern_points = []
left_image_points = []

#load the images
images = glob.glob('Project_Stereo_left\left\*.jpg')

for fimage in images:
    img = cv.imread(fimage)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray_image, pattern_size)
    if found:
        #use function cornerSubPix to refine corner coordinates to subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)
        pattern_points.append(single_pattern_points)
        left_image_points.append(corners)

h, w = gray_image.shape


_, leftCameraMatrix, leftDistCoeffis, leftRvecs, leftTvecs = cv.calibrateCamera(pattern_points, left_image_points, (w, h), None,None)
leftR, _ = cv.Rodrigues(leftRvecs[2])

#read the left image
left_image = cv.imread('Project_Stereo_left\left\left03.jpg')
left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)

#store the corner points in chessboard and corresponding points in image plane.
right_image_points = []
pattern_points = []
#load the images
images = glob.glob('Project_Stereo_right/right/*.jpg')
for fimage in images:
    img = cv.imread(fimage)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray_image, pattern_size)
    if found:
        #use function cornerSubPix to refine corner coordinates to subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)
        pattern_points.append(single_pattern_points)
        right_image_points.append(corners)
h, w = gray_image.shape

_, rightCameraMatrix, rightDistCoeffis, rightRvecs, rightTvecs = cv.calibrateCamera(pattern_points, right_image_points, (w, h), None, None)
rightR, _ = cv.Rodrigues(rightRvecs[2])

right_image = cv.imread('Project_Stereo_right/right/right03.jpg')
right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)


#rotation matrix
R = np.dot(rightR, leftR.T)
#traslation vector from left to right
T = rightTvecs[2] - np.dot(R, leftTvecs[2])

#use the opencv function stereoCalibrate(), stereoRectify() in order to make comparison with the matrices get by derivation
h, w = right_image.shape
_,left_mtx, left_dis,right_mtx,right_dis,stereoR,stereoT,stereoE,stereoF = cv.stereoCalibrate(pattern_points, left_image_points,right_image_points,leftCameraMatrix,leftDistCoeffis,rightCameraMatrix,rightDistCoeffis,(w,h))
left_r, right_r, left_p, right_p, q,_,_ = cv.stereoRectify(leftCameraMatrix, leftDistCoeffis, rightCameraMatrix, rightDistCoeffis, (w, h), R, T, 0,0,(w,h))

#draw the orgin photos
plt.subplot(3, 2, 1)
plt.title('Origin Left Image')
#cv.circle(left_image, (floor(leftCorner[0]), floor(leftCorner[1])), 4, (0, 255, 0))
plt.imshow(left_image)
plt.axis('off')
plt.subplot(3, 2, 2)
plt.title('Origin Right Image')
plt.imshow(right_image)
plt.axis('off')

#use the fundamental matrix calculated by opencv function stereoCalibrate() to draw the epipolar lines
rightNewCameraMatrix, _ = cv.getOptimalNewCameraMatrix(rightCameraMatrix, rightDistCoeffis, (w, h), 0, (w, h))
leftNewCameraMatrix, _ = cv.getOptimalNewCameraMatrix(leftCameraMatrix, leftDistCoeffis, (w, h), 0, (w, h))

#calculate the roation matrix rl and rr
vectorR = cv.Rodrigues(R)
t = vectorR[0]
rightVR = -1 * vectorR[0] / 2
leftVR = vectorR[0] / 2
t = np.cross((T/np.linalg.norm(T,ord = 2)).T, np.array([-1*T[1]/(T[0]**2+T[1]**2)**0.5,T[0]/(T[0]**2+T[1]**2)**0.5, 0]))
Rrect = np.array([np.array([i[0] for i in T/np.linalg.norm(T,ord = 2).T]), np.array([i[0] for i in [-1*T[1]/(T[0]**2+T[1]**2)**0.5,T[0]/(T[0]**2+T[1]**2)**0.5, np.array([0])]]), np.array([-1*i[0] for i in np.cross((T/np.linalg.norm(T,ord = 2)).T, np.array([-1*T[1]/(T[0]**2+T[1]**2)**0.5,T[0]/(T[0]**2+T[1]**2)**0.5, 0]))[0]])])
leftHalfR = cv.Rodrigues(leftVR)
rightHalfR = cv.Rodrigues(rightVR)

#map formula
leftMapx, leftMapy = cv.initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffis,np.dot(Rrect,leftHalfR[0]), leftNewCameraMatrix, left_image.shape[::-1], cv.CV_16SC2)
rightMapx, rightMapy = cv.initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffis, np.dot(Rrect, rightHalfR[0]), leftNewCameraMatrix, right_image.shape[::-1],cv.CV_16SC2)
#the left and right image which are rectified by the matrices calculated
left_new_image = cv.remap(left_image, leftMapx, leftMapy, cv.INTER_LINEAR, cv.BORDER_DEFAULT, 0)
right_new_image = cv.remap(right_image, rightMapx, rightMapy, cv.INTER_LINEAR, cv.BORDER_DEFAULT, 0)

#the matrices calculated by opencv function stereoCalibrate() in order to make comparison
stereoLeftMapx, stereoLeftMapy = cv.initUndistortRectifyMap(left_mtx, left_dis,left_r, left_p, left_image.shape[::-1], cv.CV_16SC2)
stereoRightMapx, stereoRightMapy = cv.initUndistortRectifyMap(right_mtx, right_dis, right_r, right_p, right_image.shape[::-1],cv.CV_16SC2)
stereo_left_new_image = cv.remap(left_image, stereoLeftMapx, stereoLeftMapy, cv.INTER_LINEAR, cv.BORDER_DEFAULT, 0)
stereo_right_new_image = cv.remap(right_image, stereoRightMapx, stereoRightMapy, cv.INTER_LINEAR, cv.BORDER_DEFAULT, 0)

plt.subplot(3, 2, 3)
plt.title('Left Image')
#cv.circle(left_image, (floor(leftCorner[0]), floor(leftCorner[1])), 4, (0, 255, 0))
for i in range(0, h, 20):
    cv.line(left_new_image, (0, i),(580, i), (0, 0, 255))
plt.imshow(left_new_image)
plt.axis('off')
plt.subplot(3, 2, 4)
plt.title('Right Image')
#cv.line(right_image, (0, floor( - coeffi[2]/coeffi[1])), (600, floor(-600.*coeffi[0]/coeffi[1] - coeffi[2]/coeffi[1])), (0, 0, 255))
for i in range(0, h, 20):
    cv.line(right_new_image, (0, i),(580, i), (0, 0, 255))
plt.imshow(right_new_image)
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title('Left(stereoRectify())')
#cv.circle(left_image, (floor(leftCorner[0]), floor(leftCorner[1])), 4, (0, 255, 0))
for i in range(0, h, 20):
    cv.line(stereo_left_new_image, (0, i),(580, i), (0, 0, 255))
plt.imshow(stereo_left_new_image)
plt.axis('off')
plt.subplot(3, 2, 6)
plt.title('Right(stereoRectify())')
#cv.line(right_image, (0, floor( - coeffi[2]/coeffi[1])), (600, floor(-600.*coeffi[0]/coeffi[1] - coeffi[2]/coeffi[1])), (0, 0, 255))
for i in range(0, h, 20):
    cv.line(stereo_right_new_image, (0, i),(580, i), (0, 0, 255))
plt.imshow(stereo_right_new_image)
plt.axis('off')
plt.show()
cv.destroyAllWindows()


