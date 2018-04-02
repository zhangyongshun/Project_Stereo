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

leftCorner = left_image_points[0][3]

_, leftCameraMatrix, leftDistCoeffis, leftRvecs, leftTvecs = cv.calibrateCamera(pattern_points, left_image_points, (w, h), None,None)
leftR, _ = cv.Rodrigues(leftRvecs[0])

#read the left image
left_image = cv.imread('Project_Stereo_left\left\left01.jpg')
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
rightR, _ = cv.Rodrigues(rightRvecs[0])

right_image = cv.imread('Project_Stereo_right/right/right01.jpg')
right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

h, w = right_image.shape
_,left_mtx, left_dis,right_mtx,right_dis,stereoR,stereoT,stereoE,stereoF = cv.stereoCalibrate(pattern_points, left_image_points,right_image_points,leftCameraMatrix,leftDistCoeffis,rightCameraMatrix,rightDistCoeffis,(w,h))
#use the intrisic and extrinsic parameters of left and right camera to calculate rotation matrix, translation vector, essential matrix and fundamental matrix



#rotation matrix
R = np.dot(rightR, leftR.T)
#traslation vector from left to right
T = rightTvecs[0] - np.dot(R, leftTvecs[0])
#essential matrix

T_r = -1 * np.dot(stereoR.T, stereoT)
test =  -1 * np.dot(R.T, T)
S = np.array([[0, -1*T_r[0], T_r[1]],[T_r[2], 0, -1*T_r[0]],[-1*T_r[1], T_r[0], 0]])
E = np.dot(stereoR, S)
t = np.linalg.matrix_rank(E)
v,s,u = np.linalg.svd(E)
ms = np.eye(3)
ms[0][0] = s[0]
ms[1][1] = s[1]
ms[2][2] = 0
E = np.dot(np.dot(v, ms), u)
r = np.linalg.matrix_rank(E)

r = np.linalg.matrix_rank(stereoE)
#fundamental matrix
F = np.dot(np.linalg.inv(right_mtx).T, np.dot(E, np.linalg.inv(left_mtx)))

'''
ms = np.eye(3)
ms[0][0] = s[0]
ms[1][1] = s[1]
ms[2][2] = 0
F = np.dot(np.dot(v, ms), u)
r = np.linalg.matrix_rank(F)
'''
#draw the epipolar line
leftCorner = np.array([[leftCorner[0][0]],[leftCorner[0][1]], [1]])
coeffi = np.dot(F, leftCorner)
P = np.array([[3],[0],[0]])
leftP=  np.dot(leftCameraMatrix, np.dot(leftR, P) + leftTvecs[0])
rightP = np.dot(rightCameraMatrix, np.dot(rightR, P) + rightTvecs[0])

leftP[0] = leftP[0]/leftP[2]
leftP[1] = leftP[1]/leftP[2]
leftP[2] = 1

rightP[0] = rightP[0]/rightP[2]
rightP[1] = rightP[1]/rightP[2]
rightP[2] = 1


rst1 = np.dot(rightP.T, np.dot(F, leftP))
rst2 = np.dot(rightP.T, np.dot(stereoF, leftP))
#list = cv.initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffis, None, leftCameraMatrix, (w,h),cv.CV_16SC2)
'''
p = np.dot(np.linalg.inv(leftCameraMatrix), leftCorner)
pw = np.dot(np.linalg.inv(leftR), np.dot(np.linalg.inv(leftCameraMatrix), leftCorner) - leftTvecs[0])
'''
coeffi = np.dot(stereoF, leftCorner)
plt.subplot(1, 2, 1)
plt.title('Before undistortion')
cv.circle(left_image, (floor(leftCorner[0]), floor(leftCorner[1])), 4, (0, 255, 0))
plt.imshow(left_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('After undistortion')
cv.line(right_image, (0, floor( - coeffi[2]/coeffi[1])), (600, floor(-600.*coeffi[0]/coeffi[1] - coeffi[2]/coeffi[1])), (0, 0, 255))
plt.imshow(right_image)
plt.axis('off')
plt.show()
cv.destroyAllWindows()


