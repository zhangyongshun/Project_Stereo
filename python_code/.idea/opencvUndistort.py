import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt


#inner corners on the chessboard
pattern_size = (9, 6)

#suoppose plane of the chessboard in world coordinates is "Z = 0", and encode the points on chessboard.
single_pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
single_pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

#store the corner points in chessboard and corresponding points in image plane.
pattern_points = []
image_points = []

#load the images
images = glob.glob('Project_Stereo_left\left\*.jpg')
#name the iamge with drawed corner
num = 1

for fimage in images:
    print(fimage)     #check the name of the image
    img = cv.imread(fimage)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(gray_image.shape[::-1])
    found, corners = cv.findChessboardCorners(gray_image, pattern_size)
    if found:
        #use function cornerSubPix to refine corner coordinates to subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)
        pattern_points.append(single_pattern_points)
        image_points.append(corners)
h, w = gray_image.shape

retvla, cameraMatrix, distCoeffis, rvecs, tvecs = cv.calibrateCamera(pattern_points, image_points, (w, h), None,None)


#undistort with function getOptimalNewCameraMatrix() and function undistort()
single_image = cv.imread('Project_Stereo_left\left\left02.jpg')
single_gray_image = cv.cvtColor(single_image, cv.COLOR_BGR2GRAY)
h, w = single_gray_image.shape

newCameraMatrix, _ = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffis, (w, h), 0, (w, h))
undistort_image = cv.undistort(single_gray_image, cameraMatrix, distCoeffis, None, newCameraMatrix)

plt.subplot(1, 2, 1)
plt.title('Before undistortion')
plt.imshow(single_gray_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('After undistortion')
plt.imshow(undistort_image)
plt.axis('off')
plt.show()
cv.destroyAllWindows()


