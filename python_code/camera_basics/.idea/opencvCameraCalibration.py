import cv2 as cv
import glob
import numpy as np


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

        #draw the corners on the image, and store them under the engineering catalogue
        cv.drawChessboardCorners(gray_image, pattern_size, corners, found)
        cv.imshow('Corners', gray_image)
        cv.imwrite('Project_Stereo_left\left_corner\left'+ str(num) + '.jpg' ,gray_image)
        num += 1
        cv.waitKey(1)
h, w = gray_image.shape
print(w)
retvla, cameraMatrix, distCoeffis, rvecs, tvecs = cv.calibrateCamera(pattern_points, image_points, (w, h), None,None)

print("camera matrix is:\n", cameraMatrix)
print(rvecs)
print(tvecs[0])
#reprojection error
error = 0
for i in range(len(pattern_points)):
    new_image_points, _ = cv.projectPoints(pattern_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffis)
    error += cv.norm(image_points[i], new_image_points, cv.NORM_L2) / len(new_image_points)
print("total error is:", error/len(pattern_points))
cv.destroyAllWindows()

