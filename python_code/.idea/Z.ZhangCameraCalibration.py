import cv2 as cv
import glob
import numpy as np
from  cmath import *
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

#rows of L according to Appendiex A which is used for initialize the homography H
rows_L = []
#homography matrix, all_h stores all the h of the images
all_h = []

#matrix V_b which consists of the combination of hi, used for calculate the matrix B
V_b = []
for fimage in images:
    print(fimage)     #check the name of the image
    img = cv.imread(fimage)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray_image, pattern_size)
    if found:
        single_image_h = []
        #use function cornerSubPix to refine corner coordinates to subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)

        pattern_points.append(single_pattern_points)
        image_points.append(corners)
        #add the points to be new row of L according to Appendiex A and calculate the jacbi matrix used in LM
        for i in range(len(single_pattern_points)):
            rows_L.append(list(single_pattern_points[i][:2]) + [1] + [0,0,0]  + [x * -1 * corners[i][0][0] for x in (list ((single_pattern_points[i][:2]))+[1])])
            rows_L.append([0,0,0] + list(single_pattern_points[i][:2]) + [1] +  [x * -1 * corners[i][0][1] for x in (list ((single_pattern_points[i][:2]))+[1])])
        # using svd() function to get the right singular vector of L which is used for the initial value of H, and hit(i in [1,2,3]) represents the ith row of H
        _, _, Vt = np.linalg.svd(np.array(rows_L))
        h1t = (Vt[-1::][0][0:3])
        h2t = (Vt[-1::][0][3:6])
        h3t = (Vt[-1::][0][6:])
        #extend the H to column vector
        h = list(h1t) + list(h2t) + list(h3t)
        #Levenberg-Marquardt algorithm to get h for each image
        v = 2
        f = []
        #calculate the  Jacobi matrix
        jacobi = []
        single_pattern_points = np.array(single_pattern_points)
        single_pattern_points[:,-1] = np.ones(single_pattern_points.shape[0])
        for i in range(len(single_pattern_points)):
            jacobi.append([-1* single_pattern_points[i][0] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           -1* single_pattern_points[i][1] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           -1 / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           0,0,0,
                           single_pattern_points[i][0] * (np.dot(h1t, np.array(single_pattern_points[i]).T))/ pow(np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                           single_pattern_points[i][1] * (np.dot(h1t, np.array(single_pattern_points[i]).T))/ pow(np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                           (np.dot(h1t, np.array(single_pattern_points[i]).T))/ pow(np.dot(h3t, np.array(single_pattern_points[i]).T), 2)])
            jacobi.append([0, 0, 0,
                           -1 * single_pattern_points[i][0] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           -1 * single_pattern_points[i][1] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           -1 / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                           single_pattern_points[i][0] * (np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow(np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                           single_pattern_points[i][1] * (np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow(np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                           (np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow ( np.dot(h3t, np.array(single_pattern_points[i]).T), 2)])
            f.append(corners[i][0][0] - (np.dot(h1t, np.array(single_pattern_points[i]).T)) / (np.dot(h3t, np.array(single_pattern_points[i]).T)))
            f.append(corners[i][0][1] - (np.dot(h2t, np.array(single_pattern_points[i]).T)) / (np.dot(h3t, np.array(single_pattern_points[i]).T)))

        h = np.array(h).T
        print(h.shape)
        f = np.array(f)
        print(image_points[0][0])
        jacobi = np.array(jacobi)
        A = np.dot(jacobi.T, jacobi)
        g = np.dot(jacobi.T, -1 * f)
        threshold = e-50
        found = max(g) < threshold
        u = max(np.diag(A))
        while (not found):
            hlm = np.dot(np.linalg.inv(A + u * np.eye(A.shape[0])), g)
            if np.dot(hlm.T, hlm) < threshold * (threshold + np.dot(h.T, h)):
                found = True
            else:
                h_new = h + hlm
                h1t = h_new[:3].T
                h2t = h_new[3:6].T
                h3t = h_new[6:].T
                #calculate the new value of function f
                new_f = []
                for i in range(len(single_pattern_points)):
                    new_f.append((corners[i][0][0] - np.dot(h1t, np.array(single_pattern_points[i]).T)) / (np.dot(h3t, np.array(single_pattern_points[i]).T)))
                    new_f.append((corners[i][0][1] - np.dot(h2t, np.array(single_pattern_points[i]).T)) / (np.dot(h3t, np.array(single_pattern_points[i]).T)))
                new_f = np.array(new_f)

                p = (np.dot(f.T, f) - np.dot(new_f.T, new_f)) / np.dot(hlm.T, u * hlm + g)
                if p > 0 :
                    h = h_new
                    #re-calculate the jacobi matrix again based on the new value of h, and new value of f
                    jacobi = []
                    for i in range(len(single_pattern_points)):
                        jacobi.append(
                            [-1 * single_pattern_points[i][0] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                             -1 * single_pattern_points[i][1] / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                             -1 / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                             0, 0, 0,
                             single_pattern_points[i][0] * (np.dot(h1t, np.array(single_pattern_points[i]).T)) / pow(
                                 np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                             single_pattern_points[i][1] * (np.dot(h1t, np.array(single_pattern_points[i]).T)) / pow(
                                 np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                             (np.dot(h1t, np.array(single_pattern_points[i]).T)) / pow(
                                 np.dot(h3t, np.array(single_pattern_points[i]).T), 2)])
                        jacobi.append([0, 0, 0,
                                       -1 * single_pattern_points[i][0] / (
                                       np.dot(h3t, np.array(single_pattern_points[i]).T)),
                                       -1 * single_pattern_points[i][1] / (
                                       np.dot(h3t, np.array(single_pattern_points[i]).T)),
                                       -1 / (np.dot(h3t, np.array(single_pattern_points[i]).T)),
                                       single_pattern_points[i][0] * (
                                       np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow(
                                           np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                                       single_pattern_points[i][1] * (
                                       np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow(
                                           np.dot(h3t, np.array(single_pattern_points[i]).T), 2),
                                       (np.dot(h2t, np.array(single_pattern_points[i]).T)) / pow(
                                           np.dot(h3t, np.array(single_pattern_points[i]).T), 2)])
                    jacobi = np.array(jacobi)
                    A = np.dot(jacobi.T, jacobi)
                    g = np.dot(jacobi.T, -1 * new_f)
                    f = new_f
                    found = (max(g) < threshold) or (np.dot(f.T, f) < threshold)
                    u = u * max(1/3, 1-pow((2*p-1), 3))
                    v *= 2
                else:
                    u = u * v
                    v *= 2
        #for each image, use to homography H, to form the matrix V_b to calculate the matrix B, according to zhang's paper section 3.1
        #hi is the ith column of h
        h1 = h[:3]
        h2 = h[3:6]
        h3 = h[6:]
        V_b.append([h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1], h1[2] * h2[0] + h1[0] * h2[2],
                    h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]])
        V_b.append([h1[0] * h1[0] - h2[0] * h2[0], h1[0] * h1[1] + h1[1] * h1[0] - h2[0] * h2[1] - h2[1] * h2[0], h1[1] * h1[1] - h2[1] * h2[1], h1[2] * h1[0] + h1[0] * h1[2] - h2[2] * h2[0] + h2[0] * h2[2],
                    h1[2] * h1[1] + h1[1] * h1[2] - h2[2] * h2[1] - h2[1] * h2[2], h1[2] * h1[2] - h2[2] * h2[2]])
V_b = np.array(V_b)
_, _, Vt = np.linalg.svd(V_b)
print(Vt)
b = Vt[-1::][0].T
print(b)

#according to Appendiex B, calculate the intrinsic paraters basedo on the vector b
v_0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])
lam = b[5] - (b[3] * b[3] + v_0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
alp = sqrt(lam / b[0])
beta = sqrt(lam * b[0] / (b[0] * b[2] - b[1] * b[1]))
gam = -1 * b[1] * alp * alp * beta / lam
u_0 = gam * v_0 / beta - b[3] * alp * alp / lam
print(v_0)
print(lam)
print(lam / b[0])

#acoording to the intrinsic paremeters calculated, we can get the matrix K
K = np.array([[alp, gam, u_0], [0, beta, v_0], [0, 0, 1]])
print(K.shape)
print(K)

#according to the instric matrix K and the scale value lam, we can get the rotation matrix R
r1 = lam * np.linalg.inv(K) * h1
r2 = lam * np.linalg.inv(K) * h2
r3 = np.outer(r1, r2)
t = lam * np.linalg.inv(K) * h3

'''print(r1)
rvecs = [r1, r2, r3]

#reprojection error
error = 0
for i in range(len(pattern_points)):
    new_image_points, _ = cv.projectPoints(pattern_points[i], np.array(rvecs), t, K, None)
    error += cv.norm(image_points[i], new_image_points, cv.NORM_L2) / len(new_image_points)
print("total error is:", error/len(pattern_points))

#undistort with function getOptimalNewCameraMatrix() and function undistort()
single_image = cv.imread('Project_Stereo_left\left\left02.jpg')
single_gray_image = cv.cvtColor(single_image, cv.COLOR_BGR2GRAY)
h, w = single_gray_image.shape

newCameraMatrix, _ = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffis, (w, h), 0, (w, h))
undistort_image = cv.undistort(single_gray_image, cameraMatrix, distCoeffis, None, newCameraMatrix)

h, w = gray_image.shape



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
cv.destroyAllWindows'''


