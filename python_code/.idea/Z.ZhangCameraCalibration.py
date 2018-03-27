import cv2 as cv
import glob
import numpy as np
from  cmath import *
import matplotlib.pyplot as plt
import copy

#inner corners on the chessboard
pattern_size = (9, 6)

#suoppose plane of the chessboard in world coordinates is "Z = 0", and encode the points on chessboard.
single_pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
single_pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
single_pattern_points[:, -1] = np.ones(len(single_pattern_points[:, -1])).T

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

#bool value, draw value of function f varation, the residual of function f and its  apporximation function caused by updating iterative point, during the LM algorithm
draw_h_picture_once = False

for fimage in images:
    print(fimage)     #check the name of the image
    img = cv.imread(fimage)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray_image, pattern_size)
    if found:
        single_image_h = []
        rows_L = []
        #use function cornerSubPix to refine corner coordinates to subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)

        #copy the list
        pattern_points.append(single_pattern_points)
        image_points.append(corners)


        #normalization the matrix
        #translation and scale

        #copy the single_pattern_points and corners !!! it's important because of the python mechanism
        single_pattern_points_copy = copy.deepcopy(single_pattern_points)
        corners_copy = copy.deepcopy(corners)

        x_average = sum(single_pattern_points_copy[:,0]) / len(single_pattern_points_copy[:,0])
        single_pattern_points_copy[:, 0] = single_pattern_points_copy[:, 0] - x_average
        y_average = sum(single_pattern_points_copy[:, 1]) / len(single_pattern_points_copy[:, 1])
        single_pattern_points_copy[:, 1] = single_pattern_points_copy[:, 1] - y_average
        u_average = sum(corners_copy[:,:,0]) / len(corners_copy[:,:,0])
        corners_copy[:, :, 0] = corners_copy[:,:,0] - u_average
        v_average = sum(corners_copy[:, :, 1]) / len(corners_copy[:, :, 1])
        corners_copy[:, :, 1] = corners_copy[:, :, 1] - v_average

        #scaling
        s_w = 1
        for i in range(len(single_pattern_points_copy)):
            s_w += (single_pattern_points_copy[i, 0] ** 2 + single_pattern_points_copy[i, 1] ** 2) ** 0.5
        s_w /= (2 ** 0.5 * len(single_pattern_points_copy))
        s_i = 1
        for i in range(len(corners_copy[:, : 0])):
            s_i += (corners_copy[0][0][0] ** 2 + corners_copy[0][0][1] ** 2) ** 0.5
        s_i /= (2 ** 0.5 * len(corners_copy[:, :, 0]))
        s_i -= 1
        s_w -= 1
        single_pattern_points_copy[:, 0] =  list(np.array(single_pattern_points_copy[:, 0]) /s_w)
        single_pattern_points_copy[:, 1] = list(np.array(single_pattern_points_copy[:, 1]) / s_w)
        corners_copy[:, :, 0] = list(np.array(corners_copy[:, :, 0]) / s_i)
        corners_copy[:, :, 1] = list(np.array(corners_copy[:, :, 1]) / s_i)



        #add the points to be new row of L according to Appendiex A and calculate the jacbi matrix used in LM
        for i in range(len(single_pattern_points_copy)):
            rows_L.append(list(single_pattern_points_copy[i])  + [0,0,0]  + [x * -1 * corners_copy[i][0][0] for x in list ((single_pattern_points_copy[i]))])
            rows_L.append([0,0,0] + list(single_pattern_points_copy[i])  +  [x * -1 * corners_copy[i][0][1] for x in list ((single_pattern_points_copy[i]))])
            '''sum = 1
    for i in range(len(rows_L)):
        for j in range(len(rows_L[0])):
            sum += rows_L[i][j]
    rows_L = np.array(rows_L) - (sum / (9 * len(rows_L)))'''

        rows_L = np.array(rows_L)
        # using svd() function to get the right singular vector of L which is used for the initial value of H, and hit(i in [1,2,3]) represents the ith row of H
        _, _, Vt = np.linalg.svd(rows_L)
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
        single_pattern_points[:, -1] = np.ones(len(single_pattern_points[:, -1]))

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
        #print(h.shape)
        f = np.array(f)
       # print(image_points[0][0])
        jacobi = np.array(jacobi)
        A = np.dot(jacobi.T, jacobi)
        g = np.dot(jacobi.T, -1 * f)
        threshold = 1E-100
        found = max(g) < threshold
        u = max(np.diag(A))
        count = 0

        real_f_change = []
        esti_f_change = []
        real_f = []

        while (not found):

            count += 1
            hlm = np.dot(np.linalg.inv(A + u * np.eye(A.shape[0])), g)
           # print(threshold * (threshold + np.dot(h.T, h)))

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
                real_f.append(np.dot(f.T, f))
                real_f_change.append(np.dot(f.T, f) - np.dot(new_f.T, new_f))
                esti_f_change.append(np.dot(hlm.T, u * hlm + g))
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

        if not draw_h_picture_once:
            x = [i+1 for i in range(len(real_f_change))]
            plt.title('The Value and Residual during LM Algorithm (Calculate the Matrix H)')

            plt.xlabel('counts of iteration')
            plt.ylabel('value of function')
            plt.plot(x, real_f, color='green', label='value of f')
            plt.plot(x, real_f_change, color='red', label='difference between two iteration of f')
            plt.plot(x, esti_f_change, color='black', label='difference between two iteration of approximate f')
            plt.legend()
            plt.show()
            draw_h_picture_once = True

        #for each image, use to homography H, to form the matrix V_b to calculate the matrix B, according to zhang's paper section 3.1
        #hi is the ith column of h

        h1 = h[:3]
        h2 = h[3:6]
        h3 = h[6:]
        V_b.append([h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1], h1[2] * h2[0] + h1[0] * h2[2],
                    h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]])
        V_b.append([h1[0] * h1[0] - h2[0] * h2[0], h1[0] * h1[1] + h1[1] * h1[0] - h2[0] * h2[1] - h2[1] * h2[0], h1[1] * h1[1] - h2[1] * h2[1], h1[2] * h1[0] + h1[0] * h1[2] - h2[2] * h2[0] - h2[0] * h2[2],
                    h1[2] * h1[1] + h1[1] * h1[2] - h2[2] * h2[1] - h2[1] * h2[2], h1[2] * h1[2] - h2[2] * h2[2]])

#use function svd() to solve the equation V * b = 0, in order to get vector b, acoording to section 3.2

V_b = np.array(V_b[3:6])
_, _, Vt = np.linalg.svd(V_b)
b = Vt[-1::][0].T


#according to Appendiex B, calculate the intrinsic paraters basedo on the vector b
v_0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])
lam = b[5] - (b[3] * b[3] + v_0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
alp = (lam / b[0]) ** 0.5
beta = (lam * b[0] / (b[0] * b[2] - b[1] * b[1])) ** 0.5
gam = -1 * b[1] * alp * alp * beta / lam
u_0 = gam * v_0 / beta - b[3] * alp * alp / lam

print(alp)
print(beta)
#acoording to the intrinsic paremeters calculated, we can get the matrix K
K = np.array([[alp * 2000, gam , u_0], [0, beta * 2000, v_0], [0, 0, 1]])
#according to the instric matrix K and the scale value lam, we can get the rotation matrix R
r1 = lam * np.dot(np.linalg.inv(K),h1)
r2 = lam * np.dot(np.linalg.inv(K), h2)
r3 = np.cross(r1, r2)
t = lam * np.dot(np.linalg.inv(K),h3)

rt = [list(r1), list(r2), list(t)]
rt = np.array(rt).T
'''print(rotation)
U, _, Vt = np.linalg.svd(np.array(rotation))
rotation = np.dot(U, Vt)
'''

#calculate the matrix D according to calculate the distortion coefficient
D = []
b = []
for i in range(len(pattern_points)):
    for j in range(len(pattern_points[0])):
        #calculate the coordinates u and v
        new_image_point = np.dot(np.dot(K, rt), np.array(pattern_points[i][j]).T)
        u = new_image_point[0] / new_image_point[2]
        v = new_image_point[1] / new_image_point[2]
        D.append([(u - u_0) * (pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2), (u - u_0) * ((pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) ** 2)])
        D.append([(v - v_0) * (pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2), (v - v_0) * ((pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) ** 2)])
        b.append([image_points[i][j][0][0] - u])
        b.append([image_points[i][j][0][1] - v])
D = np.array(D)
b = np.array(b)

k = np.dot(np.dot(np.linalg.inv(np.dot(D.T, D)), D.T), b)
print(k)
print(k[1][0])

#reprojection error
error = 0
for i in range(len(pattern_points)):
    new_image_points = []
    for j in range(len(pattern_points[0])):
        new_image_point = np.dot(np.dot(K, rt), np.array(pattern_points[i][j]).T)
        u = new_image_point[0] / new_image_point[2]
        v = new_image_point[1] / new_image_point[2]
        new_image_points.append([[u + (u - u_0) * (k[0][0]*(pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) + k[1][0] * (pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) ** 2), v + (v - v_0) * (k[0][0]*(pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) + k[1][0] * (pattern_points[i][j][0] ** 2 + pattern_points[i][j][1] ** 2) ** 2)]])
    new_image_points = np.array(new_image_points)

    for z in range(len(image_points[0])):
        error += ((new_image_points[z][0][0] - image_points[i][z][0][0]) ** 2 + (new_image_points[z][0][1] - image_points[i][z][0][1]) ** 2) ** 0.5
print("total error before distortion is:", error/len(pattern_points))

'''
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
cv.destroyAllWindows

'''
