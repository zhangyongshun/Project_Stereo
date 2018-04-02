**With Python3.5**

**opencvCameraCalibration.py**

  opencvCameraCalibration.py uses OpenCV functions including findChessboardCorners(), calibrateCamera(). Function drawChessboardcorner（）is used to draw the corners on the image detected in the origin images, we put the images drawed the corners in a new folder.
   
  This program is used for camera calibration based on pinhole camera model.
  
  INPUT: images with chessboards, which have been put into folder "Project_Stereo_left/left" under the project path
  
  OUTPUT: intrinsic parameters, extrinsic parameters and reprojection error
  
  All the function of the functions have been introduce by notes. 

**opencvUndistort.py**

  opencvUndistort.py uses OpenCV functions including findChessboardCorners(), calibrateCamera(), getOptimalNewCameraMatrix(), undistort(). Function findChessboardCorners（）, calibrateCamera() is used to calculate the intrinsic and extrinsic parameters, and function getOptimalNewCameraMatrix() used to adjust the cameraMatrix when we use a subset of the input images, function undistort() is to undistort the images based on the parameters.
   
  This program is used for undistort the image after camera calibration.
  
  INPUT: images with chessboards
  
  OUTPUT: the undistort image
  
  All the function of the functions have been introduce by notes. 
  

**Z.ZhangCameraCalibration.py**

   
  This program implements the Z.Zhang's method of camera calibration, which has not been finished yet, the peogress of Z.Zhang's method   has been finished, but the parameters figured out are not correct, which need to be debug later. 

  INPUT: images with chessboards
  
  OUTPUT: intrinsic parameters, extrinsic parameters, reprojection error, the undistort image
  
