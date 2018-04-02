**With Python3.5 Opencv3.4.1**

**opencvStereoCalibration.py**

  opencvCameraCalibration.py uses OpenCV functions including stereoCalibrate(). Function stereoCalibrate（）is used to calculate the rotation matrix R, translation vector T, essential matrix E and fundamental matrix F
  In this code, we calculate the R,T,E,F by the derivation according the relationshoip between them, and finially make comparison with what the stereoCalibrate() calculates.
  This program is used for stereo camera calibration.
  
  INPUT: images with chessboards, which have been put into folder "Project_Stereo_left/left" and  "Project_Stereo_right/right" under the project path
  
  OUTPUT: the outcomes which are that we calculate the Xr'T*F*Xl, matrices R,T,E,F, picture with epipolar line.
  
  All the function of the functions have been explained in the file by notes. 

