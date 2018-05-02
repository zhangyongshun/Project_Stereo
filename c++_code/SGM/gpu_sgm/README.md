*With vs2013 OpenCV 3.4.1*

**path.cu**
  
  path.cu implement 8 paths, which are used for aggregating costs. For each pixel, the path directions contains "left to right"
  , "left top to right bottom", "top to dbottom", "right top to left bottom", "right to left", "right bottom to left top", 
  "bottom to top" and "left bottom to right top".  
  
  All the paths functions are declared as __global__, which means the functions run on GPU.
  
  There is also a aggregateL() function, which is used to caculate cost in each path, which is declared as __device__
  
  
**path.h**
  
  path.h  declares all path functions, which are implemented in paths.cpp.
  
  
**main.cu**
  
  main.cu includes path.h, and there are calculate mutual information cost functions and calculate disparity maps functions in main.cu,
  all the function have been noted in the cpp file.
  
  KITTI images are used in the program, and "devkit\cpp\io_disp.h" is used for generating error image.
  
  INPUT: left calibrated image, right calibrated image.
  
  OUTPUT: disparity image, error image
