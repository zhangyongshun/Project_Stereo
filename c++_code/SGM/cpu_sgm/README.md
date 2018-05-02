*With vs2013 OpenCV 3.4.1*

**paths.cpp**
  
  paths.cpp implement 8 paths, which are used for aggregating costs. For each pixel, the path directions contains "left to right"
  , "left top to right bottom", "top to dbottom", "right top to left bottom", "right to left", "right bottom to left top", 
  "bottom to top" and "left bottom to right top".
  
  All the functions are declared in path.h.
  
  
**path.h**
  
  path.h  declares all path functions, which are implemented in paths.cpp.
  
  
**main.cpp**
  
  main.cpp includes path.h, and there are calculate mutual information cost functions and calculate disparity maps functions in main.cpp,
  all the function have been noted in the cpp file.
  
  INPUT: left calibrated image, right calibrated image.
  
  OUTPUT: disparity image
