#ifndef PATH_H
#define PATH_H

#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

#define MAX_FLOAT  3.4028235E38
#define P1 8
#define P2 32

// calculate fist way cost from top left corner to bottm right corner
void firstWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void secondWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void thirdWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void fourthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void fifthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void sixthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void seventhWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);
void eighthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image);

#endif