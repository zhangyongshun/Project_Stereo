#include <iostream>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace std;

int main()
{
	//load the image
	cv::Mat left_image= cv::imread("left1.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat right_image = cv::imread("right1.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat disparity_image, disparity_image8;
	//get the channels of images and set SAD aggregating cost window size
	int channel = left_image.channels();
	int SAD_window_size = 7*7;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
	sgbm->setP1(8 * channel*SAD_window_size);
	sgbm->setP2(32 * channel*SAD_window_size);
	sgbm->setSpeckleRange(32);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//disparity_image = cv::Mat(cv::Size(480, 760), CV_8U, cv::Scalar(0));
	sgbm->compute(left_image, right_image, disparity_image);
	cv::imwrite("out1.png", disparity_image);
	disparity_image.convertTo(disparity_image8, CV_8U, 255 / 16*(16.));
	cv::imwrite("out81.png", disparity_image8);
	/*
	cv::namedWindow("results", CV_WINDOW_AUTOSIZE);
	cv::imshow("results", disparity_image);
	cv::waitKey(0);
	cv::destroyAllWindows();*/
	return 0;
}