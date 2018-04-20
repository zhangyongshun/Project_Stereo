#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

int main()
{
	cout << "OpenCV Version: " << CV_VERSION << endl;
	Mat image = imread("img.jpg");
	imshow("img", image);
	waitKey(0);

	return 0;
}