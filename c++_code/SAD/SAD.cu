
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>   
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "devkit\cpp\io_disp.h"

#define MAX_INT 888888

using namespace std;

//load left and right images which have been rectified
void readImage(cv::Mat &left_image, cv::Mat &right_image, int &width, int &height)
{
	left_image = cv::imread("left.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	right_image = cv::imread("right.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	width = left_image.cols;
	height = left_image.rows;
	cv::imshow("l", left_image);
	cv::imshow("2", right_image);
	cv::waitKey(0);
}

__global__ void calculateCost(const uchar *left_image, const uchar *right_image,
	unsigned int *cost,
	unsigned int max,
	const int width, const int height, const int disparity_range, const int half_window)
{
	unsigned int d = threadIdx.x;
	unsigned int row = blockIdx.y;
	unsigned int col = blockIdx.x;
	
	unsigned int sub_pixel = 0;
	unsigned int min_sum;
	unsigned int sum;

	unsigned char left_pixel, right_pixel;

	if (row >= height || d >= disparity_range || col >= width)
		return;

	min_sum = max;
	sum = 0;

	for (int m = row - half_window; m <= row + half_window; m++)
	{
		for (int n = col - half_window; n <= col + half_window; n++)
		{
			if (m < 0 || m >= height || n < 0 || n >= width)
			{
				sub_pixel = 0;
			}
			else if (n + d >= width)
			{
				sub_pixel = 0;
			}
			else
			{
				left_pixel = left_image[m*width+n];
				right_pixel = right_image[m*width + n + d];
				if (left_pixel > right_pixel)
				{
					sub_pixel = left_pixel - right_pixel;
				}
				else
				{
					sub_pixel = right_pixel - left_pixel;
				}

			}
			sum += sub_pixel;
		}
	}
	cost[(row*width + col)*disparity_range + d] = sum;
}
 

int main(){
	/*Half of the window size for the census transform*/
	int half_window = 11;
	int compare_length = (2 * half_window + 1)*(2 * half_window + 1);
	
	cv::Mat left_image, right_image, sad_image;

	int width;
	int height;
	
	readImage(left_image, right_image, width, height);

	sad_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
	
	int disparity_range = 32;

	int sum = 0;
	int min_k = 0;
	int min_sum = 0;

	unsigned char left_pixel = 0;
	unsigned char right_pixel = 0;
	unsigned char sub_pixel = 0;

	

	unsigned int *cost;
	unsigned int *dev_cost;
	unsigned char *dev_left_image;
	unsigned char *dev_right_image;

	unsigned int cost_size = height*width*disparity_range*sizeof(unsigned int);
	unsigned int image_size = height*width*sizeof(unsigned char);
	cost = new unsigned int[height*width*disparity_range];

	cudaMalloc((void **)&dev_left_image, image_size);
	cudaMalloc((void **)&dev_right_image, image_size);
	cudaMalloc((void **)&dev_cost, cost_size);

	cudaMemcpy(dev_left_image, left_image.ptr(), image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right_image, right_image.ptr(), image_size, cudaMemcpyHostToDevice);
	cudaMemset(dev_cost, 0, cost_size);

	dim3 block_c(disparity_range, 1);
	dim3 grid_c(width, height);
	
	calculateCost << <grid_c, block_c >> >(dev_left_image, dev_right_image, dev_cost, MAX_INT, width, height, disparity_range, half_window);
	cudaMemcpy(cost, dev_cost, cost_size, cudaMemcpyDeviceToHost);

	int min_d;
	int min_cost;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			min_d = 0;
			min_cost = MAX_INT;
			for (int d = 0; d < disparity_range; d++)
			{
				if (cost[(row*width + col)*disparity_range + d] < min_cost)
				{
					min_cost = cost[(row*width + col)*disparity_range + d];
					min_d = d;
				}
			}
			sad_image.at<uchar>(row, col) = min_d;
		}
	}

	float factor = 256. / (disparity_range);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			sad_image.at<uchar>(i, j) *= factor;
		}
	}
	cv::imshow("new_d", sad_image);
	cv::imwrite("out.png", sad_image);
	DisparityImage calculate_d;
	//kitti_d.writeColor("kitti.png");
	//kitti_d.read("ground.png");

	//kitti_d.writeColor("color.png");
	calculate_d.read("out.png");
	calculate_d.writeColor("color.png");

	cudaFree(dev_left_image);
	cudaFree(dev_right_image);
	cudaFree(dev_cost);

	cvWaitKey(0);
	return 0;
}