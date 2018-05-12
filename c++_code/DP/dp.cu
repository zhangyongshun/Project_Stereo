
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>   
#include <cstring>   
#include <iostream>   
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <cmath>   

#define MAX_DIFF 10

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

__global__ void calculateCost(const uchar *left_image, const uchar *right_image, uchar *disparity_image, unsigned int *cost, int disparity_range, int width, int height, int max_diff)
{
	int d = threadIdx.x;
	int row = blockIdx.y;
	
	if (d >= disparity_range || row >= height)
		return;
	for (int col = 0; col < width; col++)
	{
		int k = d + col;
		if (k <0 || k > width - 1)
		{
			continue;
		}
		else
		{
			if (abs(left_image[row*width + k] - right_image[row*width + col]) <= max_diff)
			{
				cost[(row+1)*((col + 1)*width + k + 1)] = cost[(row+1)*(col*width + k)] + 1;
			}
			else if (cost[(row+1)*(col*width + k + 1)] > cost[(row+1)*((col + 1)*width + k)])
			{
				cost[(row+1)*((col + 1)*width + k + 1)] = cost[(row+1)*(col*width + k + 1)];
			}
			else{
				cost[(row+1)*((col + 1)*width + k + 1)] = cost[(row+1)*((col + 1)*width + k)];
			}
		}
		__syncthreads();
	}
	//re_search after the cost calculated 
	int m = width;
	int n = width;
	unsigned int l = cost[(row+1) * (width*width)];
	while (l>0)
	{
		if (cost[(row+1)*(m*width+n)] == cost[(row+1)*((m - 1)*width + n)])
			m--;
		else if (cost[(row+1)*(m * width + n)] == cost[(row+1)*(m * width + n - 1)])
			n--;
		else
		{
			disparity_image[row*width + m] = (uchar)(n - m);
			l--;
			m--;
			n--;
		}
	}
}

int  main()
{
	/*Half of the window size for the census transform*/
	int half_window = 11;
	int compare_length = (2 * half_window + 1)*(2 * half_window + 1);

	int width, height;
	int disparity_range = 32;
	cv::Mat left_image, right_image, disparity_image;

	readImage(left_image, right_image, width, height);
	disparity_image = cv::Mat(height, width, CV_8UC1);
	unsigned int *dev_cost;
	int cost_size = height * width * width * sizeof(unsigned int);
	int image_size = height*width*sizeof(uchar);
	uchar *dev_left_image;
	uchar *dev_right_image;
	uchar *dev_disparity_image;

	cudaMalloc((void **)&dev_left_image, image_size);
	cudaMalloc((void**)&dev_right_image, image_size);
	cudaMalloc((void **)&dev_disparity_image, image_size);
	cudaMalloc((void **)&dev_cost, cost_size);
	cudaMemset(dev_cost, 0, cost_size);
	cudaMemcpy(dev_left_image, left_image.ptr(), image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right_image, right_image.ptr(), image_size, cudaMemcpyHostToDevice);

	dim3 block(disparity_range, 1);
	dim3 grid(1, height);

	calculateCost<<<grid, block>>>(dev_left_image, dev_right_image, dev_disparity_image, dev_cost, disparity_range, width, height, MAX_DIFF);

	cudaMemcpy(disparity_image.ptr(), dev_disparity_image, image_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << (unsigned int)disparity_image.at<uchar>(i,j) << endl;
		}
	}
	cv::imwrite("out.png", disparity_image);
	getchar();
	return 0;
}