#include "path.h"
#include <iostream>
#include <stdio.h>
#include "devkit\cpp\io_disp.h"
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
				left_pixel = left_image[m*width + n];
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
			sum += (sub_pixel );
		}
	}
	cost[(row*width + col)*disparity_range + d] = sum;
}

//cost aggregation
void calculateOneWayCost(const int row_diff, const int col_diff, uchar *left_image, const uchar *right_image,
	unsigned int *cost, unsigned int *dev_one_path_cost, const int width, const int height,
	const int disparity_range)
{
	unsigned int max = MAX_INT;

	dim3 block(disparity_range, 1);
	dim3 grid_top_down(1, width);
	dim3 grid_left_right(1, height);
	dim3 grid_slam(1, width + height);
	//choose one of ways
	//cout << row_diff + (col_diff + 1) * 3 + 1 << endl;
	switch (row_diff + (col_diff + 1) * 3 + 1)
	{
	case 0:
		//firstPath << <grid_slam, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 1:
		secondPath << <grid_top_down, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 2:
		//thirdPath << <grid_slam, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 3:
		fourthPath << <grid_left_right, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 8:
		fifthPath << <grid_left_right, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 5:
		//sixthPath << <grid_slam, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 6:
		seventhPath << <grid_top_down, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 7:
		//eighthPath << <grid_slam, block >> >(left_image, right_image, cost, dev_one_path_cost, max, width, height, disparity_range);
		break;
	default:
		printf("path does not exist!\n");
	}

}
__global__ void aggregateCost(unsigned int * dev_aggregate_cost, const unsigned int * dev_one_path_cost, int cost_size)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int id = i + (j * blockDim.x);

	if (id < cost_size){
		dev_aggregate_cost[id] += dev_one_path_cost[id];
	}

}

__global__ void generateDisparity(const unsigned int *cost, unsigned char *disparity_image, const int width, const int height, const int disparity_range, unsigned int min_cost){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= width && row >= height)
		return;
	unsigned int min_cost_d = min_cost;
	if (col < width && row < height)
	{
		int min_d = disparity_range;
		const unsigned int *v = &cost[(row*width + col)*disparity_range];
		for (int d = 0; d < disparity_range; d++)
		{
			if (v[d] < min_cost_d) {
				min_cost_d = v[d];
				min_d = d;
			}
		}
		disparity_image[col + row*width] = min_d;
		//printf("mind = %d, disparity=%f\n", min_d, disparity_image[col + row*width]);
	}
}

//sgm with cuda accelerate
int main()
{
	//cost define
	int half_window = 11;
	int compare_length = (2 * half_window + 1)*(2 * half_window + 1);

	cv::Mat left_image, right_image, sad_image;

	int width;
	int height;

	readImage(left_image, right_image, width, height);

	sad_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));

	int disparity_range =  32;

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
	
	//calculate one path cost
	unsigned int *dev_one_path_cost;
	//aggregate all the paths' cost
	unsigned int *dev_aggregate_cost;

	unsigned int cost_size = height*width*disparity_range*sizeof(unsigned int);
	unsigned int image_size = height*width*sizeof(unsigned char);

	clock_t start = clock();

	cost = new unsigned int[height*width*disparity_range];

	cudaMalloc((void **)&dev_left_image, image_size);
	cudaMalloc((void **)&dev_right_image, image_size);
	cudaMalloc((void **)&dev_cost, cost_size);

	cudaMalloc((void **)&dev_one_path_cost, cost_size);
	cudaMalloc((void **)&dev_aggregate_cost, cost_size);

	cudaMemcpy(dev_left_image, left_image.ptr(), image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right_image, right_image.ptr(), image_size, cudaMemcpyHostToDevice);
	cudaMemset(dev_cost, 0, cost_size);
	cudaMemset(dev_aggregate_cost, 0, cost_size);

	dim3 block_c(disparity_range, 1);
	dim3 grid_c(width, height);

	calculateCost << <grid_c, block_c >> >(dev_left_image, dev_right_image, dev_cost, MAX_INT, width, height, disparity_range, half_window);
	//

	dim3 block(512, 1);
	dim3 grid(1, ceil(width*height*disparity_range / 512.));
	//cudaMemcpy(dev_cost, cost, cost_size, cudaMemcpyHostToDevice);

	//8 paths
	int row_diff, col_diff;
	for (row_diff = -1; row_diff != 2; row_diff++)
	{
		for (col_diff = -1; col_diff != 2; col_diff++)
		{
			if (!row_diff && !col_diff)
				continue;
			cudaMemset(dev_one_path_cost, 0, cost_size);
			calculateOneWayCost(row_diff, col_diff, dev_left_image, dev_right_image, dev_cost, dev_one_path_cost, width, height, disparity_range);
			aggregateCost << <grid, block >> >(dev_aggregate_cost, dev_one_path_cost, width*height*disparity_range);
		}
	}

	//dim3 grid_d(ceil((float)down_width / 32.), ceil((float)down_height / 16.));
	//dim3 block_d(32, 16);

	//float *dev_disparity_image;
	//int disparity_size = left_disparity_image.cols*left_disparity_image.rows*sizeof(float);

	//cudaMalloc((void **)&dev_disparity_image, disparity_size);
	//cudaMemcpy(dev_disparity_image, left_disparity_image.ptr(), disparity_size, cudaMemcpyHostToDevice);
	//generateDisparity << <grid_d, block_d >> >(dev_aggregate_cost, dev_disparity_image, down_width, down_height, down_disparity_range, MAX_INT);
	//cudaMemcpy(left_disparity_image.ptr(), dev_disparity_image, disparity_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cost, dev_aggregate_cost, cost_size, cudaMemcpyDeviceToHost);

	

	clock_t stop = clock();

	cout << "Time: " << (double)(stop - start) / CLK_TCK << endl;

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
//	cv::medianBlur(sad_image, sad_image, (3, 3));
	cv::imshow("new_d", sad_image);
	cv::waitKey();
	cv::imwrite("out.png", sad_image);
	DisparityImage calculate_d;
	//kitti_d.writeColor("kitti.png");
	//kitti_d.read("ground.png");

	//kitti_d.writeColor("color.png");
	calculate_d.read("out.png");
	calculate_d.writeColor("color.png");

	cudaFree(dev_aggregate_cost);
	cudaFree(dev_one_path_cost);
	cudaFree(dev_left_image);
	cudaFree(dev_right_image);
	cudaFree(dev_cost);

	getchar();
	return 0;
}

