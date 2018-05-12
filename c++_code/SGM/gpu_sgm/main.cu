#include "path.h"
#include <iostream>
#include <stdio.h>
#include "devkit\cpp\io_disp.h"
using namespace std;

//load left and right images which have been rectified
void readImage(cv::Mat &left_image, cv::Mat &right_image, int &width, int &height)
{
	left_image = cv::imread("left0.png", CV_LOAD_IMAGE_GRAYSCALE);
	right_image = cv::imread("right0.png", CV_LOAD_IMAGE_GRAYSCALE);
	width = left_image.cols;
	height = left_image.rows;
}
void censusPro(int &total_match_pixels, int disparity_range, cv::Mat &prob, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &left_disparity_image)
{
	total_match_pixels = 0;

	int width = left_image.cols;
	int height = left_image.rows;
	int d;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int d = col - left_disparity_image.at<float>(row, col);
			if (d >= 0 && d < width)
			{
				prob.at<float>(left_image.at<uchar>(row, col), right_image.at<uchar>(row, d)) += 1;
				total_match_pixels += 1;
			}
		}
	}
	for (int row = 0; row < 256; row++)
		for (int col = 0; col < 256; col++)
		{
			prob.at<float>(row, col) = prob.at<float>(row, col) / float(total_match_pixels);
			//cout << (float)prob.at<float>(row, col) << endl;
		}
}
void calculateMI(cv::Mat &left_image, cv::Mat &right_image, cv::Mat &left_disparity_image, cv::Mat &prob, cv::Mat &left_image_prob, cv::Mat &right_image_prob, int total_match_pixels)
{
	float left_sum;
	float right_sum;

	//calculate P_I_1 and P_I_2
	for (int i = 0; i != 256; i++)
	{
		left_sum = right_sum = 0;
		for (int j = 0; j != 256; j++)
		{
			left_sum += prob.at<float>(i, j);
			right_sum += prob.at<float>(j, i);
		}
		left_image_prob.at<float>(0, i) = left_sum;
		right_image_prob.at<float>(0, i) = right_sum;
	}	


	//calculate -log(P \otimes g) for P_I_1, P_I_2,  P_I_1_I_2
	float left_gb;
	float right_gb;
	float union_lr_gb;
	cv::GaussianBlur(prob, prob, cv::Size(7, 7), 0, 0);
	cv::GaussianBlur(left_image_prob, left_image_prob, cv::Size(7, 1), 0, 0);
	cv::GaussianBlur(right_image_prob, right_image_prob, cv::Size(7, 1), 0, 0);

	for (int i = 0; i != 256; i++)
	{
		for (int j = 0; j != 256; j++)
		{
			if (prob.at<float>(i, j) < 1E-6)prob.at<float>(i, j) = 1E-6;
			prob.at<float>(i, j) = -log(prob.at<float>(i, j));
		}
		if (left_image_prob.at<float>(0, i) < 1E-6)left_image_prob.at<float>(0, i) = 1E-6;
		left_image_prob.at<float>(0, i) = -log(left_image_prob.at<float>(0, i));
		if (right_image_prob.at<float>(0, i) < 1E-6)right_image_prob.at<float>(0, i) = 1E-6;
		right_image_prob.at<float>(0, i) = -log(right_image_prob.at<float>(0, i)) ;
	}
	
	cv::GaussianBlur(prob, prob, cv::Size(7, 7), 0, 0);
	cv::GaussianBlur(left_image_prob, left_image_prob, cv::Size(7, 1), 0, 0);
	cv::GaussianBlur(right_image_prob, right_image_prob, cv::Size(7, 1), 0, 0);
	
}

void saveDisparityImage(int disparity_range, cv::Mat &disparity_image)
{
	cv::normalize(disparity_image, disparity_image, 1.0, 0.0, cv::NORM_MINMAX);//归一到0~1之间
	
	cv::Mat B = cv::Mat(disparity_image.rows, disparity_image.cols, CV_8UC1);
	disparity_image.convertTo(B, CV_8UC1, 255, 0); //转换为0~255之间的整数
	//for (int i = 0; i < B.rows; i++)
	//{
	//	for (int j = 0; j < B.cols; j++)
	//	//	cout << (int)B.at<uchar>(i, j) << endl;
	//}
	cv::GaussianBlur(B, B, cv::Size(7, 1), 0, 0);
	cv::imshow("B", B);//显示

	cv::waitKey();
	cv::imwrite("out2.png", B);
}

//cost aggregation
void calculateOneWayCost(const int row_diff, const int col_diff, uchar *left_image, const uchar *right_image,
	float *hlr, float *hl, float *hr, float*dev_one_path_cost, const int width, const int height,
	const int disparity_range)
{
	float max = MAX_FLOAT;

	dim3 block(disparity_range, 1);
	dim3 grid_top_down(1, width);
	dim3 grid_left_right(1, height);
	dim3 grid_slam(1, width + height);
	//choose one of ways
	//cout << row_diff + (col_diff + 1) * 3 + 1 << endl;
	switch (row_diff + (col_diff + 1) * 3 + 1)
	{
	case 0:
		//firstPath << <grid_slam, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 1:
		secondPath << <grid_top_down, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 2:
		//thirdPath << <grid_slam, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 3:
		fourthPath << <grid_left_right, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 8:
		fifthPath << <grid_left_right, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 5:
		//sixthPath << <grid_slam, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 6:
		seventhPath << <grid_top_down, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	case 7:
		//eighthPath << <grid_slam, block >> >(left_image, right_image, hlr, hl, hr, dev_one_path_cost, max, width, height, disparity_range);
		break;
	default:
		printf("path does not exist!\n");
	}

}
__global__ void aggregateCost(float * dev_aggregate_cost, const float * dev_one_path_cost, int cost_size)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int id = i + (j * blockDim.x);

	if (id < cost_size){
		dev_aggregate_cost[id] += dev_one_path_cost[id];
	}

}

__global__ void generateDisparity(const float *cost, float *disparity_image, const int width, const int height, const int disparity_range, float min_cost){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= width && row >= height)
		return;
	
	if (col < width && row < height)
	{
		float min_cost_d = min_cost;
		int min_d = 16;
		const float *v = &cost[(row*width + col)*disparity_range];
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
	int width, height;
	int total_match_pixels;

	int disparity_range = 32;

	cv::Mat original_left_image, original_right_image, down_left_image, down_right_image;
	cv::Mat prob, left_image_prob, right_image_prob;
	cv::Mat left_disparity_image, right_disparity_image;

	//read the image
	readImage(original_left_image, original_right_image, width, height);

	int cost_size = width * height * disparity_range*sizeof(float);
	int hl_size = 256 * sizeof(float);
	int hlr_size = 256 * hl_size;
	float *hlr, *hl, *hr;
	uchar *dev_left_image;
	uchar *dev_right_image;
	float  * dev_aggregate_cost;
	float  *dev_one_path_cost;

	clock_t start = clock();

	cudaMalloc((void **)&dev_one_path_cost, cost_size);
	cudaMalloc((void **)&dev_aggregate_cost, cost_size);
	cudaMalloc((void **)&hlr, hlr_size);
	cudaMalloc((void **)&hl, hl_size);
	cudaMalloc((void**)&hr, hl_size);

	//cudaMalloc((void **)dev_left_image, )


	int count = 2; // make sure 1/16 cycles 3 times
	int c1 = 2;
	int down_height = height / 16;
	int down_width = width / 16;
	int down_disparity_range = disparity_range / 16;
	left_disparity_image = cv::Mat(height / 16, width / 16, CV_32FC1, cv::Scalar(0));

	for (int c = 16; c >= 1; c /= 2)
	{
		cout << c << endl;
		prob = cv::Mat(cv::Size(256, 256), CV_32FC1, cv::Scalar(1E-4));
		left_image_prob = cv::Mat(cv::Size(256, 1), CV_32FC1, cv::Scalar(0));
		right_image_prob = cv::Mat(cv::Size(256, 1), CV_32FC1, cv::Scalar(0));

		if (c == 1)
		{
			down_left_image = original_left_image;
			down_right_image = original_right_image;
			cv::resize(left_disparity_image, left_disparity_image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
		}
		else
		{
			cv::resize(original_left_image, down_left_image, cv::Size(down_width, down_height), 0, 0, cv::INTER_AREA);
			cv::resize(original_right_image, down_right_image, cv::Size(down_width, down_height), 0, 0, cv::INTER_AREA);
			cv::resize(left_disparity_image, left_disparity_image, cv::Size(down_width, down_height), 0, 0, cv::INTER_AREA);
		}
		censusPro(total_match_pixels, down_disparity_range, prob, down_left_image, down_right_image, left_disparity_image);


		//calculateMI(down_left_image, down_right_image, left_disparity_image, prob, left_image_prob, right_image_prob, gaussian_blur2D, gaussian_blur1D, total_match_pixels);
		calculateMI(down_left_image, down_right_image, left_disparity_image, prob, left_image_prob, right_image_prob, total_match_pixels);

		int image_size = down_width*down_height*sizeof(char);
		int down_cost_size = down_width*down_height*down_disparity_range*sizeof(float);

		cudaMalloc((void **)&dev_left_image, image_size);
		cudaMalloc((void **)&dev_right_image, image_size);

		cudaMemcpy(dev_left_image, down_left_image.ptr(), image_size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_right_image, down_right_image.ptr(), image_size, cudaMemcpyHostToDevice);
		cudaMemset(dev_aggregate_cost, 0, down_cost_size);
		cudaMemcpy(hlr, prob.ptr(), hlr_size, cudaMemcpyHostToDevice);
		cudaMemcpy(hl, left_image_prob.ptr(), hl_size, cudaMemcpyHostToDevice);
		cudaMemcpy(hr, right_image_prob.ptr(), hl_size, cudaMemcpyHostToDevice);


		dim3 block(512, 1);
		dim3 grid(1, ceil(down_width*down_height*down_disparity_range / 512.));

		//8 paths
		int row_diff, col_diff;
		for (row_diff = -1; row_diff != 2; row_diff++)
		{
			for (col_diff = -1; col_diff != 2; col_diff++)
			{
				if (!row_diff && !col_diff)
					continue;
				cudaMemset(dev_one_path_cost, 0, down_cost_size);
				calculateOneWayCost(row_diff, col_diff, dev_left_image, dev_right_image, hlr, hl, hr, dev_one_path_cost, down_width, down_height, down_disparity_range);
				aggregateCost << <grid, block >> >(dev_aggregate_cost, dev_one_path_cost, down_width*down_height*down_disparity_range);
			}
		}

		dim3 grid_d(ceil((float)down_width / 32.), ceil((float)down_height / 16.));
		dim3 block_d(32, 16);

		float *dev_disparity_image;
		int disparity_size = left_disparity_image.cols*left_disparity_image.rows*sizeof(float);

		cudaMalloc((void **)&dev_disparity_image, disparity_size);
		cudaMemcpy(dev_disparity_image, left_disparity_image.ptr(), disparity_size, cudaMemcpyHostToDevice);
		generateDisparity << <grid_d, block_d >> >(dev_aggregate_cost, dev_disparity_image, down_width, down_height, down_disparity_range, MAX_FLOAT);
		cudaMemcpy(left_disparity_image.ptr(), dev_disparity_image, disparity_size, cudaMemcpyDeviceToHost);


		cv::medianBlur(left_disparity_image, left_disparity_image, (3, 3));
		
		//cv::imshow("d", left_disparity_image);
		//cv::waitKey();

		cudaFree(dev_left_image);
		cudaFree(dev_right_image);


		if (c == 1)
		{
			break;
		}
		else if (c == 16 && count)
		{
			count--;
			c = 32;
		}
        else
		{
			down_width = down_width * 2;
			down_height = down_height * 2;
			down_disparity_range = down_disparity_range * 2;
		}

	}

	//cv::imshow("d", left_disparity_image);
	////cv::imshow("r", right_disparity_image);
	//   cv::waitKey();
	cv::GaussianBlur(left_disparity_image, left_disparity_image, cv::Size(7, 1), 0, 0);
	saveDisparityImage(disparity_range, left_disparity_image);


	clock_t stop = clock();

	cout << "Time: " << (double)(stop - start) / CLK_TCK << endl;

	DisparityImage kitti_d, calculate_d, c;
	//kitti_d.writeColor("kitti.png");
    kitti_d.read("disp0.png");

	kitti_d.writeColor("tcolor.png");
	calculate_d.read("out2.png");
	calculate_d.writeColor("color.png");
	//c.read("out2.png");
	//c.writeColor("color2.png");
    kitti_d.errorImage(kitti_d, calculate_d, true).write("error.png");

	cudaFree(dev_aggregate_cost);
	cudaFree(dev_one_path_cost);
	cudaFree(hlr);
	cudaFree(hl);
	cudaFree(hr);
	getchar();
	return 0;
}

