#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

#define PI 3.1419926535
#define PATH 8
#define P1 3
#define P2 23
#define MAX_FLOAT  3.4028235E38

using namespace std;



//\theta = 1, \delta = 0
float gaussian2D(float x, float y)
{
	return exp(-0.5 * (pow(x, 2) + pow(y, 2))) / (2 * PI);
}
void calculateGaussianBlur2D(float **gaussian_blur2D)
{
	float sum = 0;
	for (int i = 0; i < 7; i++)
		for (int j = 0; j < 7; j++)
		{
			gaussian_blur2D[i][j] = gaussian2D(i - 3, j - 3);
			sum += gaussian_blur2D[i][j];
		}
	for (int i = 0; i < 7; i++)
		for (int j = 0; j < 7; j++)
			gaussian_blur2D[i][j] /= sum;
}
float gaussian1D(float x)
{
	return exp(-0.5 * pow(x, 2)) / sqrt(2 * PI);
}
void calculateGaussianBlur1D(float *gaussian_blur1D)
{
	float sum = 0;
	for (int i = 0; i < 7; i++)
	{
		gaussian_blur1D[i] = gaussian1D(i - 3);
		sum += gaussian_blur1D[i];
	}
	for (int i = 0; i < 7; i++)
		gaussian_blur1D[i] /= sum;
}

//load left and right images which have been rectified
void readImage(cv::Mat &left_image, cv::Mat &right_image, int &width, int &height)
{
	left_image = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	right_image = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	width = left_image.cols;
	height = left_image.rows;
}

//for each image, calculate the image by down sampling, which use to make hierarchical calculation
void downSampling(int width, int height, int down_scale, cv::Mat &original_image, cv::Mat &down_image)
{
	down_image = cv::Mat(height / down_scale, width / down_scale, CV_8U, cv::Scalar(0));

	int sum = 0;
	int size = down_scale * down_scale;
	for (int i = 0; i < height; i += down_scale)
	{
		sum = 0;
		for (int j = 0; j < width; j += down_scale)
		{
			for (int k = i; k < 2 + i; k++)
				for (int w = j; w < 2 + j; w++)
					sum += original_image.at<uchar>(k, w);

			down_image.at<uchar>(i / down_scale, j / down_scale) = sum / size;
		}
	}
}
void censusPro(int &total_match_pixels,int disparity_range, float prob[][256], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &left_disparity_image)
{
	total_match_pixels = 0;
	memset(prob, 0, sizeof(float) * 256 * 256);   //test
	int width = left_image.cols;
	int height = left_image.rows;
	int d;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d = left_disparity_image.at<uchar>(row, col);
			if (d <  disparity_range && row - d >= 0)
			{
				prob[left_image.at<uchar>(row, col)][right_image.at<uchar>(row - d, col)] += 1;
				total_match_pixels += 1;
			}
		}
	}
	for (int row = 0; row < 256; row++)
		for (int col = 0; col < 256; col++)
			prob[row][col] /= total_match_pixels;
}

void calculateMI(cv::Mat &left_image, cv::Mat &right_image, cv::Mat &left_disparity_image, float prob[][256], float left_image_prob[], float right_image_prob[], float gaussian_blur2D[][7], float gaussian_blur1D[7], int total_match_pixels)
{
	float left_sum;
	float right_sum;

	//calculate P_I_1 and P_I_2
	for (int i = 0; i < 256; i++)
	{
		left_sum = right_sum = 0;
		for (int j = 0; j < 256; j++)
		{
			left_sum += prob[i][j];
			right_sum += prob[j][i];
		}
		left_image_prob[i] = left_sum;
		right_image_prob[i] = right_sum;
	}

	//calculate -log(P \otimes g) for P_I_1, P_I_2,  P_I_1_I_2
	float left_gb;
	float right_gb;
	float union_lr_gb;

	for (int i = 0; i < 256; i++)
	{
		left_gb = right_gb = union_lr_gb = 0;

		for (int j = 0; j < 256; j++)
		{
			union_lr_gb = 0;
			if ((i < 3 || i > 252) && (j < 3 || j > 252))
			{
				for (int s = 0; s < 7; s++)
				{
					for (int k = 0; k < 7; k++)
					{
						if (s - 3 + i <= 255 && s - 3 + i >= 0 && k - 3 + j >= 0 && k - 3 + j <= 255)
							union_lr_gb += prob[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
						else if (s - 3 + i <= 255 && s - 3 + i >= 0 && (k - 3 + j < 0 || k - 3 + j > 255))
							union_lr_gb += prob[s - 3 + i][j - k + 3] * gaussian_blur2D[s][k];
						else if (k - 3 + j <= 255 && k - 3 + j >= 0 && (s - 3 + i < 0 || s - 3 + i > 255))
							union_lr_gb += prob[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
						else
							union_lr_gb += prob[i - s + 3][j - k + 3] * gaussian_blur2D[s][k];
					}
				}
			}
			else
			{
				for (int s = 0; s < 7; s++)
					for (int k = 0; k < 7; k++)
						union_lr_gb += prob[i - 3 + s][j - 3 + k] * gaussian_blur2D[s][k];
			}
			prob[i][j] = -log(union_lr_gb);
		}
		if (i < 3 || i > 252)
		{
			for (int s = 0; s < 7; s++)
			{
				if (s - 3 + i >= 0 && s - 3 + i <= 255)
				{
					left_gb += left_image_prob[s - 3 + i] * gaussian_blur1D[s];
					right_gb += right_image_prob[s - 3 + i] * gaussian_blur1D[s];
				}
				else
				{
					left_gb += left_image_prob[i - s + 3] * gaussian_blur1D[s];
					right_gb += right_image_prob[i - s + 3] * gaussian_blur1D[s];
				}
			}
		}
		else
		{
			for (int s = 0; s < 7; s++)
			{
				left_gb += left_image_prob[i - 3 + s] * gaussian_blur1D[s];
				right_gb += right_image_prob[i - 3 + s] * gaussian_blur1D[s];
			}
		}
		left_image_prob[i] = -log(left_gb);
		right_image_prob[i] = -log(right_gb);
	}

	//calculate -log(P \otimes g) \otimes g
	for (int i = 0; i < 256; i++)
	{
		left_gb = right_gb = 0;

		for (int j = 0; j < 256; j++)
		{
			union_lr_gb = 0;
			if ((i < 3 || i > 252) && (j < 3 || j > 252))
			{
				for (int s = 0; s < 7; s++)
				{
					for (int k = 0; k < 7; k++)
					{
						if (s - 3 + i <= 255 && s - 3 + i >= 0 && k - 3 + j >= 0 && k - 3 + j <= 255)
							union_lr_gb += prob[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
						else if (s - 3 + i <= 255 && s - 3 + i >= 0 && (k - 3 + j < 0 || k - 3 + j > 255))
							union_lr_gb += prob[s - 3 + i][j - k + 3] * gaussian_blur2D[s][k];
						else if (k - 3 + j <= 255 && k - 3 + j >= 0 && (s - 3 + i < 0 || s - 3 + i > 255))
							union_lr_gb += prob[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
						else
							union_lr_gb += prob[i - s + 3][j - k + 3] * gaussian_blur2D[s][k];
					}
				}
			}
			else
			{
				for (int s = 0; s < 7; s++)
					for (int k = 0; k < 7; k++)
						union_lr_gb += prob[i - 3 + s][j - 3 + k] * gaussian_blur2D[s][k];
			}
			prob[i][j] = union_lr_gb / total_match_pixels;   //union entropy
		}
		if (i < 3 || i > 252)
		{
			for (int s = 0; s < 7; s++)
			{
				if (s - 3 + i >= 0 && s - 3 + i <= 255)
				{
					left_gb += left_image_prob[s - 3 + i] * gaussian_blur1D[s];
					right_gb += right_image_prob[s - 3 + i] * gaussian_blur1D[s];
				}
				else
				{
					left_gb += left_image_prob[i - s + 3] * gaussian_blur1D[s];
					right_gb += right_image_prob[i - s + 3] * gaussian_blur1D[s];
				}
			}
		}
		else
		{
			for (int s = 0; s < 7; s++)
			{
				left_gb += left_image_prob[i - 3 + s] * gaussian_blur1D[s];
				right_gb += right_image_prob[i - 3 + s] * gaussian_blur1D[s];
			}
		}
		left_image_prob[i] = left_gb / total_match_pixels;    //result of P_I_1's entropy
		right_image_prob[i] = right_gb / total_match_pixels;   //result of P_I_2's entropy
	}
}
// calculate fist way cost from top left corner to bottm right corner
void firstWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[0][i][0][d] = d ? -left_image_prob[left_image.at<uchar>(i, 0)] : prob[left_image.at<uchar>(i, 0)][right_image.at<uchar>(i, 0)] - left_image_prob[left_image.at<uchar>(i, 0)] - right_image_prob[right_image.at<uchar>(i, 0)];
		for (int j = 1; j < width; j++)
			L[0][0][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(0, j)][right_image.at<uchar>(0, j - d)] - left_image_prob[left_image.at<uchar>(0, j)] - right_image_prob[right_image.at<uchar>(0, j - d)] : -left_image_prob[left_image.at<uchar>(0, j)];
	}
	for (int i = 1; i < height; i++)
	{ 
		for (int d = 0; d < disparity_range; d++)
		{
			if (i - d >= 0)
			{
				for (int c = i + 1; c < height; c++)
				{
					L[0][c][i][d] = prob[left_image.at<uchar>(c, i)][right_image.at<uchar>(c, i - d)] - left_image_prob[left_image.at<uchar>(c, i)] - right_image_prob[right_image.at<uchar>(c, i - d)];
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d ++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[0][c - 1][i - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[0][c - 1][i - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[0][c-1][i-1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[0][c - 1][i - 1][pre_d]+P1);
						}
						else
						{
							if (i - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c - 1, i - 1) - right_image.at<uchar>(c - 1, i - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i ,gradient_intensity ? L[0][c - 1][i - 1][pre_d] + P2 / gradient_intensity : L[0][c - 1][i - 1][pre_d]);
						}
					}
					L[0][c][i][d] = L[0][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = i+1; j < width; j++)
				{
					L[0][i][j][d] = prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[0][i - 1][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[0][i - 1][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[0][i - 1][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[0][i - 1][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i - 1, j - 1) - right_image.at<uchar>(i - 1, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][i - 1][j - 1][pre_d] + P2 / gradient_intensity : L[0][i - 1][j - 1][pre_d]);
						}
					}
					L[0][i][j][d] = L[0][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for(int c = i + 1; c < height; c++)
				{
					L[0][c][i][d] = -left_image_prob[left_image.at<uchar>(c, i)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[0][c - 1][i - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[0][c - 1][i - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[0][c - 1][i - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[0][c - 1][i - 1][pre_d] + P1);
						}
						else
						{
							if (i - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c - 1, i - 1) - right_image.at<uchar>(c - 1, i - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][c - 1][i - 1][pre_d] + P2 / gradient_intensity : L[0][c - 1][i - 1][pre_d]);
						}
					}
					L[0][c][i][d] = L[0][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = i+1; j < width; j++)
				{
					L[0][i][j][d] = j - d >= 0?prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)]:-left_image.at<uchar>(i, j);
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[0][i - 1][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[0][i - 1][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[0][i - 1][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[0][i - 1][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i - 1, j - 1) - right_image.at<uchar>(i - 1, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][i - 1][j - 1][pre_d] + P2 / gradient_intensity : L[0][i - 1][j - 1][pre_d]);
						}
					}
					L[0][i][j][d] = L[0][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//calculate second way cost from top to bottom 
void secondWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int j = 0; j < width; j++)
			L[1][0][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(0, j)][right_image.at<uchar>(0, j - d)] - left_image_prob[left_image.at<uchar>(0, j)] - right_image_prob[right_image.at<uchar>(0, j - d)] : -left_image_prob[left_image.at<uchar>(0, j)];
	}
	for (int i = 1; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < disparity_range; d++)
			{
				L[1][i][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)] : -left_image.at<uchar>(i, j);
				pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
				gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
				for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
				{
					pre_min_d_all = min(pre_min_d_all, L[1][i - 1][j][pre_d]);
					if (pre_d == d)
					{
						pre_min_d = min(pre_min_d, L[1][i - 1][j][pre_d]);
					}
					else if (pre_d == d - 1)
					{
						pre_min_d__1 = min(pre_min_d__1, L[1][i - 1][j][pre_d] + P1);
					}
					else if (pre_d == d + 1)
					{
						pre_min_d_1 = min(pre_min_d_1, L[1][i - 1][j][pre_d] + P1);
					}
					else
					{
						if (j - pre_d >= 0)
							gradient_intensity = abs(left_image.at<uchar>(i - 1, j) - right_image.at<uchar>(i - 1, j - pre_d));
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[1][i - 1][j][pre_d] + P2 / gradient_intensity : L[1][i - 1][j][pre_d]);
					}
				}
				L[1][i][j][d] = L[1][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}

//calculate third way from top right corner to bottom left corner
void thirdWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[2][i][width - 1][d] = prob[left_image.at<uchar>(i, width - 1)][right_image.at<uchar>(i, width - 1 - d)] - left_image_prob[left_image.at<uchar>(i, width - 1)] - right_image_prob[right_image.at<uchar>(i, width - 1 - d)];
		for (int j = 1; j < width; j++)
			L[2][0][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(0, j)][right_image.at<uchar>(0, j - d)] - left_image_prob[left_image.at<uchar>(0, j)] - right_image_prob[right_image.at<uchar>(0, j - d)] : -left_image_prob[left_image.at<uchar>(0, j)];
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (width - 1 - i - d >= 0)
			{
				for (int c = i + 1; c < height; c++)
				{
					L[2][c][width - 1 - i][d] = prob[left_image.at<uchar>(c, width - 1 - i)][right_image.at<uchar>(c, width - 1 - i - d)] - left_image_prob[left_image.at<uchar>(c, width - 1 - i)] - right_image_prob[right_image.at<uchar>(c, width - 1 - i - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[2][c - 1][width - i][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[2][c - 1][width - i][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[2][c - 1][width - i][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[2][c - 1][width - i][pre_d] + P1);
						}
						else
						{
							if (width - i - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c - 1, width - i) - right_image.at<uchar>(c - 1, width - i - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][c - 1][width - i][pre_d] + P2 / gradient_intensity : L[2][c - 1][width - i][pre_d]);
						}
					}
					L[2][c][width - i - 1][d] = L[2][c][width - i-1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = width-2-i; j >= 0; j --)
				{
					L[2][i][j][d] = prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[2][i - 1][j + 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[2][i - 1][j + 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[2][i - 1][j+ 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[2][i - 1][j + 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i - 1, j + 1) - right_image.at<uchar>(i - 1, j + 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][i - 1][j + 1][pre_d] + P2 / gradient_intensity : L[2][i - 1][j + 1][pre_d]);
						}
					}
					L[2][i][j][d] = L[2][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int c = i + 1; c < height; c++)
				{
					L[2][c][width - 1 - i][d] = -left_image_prob[left_image.at<uchar>(c, width - 1 - i)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[2][c - 1][width - i][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[2][c - 1][width-i][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[2][c - 1][width-i][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[2][c - 1][width-i][pre_d] + P1);
						}
						else
						{
							if (width-i - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c - 1, width-i) - right_image.at<uchar>(c - 1, width-i - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][c - 1][width-i][pre_d] + P2 / gradient_intensity : L[0][c - 1][width-i][pre_d]);
						}
					}
					L[2][c][width-i-1][d] = L[2][c][width-i-1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = width-2-i; j >= 0; j --)
				{
					L[2][i][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)] : -left_image.at<uchar>(i, j);
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[2][i - 1][j + 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[2][i - 1][j + 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[2][i - 1][j + 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[2][i - 1][j + 1][pre_d] + P1);
						}
						else
						{
							if (j + 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i - 1, j + 1) - right_image.at<uchar>(i - 1, j + 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][i - 1][j + 1][pre_d] + P2 / gradient_intensity : L[2][i - 1][j + 1][pre_d]);
						}
					}
					L[2][i][j][d] = L[2][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//path from left to right
void fourthWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[3][i][0][d] = d ? -left_image_prob[left_image.at<uchar>(i, 0)] : prob[left_image.at<uchar>(i, 0)][right_image.at<uchar>(i, 0)] - left_image_prob[left_image.at<uchar>(i, 0)] - right_image_prob[right_image.at<uchar>(i, 0)];
	}
	for (int d= 0; d < disparity_range; d++)
	{
		for (int j = 1; j < width; j++)
		{
			if (j - d >= 0)
			{
				for (int i = 0; i < height; i++)
				{
					L[3][i][j][d] = prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[3][i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[3][i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[3][i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[3][i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i, j - 1) - right_image.at<uchar>(i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[3][i][j - 1][pre_d] + P2 / gradient_intensity : L[3][i][j - 1][pre_d]);
						}
					}
					L[3][i][j][d] = L[3][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					L[3][i][j][d] = -left_image.at<uchar>(i, j);
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[3][i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[3][i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[3][i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[3][i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i, j - 1) - right_image.at<uchar>(i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[3][i][j - 1][pre_d] + P2 / gradient_intensity : L[3][i][j - 1][pre_d]);
						}
					}
					L[3][i][j][d] = L[3][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}
//path from right to left
void fifthWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[4][i][width - 1][d] = prob[left_image.at<uchar>(i, width - 1)][right_image.at<uchar>(i, width - 1 - d)] - left_image_prob[left_image.at<uchar>(i, width - 1)] - right_image_prob[right_image.at<uchar>(i, width - 1 - d)];
	}
	for (int d = 0; d < disparity_range; d++)
	{
		for (int j = width - 2; j >= 0; j--)
		{
			if (j - d >= 0)
			{
				for (int i = 0; i < height; i++)
				{
					L[4][i][j][d] = prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j)] - right_image_prob[right_image.at<uchar>(i, j - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[4][i][j + 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[4][i][j + 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[4][i][j + 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[4][i][j + 1][pre_d] + P1);
						}
						else
						{
							if (j + 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i, j + 1) - right_image.at<uchar>(i, j + 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[4][i][j + 1][pre_d] + P2 / gradient_intensity : L[4][i][j + 1][pre_d]);
						}
					}
					L[4][i][j][d] = L[4][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					L[4][i][j][d] = -left_image_prob[left_image.at<uchar>(i, j)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[4][i][j + 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[4][i][j + 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[4][i][j + 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[4][i][j + 1][pre_d] + P1);
						}
						else
						{
							if (j + 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i, j + 1) - right_image.at<uchar>(i, j + 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[4][i][j + 1][pre_d] + P2 / gradient_intensity : L[4][i][j + 1][pre_d]);
						}
					}
					L[4][i][j][d] = L[4][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//from bottom left to top right
void sixthWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[5][i][0][d] = d ? -left_image_prob[left_image.at<uchar>(i, 0)] : prob[left_image.at<uchar>(i, 0)][right_image.at<uchar>(i, 0)] - left_image_prob[left_image.at<uchar>(i, 0)] - right_image_prob[right_image.at<uchar>(i, 0)];
		for (int j = 1; j < width; j++)
			L[5][height - 1][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(height - 1, j)][right_image.at<uchar>(height - 1, j - d)] - left_image_prob[left_image.at<uchar>(height - 1, j)] - right_image_prob[right_image.at<uchar>(height - 1, j - d)] : -left_image_prob[left_image.at<uchar>(height - 1, j)];
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (i - d >= 0)
			{
				for (int c = height-1-i; c >= 0; c --)
				{
					L[5][c][i][d] = prob[left_image.at<uchar>(c, i)][right_image.at<uchar>(c, i - d)] - left_image_prob[left_image.at<uchar>(c, i)] - right_image_prob[right_image.at<uchar>(c, i - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][c + 1][i - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][c + 1][i - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][c + 1][i - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][c + 1][i - 1][pre_d] + P1);
						}
						else
						{
							if (i - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c + 1, i - 1) - right_image.at<uchar>(c + 1, i - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][c + 1][i - 1][pre_d] + P2 / gradient_intensity : L[5][c + 1][i - 1][pre_d]);
						}
					}
					L[5][c][i][d] = L[5][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = i+1; j < width; j++)
				{
					L[5][height - 1 - i][j][d] = prob[left_image.at<uchar>(height - 1 - i, j)][right_image.at<uchar>(height - 1 - i, j - d)] - left_image_prob[left_image.at<uchar>(height - 1 - i, j)] - right_image_prob[right_image.at<uchar>(height - 1 - i, j - d)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][height - i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][height-i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][height-i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][height-i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(height-i, j - 1) - right_image.at<uchar>(height-i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][height-i][j - 1][pre_d] + P2 / gradient_intensity : L[5][height-i][j - 1][pre_d]);
						}
					}
					L[5][height-1-i][j][d] = L[5][height-1-i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					L[5][c][i][d] = -left_image_prob[left_image.at<uchar>(c, i)];
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][c + 1][i - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][c + 1][i - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][c + 1][i - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][c + 1][i - 1][pre_d] + P1);
						}
						else
						{
							if (i - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c + 1, i - 1) - right_image.at<uchar>(c + 1, i - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][c + 1][i - 1][pre_d] + P2 / gradient_intensity : L[5][c + 1][i - 1][pre_d]);
						}
					}
					L[5][c][i][d] = L[5][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = i+1; j < width; j++)
				{
					L[5][height - 1 - i][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(height - 1 - i, j)][right_image.at<uchar>(height - 1 - i, j - d)] - left_image_prob[left_image.at<uchar>(height - 1 - i, j)] - right_image_prob[right_image.at<uchar>(height - 1 - i, j - d)] : -left_image.at<uchar>(height - 1 - i, j);
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][height-i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][height-i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][height-i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][height-i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(height-i, j - 1) - right_image.at<uchar>(height-i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][height - i][j - 1][pre_d] + P2 / gradient_intensity : L[5][height-i][j - 1][pre_d]);
						}
					}
					L[5][height-1-i][j][d] = L[5][height-1-i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}
//from bottom to top
void seventhWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int j = 0; j < width; j++)
			L[6][height-1][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(0, j)][right_image.at<uchar>(0, j - d)] - left_image_prob[left_image.at<uchar>(0, j)] - right_image_prob[right_image.at<uchar>(0, j - d)] : -left_image_prob[left_image.at<uchar>(0, j)];
	}
	for (int i = height - 2; i >= 0; i--)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < disparity_range; d++)
			{
				L[6][i][j][d] = j-d>=0?prob[left_image.at<uchar>(i, j)][right_image.at<uchar>(i, j - d)] - left_image_prob[left_image.at<uchar>(i, j-d)] - right_image_prob[right_image.at<uchar>(i, j - d)]:-left_image.at<uchar>(i,j);
				pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
				gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
				for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
				{
					pre_min_d_all = min(pre_min_d_all, L[6][i + 1][j][pre_d]);
					if (pre_d == d)
					{
						pre_min_d = min(pre_min_d, L[6][i + 1][j][pre_d]);
					}
					else if (pre_d == d - 1)
					{
						pre_min_d__1 = min(pre_min_d__1, L[6][i + 1][j][pre_d] + P1);
					}
					else if (pre_d == d + 1)
					{
						pre_min_d_1 = min(pre_min_d_1, L[6][i + 1][j][pre_d] + P1);
					}
					else
					{
						if (j - pre_d >= 0)
							gradient_intensity = abs(left_image.at<uchar>(i + 1, j) - right_image.at<uchar>(i + 1, j - pre_d));
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[6][i + 1][j][pre_d] + P2 / gradient_intensity : L[6][i][j][pre_d]);
					}
				}
				L[6][i][j][d] = L[6][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}
//from bottom right to left
void eighthWayCost(int height, int width, int disparity_range, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[7][i][width - 1][d] = prob[left_image.at<uchar>(i, width - 1)][right_image.at<uchar>(i, width - 1 - d)] - left_image_prob[left_image.at<uchar>(i, width - 1)] - right_image_prob[right_image.at<uchar>(i, width-1-d)];
		for (int j = 1; j < width; j++)
			L[7][height-1][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(0, j)][right_image.at<uchar>(0, j - d)] - left_image_prob[left_image.at<uchar>(0, j)] - right_image_prob[right_image.at<uchar>(0, j - d)] : -left_image_prob[left_image.at<uchar>(0, j)];
	}
	for (int i = 1; i < height; i ++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (width - 1 - i - d >= 0)
			{
				for (int c = height-1-i; c >= 0; c --)
				{
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					L[7][c][width - 1 - i][d] = prob[left_image.at<uchar>(c, width - 1 - i)][right_image.at<uchar>(c, width - 1 - i - d)] - left_image_prob[left_image.at<uchar>(c, width - 1 - i)] - right_image_prob[right_image.at<uchar>(c, width - 1 - i - d)];
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[7][c + 1][width - i][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[7][c + 1][width - i][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[7][c + 1][width - i ][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[7][c + 1][width - i][pre_d] + P1);
						}
						else
						{
							if (width - i - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c + 1, width - i) - right_image.at<uchar>(c + 1, width - i - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][c + 1][width - i][pre_d] + P2 / gradient_intensity : L[7][c + 1][width - i][pre_d]);
						}
					}
					L[7][c][width - i - 1][d] = L[7][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
			}
			else
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					L[7][c][width - 1 - i][d] =  - left_image_prob[left_image.at<uchar>(c, width - 1 - i)] ;
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[7][c + 1][width - i][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[7][c + 1][width - i][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[7][c + 1][width - i][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[7][c + 1][width - i][pre_d] + P1);
						}
						else
						{
							if (width - i - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(c + 1, width - i) - right_image.at<uchar>(c + 1, width - i - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][c + 1][width - i][pre_d] + P2 / gradient_intensity : L[7][c + 1][width - i][pre_d]);
						}
					}
					L[7][c][width - i - 1][d] = L[7][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			for (int j = width-2-i; j >= 0; j --)
			{
				L[7][height - 1 - i][j][d] = j - d >= 0 ? prob[left_image.at<uchar>(height - 1 - i, j)][right_image.at<uchar>(height - 1 - i, j - d)] - left_image_prob[left_image.at<uchar>(height - 1 - i, j)] - right_image_prob[right_image.at<uchar>(height - 1 - i, j - d)] : -left_image.at<uchar>(height - 1 - i, j);
				pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
				gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
				for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
				{
					pre_min_d_all = min(pre_min_d_all, L[7][height-i][j + 1][pre_d]);
					if (pre_d == d)
					{
						pre_min_d = min(pre_min_d, L[7][height-i][j + 1][pre_d]);
					}
					else if (pre_d == d - 1)
					{
						pre_min_d__1 = min(pre_min_d__1, L[7][height-i][j + 1][pre_d] + P1);
					}
					else if (pre_d == d + 1)
					{
						pre_min_d_1 = min(pre_min_d_1, L[7][height-i][j + 1][pre_d] + P1);
					}
					else
					{
						if (j + 1 - pre_d >= 0)
							gradient_intensity = abs(left_image.at<uchar>(height-i, j + 1) - right_image.at<uchar>(height-i, j + 1 - pre_d));
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][height-i][j + 1][pre_d] + P2 / gradient_intensity : L[7][height-i][j + 1][pre_d]);
					}
				}
				L[7][height-1-i][j][d] = L[7][height-i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}
void aggregateCosts(int width, int height, float ****L, float prob[][256], float left_image_prob[], float right_image_prob[], int disparity_range, cv::Mat &left_image, cv::Mat &right_image)
{
	//initialize the border pixels in each path cost L with value C(p,d):0
	
	//
	int rowDiff;
	int colDiff;

	for (int p = 0; p < PATH; p++)
		calculateOneWayCost();
}

int main()
{
	float ***C;
	float ****L;
	float ***S;

	float gaussian_blur2D[7][7];
	float gaussian_blur1D[7];

	//cv::Matrix of probability used for store the entropy after function calculateMI()
	float prob[256][256]; //prob 256*256
	float left_image_prob[256];
	float right_image_prob[256];

	int width, height;
	int total_match_pixels;

	cv::Mat original_left_image, left_image_1_2, left_image_1_4, left_image_1_8, left_image_1_16;
	cv::Mat original_right_image, right_image_1_2, right_image_1_4, right_image_1_8, right_image_1_16;
	cv::Mat left_disparity_image;
	cv::Mat right_disparity_image;
	int disparity_range = 10;

	readImage(original_left_image, original_right_image, width, height);

	left_disparity_image = cv::Mat(height/16, width/16, CV_8U, Scalar(0));

	//malloc the memory
	C = new float **[height];    //cost of per pixel: H*W*D
	S = new float **[height];     //aggregating cost: H*W*D
	for (int row = 0; row < height; row++)
	{
		C[row] = new float *[width];
		S[row] = new float *[width];
		for (int col = 0; col < width; col++)
		{
			C[row][col] = new float[disparity_range];
			S[row][col] = new float[disparity_range];
		}
	}
	
	L = new float ***[PATH];         //cost along path: P*H*D*D
	for (int p = 0; p < PATH; p++)
	{
		L[p] = new float **[height];
		for (int row = 0; row < height; row++)
		{
			L[p][row] = new float *[width];
			for (int col = 0; col < width; col++)
			{
				L[p][row][col] = new float[disparity_range];
			}
		}
	}

	downSampling(width, height, 16, original_left_image, left_image_1_16);
	downSampling(width, height, 16, original_right_image, right_image_1_16);
	censusPro(total_match_pixels, disparity_range, prob, left_image_1_16, right_image_1_16, left_disparity_image);
	calculateMI(left_image_1_16, right_image_1_16, left_disparity_image, prob, left_image_prob, right_image_prob, gaussian_blur2D, gaussian_blur1D, total_match_pixels);
	cout << prob[1][1] << endl;
	
	getchar();
	return 0;
}

