#include "path.h"
#include "gaussian.h"

#define PATH 8

using namespace std;


//load left and right images which have been rectified
void readImage(cv::Mat &left_image, cv::Mat &right_image, int &width, int &height)
{
	left_image = cv::imread("left_image.png", CV_LOAD_IMAGE_GRAYSCALE);
	right_image = cv::imread("right_image.png", CV_LOAD_IMAGE_GRAYSCALE);
	width = left_image.cols;
	height = left_image.rows;
}

void censusPro(int &total_match_pixels,int disparity_range, cv::Mat &prob, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &left_disparity_image)
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
			prob.at<float>(row, col) = prob.at<float>(row, col)/ float(total_match_pixels);
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
			left_sum += prob.at<float>(i,j);
			right_sum += prob.at<float>(j,i);
		}
		left_image_prob.at<float>(0,i) = left_sum;
		right_image_prob.at<float>(0,i) = right_sum;
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
			prob.at<float>(i,j) = -log(prob.at<float>(i,j));
		}
		left_image_prob.at<float>(0, i) = -log(left_image_prob.at<float>(0, i));
		right_image_prob.at<float>(0, i) = -log(right_image_prob.at<float>(0, i));
	}
	cv::GaussianBlur(prob, prob, cv::Size(7, 7), 0, 0);
	cv::GaussianBlur(left_image_prob, left_image_prob, cv::Size(7, 1), 0, 0);
	cv::GaussianBlur(right_image_prob, right_image_prob, cv::Size(7, 1), 0, 0);
	//cout << prob.at<float>(0, 1) << endl;
	//for (int i = 0; i < 256; i++)
	//{
	//	left_gb = right_gb = union_lr_gb = 0;

	//	for (int j = 0; j < 256; j++)
	//	{
	//		union_lr_gb = 0;
	//		if ((i < 3 || i > 252) && (j < 3 || j > 252))
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//					if (s - 3 + i <= 255 && s - 3 + i >= 0 && k - 3 + j >= 0 && k - 3 + j <= 255)
	//						union_lr_gb += prob[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
	//					else if (s - 3 + i <= 255 && s - 3 + i >= 0 && (k - 3 + j < 0 || k - 3 + j > 255))
	//						union_lr_gb += prob[s - 3 + i][j - k + 3] * gaussian_blur2D[s][k];
	//					else if (k - 3 + j <= 255 && k - 3 + j >= 0 && (s - 3 + i < 0 || s - 3 + i > 255))
	//						union_lr_gb += prob[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += prob[i - s + 3][j - k + 3] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else if (i < 3 || i > 252)
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//					if (s - 3 + i <= 255 && s - 3 + i >= 0)
	//						union_lr_gb += prob[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += prob[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else if (j < 3 || j > 252)
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//				    if (k - 3 + j <= 255 && k - 3 + j >= 0)
	//						union_lr_gb += prob[i + s - 3][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += prob[i + s - 3][j - k + 3] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else
	//		{
	//			for (int s = 0; s < 7; s++)
	//				for (int k = 0; k < 7; k++)
	//					union_lr_gb += prob[i - 3 + s][j - 3 + k] * gaussian_blur2D[s][k];
	//		}
	//		if (union_lr_gb < 1E-10)
	//			union_lr_gb = 1E-10;
	//		cost_hlr[i][j] = -log(union_lr_gb);
	//	}
	//	if (i < 3 || i > 252)
	//	{
	//		for (int s = 0; s < 7; s++)
	//		{
	//			if (s - 3 + i >= 0 && s - 3 + i <= 255)
	//			{
	//				left_gb += left_image_prob[s - 3 + i] * gaussian_blur1D[s];
	//				right_gb += right_image_prob[s - 3 + i] * gaussian_blur1D[s];
	//			}
	//			else
	//			{
	//				left_gb += left_image_prob[i - s + 3] * gaussian_blur1D[s];
	//				right_gb += right_image_prob[i - s + 3] * gaussian_blur1D[s];
	//			}
	//		}
	//	}
	//	else
	//	{
	//		for (int s = 0; s < 7; s++)
	//		{
	//			left_gb += left_image_prob[i - 3 + s] * gaussian_blur1D[s];
	//			right_gb += right_image_prob[i - 3 + s] * gaussian_blur1D[s];
	//		}
	//	}
	//	if (left_gb < 1E-10)
	//		left_gb = 1E-10;
	//	if (right_gb < 1E-10)
	//		right_gb = 1E-10;
	//	cost_hl[i] = -log(left_gb);
	//	cost_hr[i] = -log(right_gb);
	//}

	////calculate -log(P \otimes g) \otimes g
	//for (int i = 0; i < 256; i++)
	//{
	//	left_gb = right_gb = 0;

	//	for (int j = 0; j < 256; j++)
	//	{
	//		union_lr_gb = 0;
	//		if ((i < 3 || i > 252) && (j < 3 || j > 252))
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//					if (s - 3 + i <= 255 && s - 3 + i >= 0 && k - 3 + j >= 0 && k - 3 + j <= 255)
	//						union_lr_gb += cost_hlr[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
	//					else if (s - 3 + i <= 255 && s - 3 + i >= 0 && (k - 3 + j < 0 || k - 3 + j > 255))
	//						union_lr_gb += cost_hlr[s - 3 + i][j - k + 3] * gaussian_blur2D[s][k];
	//					else if (k - 3 + j <= 255 && k - 3 + j >= 0 && (s - 3 + i < 0 || s - 3 + i > 255))
	//						union_lr_gb += cost_hlr[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += cost_hlr[i - s + 3][j - k + 3] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else if (i < 3 || i > 252)
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//					if (s - 3 + i <= 255 && s - 3 + i >= 0)
	//						union_lr_gb += cost_hlr[s - 3 + i][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += cost_hlr[i - s + 3][k - 3 + j] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else if (j < 3 || j > 252)
	//		{
	//			for (int s = 0; s < 7; s++)
	//			{
	//				for (int k = 0; k < 7; k++)
	//				{
	//					if (k - 3 + j <= 255 && k - 3 + j >= 0)
	//						union_lr_gb += cost_hlr[i + s - 3][k - 3 + j] * gaussian_blur2D[s][k];
	//					else
	//						union_lr_gb += cost_hlr[i + s - 3][j - k + 3] * gaussian_blur2D[s][k];
	//				}
	//			}
	//		}
	//		else
	//		{
	//			for (int s = 0; s < 7; s++)
	//				for (int k = 0; k < 7; k++)
	//					union_lr_gb += cost_hlr[i - 3 + s][j - 3 + k] * gaussian_blur2D[s][k];
	//		}
	//		prob.at<float>(i,j) = union_lr_gb / total_match_pixels;   //union entropy
	//	}
	//	if (i < 3 || i > 252)
	//	{
	//		for (int s = 0; s < 7; s++)
	//		{
	//			if (s - 3 + i >= 0 && s - 3 + i <= 255)
	//			{
	//				left_gb += cost_hl[s - 3 + i] * gaussian_blur1D[s];
	//				right_gb += cost_hr[s - 3 + i] * gaussian_blur1D[s];
	//			}
	//			else
	//			{
	//				left_gb += cost_hl[i - s + 3] * gaussian_blur1D[s];
	//				right_gb += cost_hr[i - s + 3] * gaussian_blur1D[s];
	//			}
	//		}
	//	}

	//	else
	//	{
	//		for (int s = 0; s < 7; s++)
	//		{
	//			left_gb += cost_hl[i - 3 + s] * gaussian_blur1D[s];
	//			right_gb += cost_hr[i - 3 + s] * gaussian_blur1D[s];
	//		}
	//	}
	//	left_image_prob[i] = left_gb / total_match_pixels;    //result of P_I_1's entropy
	//	right_image_prob[i] = right_gb / total_match_pixels;   //result of P_I_2's entropy
	//}
}

void aggregateCosts(int height, int width, int disparity_range, float ****L, cv::Mat &prob, cv::Mat &left_image_prob, cv::Mat &right_image_prob, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &disparity_image)
{
	//calculate the aggregating the cost for all pixels 
	//cout << prob.at<float>(0, 1) << endl;
	firstWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	secondWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	thirdWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	fourthWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	fifthWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	sixthWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	seventhWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
	eighthWayCost(height, width, disparity_range, L, prob, left_image_prob, right_image_prob, left_image, right_image, disparity_image);
}
void generateDisparityImageLsat(int disparity_range, float ****L, cv::Mat &left_disparity_image, cv::Mat &right_disparity_image)
{
	float left_min_cost;
	int right_min_cost;
	int left_min_d;
	int right_min_d;
	float left_total_cost;
	float right_total_cost;
	int height = left_disparity_image.rows;
	int width = right_disparity_image.cols;
	for (int i = 0; i < height; i++)
	{
		right_min_d = left_min_d = disparity_range;
		for (int j = 0; j < width; j++)
		{
			right_min_cost = MAX_FLOAT;
			left_min_d = left_disparity_image.at<float>(i, j);
			if (j + left_min_d < width)
			{
				for (int d = 0; d < disparity_range; d++)
				{
					right_total_cost = 0;
					for (int p = 0; p < PATH; p++)
					{
						right_total_cost += L[p][i][j + left_min_d][d];
					}
					if (right_total_cost < right_min_cost)
					{
						right_min_cost = right_total_cost;
						right_min_d = d;
					}
				}
				right_disparity_image.at<float>(i, j) = right_min_d;
			}
			else
			{
				right_disparity_image.at<float>(i, j) = 0;
			}
		}
	}
}
void generateDisparityImage(int disparity_range, float ****L, cv::Mat &disparity_image)
{
	float left_min_cost;
	int left_min_d;
	float left_total_cost;
	int height = disparity_image.rows;
	int width = disparity_image.cols;
	for (int i = 0; i < height; i++)
	{
		left_min_d = 0;
		for (int j = 0; j < width; j++)
		{
			left_min_cost = MAX_FLOAT;
			for (int d = 0; d < disparity_range; d++)
			{
				left_total_cost = 0;
				for (int p = 0; p < PATH; p++)
				{
					left_total_cost += L[p][i][j][d];
				}
				if (left_total_cost < left_min_cost)
				{
					left_min_cost = left_total_cost;
					left_min_d = d;
				}
			}
			disparity_image.at<float>(i, j) = left_min_d;
		}
	}
}
void saveDisparityImage(int disparity_range, cv::Mat &disparity_image)
{
	cv::imwrite("out1.png", disparity_image);
	float factor = 256.0 / (disparity_range);
	for (int row = 0; row < disparity_image.rows; ++row) {
		for (int col = 0; col < disparity_image.cols; ++col) {
			disparity_image.at<float>(row, col) *= factor;
		}
	}
	cv::imwrite("out2.png", disparity_image);
}
int main()
{
	//float ***C;
	float ****L; 
	float ***S;

	//cv::Matrix of probability used for store the entropy after function calculateMI()
	//float prob[256][256]; //prob 256*256
	//float left_image_prob[256];
	//float right_image_prob[256];

	int width, height;
	int total_match_pixels;

	cv::Mat original_left_image, down_left_image;
	cv::Mat original_right_image, down_right_image;
	cv::Mat left_disparity_image;
	cv::Mat right_disparity_image;
	cv::Mat prob, left_image_prob, right_image_prob;
	

	readImage(original_left_image, original_right_image, width, height);
	int disparity_range = 16;

	//malloc the memory
	//C = new float **[height];    //cost of per pixel: H*W*D
	S = new float **[height];     //aggregating cost: H*W*D
	for (int row = 0; row < height; row++)
	{
		//C[row] = new float *[width];
		S[row] = new float *[width];
		for (int col = 0; col < width; col++)
		{
			//C[row][col] = new float[disparity_range];
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
				for (int i = 0; i < disparity_range; i++)
				{
					L[p][row][col][i] = MAX_FLOAT;
				}
			}
		}
	}
	int count = 2; // make sure 1/16 cycles 3 times
	int down_height = height / 16;
	int down_width = width / 16;
	left_disparity_image = cv::Mat(height/16, width/16, CV_32FC1, cv::Scalar(0));
	for (int c = 16; c >= 1; c /= 2)
	{
		cout << c << endl;
		prob = cv::Mat(cv::Size(256, 256), CV_32FC1, cv::Scalar(1E-6));
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
		censusPro(total_match_pixels, disparity_range/c, prob, down_left_image, down_right_image, left_disparity_image);

		
		//calculateMI(down_left_image, down_right_image, left_disparity_image, prob, left_image_prob, right_image_prob, gaussian_blur2D, gaussian_blur1D, total_match_pixels);
		calculateMI(down_left_image, down_right_image, left_disparity_image, prob, left_image_prob, right_image_prob, total_match_pixels);
		aggregateCosts(down_height, down_width, disparity_range/c, L, prob, left_image_prob, right_image_prob, down_left_image, down_right_image, left_disparity_image);
	

		generateDisparityImage(disparity_range/c, L, left_disparity_image);

		cv::medianBlur(left_disparity_image, left_disparity_image, (3, 3));
		cv::imshow("l", down_left_image);
		cv::imshow("d", left_disparity_image);
		cv::waitKey();
		
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
		}
	}
	right_disparity_image = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
	
	generateDisparityImageLsat(disparity_range, L, left_disparity_image, right_disparity_image);
	cv::imshow("d", left_disparity_image);
	cv::imshow("r", right_disparity_image);
	cv::waitKey();
	saveDisparityImage(disparity_range, left_disparity_image);
	
	getchar();
	return 0;
}

