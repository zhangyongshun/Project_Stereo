#include "path.h"

using namespace std;

// calculate fist way cost from top left corner to bottm right corner
void firstWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[0][i][0][0] = (float)(cost_hlr.at<float>(left_image.at<uchar>(i, 0), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, 0)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0)));
		for (int j = 1; j < width; j++)
			L[0][0][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, j - d)) : L[0][0][j][d];// cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, 0));
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (i - d >= 0)
			{
				for (int c = i; c < height; c++)
				{
					L[0][c][i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, i), right_image.at<uchar>(c, i - d)) - cost_hl.at<float>(0, left_image.at<uchar>(c, i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, i - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][c - 1][i - 1][pre_d] + P2 / (gradient_intensity + 1) : L[0][c - 1][i - 1][pre_d]+P2);
						}
					}
					L[0][c][i][d] = L[0][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = i + 1; j < width; j++)
				{
					L[0][i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][i - 1][j - 1][pre_d] + P2 / (gradient_intensity + 1) : L[0][i - 1][j - 1][pre_d]+P2);
						}
					}
					L[0][i][j][d] = L[0][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int c = i; c < height; c++)
				{
					//L[0][c][i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, i), right_image.at<uchar>(c, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(c, i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][c - 1][i - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2): L[0][c - 1][i - 1][pre_d]+P2);
						}
					}
					L[0][c][i][d] = L[0][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = i + 1; j < width; j++)
				{
					L[0][i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d)) : L[0][i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[0][i - 1][j - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[0][i - 1][j - 1][pre_d] + P2);
						}
					}
					L[0][i][j][d] = L[0][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//calculate second way cost from top to bottom 
void secondWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int j = 0; j < width; j++)
			L[1][0][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, j - d)) : L[1][0][j][d];// cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, 0));
	}
	for (int i = 1; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < disparity_range; d++)
			{
				L[1][i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d)) : L[1][i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[1][i - 1][j][pre_d] + max(P2 / (gradient_intensity + 1), P1+2) : L[1][i - 1][j][pre_d]+P2);
					}
				}
				L[1][i][j][d] = L[1][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}

//calculate third way from top right corner to bottom left corner
void thirdWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[2][i][width - 1][d] = cost_hlr.at<float>(left_image.at<uchar>(i, width - 1), right_image.at<uchar>(i, width - 1 - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, width - 1)) - cost_hr.at<float>(0, right_image.at<uchar>(i, width - 1 - d));
		for (int j = 1; j < width; j++)
			L[2][0][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, j - d)) : L[2][0][j][d];// cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, 0));
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (width - 1 - i - d >= 0)
			{
				for (int c = i; c < height; c++)
				{
					L[2][c][width - 1 - i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, width - 1 - i), right_image.at<uchar>(c, width - 1 - i - d)) - cost_hl.at<float>(0, left_image.at<uchar>(c, width - 1 - i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, width - 1 - i - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][c - 1][width - i][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[2][c - 1][width - i][pre_d] + P2);
						}
					}
					L[2][c][width - i - 1][d] = L[2][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = width - 2 - i; j >= 0; j--)
				{
					L[2][i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d)) : L[2][i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(i - 1, j + 1) - right_image.at<uchar>(i - 1, j + 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][i - 1][j + 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[2][i - 1][j + 1][pre_d] + P2);
						}
					}
					L[2][i][j][d] = L[2][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int c = i; c < height; c++)
				{
				//	L[2][c][width - 1 - i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, width - 1 - i), right_image.at<uchar>(c, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(c, width - 1 - i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][c - 1][width - i][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[0][c - 1][width - i][pre_d] + P2);
						}
					}
					L[2][c][width - i - 1][d] = L[2][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = width - 2 - i; j >= 0; j--)
				{
					L[2][i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d)) : L[2][i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[2][i - 1][j + 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[2][i - 1][j + 1][pre_d] + P2);
						}
					}
					L[2][i][j][d] = L[2][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//path from left to right
void fourthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[3][i][0][0] = cost_hlr.at<float>(left_image.at<uchar>(i, 0), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, 0)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
	}
	for (int j = 1; j < width; j++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (j - d >= 0)
			{
				for (int i = 0; i < height; i++)
				{
					L[3][i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[3][i][j - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[3][i][j - 1][pre_d] + P2);
						}
					}
					L[3][i][j][d] = L[3][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					//L[3][i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[3][i][j - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[3][i][j - 1][pre_d] + P2);
						}
					}
					L[3][i][j][d] = L[3][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}
//path from right to left
void fifthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[4][i][width - 1][d] = cost_hlr.at<float>(left_image.at<uchar>(i, width - 1), right_image.at<uchar>(i, width - 1 - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, width - 1)) - cost_hr.at<float>(0, right_image.at<uchar>(i, width - 1 - d));
	}
	for (int j = width - 2; j >= 0; j--)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (j - d >= 0)
			{
				for (int i = 0; i < height; i++)
				{
					L[4][i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[4][i][j + 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[4][i][j + 1][pre_d] + P2);
						}
					}
					L[4][i][j][d] = L[4][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					//L[4][i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[4][i][j + 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[4][i][j + 1][pre_d] + P2);
						}
					}
					L[4][i][j][d] = L[4][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}

//from bottom left to top right
void sixthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[5][i][0][0] = cost_hlr.at<float>(left_image.at<uchar>(i, 0), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, 0)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
		for (int j = 1; j < width; j++)
			L[5][height - 1][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(height - 1, j), right_image.at<uchar>(height - 1, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1, j - d)) : L[5][height - 1][j][d];// cost_hlr.at<float>(left_image.at<uchar>(height - 1, j), right_image.at<uchar>(height - 1, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1, 0));
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (i - d >= 0)
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					L[5][c][i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, i), right_image.at<uchar>(c, i - d)) - cost_hl.at<float>(0, left_image.at<uchar>(c, i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, i - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][c + 1][i - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[5][c + 1][i - 1][pre_d] + P2);
						}
					}
					L[5][c][i][d] = L[5][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
				for (int j = i; j < width; j++)
				{
					L[5][height - 1 - i][j][d] = cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, j), right_image.at<uchar>(height - 1 - i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, j - d));
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][height - i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][height - i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][height - i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][height - i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(height - i, j - 1) - right_image.at<uchar>(height - i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][height - i][j - 1][pre_d] + P2 / (gradient_intensity + 1) : L[5][height - i][j - 1][pre_d]+P2);
						}
					}
					L[5][height - 1 - i][j][d] = L[5][height - 1 - i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			else
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					//L[5][c][i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, i), right_image.at<uchar>(c, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(c, i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][c + 1][i - 1][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[5][c + 1][i - 1][pre_d] + P2);
						}
					}
					L[5][c][i][d] = L[5][c][i][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
				for (int j = i; j < width; j++)
				{
					L[5][height - 1 - i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, j), right_image.at<uchar>(height - 1 - i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, j - d)) : L[5][height - 1 - i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, j), right_image.at<uchar>(height - 1 - i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, 0));
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
					for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
					{
						pre_min_d_all = min(pre_min_d_all, L[5][height - i][j - 1][pre_d]);
						if (pre_d == d)
						{
							pre_min_d = min(pre_min_d, L[5][height - i][j - 1][pre_d]);
						}
						else if (pre_d == d - 1)
						{
							pre_min_d__1 = min(pre_min_d__1, L[5][height - i][j - 1][pre_d] + P1);
						}
						else if (pre_d == d + 1)
						{
							pre_min_d_1 = min(pre_min_d_1, L[5][height - i][j - 1][pre_d] + P1);
						}
						else
						{
							if (j - 1 - pre_d >= 0)
								gradient_intensity = abs(left_image.at<uchar>(height - i, j - 1) - right_image.at<uchar>(height - i, j - 1 - pre_d));
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[5][height - i][j - 1][pre_d] + max(P2 / (gradient_intensity+1),P1+2) : L[5][height - i][j - 1][pre_d] + P2);
						}
					}
					L[5][height - 1 - i][j][d] = L[5][height - 1 - i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
		}
	}
}
//from bottom to top
void seventhWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int j = 0; j < width; j++)
			L[6][height - 1][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, j - d)) : L[6][height - 1][j][d];// cost_hlr.at<float>(left_image.at<uchar>(0, j), right_image.at<uchar>(0, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(0, j)) - cost_hr.at<float>(0, right_image.at<uchar>(0, 0));
	}
	for (int i = height - 2; i >= 0; i--)
	{
		for (int j = 0; j < width; j++)
		{
			for (int d = 0; d < disparity_range; d++)
			{
				L[6][i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, j - d)) : L[6][i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(i, j), right_image.at<uchar>(i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(i, 0));
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
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[6][i + 1][j][pre_d] + max(P2 / (gradient_intensity + 1),P1+2) : L[6][i][j][pre_d] +P2);
					}
				}
				L[6][i][j][d] = L[6][i][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}
//from bottom right to left
void eighthWayCost(int height, int width, int disparity_range, float ****L, cv::Mat &cost_hlr, cv::Mat &cost_hl, cv::Mat &cost_hr, cv::Mat &left_image, cv::Mat &right_image, cv::Mat &dispatity_image)
{
	float pre_min_d, pre_min_d_1, pre_min_d__1, pre_min_d_i, pre_min_d_all;
	int gradient_intensity;
	for (int d = 0; d < disparity_range; d++)                      //initailize the border of the image with search path
	{
		for (int i = 0; i < height; i++)
			L[7][i][width - 1][d] = cost_hlr.at<float>(left_image.at<uchar>(i, width - 1), right_image.at<uchar>(i, width - 1 - d)) - cost_hl.at<float>(0, left_image.at<uchar>(i, width - 1)) - cost_hr.at<float>(0, right_image.at<uchar>(i, width - 1 - d));
		for (int j = 0; j < width - 1; j++)
			L[7][height - 1][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(height - 1, j), right_image.at<uchar>(height - 1, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1, j - d)) : L[7][height - 1][j][d];// cost_hlr.at<float>(left_image.at<uchar>(height - 1, j), right_image.at<uchar>(height - 1, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1, 0));
	}
	for (int i = 1; i < height; i++)
	{
		for (int d = 0; d < disparity_range; d++)
		{
			if (width - 1 - i - d >= 0)
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
					L[7][c][width - 1 - i][d] = cost_hlr.at<float>(left_image.at<uchar>(c, width - 1 - i), right_image.at<uchar>(c, width - 1 - i - d)) - cost_hl.at<float>(0, left_image.at<uchar>(c, width - 1 - i)) - cost_hr.at<float>(0, right_image.at<uchar>(c, width - 1 - i - d));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][c + 1][width - i][pre_d] + P2 / (gradient_intensity + 1) : L[7][c + 1][width - i][pre_d]+P2);
						}
					}
					L[7][c][width - i - 1][d] = L[7][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;

				}
			}
			else
			{
				for (int c = height - 1 - i; c >= 0; c--)
				{
					//L[7][c][width - 1 - i][d] = cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, width - 1 - i), right_image.at<uchar>(height - 1 - i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, width - 1 - i)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, 0));
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
							pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][c + 1][width - i][pre_d] + max(P2 / (gradient_intensity + 1), P1 + 2) : L[7][c + 1][width - i][pre_d] + P2);
						}
					}
					L[7][c][width - i - 1][d] = L[7][c][width - i - 1][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
				}
			}
			for (int j = width - 2 - i; j >= 0; j--)
			{
				L[7][height - 1 - i][j][d] = j - d >= 0 ? cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, j), right_image.at<uchar>(height - 1 - i, j - d)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, j - d)) : L[7][height - 1 - i][j][d];// cost_hlr.at<float>(left_image.at<uchar>(height - 1 - i, j), right_image.at<uchar>(height - 1 - i, 0)) - cost_hl.at<float>(0, left_image.at<uchar>(height - 1 - i, j)) - cost_hr.at<float>(0, right_image.at<uchar>(height - 1 - i, 0));
				pre_min_d = pre_min_d_1 = pre_min_d__1 = pre_min_d_i = pre_min_d_all = MAX_FLOAT;
				gradient_intensity = 0;          //gradient of intensitiy, used to change the value of P2
				for (int pre_d = 0; pre_d < disparity_range; pre_d++)              //census neighbor's cost with different disparity
				{
					pre_min_d_all = min(pre_min_d_all, L[7][height - i][j + 1][pre_d]);
					if (pre_d == d)
					{
						pre_min_d = min(pre_min_d, L[7][height - i][j + 1][pre_d]);
					}
					else if (pre_d == d - 1)
					{
						pre_min_d__1 = min(pre_min_d__1, L[7][height - i][j + 1][pre_d] + P1);
					}
					else if (pre_d == d + 1)
					{
						pre_min_d_1 = min(pre_min_d_1, L[7][height - i][j + 1][pre_d] + P1);
					}
					else
					{
						if (j + 1 - pre_d >= 0)
							gradient_intensity = abs(left_image.at<uchar>(height - i, j + 1) - right_image.at<uchar>(height - i, j + 1 - pre_d));
						pre_min_d_i = min(pre_min_d_i, gradient_intensity ? L[7][height - i][j + 1][pre_d] + max(P2 / (gradient_intensity+1), P1+2) : L[7][height - i][j + 1][pre_d]+P2);
					}
				}
				L[7][height - 1 - i][j][d] = L[7][height - i-1][j][d] + min(min(min(pre_min_d, pre_min_d_1), pre_min_d__1), pre_min_d_i) - pre_min_d_all;
			}
		}
	}
}