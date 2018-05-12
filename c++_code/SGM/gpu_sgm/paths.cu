#include "path.h"


/*****
there are total 8 paths, which can be discribed as

row\col  -1         0       1
-1      (-1,-1)  (-1,0)    (-1,1)
0       (0,-1)  ` invalid   (0,1)
1       (1,-1)    (1,0)     (1,1)

8 paths can be calculate by eight ways
*****/

__device__ void calculateL(const float *pre_cost,
	int gradient_intensity, float *curr_cost,
	const int width, const int height, const int disparity_range, float max_float)
{
	int min_cost = max_float;
	int d = threadIdx.x;

	int pre_min_cost = max_float;
	for (int pre_d = 0; pre_d < disparity_range; pre_d++)
	{
		min_cost = MIN(pre_cost[pre_d], min_cost);
		if (pre_d - d == 0) {
			pre_min_cost = MIN(pre_min_cost, pre_cost[pre_d]);
		}
		else if (abs(pre_d - d) == 1) {
			pre_min_cost = MIN(pre_min_cost, pre_cost[pre_d] + P1);
		}
		else {
			pre_min_cost = MIN(pre_min_cost, pre_cost[pre_d] + MAX(P1 + 2, gradient_intensity ? P2 / gradient_intensity : P2));
		}
	}
	curr_cost[d] += pre_min_cost - min_cost;
	//printf("%f\n", curr_cost[d]);
}


__global__ void firstPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col_height = blockIdx.y * blockDim.y + threadIdx.y;

	if (col_height >= width + height || d >= disparity_range)
		return;
	if (col_height < width)
	{
		for (int index = 0; index + col_height < width && index < height; index++)
		{
			dev_one_path_cost[(index + col_height + (index)*width)*disparity_range + d] = 0;
			if (index + col_height - d >= 0)dev_one_path_cost[(index + col_height + (index)*width)*disparity_range + d] = hlr[left_image[(index)*width + (index + col_height)] * 256 + right_image[(index)*width + ((index + col_height) - d)]] - hl[left_image[(index)*width + (index + col_height)]] - hr[right_image[(index)*width + (index + col_height) - d]];
			dev_one_path_cost[(index + col_height + (index)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(index + col_height + (index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(index + col_height - 1 + (index - 1) * width)*disparity_range], abs(left_image[(index - 1)*width + index + col_height - 1] - left_image[index*width + index + col_height]),
				&dev_one_path_cost[(index + col_height + index*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
	else
	{
		col_height -= width;
		for (int index = 0; index < width && index + col_height < height; index++)
		{
			dev_one_path_cost[(index + (index + col_height)*width)*disparity_range + d] = 0;
			if (index - d >= 0)dev_one_path_cost[(index + (index + col_height)*width)*disparity_range + d] = hlr[left_image[(index + col_height)*width + (index)] * 256 + right_image[(index + col_height)*width + ((index)-d)]] - hl[left_image[(index + col_height)*width + (index)]] - hr[right_image[(index + col_height)*width + (index)-d]];
			dev_one_path_cost[(index + (index + col_height)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(index + (index + col_height)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(index - 1 + (index + col_height - 1) * width)*disparity_range], abs(left_image[(index + col_height - 1)*width + index - 1] - left_image[(index + col_height)*width + index]),
				&dev_one_path_cost[(index + (index + col_height)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
}
__global__ void secondPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= width || d >= disparity_range)
		return;
	for (int row = 0; row < height; row++)
	{
		dev_one_path_cost[(col + row*width)*disparity_range + d] = 0;
		if (col - d >= 0)dev_one_path_cost[(col + row*width)*disparity_range + d] = hlr[left_image[row*width + col] * 256 + right_image[row*width + (col - d)]] - hl[left_image[row*width + col]] - hr[right_image[row*width + col - d]];
		dev_one_path_cost[(col + row*width)*disparity_range + d] += 3;
		dev_one_path_cost[(col + row*width)*disparity_range + d] *= 10;
		if (row)
			calculateL(&dev_one_path_cost[(col + (row - 1) * width)*disparity_range], abs(left_image[(row - 1)*width + col] - left_image[row*width + col]),
			&dev_one_path_cost[(col + row*width)*disparity_range], col, row - 1, disparity_range, MAX_FLOAT);
		__syncthreads();
	}
}
__global__ void thirdPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col_height = blockIdx.y * blockDim.y + threadIdx.y;

	if (col_height >= width + height || d >= disparity_range)
		return;
	if (col_height < width)
	{
		for (int index = 0; col_height - index >= 0 && index < height; index++)
		{
			dev_one_path_cost[(col_height - index + (index)*width)*disparity_range + d] = 0;
			if (col_height - index - d >= 0)dev_one_path_cost[(col_height - index + (index)*width)*disparity_range + d] = hlr[left_image[(index)*width + (col_height - index)] * 256 + right_image[(index)*width + ((col_height - index) - d)]] - hl[left_image[(index)*width + (col_height - index)]] - hr[right_image[(index)*width + (col_height - index) - d]];
			dev_one_path_cost[(col_height - index + (index)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(col_height - index + (index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(col_height - index + 1 + (index - 1) * width)*disparity_range], abs(left_image[(index - 1)*width + col_height - index + 1] - left_image[index*width + col_height - index]),
				&dev_one_path_cost[(col_height - index + index*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
	else
	{
		col_height -= width;
		for (int index = 0; width - 1 - index >= 0 && index + col_height < height; index++)
		{
			dev_one_path_cost[(width - 1 - index + (index + col_height)*width)*disparity_range + d] = 0;
			if (width - 1 - index - d >= 0)dev_one_path_cost[(width - 1 - index + (index + col_height)*width)*disparity_range + d] = hlr[left_image[(index + col_height)*width + (width - 1 - index)] * 256 + right_image[(index + col_height)*width + ((width - 1 - index) - d)]] - hl[left_image[(index + col_height)*width + (width - 1 - index)]] - hr[right_image[(index + col_height)*width + (width - 1 - index) - d]];
			dev_one_path_cost[(width - 1 - index + (index + col_height)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(width - 1 - index + (index + col_height)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(width - 1 - index + 1 + (index + col_height - 1) * width)*disparity_range], abs(left_image[(index + col_height - 1)*width + width - 1 - index + 1] - left_image[(index + col_height)*width + width - 1 - index]),
				&dev_one_path_cost[(index + (index + col_height)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
}
__global__ void fourthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= height || d >= disparity_range)
		return;
	for (int col = 0; col < width; col++)
	{
		dev_one_path_cost[(col + row*width)*disparity_range + d] = 0;
		if (col - d >= 0)dev_one_path_cost[(col + row*width)*disparity_range + d] = hlr[left_image[row*width + col] * 256 + right_image[row*width + (col - d)]] - hl[left_image[row*width + col]] - hr[right_image[row*width + col - d]];
		dev_one_path_cost[(col + row*width)*disparity_range + d] += 3;
		dev_one_path_cost[(col + row*width)*disparity_range + d] *= 10;
		//printf("%f\n", dev_one_path_cost[(col + row*width)*disparity_range + d]);
		if (col)
			calculateL(&dev_one_path_cost[(col - 1 + row * width)*disparity_range], abs(left_image[row*width + col] - left_image[row*width + col - 1]),
			&dev_one_path_cost[(col + row*width)*disparity_range], col - 1, row, disparity_range, MAX_FLOAT);
		__syncthreads();
	}
}
__global__ void fifthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= height || d >= disparity_range)
		return;
	for (int col = width - 1; col >= 0; col--)
	{
		dev_one_path_cost[(col + row*width)*disparity_range + d] = 0;
		if (col - d >= 0)dev_one_path_cost[(col + row*width)*disparity_range + d] = hlr[left_image[row*width + col] * 256 + right_image[row*width + (col - d)]] - hl[left_image[row*width + col]] - hr[right_image[row*width + col - d]];
		dev_one_path_cost[(col + row*width)*disparity_range + d] += 3;
		dev_one_path_cost[(col + row*width)*disparity_range + d] *= 10;
		if (col != width - 1)
			calculateL(&dev_one_path_cost[(col + 1 + row * width)*disparity_range], abs(left_image[row*width + col] - left_image[row*width + col + 1]),
			&dev_one_path_cost[(col + row*width)*disparity_range], col + 1, row, disparity_range, MAX_FLOAT);
		__syncthreads();
	}
}
__global__ void sixthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col_height = blockIdx.y * blockDim.y + threadIdx.y;

	if (col_height >= width + height || d >= disparity_range)
		return;
	if (col_height < width)
	{
		for (int index = 0; col_height + index <width && height - 1 - index >= 0; index++)
		{
			dev_one_path_cost[(col_height + index + (height - 1 - index)*width)*disparity_range + d] = 0;
			if (col_height + index - d >= 0)dev_one_path_cost[(col_height + index + (height - 1 - index)*width)*disparity_range + d] = hlr[left_image[(height - 1 - index)*width + (col_height + index)] * 256 + right_image[(height - 1 - index)*width + ((col_height + index) - d)]] - hl[left_image[(height - 1 - index)*width + (col_height + index)]] - hr[right_image[(height - 1 - index)*width + (col_height + index) - d]];
			dev_one_path_cost[(col_height + index + (height - 1 - index)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(col_height + index + (height - 1 - index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(col_height + index - 1 + (height - index) * width)*disparity_range], abs(left_image[(height - index)*width + col_height + index - 1] - left_image[(height - index - 1)*width + col_height + index]),
				&dev_one_path_cost[(col_height + index + (height - 1 - index)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
	else
	{
		col_height -= width;
		for (int index = 0; index < width && col_height - index >= 0; index++)
		{
			dev_one_path_cost[(index + (col_height - index)*width)*disparity_range + d] = 0;

			if (index - d >= 0)dev_one_path_cost[(index + (col_height - index)*width)*disparity_range + d] = hlr[left_image[(col_height - index)*width + (index)] * 256 + right_image[(col_height - index)*width + ((index)-d)]] - hl[left_image[(col_height - index)*width + (index)]] - hr[right_image[(col_height - index)*width + (index)-d]];
			dev_one_path_cost[(index + (col_height - index)*width)*disparity_range + d] += 3;
			dev_one_path_cost[(index + (col_height - index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(index - 1 + (col_height - index + 1) * width)*disparity_range], abs(left_image[(col_height - index + 1)*width + index - 1] - left_image[(col_height - index)*width + index]),
				&dev_one_path_cost[(index + (col_height - index)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
}
__global__ void seventhPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= width || d >= disparity_range)
		return;
	for (int row = height - 1; row >= 0; row--)
	{
		dev_one_path_cost[(col + row*width)*disparity_range + d] = 0;
		if (col - d >= 0)dev_one_path_cost[(col + row*width)*disparity_range + d] = hlr[left_image[row*width + col] * 256 + right_image[row*width + (col - d)]] - hl[left_image[row*width + col]] - hr[right_image[row*width + col - d]];
		dev_one_path_cost[(col + row*width)*disparity_range + d] += 3;
		//dev_one_path_cost[(col + row*width)*disparity_range + d] *= 10;
		if (row != height - 1)
			calculateL(&dev_one_path_cost[(col + (row + 1)* width)*disparity_range], abs(left_image[row*width + col] - left_image[(row + 1)*width + col]),
			&dev_one_path_cost[(col + row*width)*disparity_range], col, row + 1, disparity_range, MAX_FLOAT);
		__syncthreads();
	}
}
__global__ void eighthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int col_height = blockIdx.y * blockDim.y + threadIdx.y;

	if (col_height >= width + height || d >= disparity_range)
		return;
	if (col_height < width)
	{
		for (int index = 0; col_height - index >= 0 && height - 1 - index >= 0; index++)
		{
			dev_one_path_cost[(col_height - index + (height - 1 - index)*width)*disparity_range + d] = 0;
			if (col_height - index - d >= 0)dev_one_path_cost[(col_height - index + (height - 1 - index)*width)*disparity_range + d] = hlr[left_image[(height - 1 - index)*width + (col_height - index)] * 256 + right_image[(height - 1 - index)*width + ((col_height - index) - d)]] - hl[left_image[(height - 1 - index)*width + (col_height - index)]] - hr[right_image[(height - 1 - index)*width + (col_height - index) - d]];
			dev_one_path_cost[(col_height - index + (height - 1 - index)*width)*disparity_range + d] += 3;
			//dev_one_path_cost[(col_height - index + (height - 1 - index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(col_height - index + 1 + (height - index + 1) * width)*disparity_range], abs(left_image[(height - index)*width + col_height - index + 1] - left_image[(height - index - 1)*width + col_height - index]),
				&dev_one_path_cost[(col_height - index + (height - 1 - index)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
	else
	{
		col_height -= width;
		for (int index = 0; width - 1 - index >= 0 && col_height - index >= 0; index++)
		{
			dev_one_path_cost[(width - 1 - index + (col_height - index)*width)*disparity_range + d] = 0;
			if (width - 1 - index - d >= 0)dev_one_path_cost[(width - 1 - index + (col_height - index)*width)*disparity_range + d] = hlr[left_image[(col_height - index)*width + (width - 1 - index)] * 256 + right_image[(col_height - index)*width + ((width - 1 - index) - d)]] - hl[left_image[(col_height - index)*width + (width - 1 - index)]] - hr[right_image[(col_height - index)*width + (width - 1 - index) - d]];
			dev_one_path_cost[(width - 1 - index + (col_height - index)*width)*disparity_range + d] += 3;
			//dev_one_path_cost[(width - 1 - index + (col_height - index)*width)*disparity_range + d] *= 10;
			if (index)
				calculateL(&dev_one_path_cost[(width - index + (col_height - index + 1) * width)*disparity_range], abs(left_image[(col_height - index + 1)*width + width - index] - left_image[(col_height - index)*width + width - 1 - index]),
				&dev_one_path_cost[(width - 1 - index + (col_height - index)*width)*disparity_range], width, height, disparity_range, MAX_FLOAT);
			__syncthreads();
		}
	}
}