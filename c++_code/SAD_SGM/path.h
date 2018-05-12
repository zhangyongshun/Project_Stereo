#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define MAX_INT  888888
#define P1 12
#define P2 32
#define PATH 8

#define MIN(a, b) a>b?b:a
#define MAX(a, b) a>b?a:b

__global__ void firstPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void secondPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void thirdPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void fourthPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void fifthPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void sixthPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void seventhPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
__global__ void eighthPath(const uchar *left_image, const uchar *right_image,
	const unsigned int *cost,
	unsigned int *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);