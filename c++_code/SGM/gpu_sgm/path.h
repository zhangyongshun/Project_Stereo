#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define MAX_FLOAT  3.4028235E38
#define P1 1
#define P2 8
#define PATH 8

#define MIN(a, b) a>b?b:a
#define MAX(a, b) a>b?a:b


/*__global__ void firstPath(const uchar *left_image, const uchar *right_image,
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
	const unsigned int * cost,
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
	unsigned int  *dev_one_path_cost,
	unsigned int max,
	const int width, const int height, const int disparity_range);
	*/
__global__ void firstPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void secondPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void thirdPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void fourthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void fifthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void sixthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void seventhPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);
__global__ void eighthPath(const uchar *left_image, const uchar *right_image,
	const float *hlr, const float *hl, const float *hr,
	float  *dev_one_path_cost,
	float max,
	const int width, const int height, const int disparity_range);