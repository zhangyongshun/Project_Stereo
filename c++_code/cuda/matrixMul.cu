
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define TILE_WIDTH 16

using namespace std;

__global__ void matrixMulKernel(float *M, float *N, float *D, int width_a, int width_b)
{
	__shared__ float shared_m[TILE_WIDTH][TILE_WIDTH];
	__shared__ float shared_n[TILE_WIDTH][TILE_WIDTH];
	
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int threads_x = threadIdx.x;
	int threads_y = threadIdx.y;
	
	int row = threads_y + block_y * blockDim.y;
	int column = threads_x + block_x * blockDim.x;
    
	float temp_multi = 0.0;

	int a_begin = width_a * TILE_WIDTH * block_y;
	int a_step = TILE_WIDTH;
	int a_end = a_begin + width_a-1;

	int b_begin = TILE_WIDTH * block_x;
	int b_step = TILE_WIDTH * width_b;

	for (int a = a_begin, b = b_begin; a < a_end; a+=a_step, b+=b_step)
	{
		shared_m[threads_y][threads_x] = M[a + width_a * threads_y + threads_x];
		shared_n[threads_y][threads_x] = N[b + threads_y * width_b + threads_x];
		__syncthreads();

		for (int j = 0; j<TILE_WIDTH; j++)
			temp_multi += shared_m[threads_y][j] * shared_n[j][threads_x];
		__syncthreads();
	}
	D[width_b * TILE_WIDTH * block_y + TILE_WIDTH * block_x + threads_x + width_b * threads_y] = temp_multi;
}

/*
*function matrixMul() use to accelerate the matrix multiply by gpu

matrix_a, matrix_b are the matrices input, which dimensions are n*m and m*w
matrix_c is the output of matrix multiplication
*/
void matirxMul(float *matrix_a, float *matrix_b, float *matrix_c, int width_a, int width_b, int height_a)
{
	float *cuda_a, *cuda_b, *cuda_c;

	unsigned int size_a = height_a * width_a;
	unsigned int size_b = width_a * width_b;
	unsigned int size_c = height_a * width_b;
	
	cudaError_t cudaStatus;
    // malloc the memory of cuda matrices
	cudaStatus = cudaMalloc((void **) &cuda_a, size_a * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda_a memory alloc error"<<endl;
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMalloc((void **)&cuda_b, size_b * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda_b memory alloc error" << endl;
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMalloc((void **)&cuda_c, size_c * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda_c memory alloc error" << endl;
		exit(EXIT_FAILURE);
	}
	//copy data
	cudaStatus = cudaMemcpy(cuda_a, matrix_a, size_a * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda_a memory copy error" << endl;
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMemcpy(cuda_b, matrix_b, size_b * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda_b memory copy error" << endl;
		exit(EXIT_FAILURE);
	}
	
	dim3 threads(TILE_WIDTH, TILE_WIDTH);
	dim3 grid(width_b / TILE_WIDTH, height_a / TILE_WIDTH);

	matrixMulKernel << <grid, threads >> >(cuda_a, cuda_b, cuda_c, width_a, width_b);
	cout << "cuda_c" << endl;
	cudaStatus = cudaMemcpy(matrix_c, cuda_c, size_c * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "matrix_c copy error" << endl;
	}
	for (int i = 0; i < height_a; i++)
	{
		for (int j = 0; j < width_b; j++)
			cout << matrix_c[i * width_b + j];
		cout << endl;
	}	
}
int main()
{

    const int arraySize = 16*16;
	float a[arraySize];
	float b[arraySize];
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			a[16 * i + j] = 1;
			b[16 * i + j] = 1;
		}
	}
    float c[arraySize] = { 0 };

	matirxMul(a, b, c, 16, 16, 16);
	getchar();
    return 0;
}

