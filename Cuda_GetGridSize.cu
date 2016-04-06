#include "cuda_runtime.h"

/**********************************************************************************************//**
 * \class	Cuda_GetGridSize
 *
 * \brief	this class is for process the dimemsion of the grids for the kernel of the cuda
 * 
 * \author	Bxiaoke
 * \date	4/4/2016
 **************************************************************************************************/

class Cuda_GetGridSize
{

public:
	__device__ __host__ Cuda_GetGridSize() :ThreadNum(0), BlockSize(16, 16)
	{

	}

	__device__ __host__ void SetImgSize(dim3 m_size)
	{
		ImgSize = m_size;
	}


	__device__ __host__ void SetThreadNum(int num)
	{
		ThreadNum = num;
	}

	__device__ __host__ void SetBlockSize(dim3 b_size)
	{
		BlockSize = b_size;
	}


	__device__ __host__ dim3 GetGridSize()
	{
		// calc the num of the patch for the limitd thread
		int num_patch = ThreadNum / (BlockSize.x*BlockSize.y);
		GridSize = dim3(num_patch, 1, 1);
		return GridSize;
	}


	__device__ __host__ int GetNumIter()
	{
		int num_patch = ThreadNum / (BlockSize.x*BlockSize.y);

		int I_Heigh = (ImgSize.y - 1) / BlockSize.y + 1;
		int I_width = (ImgSize.x - 1) / BlockSize.x + 1;

		int iter_num = (I_Heigh*I_width - 1) / num_patch + 1;
		return iter_num;
	}

	__device__ __host__ dim3 GetIndex(int Iter, int thread_x, int thread_y)
	{
		//thread id to patch index
		int num_patch = ThreadNum / (BlockSize.x*BlockSize.y);
		int I_Heigh = (ImgSize.y - 1) / BlockSize.y + 1;
		int I_width = (ImgSize.x - 1) / BlockSize.x + 1;



		int a = Iter*num_patch;
		int b = thread_x / BlockSize.x;

		int row = (a + b) / I_width*BlockSize.y + thread_y;
		int col = (a + b) % I_width*BlockSize.x + thread_x%BlockSize.x;

		return dim3(col, row);
	}

private:
	dim3 ImgSize;
	int ThreadNum;
	dim3 BlockSize;
	dim3 GridSize;

};















/*
Cuda_GetGridSize gd;
gd.SetImgSize(dim3(128, 256));
gd.SetThreadNum(2048);
gd.SetBlockSize(dim3(15, 13));

printf("iter num: %d\n", gd.GetNumIter());
printf("get grid size: %d,%d.\n",gd.GetGridSize().x,gd.GetGridSize().y);

printf("index: %d,%d.\n", gd.GetIndex(1, 135, 1).x, gd.GetIndex(1, 135, 1).y);

*/






