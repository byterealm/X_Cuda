//#include "stdlib.h"
//#include "stdio.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <curand.h>
#include <cstdlib>
#include <time.h>

#include "Cuda_Matrix.h"
#include "Cuda_HashTable.cu"
#include <curand_kernel.h>
#include <curand.h>
#include <ctime>
#include "Cuda_GetGridSize.cu"
#include "Cuda_Info.cu"
#include "XType.h"


#define XYZ_TO_INT(x, y, z) (((z)<<20)|((y)<<10)|(x))
#define INT_TO_X(v) ((v)&((1<<10)-1))
#define INT_TO_Y(v) (((v)>>10)&((1<<10)-1))
#define INT_TO_Z(v) ((v)>>20)

#define point_dist 500





__device__ bool PointDistBound(dim3 p1, dim3 p2)
{
	if ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z) < point_dist)
		return true;
	return false;
}


__host__ dim3 calcGridSize(dim3 block_size, int first, int second)
{
	int x = (first % block_size.x != 0) ? (first / block_size.x + 1) : (first / block_size.x);
	int y = (second % block_size.y != 0) ? (second / block_size.y + 1) : (second / block_size.y);
	return dim3(x, y);
}


__global__ void knn_init_device(Cuda_X_Matrix Mnn, dim3 dim_a, dim3 dim_b, int layer, int iter_num, Cuda_GetGridSize cuGSize, int knn)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	dim3 Coord = cuGSize.GetIndex(iter_num, idx, idy);

	if (Coord.x >= dim_a.x || Coord.y >= dim_a.y)
		return;


	SetElem_T_D<int>(Coord.x + knn * Coord.y, Coord.y, layer, Mnn, XYZ_TO_INT(Coord.x, Coord.y, layer));

	/*
	HashTable Hash(knn);
	
	dim3 a_p(Coord.x, Coord.y, layer);
	
	// for random 
	unsigned int seed = (unsigned int)clock64();
	curandState s;
	curand_init(seed, 0, 0, &s);
		
	for (int i = 0; i < knn; i++)
	{
		for (;;)
		{
			int xv = (1 - curand_uniform(&s)) * dim_b.x;
			int yv = (1 - curand_uniform(&s)) * dim_b.y;
			int zv = (1 - curand_uniform(&s)) * dim_b.z;
			dim3 b_p(xv, yv, zv);
			if (!PointDistBound(b_p, a_p)&& Hash.try_insert(XYZ_TO_INT(xv,yv,zv)))
			{				
				SetElem_T_D<int>(Coord.x + knn * Coord.y, Coord.y, layer, Mnn, XYZ_TO_INT(xv, yv, zv));
				break;
			}
		}
	}
	*/
	
}


extern "C"
void knn_init_nn(void* vol_a, dim3 dim_a, Cuda_X_Matrix& M_a,
                 void* vol_b, dim3 dim_b, Cuda_X_Matrix& M_b,
				 Cuda_X_Matrix& M_ann, XParams para)
{
	
	/** \brief	load the volume vol_a. */
	dim_a = dim3(dim_a.x,dim_a.y, dim_a.z);
	M_a = GenerateData(dim_a, m_Int);
	CopyHost2Device(vol_a, M_a, m_Int);


	/** \brief	laod the volume vol_b. */
	dim_b = dim3(dim_b.x, dim_b.y, dim_b.z);
	M_b = GenerateData(dim_b, m_Int);
	CopyHost2Device(vol_b, M_b, m_Int);



	/** \brief	creat the ann map */
	dim3 dim_ann(dim_a.x * para.knn, dim_a.y, dim_a.z);
	M_ann = GenerateData(dim_ann, m_Int);



	dim3 m_dim_a(dim_a.x - para.patch_w, dim_a.y - para.patch_w, dim_a.z - para.patch_w);
	dim3 m_dim_b(dim_b.x - para.patch_w, dim_b.y - para.patch_w, dim_b.z - para.patch_w);

	


	Cuda_Info m_info;
	int num_thread = m_info.getNumThread();
	Cuda_GetGridSize Cuda_Grid;
	Cuda_Grid.SetThreadNum(num_thread);
	Cuda_Grid.SetImgSize(dim3(dim_a.x,dim_a.y));
	dim3 block_size(16, 16);
	Cuda_Grid.SetBlockSize(block_size);
	dim3 gridSize = Cuda_Grid.GetGridSize();
	int iter_num = Cuda_Grid.GetNumIter();



	double dur;
	clock_t start, end;
	start = clock();


	for (int layer = 0; layer < dim_a.z;layer++)
	for (int iter = 0; iter < iter_num; iter++)
	{
		knn_init_device <<<gridSize, block_size >>>(M_ann, m_dim_a, m_dim_b, layer, iter, Cuda_Grid, para.knn);
	}
		

	end = clock();
	dur = static_cast<double>(end - start);
	printf("knn init volume cost: %f\n", (dur / CLOCKS_PER_SEC));

	

	
	//int ap;
	//GetElem(0, 0, 0, M_ann, &ap, m_Int);
	//printf("data range: %d\n", ap);

}
































#define NUM_TEX 10
//cudaTextureObject_t TextObject[NUM_TEX];

texture<int, 3, cudaReadModeElementType> vol_a;
cudaArray* volume_A;

int* Map;
cudaPitchedPtr devicePitchedPointer;


extern "C"
void CopyData(const int* volume, cudaExtent volumeSize, int index)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	cudaError_t error = cudaMalloc3DArray(&volume_A, &channelDesc, volumeSize);
	if (error != cudaSuccess)
	{
		printf("malloc failed!");
		system("pause");
		exit(0);
	}

	// copy daat to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, volumeSize.width * sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = volume_A;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	error = cudaMemcpy3D(&copyParams);
	if (error != cudaSuccess)
	{
		printf("malloc failed!");
		system("pause");
		exit(0);
	}


	// set texture parameters
	vol_a.normalized = false; // access with normalized texture coordinates
	vol_a.filterMode = cudaFilterModePoint; // linear interpolation
	vol_a.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	vol_a.addressMode[1] = cudaAddressModeClamp;
	vol_a.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	error = cudaBindTextureToArray(vol_a, volume_A, channelDesc);
	if (error != cudaSuccess)
	{
		printf("malloc failed!");
		system("pause");
		exit(0);
	}
}

extern "C"
void FreeMem()
{
	//cudaDestroyTextureObject(TextObject[0]);
	cudaFreeArray(volume_A);
}


__global__ void Knn_z(int layer)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	//printf("thread:");
	float test = tex3D(vol_a, idx, idy, layer);//by using this the error occurs	
}

/*
dim3 calcGridSize(dim3 block_size, int first, int second)
{
	int x = (first % block_size.x != 0) ? (first / block_size.x + 1) : (first / block_size.x);
	int y = (first % block_size.y != 0) ? (first / block_size.y + 1) : (first / block_size.y);
	return dim3(x, y);
}
*/


extern "C"
void PrintTexture(dim3 vol_size = dim3(16, 16, 16))
{
	dim3 block_size(16, 16);
	dim3 gridSize;

	printf("bxk");
	// z way
	gridSize = calcGridSize(block_size, vol_size.x, vol_size.y);
	for (int i = 0; i <= 99; i++)
		Knn_z<<<gridSize, block_size >>>(100);
	//cudaDeviceSynchronize();
	//cudaFreeArray(volume_A);	
}

