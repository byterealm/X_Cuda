#ifndef CUDA_MATRIX
#define  CUDA_MATRIX
#include "cuda_runtime.h"
#include <stdio.h>
#include <cstdlib>
#include <typeinfo>

enum ClassType
{
	m_Uchar,
	m_Int,
	m_Float,
	m_Double
};

/**=================================================================================================
* \class	X_Matrix
*
* \brief	A matrix.
* 			m_dim the three dimemsion is defined for width heigh and deep
*
*
* \author	Bxiaoke
* \date	4/5/2016
*===============================================================================================**/

class X_Matrix
{
public:
	cudaPitchedPtr m_devicePitchedPointer;
	dim3 m_dim;
	ClassType m_type;


	/**=================================================================================================
	* \fn	template <class Type> __host__ void GenerateData(dim3 v_size)
	*
	* \brief	Generates a data. generate the matrix with v_size dimension, this is the template, only run in host code!
	*
	* \author	Bxiaoke
	* \date	4/5/2016
	*
	* \tparam	Type	Type of the type.
	* \param	v_size	The size.
	*===============================================================================================**/

	template <class Type>
	__host__ void GenerateData(dim3 v_size)
	{
		if (typeid(Type) == typeid(int))
		{
			m_type = m_Int;
		}
		else if (typeid(Type) == typeid(unsigned char))
		{
			m_type = m_Uchar;
		}
		else if (typeid(Type) == typeid(float))
		{
			m_type = m_Float;
		}
		else if (typeid(Type) == typeid(double))
		{
			m_type = m_Double;
		}
		else
		{
			return;
		}

		if (v_size.x == 0)
		{
			printf("error! dimension is equal to 0");
			system("pause");
			exit(0);
		}


		int yy = v_size.y == 0 ? 1 : v_size.y;
		int zz = v_size.z == 0 ? 1 : v_size.z;
		v_size = dim3(v_size.x, yy, zz);

		m_dim = v_size;

		// malloc 3D data;
		cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(Type) * v_size.x, v_size.y, v_size.z);
		cudaError_t error = cudaMalloc3D(&m_devicePitchedPointer, volumeSizeBytes);
		if (error != cudaSuccess)
		{
			printf("malloc failed!");
			system("pause");
			exit(0);
		}

		// init the data
		error = cudaMemset3D(m_devicePitchedPointer, 0, volumeSizeBytes);
		if (error != cudaSuccess)
		{
			printf("memory set!");
			system("pause");
			exit(0);
		}
	}

	/**=================================================================================================
	* \fn	__host__ void CopyDevice2Host_T(void* volume)
	*
	* \brief	Copies the device 2 host described by volume.    copy the matrix to volume
	*
	* \author	Bxiaoke
	* \date	4/5/2016
	*
	* \param [in,out]	volume	If non-null, the volume.
	*===============================================================================================**/

	__host__ void CopyDevice2Host(void* volume)
	{
		// check the dimension
		if (m_dim.x == 0)
		{
			printf("error! dimension is equal to 0");
			system("pause");
			exit(0);
		}

		int yy = m_dim.y == 0 ? 1 : m_dim.y;
		int zz = m_dim.z == 0 ? 1 : m_dim.z;
		m_dim = dim3(m_dim.x, yy, zz);


		int type_size = 0;

		if (m_type == m_Uchar)
			type_size = sizeof(unsigned char);
		else if (m_type == m_Int)
			type_size = sizeof(int);
		else if (m_type == m_Float)
			type_size = sizeof(float);
		else if (m_type == m_Double)
			type_size = sizeof(double);


		// set the parameter
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = m_devicePitchedPointer;
		copyParams.dstPtr = make_cudaPitchedPtr((void*)volume, m_dim.x * type_size, m_dim.x, m_dim.y);
		copyParams.extent = make_cudaExtent(type_size * m_dim.x, m_dim.y, m_dim.z);
		copyParams.kind = cudaMemcpyDeviceToHost;

		// copy the data
		cudaError_t error = cudaMemcpy3D(&copyParams);
		if (error != cudaSuccess)
		{
			printf("copy gpu to host failed!");
			system("pause");
			exit(0);
		}
	}


	__host__  void CopyHost2Device(void* volume)
	{
		// check the dimension
		if (m_dim.x == 0)
		{
			printf("error! dimension is equal to 0");
			system("pause");
			exit(0);
		}

		int yy = m_dim.y == 0 ? 1 : m_dim.y;
		int zz = m_dim.z == 0 ? 1 : m_dim.z;
		m_dim = dim3(m_dim.x, yy, zz);


		int type_size = 0;
		if (m_type == m_Uchar)
			type_size = sizeof(unsigned char);
		else if (m_type == m_Int)
			type_size = sizeof(int);
		else if (m_type == m_Float)
			type_size = sizeof(float);
		else if (m_type == m_Double)
			type_size = sizeof(double);


		// set the parameter
		cudaMemcpy3DParms copyParams = { 0 };
		//copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, M.m_dim.x * sizeof(int), M.m_dim.x, M.m_dim.y);
		copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, m_dim.x * type_size, m_dim.x, m_dim.y);
		copyParams.dstPtr = m_devicePitchedPointer;
		//copyParams.extent = make_cudaExtent(sizeof(int) * M.m_dim.x, M.m_dim.y, M.m_dim.z);
		copyParams.extent = make_cudaExtent(type_size * m_dim.x, m_dim.y, m_dim.z);
		copyParams.kind = cudaMemcpyHostToDevice;

		// copy the data
		cudaError_t error = cudaMemcpy3D(&copyParams);
		if (error != cudaSuccess)
		{
			printf("copy host to device failed!");
			system("pause");
			exit(0);
		}
	}


	/**=================================================================================================
	* \fn	__device__ void SetElem(int ii, int jj, int k, void* data)
	*
	* \brief	Sets an element. note the index of ii,jj,kk , the layer like A[jj][ii]
	*
	* \author	Bxiaoke
	* \date	4/5/2016
	*
	* \param	ii				The ii.  is the width
	* \param	jj				The jj.  is the heigh
	* \param	kk				The int to process. is the deep
	* \param [in,out]	data	If non-null, the data.
	*===============================================================================================**/

	__device__ void SetElem(int ii, int jj, int kk, void* data)
	{
		// Get attributes from device pitched pointer
		char* devicePointer = (char *)m_devicePitchedPointer.ptr;
		size_t pitch = m_devicePitchedPointer.pitch;
		size_t slicePitch = pitch * m_dim.y;
		char* current_slice = devicePointer + kk * slicePitch;

		if (m_type == m_Uchar)
		{
			unsigned char tp = *(char*)data;
			unsigned char* current_row = (unsigned char *)(current_slice + jj * pitch);
			current_row[ii] = tp;
		}
		else if (m_type == m_Int)
		{
			int tp = *(int*)data;
			int* current_row = (int *)(current_slice + jj * pitch);
			current_row[ii] = tp;
		}
		else if (m_type == m_Float)
		{
			float tp = *(float*)data;
			float* current_row = (float *)(current_slice + jj * pitch);
			current_row[ii] = tp;
		}
		else if (m_type == m_Double)
		{
			double tp = *(double*)data;
			double* current_row = (double *)(current_slice + jj * pitch);
			current_row[ii] = tp;
		}
	}


	__device__ void GetElem(int i, int j, int k, void* data)
	{
		// Get attributes from device pitched pointer
		char* devicePointer = (char *)m_devicePitchedPointer.ptr;
		size_t pitch = m_devicePitchedPointer.pitch;
		size_t slicePitch = pitch * m_dim.y;

		char* current_slice = devicePointer + k * slicePitch;


		if (m_type == m_Uchar)
		{
			unsigned char* current_row = (unsigned char *)(current_slice + j * pitch);
			*(unsigned char*)data = current_row[i];
		}
		else if (m_type == m_Int)
		{
			int* current_row = (int *)(current_slice + j * pitch);
			*(int*)data = current_row[i];
		}
		else if (m_type == m_Float)
		{
			float* current_row = (float *)(current_slice + j * pitch);
			*(float*)data = current_row[i];
		}
		else if (m_type == m_Double)
		{
			double* current_row = (double *)(current_slice + j * pitch);
			*(double*)data = current_row[i];
		}
	}
};


#endif

#define  m_test 0

#if m_test

__global__ void Matrix_Test_G(X_Matrix M, int layer, X_Matrix N)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	int dd = idx + idy*M.m_dim.x + layer*M.m_dim.x*M.m_dim.y;
	M.SetElem(idx, idy, layer, &dd);

	int tt;
	M.GetElem(idx, idy, layer, &tt);
	N.SetElem(idx, idy, layer, &tt);

}




extern "C"
void MatrixTest()
{
	int width = 20, heigh = 10, deep = 7;

	X_Matrix M;
	M.GenerateData<int>(dim3(width, heigh, deep));

	X_Matrix MT;
	MT.GenerateData<int>(dim3(width, heigh, deep));

	/** \brief	init test */
	int *volume = new int[width*heigh*deep];
	memset(volume, '\001', sizeof(int) *width*heigh*deep);
	M.CopyDevice2Host(volume);
	for (int i = 0; i < width*heigh*deep; i++)
		if (volume[i] != 0)
		{
		printf("error: init wrong!!\n");
		system("pause");
		}
	printf("Init test successful!!\n");


	for (int i = 0; i < width*heigh*deep; i++)
		volume[i] = i;
	M.CopyHost2Device(volume);
	int *volume2 = new int[width*heigh*deep];
	memset(volume2, '\001', sizeof(int) *width*heigh*deep);
	M.CopyDevice2Host(volume2);
	for (int i = 0; i < width*heigh*deep; i++)
		if (volume2[i] != i)
		{
		printf("error: copy error!!\n");
		system("pause");
		}
	printf("Copy test successful!!\n");


	for (int i = 0; i < deep; i++)
		Matrix_Test_G << <1, dim3(width, heigh) >> > (M, i, MT);

	M.CopyDevice2Host(volume2);
	for (int i = 0; i < width*heigh*deep; i++)
	{
		if (volume2[i] != i)
		{
			printf("error: assign error!!\n");
			system("pause");
		}
	}

	printf("Assign test successful!!\n");

	MT.CopyDevice2Host(volume);
	for (int i = 0; i < width*heigh*deep; i++)
	{
		if (volume[i] != i)
		{
			printf("error: assign error!!\n");
			system("pause");
		}
	}

}


#endif

