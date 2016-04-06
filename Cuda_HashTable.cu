#ifndef CUDA_HASH_TABLE_H
#define  CUDA_HASH_TABLE_H
#include "cuda_runtime.h"
#include <stdio.h>
#include <curand.h>
#include <cstdlib>
#include "Cuda_Matrix.h"
#include "Cuda_List.cu"


struct HashTable
{
	int count;
	Cuda_X_List<int>* Entries;
	Cuda_X_List<int> Pool;


	__device__ __host__ HashTable(int entries)
	{
		count = entries;
		Entries = new Cuda_X_List<int>[entries];
	}

	__device__ __host__ ~HashTable()
	{
		//printf("auto virtual!");
		//for (int i = 0; i < count; i++)
			//delete &Entries[i];

		delete[] Entries;
	}

	__device__ __host__ int hash(int value)
	{
		return value % count;
	}

	// true for success inserted
	__device__ __host__ int try_insert(int value)
	{
		int key = hash(value);

		Cuda_X_List<int>::iterator iter = (Entries[key]).find(value);

		if (iter != nullptr)
			return 0;

		(Entries[key]).push_back(value);
		Pool.push_back(value);
		//Pool.push_back(999);
		return 1;
	}

	// true for exit, false for non exit
	__device__ __host__ int contains(int value)
	{
		int key = hash(value);

		Cuda_X_List<int>* pt;
		pt = &Entries[key];
		Cuda_X_List<int>::iterator iter = pt->find(value);

		if (iter != nullptr)
			return 1;

		return 0;
	}


	__device__ __host__ void remove(int value)
	{
		int key = hash(value);
		Cuda_X_List<int>* pt;
		pt = &Entries[key];
		pt->erase_value(value);
		Pool.erase_value(value);
	}
};


#ifdef bxktest


__global__ void HH(Cuda_X_Matrix M)
{
	HashTable Ha(5);
	Ha.try_insert(1000003);
	Ha.try_insert(1000004);
	Ha.try_insert(1000005);
	Ha.try_insert(10002303);
	if (Ha.try_insert(1000003) == 0)
	{
		printf("insert false\n");
	}
	else
	{
		printf("insert success\n");
	}

	Ha.remove(1000004);
	if (Ha.contains(1000004))
	{
		printf("good exit!!\n");
	}
	else
	{
		printf("bad!!\n");
	}


	Cuda_X_List<int> &PP = Ha.Pool;
	Cuda_X_List<int>::iterator itr;

	for (itr = PP.begin(); itr != PP.end(); itr = itr->next)
	{
		printf("Hash data: %d !\n", itr->value);
	}
}


extern "C"
void checkHash()
{
	dim3 m_dim(1, 4, 1);
	Cuda_X_Matrix gpu_data = GenerateData(m_dim, m_Int);
	HH <<<1, 1 >>>(gpu_data);


}
#endif


#endif

