#include <cstdio>
#ifndef CUDA_VECTOR_CU
#define CUDA_VECTOR_CU
#include "cuda_runtime.h"

/**********************************************************************************************//**
 * \struct	Cuda_X_Vector
 *
 * \brief	A cuda x coordinate vector.
 * use:
 * 
 * \author	Bxiaoke
 * \date	4/4/2016
 *
 * \tparam	Type	Type of the type.
 **************************************************************************************************/

template <class Type>
struct Cuda_X_Vector
{
	typedef Type* iterator;

	int m_size;// the elem num
	int length;// the located memory

	iterator head;

	__device__ __host__ iterator begin()
	{
		return head;
	}

	__device__ __host__ iterator end()
	{
		return head + m_size;
	}

	__device__ __host__ Cuda_X_Vector() : m_size(0), length(5)
	{
		head = new Type[length + 1];
	}

	__device__ __host__ Cuda_X_Vector(const Cuda_X_Vector& C) = delete;

	__device__ __host__ ~Cuda_X_Vector()
	{
		delete[] head;
	}

	__device__ __host__ int capacity()
	{
		return length;
	}

	__device__ __host__ void reserve(int size)
	{
		if (size <= length)
			return;
		iterator tp = new Type[size + 1];
		for (int i = 0; i < m_size; i++)
			tp[i] = head[i];
		delete[] head;
		head = tp;
		length = size;
	}

	__device__ __host__ void push_back(Type value)
	{
		if (m_size + 1 <= length)
		{
			head[m_size] = value;
			m_size++;
		}
		else
		{
			reserve(length + 5);
			push_back(value);
		}
	}


	__device__ __host__ void pop_back()
	{
		if (m_size > 0)
		{
			m_size--;
		}
	}

	__device__ __host__ bool empty()
	{
		if (m_size == 0)
			return true;
		return false;
	}


	/*******************************************************************************
	*
	*                                   pop heap
	*
	**********************************************************************************/


	__device__ __host__ void x_pop_heap()//(iterator First, iterator Last)
	{
		iterator First = begin();
		iterator Last = end();
		ptrdiff_t Count = Last - First;
		if (Count > 1)
		{
			Type Val = *(Last - 1);
			*(Last - 1) = *First;
			x_Adjust_heap(First, 0, Count, Val);
		}
	}


	/*******************************************************************************
	*
	*                                   push heap
	*
	**********************************************************************************/

	__device__ __host__ void x_push_heap()//(iterator First = begin(), iterator Last = end())
	{
		iterator First = begin();
		iterator Last = end();

		ptrdiff_t Count = Last - First;
		if (Count > 0)
		{
			--Last;
			Count--;
			Type Val = *Last;
			p_Push(First, Count, 0, Val);
		}
	}

	/*******************************************************************************
	*
	*                                   make heap
	*
	**********************************************************************************/

	__device__ __host__ void p_Push(iterator First, ptrdiff_t Hole, ptrdiff_t Top, Type Val)
	{
		for (ptrdiff_t Idx = (Hole - 1) / 2;
		     Top < Hole && (*(First + Idx) < Val);
		     Idx = (Hole - 1) / 2)
		{
			*(First + Hole) = *(First + Idx);
			Hole = Idx;
		}
		*(First + Hole) = Val;
	}

	__device__ __host__ void x_Adjust_heap(iterator First, ptrdiff_t Hole, ptrdiff_t Length, Type Val)
	{
		ptrdiff_t Top = Hole;
		ptrdiff_t Idx = 2 * Hole + 2;

		for (; Idx < Length; Idx = 2 * Idx + 2)
		{
			if (*(First + Idx) < *(First + (Idx - 1)))
				--Idx;
			*(First + Hole) = *(First + Idx);
			Hole = Idx;
		}

		if (Idx == Length)
		{
			*(First + Hole) = *(First + (Length - 1));
			Hole = Length - 1;
		}
		p_Push(First, Hole, Top, Val);
	}


	__device__ __host__ void x_make_heap()//(iterator First = begin(), iterator Last = end())
	{
		iterator First = begin(); 
		iterator Last = end();
		ptrdiff_t Length = Last - First;
		if (Length > 1)
		{
			for (ptrdiff_t Hole = Length / 2; 0 < Hole;)
			{
				--Hole;
				Type Val = *(First + Hole);
				x_Adjust_heap(First, Hole, Length, Val);
			}
		}
	}
};


#ifdef bxktest

__global__ void Vt()
{
	Cuda_X_Vector<int> CV;
	for (int i = 0; i < 20; i++)
	{
		CV.push_back(i);
		//printf("CVsize: %d,CV capcity:%d \n", CV.m_size, CV.capacity());
	}
	CV.x_make_heap();
	Cuda_X_Vector<int>::iterator itr;

	for (itr = CV.begin(); itr != CV.end(); itr++)
	{
		printf("data: %d\n", *itr);
	}
}

extern "C"
void testVector()
{
	Vt <<<1, 1 >>>();
}

#endif


#endif

