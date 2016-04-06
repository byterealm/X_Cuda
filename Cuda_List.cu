#ifndef CUDA_LIST_H
#define CUDA_LIST_H

#include "cuda_runtime.h"

#include <stdio.h>
//#include "Cuda_Matrix.cu"

template <class Type>
struct Cuda_X_List_Elem
{
	Type value;
	Cuda_X_List_Elem *next;
	Cuda_X_List_Elem *pre;
};



// circle bidirection list

template <class Type>
struct Cuda_X_List
{
	typedef Cuda_X_List_Elem<Type> *iterator;
	typedef Cuda_X_List_Elem<Type> Elem;
	int m_size;
	iterator head;	

	__device__ __host__ Cuda_X_List(const Cuda_X_List& C) = delete;


	__device__ __host__ Cuda_X_List() :m_size(0)
	{
		head = new Elem;
		head->next = head;
		head->pre = head;
	}
	__device__ __host__ ~Cuda_X_List()
	{
		//printf("xx===============x\n");
		iterator itr = begin();
		while (itr!=end())
		{
			iterator tp = itr;
			itr = itr->next;
			//printf("data.: %d. \n", tp->value);
			delete tp;
			tp = NULL;
		}
		delete end();
		head = NULL;
	}

	__device__ __host__ iterator begin()
	{
		return head->next;
	}

	__device__ __host__ iterator end()
	{
		return head;
	}
	
	__device__ __host__ void push_back(Type value)
	{
		iterator tp = new Elem;
		tp->next = end();
		tp->pre = end()->pre;
		end()->pre->next = tp;
		end()->pre = tp;			
		tp->value = value;
		m_size++;
	}

	__device__ __host__ void pop_back()
	{
		if (m_size >0)
		{
			iterator tp = end()->pre;
			end()->pre = tp->pre;
			tp->pre->next = end();
			m_size--;
		}
	}

	__device__ __host__ void push_front(Type value)
	{
		iterator m_g = begin();
		iterator tp = new Elem;
		tp->next = m_g;
		tp->pre = m_g->pre;
		m_g->pre->next = tp;
		m_g->pre = tp;
		tp->value = value;
		m_size++;
	}

	__device__ __host__ void pop_front()
	{
		if (m_size >0)
		{
			iterator tp = begin()->next;
			end()->next = tp;
			tp->pre = end();
			m_size--;
		}
	}	

	__device__ __host__ void erase(iterator itr)
	{
		bool flag = false;
		iterator iter_1;
		for (iter_1 = begin(); iter_1 != end(); iter_1 = iter_1->next)
		{
			if (iter_1 == itr)
			{
				flag = true;
				break;
			}
		}
		if (!flag)
			return;

		if (itr!=nullptr && itr!=end())
		{
			itr->pre->next = itr->next;
			itr->next->pre = itr->pre;
			delete itr;
			m_size--;
		}
	}


	__device__ __host__ void erase_value(Type value)
	{
		iterator iter_1;
		for (iter_1 = begin(); iter_1 != end(); iter_1 = iter_1->next)
		{
			if (iter_1->value == value)
			{
				iterator tp=iter_1->pre;
				erase(iter_1);
				iter_1 = tp;
			}
		}
	}


	__device__ __host__ iterator find(Type value)
	{
		iterator iter_1;
		for (iter_1 = begin(); iter_1 != end(); iter_1 = iter_1->next)
		{			
			if (iter_1->value == value)
				return iter_1;
		}
		return nullptr;
	}

};

//#define  bxktest 1

#ifdef  bxktest

__global__ void TT(Cuda_X_Matrix M)
{
	
	Cuda_X_List<int> AList;

	AList.push_front(1);	
	AList.erase(AList.begin());
	AList.push_back(4);
	AList.push_back(5);
	AList.pop_front();
	AList.push_front(3);
	AList.push_back(6);
	AList.erase(AList.end()->pre);

	Cuda_X_List<int>::iterator iter_1;


	
	int i = 0;
	for (iter_1 = AList.begin(); iter_1 != AList.end(); iter_1 = iter_1->next)
	{
		
		SetElem_T_D<int>(0, i, 0, M, iter_1->value);
		i++;
	}
	
}


extern "C"
void testCuList()
{
	dim3 m_dim(1, 5, 1);
	Cuda_X_Matrix gpu_data = GenerateData(m_dim, m_Int);

	

	TT<<<1, 1>>>(gpu_data);
	int *data = new int[10];
	CopyDevice2Host(data, gpu_data, m_Int);
	for (int i = 0; i < 10;i++)
	{
		printf("data88: %d!\n", data[i]);
	}
	
	
}

#endif

#endif
