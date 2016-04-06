#include <cstdio>
#include <cuda_runtime_api.h>
#ifndef  CUDA_INFO_CU
#define  CUDA_INFO_CU

struct Cuda_Info
{
	void PrintInfo()
	{
		printf("\n\n\n");

		printf("===========================GPU=========================\n");
		int gpuNum;
		cudaGetDeviceCount(&gpuNum);
		printf("The gpu device Num:\t\t\t%d\n", gpuNum);

		cudaDeviceProp P;
		cudaGetDeviceProperties(&P, 0);

		printf("Compute capability: \t\t\t%d.%d.\n", P.major, P.minor);
		int num_cores = getSPcores(P);
		printf("The number of Cores: \t\t\t%d.\n", num_cores);
		printf("The number of the multiprocessor:\t%d\n", P.multiProcessorCount);
		printf("Max thread per multiprocessor:\t\t%d\n", P.maxThreadsPerMultiProcessor);
		printf("Max thread Per Block:\t\t\t%d\n", P.maxThreadsPerBlock);
		printf("Max thread dim:\t\t\t\t%d,%d,%d;\n", P.maxThreadsDim[0], P.maxThreadsDim[1], P.maxThreadsDim[2]);
		printf("Max grid size:\t\t\t\t%d,%d;\n", P.maxGridSize[0], P.maxGridSize[1]);
		printf("thread per warp:\t\t\t%d.\n", P.warpSize);

		printf("\n\n===========================GPU Memory=========================\n");
		printf("Total global memory:\t\t\t\t%llubytes\n", P.totalGlobalMem);
		printf("Total amount of shared memory per block:\t%llu\n", P.sharedMemPerBlock);
		printf("Total registers per block:\t\t\t%d\n", P.regsPerBlock);
		printf("Maximum memory pitch:\t\t\t\t%llu\n", P.memPitch);
		printf("Total amount of constant memory:\t\t%llu\n", P.totalConstMem);
		printf("\n\n\n");
	} 
 

	int getNumThread()
	{
		cudaDeviceProp P;
		cudaGetDeviceProperties(&P, 0);
		return getSPcores(P);
	}


	int getSPcores(cudaDeviceProp devProp)
	{
		int cores = 0;
		int mp = devProp.multiProcessorCount;
		switch (devProp.major){
		case 2: // Fermi
			if (devProp.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			break;
		case 3: // Kepler
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		default:
			printf("Unknown device type\n");
			break;
		}
		return cores;
	}



};



#endif
