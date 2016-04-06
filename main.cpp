#include "cuda_runtime.h"
#include <cstdio>
#include <windows.h>
#include "volume.h"
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <direct.h>
#include <list>
#include <io.h>  
#include <process.h>  
#include "Cuda_Matrix.h"
#include "Cuda_Info.cu"

#include "XType.h"


using namespace std;
typedef int PixelType;
XParams* para = new XParams;

string filename_a = filePath + "//" + fname_a + ".raw";
string filename_b = filePath + "//" + fname_b + ".raw";

dim3  Volume_Dimensions = dim3(128, 128, 128);

extern "C"
void knn_init_nn(void* vol_a, dim3 dim_a, Cuda_X_Matrix& M_a, void* vol_b, dim3 dim_b, Cuda_X_Matrix& M_b, Cuda_X_Matrix& M_ann, XParams para);

extern "C" void CopyData(const int* volume, cudaExtent volumeSize, int index);
extern "C" void PrintTexture(dim3 vol_size = dim3(16, 16, 16));

int product(dim3 v_size)
{
	if (v_size.x == 0)
	{
		printf("error! dimension is equal to 0");
		system("pause");
		exit(0);
	}

	int yy = v_size.y == 0 ? 1 : v_size.y;
	int zz = v_size.z == 0 ? 1 : v_size.z;
	return v_size.x*yy*zz;
}





int main()
{


	Cuda_Info CC;
	CC.PrintInfo();

	
	if (_access(r_path.c_str(), 0))
	{
		mkdir(r_path.c_str());
		printf("result make success£¡\n");
	}

	para->cores = 1;
	para->knn = 5;
	para->nn_iters = 20;
	para->patch_w = 16;
	para->ToSelf = true;
	para->point_dist = 500;

	srand(time(0));

	double dur;
	clock_t start, end;
	

	/**************************************************************************
	*                     load volume start
	****************************************************************************/



	
	start = clock();	
	auto vol_a = LoadVolume(filename_a, Volume_Dimensions.x, Volume_Dimensions.y, Volume_Dimensions.z, 0);
	auto vol_b = LoadVolume(filename_b, Volume_Dimensions.x, Volume_Dimensions.y, Volume_Dimensions.z, 0);

	end = clock();
	dur = static_cast<double>(end - start);
	printf("knn load volume: %f\n", (dur / CLOCKS_PER_SEC));


	/**************************************************************************
	*                     load volume end
	****************************************************************************/

	
	Cuda_X_Matrix M_a,M_b,M_ann;
	knn_init_nn(vol_a->data, Volume_Dimensions, M_a, vol_b->data, Volume_Dimensions, M_b, M_ann, *para);
	VecVol *Matrix_ann =new VecVol(Volume_Dimensions.x, Volume_Dimensions.y, Volume_Dimensions.z, para->knn);

	CopyDevice2Host(Matrix_ann->data, M_ann, m_Int);
	writeVecVol("G://aa.vec", Matrix_ann);


	


	//CopyDevice2Host(Matrix_ann->data, M_ann, m_Int);
	//writeVecVol("C://Users//X-Lab//Desktop//vec.vec", Matrix_ann);

	/*
	cudaExtent a_size = make_cudaExtent(vol_a->w, vol_a->h, vol_a->d);
	CopyData((int*)vol_a->data, a_size,0);
	PrintTexture(Volume_Dimensions);



	dim3 threads_per_block = dim3(32, 32, 1);
	dim3 blocks_per_grid = dim3(32, 1, 1);

	*/
	
	//cudaFree(&dstPtr_B);

	system("pause");



}