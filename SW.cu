#include"SW.h"

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
extern "C" void calculate(float *x,float *y, int N)
{
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
 
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: c++ string
	output: char array
-Function to allocate unified memory
 and copy to unified memory
*/
extern "C" void allocateMemory(std::string& x, char* memory)
{
	const char *cstr = x.c_str();
	cudaMallocManaged(&memory, x.length()*(sizeof(char)+1));
	//x.copy( memory, x.length() );
	//for(int i=0;i<x.length();++i) memory[i]=cstr[i];
	strcpy(memory, cstr);	
	memory[x.length()]='\0';
	//for(int i=0;i<x.length();++i) std::cout<<cstr[i];
	std::cout<<"Memory allocated"<<std::endl;
	for(int i=0;i<x.length();++i) std::cout<<memory[i];
	std::cout<<std::endl;
	return;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: char array
	output: -
-Function to release unified memory
*/
extern "C" void releaseMemory(char* memory)
{
	cudaFree(memory);
	std::cout<<"Memory released"<<std::endl;
	return;
}
