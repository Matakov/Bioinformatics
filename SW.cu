#include"SW.h"

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
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
extern "C" char* allocateMemory(std::string& x)
{
	char* memory;
	const char *cstr = x.c_str();
	cudaMallocManaged(&memory, x.length()*(sizeof(char)+1));
	//x.copy( memory, x.length() );
	//for(int i=0;i<x.length();++i) memory[i]=cstr[i];
	strcpy(memory, cstr);	
	memory[x.length()]='\0';
	//for(int i=0;i<x.length();++i) std::cout<<cstr[i];
	//std::cout<<"Memory allocated"<<std::endl;
	//for(int i=0;i<x.length();++i) std::cout<<memory[i];
	//std::cout<<std::endl;
	return memory;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: c++ string
	output: char array
-Function to allocate cuda memory
*/
extern "C" float* allocateMatrixMemory(const std::string& x,const std::string& y)
{
	float* memory;
	//cudaMallocManaged(&memory, (x.length()+1)*(y.length()+1)*(sizeof(float)));
	//x.copy( memory, x.length() );
	//for(int i=0;i<x.length();++i) memory[i]=cstr[i];
  	cudaMalloc((float **)&memory, (x.length()+1) * (y.length()+1) * sizeof(float));	
	//for(int i=0;i<x.length();++i) std::cout<<cstr[i];
	//std::cout<<"Memory allocated"<<std::endl;
	//for(int i=0;i<x.length();++i) std::cout<<memory[i];
	//std::cout<<std::endl;
	return memory;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: c++ string
	output: char array
-Function to allocate memory
*/
extern "C" float* allocateMatrixMemoryCPU(const std::string& x,const std::string& y)
{
	float* memory;
	//cudaMallocManaged(&memory, (x.length()+1)*(y.length()+1)*(sizeof(float)));
	//x.copy( memory, x.length() );
	//for(int i=0;i<x.length();++i) memory[i]=cstr[i];
  	memory =(float *) malloc((x.length()+1) * (y.length()+1) * sizeof(float));	
	//for(int i=0;i<x.length();++i) std::cout<<cstr[i];
	//std::cout<<"Memory allocated"<<std::endl;
	//for(int i=0;i<x.length();++i) std::cout<<memory[i];
	//std::cout<<std::endl;
	return memory;
}

/*
Authors: Franjo Matkovic

Parameters:
*/
extern "C" float* initializeMemoryMatrixCPU(const std::string& x,const std::string& y, double penalty)
{
	double d=penalty;
	double e=penalty;
	float* memory = allocateMatrixMemoryCPU( x, y);
	for(int i=0;i<x.length()+1;i++)
	{
		for(int j=0;j<y.length()+1;j++)
		{
			//printf("%d,%d\n",i,j);
			if(i==0)
			{
				memory[i*(x.length()+1)+j] = -(d+e*(j-1));
			}
			else if(j==0)
			{	
				memory[i*(x.length()+1)+j] = -(d+e*(i-1));
			}
			else
			{
				memory[i*(x.length()+1)+j] = 0;
			}
			//printf("%d,%d: %f\n",i,j,memory[i*x.length()+j]);
		}
	}
	memory[0] = 0;
	return memory;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: array pointer
	output: -
-Function to release unified memory
*/
extern "C" void releaseMemory(char* memory)
{
	cudaFree(memory);
	//std::cout<<"Memory released"<<std::endl;
	return;
}

/*
Authors: Franjo Matkovic

Parameters:
	input: array pointer
	output: -
-Function to initialize memory array for calculating Needlmen-Wunsch
*/
__global__ void initializeNWS(float** memory, double penalty, float M, float N, float n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	double d=penalty;
	double e=penalty;
	//int idx,c;
	for (int i = index; i < n; i += stride)
	{
		//The following max-min is to avoid if branching
		//idx = max( i, 0);
        	//idx = min( (float)idx, M);
		//memory[idx] = -(d+e*(idx-1));
		//c = (int)i%(int)N;
		//c = (c>>31) – (-c>>31); //result is 0 if c in a row before is zero ,1 if positive ,-1 if negative
		//c = 1 - c;
		//c = max(c,-1)+min(c,1s);
		
		if(i<M || (int)i%(int)N)
		{
			*memory[i] = -(d+e*(i-1));
		}
		else
		{
			*memory[i] = 0;
		}
		//printf("Hello thread %d, f=%f\n", threadIdx.x, memory[i]);
	}
	memory[0]=0;
	return;
}

extern "C" void initialize(float** memory, double penalty, float M, float N, float n)
{
	printf("Time to initialize\n");
	printf("Memory size %f\n",n);
	printf("Memory address %p\n",(void *) memory);
	printf("Memory address %p\n",(void **) memory);
	initializeNWS<<<4,16>>>(memory,penalty,M,N,n);
	return;
}



extern "C" void NWS(char* x, char*y, float* memory)
{
	//int numberOfKernels = gridDim.x;
	
	return;
} 
