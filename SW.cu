#include"SW.h"
#include"utility.h"

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

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock)
{
    while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock)
{
    *lock = 0;
    __threadfence();
}

__global__ void initmemoryHNW(double *memory,long int const m,long int const n, double const d, double const e, double const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //printf("Hello from block %d, thread %d, index %d, Memory is %f\n", blockIdx.x, threadIdx.x,index,memory[index]);
   
    for (int i = index; i < N; i += stride)
    {
        if(i<n)
        {
            memory[i]=-(d+e*(i-1));
        }
        else if(i%n==0)
        {
            memory[i]=-(d+e*(i/n-1));
        }
        else
        {
            memory[i]=0;
        }
    }
    return;
}
/*
__global__ void initmemoryHSW(double *memory,double const m,double const n, double const d, double const e, double const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        memory[i]=0;
    }
    return;
}
*/
/*
__global__ void initmemoryAR(double *memory,double const m,double const n, double const d, double const e, long int const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        if(i%(int)m==0 || i<(int)n)
        {
            memory[i]=-(d+e*(i-1));
        }
        else
        {
            memory[i]=0;
        }
    }
    return;
}
*/
/*__global__ void NW_GPU(double* memory,double const m,double const n, double const d, double const e, double const N)
{
    extern __shared__ int s[];
    
    

    return;
}
*/
void NeedlmanWunschGPU(std::string const& s1, std::string const& s2, double const d, double const e,double (*sim)(char,char))
{
    double *Gi,*Gd,*F,*E;
    double *memory;
    char*M;
    long int m = s1.length();
    long int n = s2.length();
    long int N = (s1.length()+1)*(s2.length()+1);

    cudaMallocManaged(&memory, N*sizeof(double));
    cudaMallocManaged(&M, N*sizeof(char));
    cudaMallocManaged(&Gi, N*sizeof(double));
    cudaMallocManaged(&Gd, N*sizeof(double));
    cudaMallocManaged(&F, N*sizeof(double));
    cudaMallocManaged(&E, N*sizeof(double));
    
    int blockSize = 2;
    int numBlocks = (N + blockSize - 1) / blockSize;

    initmemoryHNW<<<numBlocks, blockSize>>>(memory,m+1,n+1,d,e,N);
    cudaDeviceSynchronize();
    //NW_GPU<<<numBlocks, blockSize,numBlocks*sizeof(int)>>>(memory,m+1,n+1,d,e,sim,N,M,Gi,Gd,F,E); 
    /*  
    for(int i=0;i<m+1;i++)
    {
        for(int j=0;j<n+1;j++)
        {
            if((i*(n+1)+j)%(n+1)==0) memory[i*(n+1)+j]=-(d+e*(i-1));
            if((i*(n+1)+j)<(n+1)) memory[i*(n+1)+j]=-(d+e*(j-1));
        }   
    }
    */  

    for(int i=0;i<m+1;i++)
    {
        for(int j=0;j<n+1;j++)
        {
            std::cout<<memory[i*(n+1)+j]<<" ";  
        }
        std::cout<<std::endl;   
    }
    //memory freeing
    cudaFree(memory);
    cudaFree(M);
    cudaFree(Gi);
    cudaFree(Gd);
    cudaFree(F);
    cudaFree(E);
    return;
} 
