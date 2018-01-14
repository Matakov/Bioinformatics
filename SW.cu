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
char* allocateMemory(std::string const& x)
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
Authors: Matej Crnac

Parameters:
    input:  semaphor - pointer to semaphor list
            n - seamphor length
    output: - initialised semaphor.
*/
__global__ void initsemaphor(int *semaphore, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //printf("Hello from block %d, thread %d, index %d, Memory is %f\n", blockIdx.x, threadIdx.x,index,memory[index]);
   
    for (int i = index; i < N; i += stride)
    {
            semaphore[i]=0;
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


/*
Authors: Matej Crnac

Parameters:
    input:  index - matrix index
            n - s2 length + 1
    output: - index of character in left string for given matrix index.
*/
__device__ long int find_index_left(long int index,long int n)
{
    return index/n - 1;
}

/*
Authors: Matej Crnac

Parameters:
    input:  index - matrix index
            n - s2 length + 1
    output: - index of character in upper string for given matrix index.
*/
__device__ long int find_index_upper(long int index,long int n)
{
    return index%n - 1;
}

/*
Authors: Matej Crnac, Franjo Matković

Parameters:
    input:  memory - pointer to matrix
            m - s1 length + 1
            n - s2 length + 1
            d - penalty
            e - penalty
            N - matrix size
            sim - similarity function
            s1 - string 1
            s2 - string 2
    output: - solved cost matrix
-Function to solve NeedlemanWunsch using GPU
*/
__global__ void NW_GPU(double* memory,long int const m,long int const n, double const d, double const e, long int const N,double (*sim)(char,char),const char* s1, const char* s2,int* semaphore)
{
    //extern __shared__ int s[];
    printf("Bez: %d, blockDim: %d, n = %ld\n",(int)n,blockDim.x,n);
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    printf("index: %d Stride: %d N: %ld\n",index,stride,N);
    //printf("%f\n",(float)(m-1)/(blockDim.x/2));
    //printf("blockIdx.x-int((double)(n-1)/(blockDim.x/2)) = %f\n",blockIdx.x-int((double)(n-1)/(blockDim.x/2)));
    //printf("semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))] = %f \n",semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]);
    //printf("index = %d, blockidx = %d, Semaphor = %d\n",index,blockIdx.x,semaphore[blockIdx.x]);
    while(1)
    {
        __syncthreads();
        if(blockIdx.x==0) break;
        //if(blockIdx.x<(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1) break;
        //if(blockIdx.x>=(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;
        //if(blockIdx.x%(int)(double)(n-1)/(blockDim.x/2)==0 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;

        if(blockIdx.x<(int)n && semaphore[blockIdx.x-1]==1) break;
        if(blockIdx.x>=(int)n && semaphore[blockIdx.x-1]==1 && semaphore[blockIdx.x-int(n)]==1) break;
        if(blockIdx.x%(int)n==0 && semaphore[blockIdx.x-int(n)]==1) break;
    }
    for (int i = index; i < N; i += stride)
    {
        if(i%n!=0 && i > n)
        {   
            double simil;
            if(s1[find_index_left(i,n)]==s2[find_index_upper(i,n)]) simil = 1;
            else simil = -3;
            //printf("Hello from block %d, thread %d, index %d, Memory is %f\n", blockIdx.x, threadIdx.x,index,memory[index]);
            //printf("Index: %d\n", i);
            //printf("memory[i-n-1] = %f\n", memory[i-n-1]);
            //printf("find_L:\n");
            //printf("find_ind_l = %d\n",find_index_left(i,n));
            //printf("s1[find_index_left(i,n)] = %c\n",s1[find_index_left(i,n)]);
            //printf("s1 finished\n");
            //printf("s2[find_index_upper(i,n)] = %c\n",s2[find_index_upper(i,n)]);
            //printf("s2 finished\n");
            //printf("sim: = %f\n",simil);
            //printf("sim finished\n");
            //printf("Index: %d, memory[i-n-1] = %f, sim: %d find_ind_l = %d, find_ind_u = %d, memory[i-n] = %f, memory[i-1] = %f\n",i, memory[i-n-1],simil, find_index_left(i,n), find_index_upper(i,n), memory[i-n], memory[i-1]);
            memory[i]=max(memory[i-n-1]+simil,max(memory[i-n] - d,memory[i-1] - d));
        }
    }
    
    semaphore[blockIdx.x]=1;
    //semaphore[blockIdx.x+1]=1;
    //semaphore[blockIdx.x+n]=1;
    

    return;
}


/*
Authors: Franjo Matkovic

Parameters:
    input:  s1 - string 1
            s2 - string 2
            d  - penalty
            e  - penalty
            sim- similarity function
    output: - solved cost matrix
-Function to solve NeedlemanWunsch
*/
void NeedlemanWunschGPU(std::string const& s1, std::string const& s2, double const d, double const e,double (*sim)(char,char))
{
    double *Gi,*Gd,*F,*E;
    double *memory;
    char *M;
    long int m = s1.length();
    long int n = s2.length();
    long int N = (s1.length()+1)*(s2.length()+1);
    long int N_orig = n*m;
    cudaMallocManaged(&memory, N*sizeof(double));
    cudaMallocManaged(&M, N*sizeof(char));
    cudaMallocManaged(&Gi, N*sizeof(double));
    cudaMallocManaged(&Gd, N*sizeof(double));
    cudaMallocManaged(&F, N*sizeof(double));
    cudaMallocManaged(&E, N*sizeof(double));
    
    int blockSize = 1;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout<<numBlocks<<std::endl;


    int *semaphore;

    cudaMallocManaged(&semaphore, numBlocks);

    initsemaphor<<<numBlocks, blockSize>>>(semaphore, numBlocks);
    cudaDeviceSynchronize();

    initmemoryHNW<<<numBlocks, blockSize>>>(memory,m+1,n+1,d,e,N);
    cudaDeviceSynchronize();
    
    //padding(s1,s2,,);
    const char* x1 = allocateMemory(s1);
    const char* x2 = allocateMemory(s2);
    int i = 0;

    /*while( x2[i] != '\0')
    {
        std::cout<<x1[i];
        i++;
    }*/
    
    std::cout<<"Seamphor before:"<<" ";
    for(int i=0;i<numBlocks;i++)
    {
        std::cout<<semaphore[i]<<" ";    
    }
    std::cout<<std::endl;
 
    NW_GPU<<<numBlocks, blockSize>>>(memory,m+1,n+1,d,e,N,sim,x1,x2,semaphore); 
    cudaDeviceSynchronize();
    
    std::cout<<"Seamphor after:"<<" ";
    for(int i=0;i<numBlocks;i++)
    {
        std::cout<<semaphore[i]<<" ";    
    }
    std::cout<<std::endl;

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
