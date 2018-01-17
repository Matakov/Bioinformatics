#include"SW.h"
#include <math.h>

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

__global__ void initmemoryHNW(int *memory,int const m,int const n, int const d, int const e, int const N)
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
__global__ void initmemoryHSW(double *memory,long int const m,long int const n, double const d, double const e, double const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //printf("Hello from block %d, thread %d, index %d, Memory is %f\n", blockIdx.x, threadIdx.x,index,memory[index]);
   
    for (int i = index; i < N; i += stride)
    {
        memory[i]=0;
    }
    return;
}
*/
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
    
   
    for (int i = index; i < N; i += stride)
    {
            //printf("Hello from block %d, thread %d, i %d, stride: %d, N: %d\n", blockIdx.x, threadIdx.x,i,stride,N);
            semaphore[i]=0;
    }
    //printf("Izlazim!!!!!!");
    return;
}

/*
Authors: Matej Crnac

Parameters:
    input:  semaphor - pointer to semaphor list
            n - seamphor length
    output: - initialised semaphor.
*/
__device__ void initsemaphorDevice(int *semaphore, int N)
{
    int index = 0;
    //int stride = blockDim.x * gridDim.x;
    
   
    for (int i = index; i < N; i += 1)
    {
            //printf("Hello from block %d, thread %d, i %d, stride: %d, N: %d\n", blockIdx.x, threadIdx.x,i,stride,N);
            semaphore[i]=0;
    }
    //printf("Izlazim!!!!!!");
    return;
}
__global__ void initmemoryHSW(int *memory,int const m,int const n, int const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        memory[i]=0;
    }
    return;
}

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
/*
void NeedlemanWunschGPU(std::string const& s1, std::string const& s2, double const d, double const e,double (*sim)(char,char))
{
    int *Gi,*Gd,*F,*E;
    int *memory;
    char *M;
    int m = s1.length();
    int n = s2.length();
    int N = (s1.length()+1)*(s2.length()+1);
    //long int N_orig = n*m;
    cudaMallocManaged(&memory, N*sizeof(int));
    cudaMallocManaged(&M, N*sizeof(char));
    cudaMallocManaged(&Gi, N*sizeof(int));
    cudaMallocManaged(&Gd, N*sizeof(int));
    cudaMallocManaged(&F, N*sizeof(int));
    cudaMallocManaged(&E, N*sizeof(int));
    
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
    //int i = 0;

    /*while( x2[i] != '\0')
    {
        std::cout<<x1[i];
        i++;
    }*/
    /*
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
*/
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
-Function to solve SmithWaterman using GPU
*/
__global__ void SW_GPU(double* memory,long int const m,long int const n, double const d, double const e, long int const N,const char* s1, const char* s2,int* semaphore)
{
    //extern __shared__ int s[];
    //printf("Bez: %d, blockDim: %d, n = %ld\n",(int)n,blockDim.x,n);
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //printf("index: %d Stride: %d N: %ld\n",index,stride,N);
    //printf("%f\n",(float)(m-1)/(blockDim.x/2));
    //printf("blockIdx.x-int((double)(n-1)/(blockDim.x/2)) = %f\n",blockIdx.x-int((double)(n-1)/(blockDim.x/2)));
    //printf("semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))] = %f \n",semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]);
    //printf("index = %d, blockidx = %d, Semaphor = %d\n",index,blockIdx.x,semaphore[blockIdx.x]);
    //printf("gridDim: %d\n",gridDim.x);
    for (int i = index; i < N; i += stride)
    {
        __syncthreads();
        while(1)
        {
            
            //printf("index = %d i = %d, blockidx = %d, threadID = %d, Semaphor = %d, memory = %f\n",index,i,blockIdx.x,threadIdx.x,semaphore[blockIdx.x],memory[i-1]);
            if(i==0) break;
            //if(blockIdx.x<(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1) break;
            //if(blockIdx.x>=(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;
            //if(blockIdx.x%(int)(double)(n-1)/(blockDim.x/2)==0 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;

            if(i<(int)n && semaphore[(i-1)%gridDim.x]>0)
            {
                semaphore[(i-1)%gridDim.x]--;
                break;
            }
            if(i>=(int)n && semaphore[(i-1)%gridDim.x]>0 && semaphore[(i-int(n))%gridDim.x]>0)
            {
                semaphore[(i-int(n))%gridDim.x]--;
                semaphore[(i-1)%gridDim.x]--;
                break;
            }
            if(i%(int)n==0 && semaphore[(i-int(n))%gridDim.x]>0)
            {
                semaphore[(i-int(n))%gridDim.x]--;
                break;
            }
        }
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
            memory[i]=max((double)0,max(memory[i-n-1]+simil,max(memory[i-n] - d,memory[i-1] - d)));
        }
        semaphore[(blockIdx.x)%gridDim.x]=2;
    }
    
    
    //semaphore[blockIdx.x+1]=1;
    //semaphore[blockIdx.x+n]=1;
    

    return;
}

__global__ void SW_GPU_thread(float* memory,long int const m,long int const n, double const d, double const e, long int const N,const char* s1, const char* s2,int* semaphore)
{
    //extern __shared__ int s[];
    //printf("Bez: %d, blockDim: %d, n = %ld\n",(int)n,blockDim.x,n);
    int index = threadIdx.x;
    int stride = 1;
    printf("index: %d Stride: %d N: %ld\n",index,stride,N);
    printf("blockDim.x %d\n",blockDim.x);
    printf("(i-1)%blockDim.x = %d\n",(index-1)%blockDim.x);
    printf("semaphore[(i-1)%blockDim.x] = %d \n",semaphore[(index-1)%blockDim.x]);
    //printf("index = %d, blockidx = %d, Semaphor = %d\n",index,blockIdx.x,semaphore[blockIdx.x]);
    //printf("gridDim: %d\n",gridDim.x);
    for (int i = index; i < N; i += stride)
    {
        //__syncthreads();
        while(1)
        {
            
            //printf("index = %d i = %d, blockidx = %d, threadID = %d, Semaphor = %d, memory = %f\n",index,i,blockIdx.x,threadIdx.x,semaphore[blockIdx.x],memory[i-1]);
            if(i==0) break;
            //if(blockIdx.x<(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1) break;
            //if(blockIdx.x>=(int)(double)(n-1)/(blockDim.x/2) && semaphore[blockIdx.x-1]==1 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;
            //if(blockIdx.x%(int)(double)(n-1)/(blockDim.x/2)==0 && semaphore[blockIdx.x-int((double)(n-1)/(blockDim.x/2))]==1) break;

            if(i<(int)n && semaphore[(i-1)%blockDim.x]>0)
            {
                semaphore[(i-1)%blockDim.x]--;
                break;
            }
            if(i>=(int)n && semaphore[(i-1)%blockDim.x]>0 && semaphore[(i-int(n))%blockDim.x]>0)
            {
                semaphore[(i-int(n))%blockDim.x]--;
                semaphore[(i-1)%blockDim.x]--;
                break;
            }
            if(i%(int)n==0 && semaphore[(i-int(n))%blockDim.x]>0)
            {
                semaphore[(i-int(n))%blockDim.x]--;
                break;
            }
        }
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
            memory[i]=max((double)0,max(memory[i-n-1]+simil,max(memory[i-n] - d,memory[i-1] - d)));
        }
        semaphore[(i)%blockDim.x]=2;
    }
    
    
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
-Function to solve SmithWaterman
*/
__device__ void threadFunction(int* memory,int subN, int subM,  int n, int m_orig, int n_orig, int localSubN,  int localSubM, int const blockSize_n, int threadSize, int threadSize_m, int threadSize_n,  int N, char *s1, char *s2,Scorer score, int* maxBlockInside, int* positionmaxBlockInside, int threadNumber)
{
    //tu negdje treba ubaciti provajru za prvi red i prvi stupac
    int totalElem = (int)threadSize;
    int coordx, coordy;
    int simil;
    int position;
    for(int i=0;i<totalElem;i++)
    {
        
        coordx = i % (int)threadSize_n;
        coordy = i / (int)threadSize_n;
        if((subM+localSubM+coordy) == 0 || (subN + localSubN + coordx) == 0) continue;

        int n_position = subN*blockSize_n + localSubN*threadSize_n + coordx;
        int m_position = subM*blockSize_n+localSubM*threadSize_m+coordy;
        if(n_position >= n_orig || m_position >= m_orig) continue;

        if(s1[subM*blockSize_n+localSubM*threadSize_m+coordy-1]==s2[subN*blockSize_n + localSubN*threadSize_n + coordx-1]) simil = score.m;
        else simil = score.mm;
        position = (int)n * ((int)subM*(int)blockSize_n + (int)localSubM*(int)threadSize_m + coordy ) + ((int)subN*(int)blockSize_n + (int)localSubN*(int)threadSize_n + coordx);
        //printf("Position: %d\n",position);
        //standard Smith Waterman for scoring
        int newScore = (int)max(memory[position-n-1]+simil,max(memory[position-n]-(int)score.d,max(0,memory[position-1]-(int)score.d)));
        memory[position] = newScore;
        //in if we check for the maximum value in thread block and save it along with a position in a matrix
        if(newScore > maxBlockInside[threadIdx.x])
        {
            //printf("Thread %d sranje je uslo s trenutnim maxBlockInside[threadIdx.x]: %d\n",threadIdx.x,maxBlockInside[threadIdx.x]);
            maxBlockInside[threadIdx.x]=newScore;
            //printf("Thread's %d maxBlockInside[threadIdx.x] after: %d\n",threadIdx.x,maxBlockInside[threadIdx.x]);
            positionmaxBlockInside[threadIdx.x]=position;
            
        }   
        //printf("blockid = %d Dretva %d, vrijednost %d\n, ",blockIdx.x,threadIdx.x,maxBlockInside[threadIdx.x]);    
    }
    return;
}

/*
Authors: Dario Sitnik, Franjo Matković

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
-Function to solve SmithWaterman using GPU on thread level
*/

__global__ void threadSolver(int *memory,int subM,int subN, int const n, int const m_orig,int const n_orig,int const blockSize_m ,int const blockSize_n,int threadNumber,int threadNumber_m,int threadNumber_n,int threadSize,int threadSize_m,int threadSize_n ,char *s1, char *s2, int* semaphore, int N, Scorer scorer, int* maxBlockInside, int* positionmaxBlockInside)
{
    int localSubN = (threadIdx.x % (int)threadNumber_n);
    int localSubM = (threadIdx.x / (int)threadNumber_n);
    int flag = 1;
    while(flag==1){
        if(threadIdx.x==0){
            threadFunction(memory,subN,subM,n,m_orig,n_orig,localSubN,localSubM,blockSize_n,threadSize,threadSize_m,threadSize_n, N,s1,s2,scorer,maxBlockInside,positionmaxBlockInside,threadNumber);
            semaphore[(threadIdx.x)%(int)threadNumber]=2;
            flag = 0;
            break;
        }
        if(threadIdx.x<(int)threadNumber_n && semaphore[(threadIdx.x-1)%(int)threadNumber]>0)
        {
            threadFunction(memory,subN,subM,n,m_orig,n_orig,localSubN,localSubM,blockSize_n,threadSize,threadSize_m,threadSize_n, N,s1,s2,scorer,maxBlockInside,positionmaxBlockInside,threadNumber);
            semaphore[(threadIdx.x-1)%(int)threadNumber]--;
            semaphore[(threadIdx.x)%(int)threadNumber]=2;
            flag = 0;
            break;
        }

        if(threadIdx.x>=(int)threadNumber_n && semaphore[(threadIdx.x-1)%(int)threadNumber]>0 && semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]>0)
        {
            threadFunction(memory,subN,subM,n,m_orig,n_orig,localSubN,localSubM,blockSize_n,threadSize,threadSize_m,threadSize_n, N,s1,s2,scorer,maxBlockInside,positionmaxBlockInside,threadNumber);
            semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]--;
	        semaphore[(threadIdx.x-1)%(int)threadNumber]--;
            semaphore[(threadIdx.x)%(int)threadNumber]=2;
            flag = 0;
            break;
        }
        if(threadIdx.x%(int)threadNumber_n==0 && semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]>0)
        {
            threadFunction(memory,subN,subM,n,m_orig,n_orig,localSubN,localSubM,blockSize_n,threadSize,threadSize_m,threadSize_n, N,s1,s2,scorer,maxBlockInside,positionmaxBlockInside,threadNumber);
            semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]--;
            semaphore[(threadIdx.x)%(int)threadNumber]=2;
            flag = 0;
            break;
        }
    }
    //printf("POSLJE blockIdx = %d Dretva %d, vrijednost %d\n, ",blockIdx.x,threadIdx.x,maxBlockInside[threadIdx.x]);
    __syncthreads();
	return;
}

//block 
__global__ void kernelCallsKernel(int *memory, int const m, int const n, int const m_orig, int const n_orig, int N, char *s1, char *s2, int numBlocks,int numBlocks_m,int numBlocks_n, int blockSize_m, int blockSize_n, int threadNumber, int threadNumber_m, int threadNumber_n, int threadSize, int threadSize_m, int threadSize_n,int* semaphore, Scorer scorer, int *maxBlock, int *postionMaxBlock)
{
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int subN = blockIdx.x%(int)numBlocks_n;
    int subM = blockIdx.x/(int)numBlocks_n;
    int *semaphoreInside;
    int *maxBlockInside;
    int *positionmaxBlockInside;
    
    semaphoreInside = (int*)malloc((int)threadNumber*sizeof(int));
	initsemaphorDevice(semaphoreInside, (int)threadNumber);
    
    maxBlockInside = (int*)malloc((int)threadNumber*sizeof(int));
	initsemaphorDevice(maxBlockInside, (int)threadNumber);
    positionmaxBlockInside = (int*)malloc((int)threadNumber*sizeof(int));
	initsemaphorDevice(positionmaxBlockInside, (int)threadNumber);

    while(1)
    {
        if(index==0) 
        {
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);
            //__syncthreads();    
            semaphore[blockIdx.x]=2;
            //__syncthreads();
            for(int z=0;z<(int)threadNumber;z++)
            {

                //printf("MaxBlockinside %d maxBlock[index] %d\n",maxBlockInside[z],maxBlock[index]);
                if(maxBlockInside[z]>maxBlock[index])
                {

                    //printf("Blok %d sranje je uslo s prethodnim maxom %d\n",blockIdx.x,maxBlock[blockIdx.x]);
                    maxBlock[index]=maxBlockInside[z];
                    //printf("Maxblock nakon je %d\n",maxBlock[blockIdx.x]);
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }

        if(index<(int)numBlocks_n && semaphore[(index-1)%gridDim.x]>0)
        {
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);
            semaphore[(index-1)%gridDim.x]--;
            //__syncthreads();
            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                 //printf("MaxBlockinside %d maxBlock[index] %d\n",maxBlockInside[z],maxBlock[index]);
                if(maxBlockInside[z]>maxBlock[index])
                {

                    //printf("Blok %d sranje je uslo s prethodnim maxom %d\n",blockIdx.x,maxBlock[blockIdx.x]);
                    maxBlock[index]=maxBlockInside[z];
                    //printf("Maxblock nakon je %d\n",maxBlock[blockIdx.x]);
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }
        if(index>=(int)numBlocks_n && semaphore[(index-1)%gridDim.x]>0 && semaphore[(index-int(numBlocks_n))%gridDim.x]>0)
        {
            
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);    
            semaphore[(index-int(numBlocks_n))%gridDim.x]--;
            semaphore[(index-1)%gridDim.x]--;
            //__syncthreads();    
            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {

                 //printf("MaxBlockinside %d maxBlock[index] %d\n",maxBlockInside[z],maxBlock[index]);
                if(maxBlockInside[z]>maxBlock[index])
                {
                    //printf("MaxBlockinside %d\n",maxBlockInside[z]);
                    //printf("Blok %d sranje je uslo s prethodnim maxom %d\n",blockIdx.x,maxBlock[blockIdx.x]);
                    maxBlock[index]=maxBlockInside[z];
                    //printf("Maxblock nakon je %d\n",maxBlock[blockIdx.x]);
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }
        if(index%(int)numBlocks_n==0 && semaphore[(index-int(numBlocks_n))%gridDim.x]>0)
        {
            
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);
            semaphore[(index-int(numBlocks_n))%gridDim.x]--;
            //__syncthreads();    
            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                 //printf("MaxBlockinside %d maxBlock[index] %d\n",maxBlockInside[z],maxBlock[index]);
                if(maxBlockInside[z]>maxBlock[index])
                {
                    //printf("Blok %d sranje je uslo s prethodnim maxom %d\n",blockIdx.x,maxBlock[blockIdx.x]);
                    maxBlock[index]=maxBlockInside[z];
                    //printf("Maxblock nakon je %d\n",maxBlock[blockIdx.x]);
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }
        
    }
    //printf("Blok %d ima max vrijednost %d i poziciju %d\n",blockIdx.x,maxBlock[blockIdx.x],postionMaxBlock[blockIdx.x]);
    //printf("IZASAO VANJSKI blockID = %d\n",blockIdx.x);
    //__syncthreads();
    cudaDeviceSynchronize();
    return;
} 


void SmithWatermanGPU(std::string const& s1, std::string const& s2, double const B, Scorer scorer)
{
    //DATA PREPARATION
    //NAORAVIT PROVJERU VELICINE STRINGOVA

	//input strings are const so we copy
	std::string string_m(s2);
	std::string string_n(s1);

	//memory locations 
	int *memory;
	char *M;

	//sizes of strings
	int m = string_m.length() + 1;
	int n = string_n.length() + 1;

    int m_orig = m;
	int n_orig = n;

	//B is the desirable number of blocks in grid
	double k = sqrt(B/((double)m/n));
	int blockNum_n = floor(k);
	int blockNum_m = floor(((double)m/n)*k);
	int blockNum = blockNum_n*blockNum_m;
    
    std::cout<<"k: "<<k<<"B: "<<B<<"m/n: "<<m/n<<"B/m/n: "<<B/(m/n)<<"blockNum_n: "<<blockNum_n<<" "<<"blockNum_m: "<<blockNum_m<<" blockNum: "<<blockNum<<std::endl;
    int N = (m)*(n);
    

	//std::cout<<k<<" "<<blockSize_n<<" "<<blockSize_m<<std::endl;
	//here we define how much will there be blocks in m and n direction
    int blockSize_n = ceil((double)n/blockNum_n);
	int blockSize = (int)pow(blockSize_n,2);
    
    std::cout<<"blockSize: "<<blockSize<<std::endl;
    std::cout<<"blockNum_n: "<<blockNum_n<<" "<<"blockNum_m: "<<blockNum_m<<" blockNum: "<<blockNum<<std::endl;
    std::cout<<"n: "<<n<<" "<<"m: "<<m<<std::endl;
	//std::cout<<"Size:"<<n<<" "<<blockSize_n<<" "<<ceil((double)n/blockSize_n)<<" "<<ceil(n/blockSize_n)<<std::endl;
	//std::cout<<"Size:"<<m<<" "<<blockSize_m<<" "<<ceil((double)m/blockSize_m)<<" "<<ceil(m/blockSize_m)<<std::endl;
	//here we are padding strings so there are no elements that will be
    
    //calculate threadNumber and threadSize
    int threadNumber = blockSize;
    int threadNumber_n = (int)pow(blockSize,0.5);
    int threadNumber_m = (int)pow(blockSize,0.5);
    int threadSize = 1;
    int threadSize_m=1;
    int threadSize_n=1;
    if (threadNumber > 1024)
    {
        
        while( !( ( (int)blockSize%(threadSize_m*threadSize_n) ) == 0 && (int)blockSize/(threadSize_m*threadSize_n) < 1024 ))
        {
            if (threadSize_m > threadSize_n) threadSize_n += 1;
            else threadSize_m += 1;
        }

        threadNumber = blockSize/(threadSize_m*threadSize_n);
        threadNumber_m = threadNumber/threadSize_m;
        threadNumber_n = threadNumber/threadSize_n;
    }
    
    std::cout<<"threadNumber: "<<threadNumber<<" threadNumber_m: "<<threadNumber_m<<" threadNumber_n; "<<threadNumber_n<<std::endl;
    //do padding
	padding(string_m,string_n,blockNum_m*(int)ceil(pow(blockSize,0.5)),blockNum_n*(int)ceil(pow(blockSize,0.5)));

	m=string_m.length()+1;
	n=string_n.length()+1;	
    std::cout<<"n: "<<n<<" "<<"m: "<<m<<std::endl;
	N = (m)*(n);
	//part of code where memory allocation is happening
	cudaMallocManaged(&memory, N*sizeof(int));
	cudaMallocManaged(&M, N*sizeof(char));

	char* x1 ;//= allocateMemory(string_m);
	
	const char *cstr = string_m.c_str();
    cudaMallocManaged(&x1, string_m.length()*(sizeof(char)+1));
    strcpy(x1, cstr);   
    x1[string_m.length()]='\0';
	
    char* x2 ;// = allocateMemory(string_n);

	const char *cstr2 = string_n.c_str();
	cudaMallocManaged(&x2, string_n.length()*(sizeof(char)+1));
    strcpy(x2, cstr2);   
    x2[string_n.length()]='\0';
	
	int *semaphore;
    int *maxBlock;
    int *postionMaxBlock;
 
    cudaMallocManaged(&semaphore, blockNum*sizeof(int));
    cudaMallocManaged(&maxBlock, blockNum*sizeof(int));
    cudaMallocManaged(&postionMaxBlock, blockNum*sizeof(int));
 
    initsemaphor<<<40, blockSize>>>(semaphore, blockNum);
    cudaDeviceSynchronize();

    initsemaphor<<<40, blockSize>>>(maxBlock, blockNum);
    cudaDeviceSynchronize();
    
    initsemaphor<<<40, blockSize>>>(postionMaxBlock, blockNum);
    cudaDeviceSynchronize();

    initmemoryHSW<<<40, blockSize>>>(memory,m,n,N);
    cudaDeviceSynchronize();

    //CALCULATION
    std::cout<<"Calculation started:"<<std::endl;

    kernelCallsKernel<<<blockNum, 1>>>(memory,m,n,m_orig,n_orig,N,x1, x2, blockNum,blockNum_m,blockNum_n,blockSize_n,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,semaphore,scorer,maxBlock,postionMaxBlock);
    cudaDeviceSynchronize();

    int maxValue=0;
    int maxPosition;
    
    for(int i=0;i<m;i++)
    {
         for(int j=0;j<n;j++)
         {
             std::cout<<memory[i*(n)+j]<<" ";  
         }
         std::cout<<std::endl;   
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"blockNum "<<blockNum<<std::endl;
    for(int z=0;z<blockNum;z++)
    {
        std::cout<<maxBlock[z]<<" ";
        if(maxBlock[z]>maxValue)
        {
            std::cout<<maxBlock[z]<<" ";
            maxValue=maxBlock[z];
            maxPosition=postionMaxBlock[z];
        }
    }
    std::cout<<std::endl;
    
    std::cout<<"Max value: "<<maxValue<<" ,position: "<<maxPosition<<std::endl;

    std::cout<<"from memory "<<memory[maxPosition]<<std::endl;        
	//memory freeing
	cudaFree(memory);
	cudaFree(M);
	cudaFree(semaphore);
    cudaFree(x1);
	cudaFree(x2);
	return;
} 


/*void SmithWatermanGPU_Basic(std::string const& s1, std::string const& s2, double const d, double const e)
 {
     float *Gi,*Gd,*F,*E;
     float *memory;
     char *M;
     long int m = s1.length();
     long int n = s2.length();
     long int N = (s1.length()+1)*(s2.length()+1);
     //long int N_orig = n*m;
     cudaMallocManaged(&memory, N*sizeof(float));
     cudaMallocManaged(&M, N*sizeof(char));
     cudaMallocManaged(&Gi, N*sizeof(float));
     cudaMallocManaged(&Gd, N*sizeof(float));
     cudaMallocManaged(&F, N*sizeof(float));
     cudaMallocManaged(&E, N*sizeof(float));
     
     int blockSize = 1;
     int numBlocks = (N + blockSize - 1) / blockSize;
     std::cout<<numBlocks<<std::endl;
 
 
     int *semaphore;
 
     cudaMallocManaged(&semaphore, numBlocks*sizeof(int));
 
     initsemaphor<<<numBlocks, blockSize>>>(semaphore, numBlocks);
     cudaDeviceSynchronize();
 
     initmemoryHSW<<<numBlocks, blockSize>>>(memory,m+1,n+1,N);
     cudaDeviceSynchronize();
     
     //padding(s1,s2,,);
     const char* x1 = allocateMemory(s1);
     const char* x2 = allocateMemory(s2);
     //int i = 0;
 
     /*while( x2[i] != '\0')
     {
         std::cout<<x1[i];
         i++;
     }*/
     /*
     std::cout<<"Seamphore before:"<<" ";
     for(int i=0;i<numBlocks;i++)
     {
         std::cout<<semaphore[i]<<" ";    
     }
     std::cout<<std::endl;
    
     //SW_GPU<<<numBlocks, blockSize>>>(memory,m+1,n+1,d,e,N,x1,x2,semaphore); 
     SW_GPU_thread<<<1, numBlocks>>>(memory,m+1,n+1,d,e,N,x1,x2,semaphore);
     //cudaDeviceSynchronize();
    
     std::cout<<"Seamphore after:"<<" ";
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
*/
