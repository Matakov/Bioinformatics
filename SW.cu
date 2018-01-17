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
    strcpy(memory, cstr);   
    memory[x.length()]='\0';
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
    cudaMalloc((float **)&memory, (x.length()+1) * (y.length()+1) * sizeof(float)); 
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
    memory =(float *) malloc((x.length()+1) * (y.length()+1) * sizeof(float));  
    return memory;
}

/*
Authors: Franjo Matkovic, Dario Sitnik
Parameters:
    input: c++ string sequence 1
	   c++ string sequence 2
	   gap penalty
    output: float array
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

        }
    }
    memory[0] = 0;
    return memory;
}

/*
Authors: Franjo Matkovic, Dario Sitnik

Parameters:
    input: array pointer
    output: void
-Function to release unified memory
*/
extern "C" void releaseMemory(char* memory)
{
    cudaFree(memory);
    return;
}

__global__ void initmemoryHNW(int *memory,int const m,int const n, int const d, int const e, int const N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
   
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
    output: - initialised semaphore.
*/
__global__ void initsemaphor(int *semaphore, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
   
    for (int i = index; i < N; i += stride)
    {
            semaphore[i]=0;
    }
    return;
}

/*
Authors: Matej Crnac

Parameters:
    input:  semaphore - pointer to semaphore list
            n - seamphor length
    output: - initialised semaphor.
*/
__device__ void initsemaphorDevice(int *semaphore, int N)
{
    int index = 0;    
   
    for (int i = index; i < N; i += 1)
    {
            semaphore[i]=0;
    }

    return;
}

/*
Authors: Franjo Matković

Parameters:
    input:  memory - pointer to memory
	    m - first sequence length + 1 
	    n - second sequence length +1
            N - m times n
    output: - initialised memory.
*/
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
Authors: Matej Crnac, Franjo Matković, Dario Sitnik

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
    printf("Bez: %d, blockDim: %d, n = %ld\n",(int)n,blockDim.x,n);
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    printf("index: %d Stride: %d N: %ld\n",index,stride,N);
    while(1)
    {
        __syncthreads();
        if(blockIdx.x==0) break;
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
            memory[i]=max(memory[i-n-1]+simil,max(memory[i-n] - d,memory[i-1] - d));
        }
    }
    
    semaphore[blockIdx.x]=1; 

    return;
}


/*
Authors: Matej Crnac, Franjo Matković, Dario Sitnik

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
	    semaphore - pointer to semaphore for sync
    output: - solved cost matrix
-Function to solve SmithWaterman using GPU
*/
__global__ void SW_GPU(double* memory,long int const m,long int const n, double const d, double const e, long int const N,const char* s1, const char* s2,int* semaphore)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        __syncthreads();
        while(1)
        {

            if(i==0) break;

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
            memory[i]=max((double)0,max(memory[i-n-1]+simil,max(memory[i-n] - d,memory[i-1] - d)));
        }
        semaphore[(blockIdx.x)%gridDim.x]=2;
    }
    
    return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory - pointer to memory matrix
            subN - x-axis block index 
            subM - y-axis block index
            n - length of sequence n
            m_orig - length of sequence m before padding
            n_orig - length of sequence n before padding
            localSubN - x-axis thread index 
            localSubM - y-axis thread index 
            blockSize_n - number of elements on block's x axis
            threadSize - size of matrix that thread is solving
            threadSize_m - number of elements on thread's y axis
            threadSize_n - number of elements on thread's x axis
            N - n*m
            s1 - pointer to sequence 1
            s2 - pointer to sequence 2
            score - score structure (match, mismatch, gap-penalty)
            maxBlockInside - vector for storing maximum values on the thread level
            positionmaxBlockInside - vector for storing positions of maximum values on the thread level 
            threadNumber - number of threads in kernel
output: - solved score matrix, max values and positions of max values on the thread level
*/
__device__ void threadFunction(int* memory,int subN, int subM,  int n, int m_orig, int n_orig, int localSubN,  int localSubM, int const blockSize_n, int threadSize, int threadSize_m, int threadSize_n,  int N, char *s1, char *s2,Scorer score, int* maxBlockInside, int* positionmaxBlockInside, int threadNumber)
{
    int totalElem = (int)threadSize;
    int coordx, coordy;
    int simil;
    int position;

    for(int i=0;i<totalElem;i++)
    {
        __syncthreads();
        coordx = i % (int)threadSize_n;
        coordy = i / (int)threadSize_n;
        if((subM+localSubM+coordy) == 0 || (subN + localSubN + coordx) == 0) continue;

        int n_position = subN*blockSize_n + localSubN*threadSize_n + coordx;
        int m_position = subM*blockSize_n+localSubM*threadSize_m+coordy;
        if(n_position >= n_orig || m_position >= m_orig) continue;

        if(s1[subM*blockSize_n+localSubM*threadSize_m+coordy-1]==s2[subN*blockSize_n + localSubN*threadSize_n + coordx-1]) simil = 1;//score.m;
        else simil = -3;//score.mm;
        position = (int)n * ((int)subM*(int)blockSize_n + (int)localSubM*(int)threadSize_m + coordy ) + ((int)subN*(int)blockSize_n + (int)localSubN*(int)threadSize_n + coordx);

        int newScore = (int)max(memory[position-n-1]+(int)simil,max(memory[position-n]-(int)score.d,max(0,memory[position-1]-(int)score.d)));
        memory[position] = newScore;

        if(newScore > maxBlockInside[threadIdx.x])
        {

            maxBlockInside[threadIdx.x]=newScore;
            positionmaxBlockInside[threadIdx.x]=position;
            
        }

    }

    return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory - pointer to memory matrix
            subN - x-axis block index 
            subM - y-axis block index
            n - length of sequence n
            m_orig - length of sequence m before padding
            n_orig - length of sequence n before padding
            blockSize_n - number of elements on block's x axis
            blockSize_m - number of elements on block's y axis
            threadSize - size of matrix that thread is solving
            threadSize_m - number of elements on thread's y axis
            threadSize_n - number of elements on thread's x axis
            N - n*m
            semaphore - pointer to semaphore for synchronization
            s1 - pointer to sequence 1
            s2 - pointer to sequence 2
            score - score structure (match, mismatch, gap-penalty)
            maxBlockInside - vector for storing maximum values on the thread level
            positionmaxBlockInside - vector for storing positions of maximum values on the thread level 
            threadNumber - number of threads in kernel
            threadNumber_n - number of threads in kernel on x axis
            threadNumber_m - number of threads in kernel on y axis
output: - score matrix, max values, positions of max values
synchronization of threads for solving score matrix, max values and positions of max values on the thread level
*/

__global__ void threadSolver(int *memory,int subM,int subN, int const n, int const m_orig,int const n_orig,int const blockSize_m ,int const blockSize_n,int threadNumber,int threadNumber_m,int threadNumber_n,int threadSize,int threadSize_m,int threadSize_n ,char *s1, char *s2, int* semaphore, int N, Scorer scorer, int* maxBlockInside, int* positionmaxBlockInside)
{
    int localSubN = (threadIdx.x % (int)threadNumber_n);
    int localSubM = (threadIdx.x / (int)threadNumber_n);
    int flag = 1;

    __syncthreads();
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
        __syncthreads();
        if(threadIdx.x%(int)threadNumber_n==0 && semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]>0)
        {
            threadFunction(memory,subN,subM,n,m_orig,n_orig,localSubN,localSubM,blockSize_n,threadSize,threadSize_m,threadSize_n, N,s1,s2,scorer,maxBlockInside,positionmaxBlockInside,threadNumber);
            semaphore[(threadIdx.x-int(threadNumber_n))%(int)threadNumber]--;
            semaphore[(threadIdx.x)%(int)threadNumber]=2;
            flag = 0;
            break;
        }
    }

    __syncthreads();
	return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory - pointer to memory matrix
            n - length of sequence n
            m - length of sequence m
            m_orig - length of sequence m before padding
            n_orig - length of sequence n before padding
            blockSize_n - number of elements on block's x axis
            blockSize_m - number of elements on block's y axis
            threadSize - size of matrix that thread is solving
            threadSize_m - number of elements on thread's y axis
            threadSize_n - number of elements on thread's x axis
            N - n*m
            numBlocks - number of blocks solving matrix
            numBlocks_n - number of blocks solving matrix on x-axis
            numBlocks_m - number of blocks solving matrix on y-axis
            semaphore - pointer to semaphore for synchronization
            s1 - pointer to sequence 1
            s2 - pointer to sequence 2
            score - score structure (match, mismatch, gap-penalty)
            maxBlock - vector for storing maximum values on the kernel level
            positionmaxBlock - vector for storing positions of maximum values on the kernel level 
            threadNumber - number of threads in kernel
            threadNumber_n - number of threads in kernel on x axis
            threadNumber_m - number of threads in kernel on y axis
output: - score matrix, max values, positions of max values
synchronization of cuda kernels for solving score matrix, max values and positions of max values on the kernel level
*/

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
    
    cudaDeviceSynchronize();
    while(1)
    {
        if(index==0) 
        {
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);

            semaphore[blockIdx.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                if(maxBlockInside[z]>maxBlock[index])
                {
                    maxBlock[index]=maxBlockInside[z];
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }

        if(index<(int)numBlocks_n && semaphore[(index-1)%gridDim.x]>0)
        {
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);
            semaphore[(index-1)%gridDim.x]--;
            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                if(maxBlockInside[z]>maxBlock[index])
                {
                    maxBlock[index]=maxBlockInside[z];
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
            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                if(maxBlockInside[z]>maxBlock[index])
                {
                    maxBlock[index]=maxBlockInside[z];
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }
        if(index%(int)numBlocks_n==0 && semaphore[(index-int(numBlocks_n))%gridDim.x]>0)
        {
            
            threadSolver<<<1,threadNumber>>>(memory,subM,subN,n,m_orig,n_orig,blockSize_m,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,s1,s2,semaphoreInside, N,scorer,maxBlockInside,positionmaxBlockInside);
            semaphore[(index-int(numBlocks_n))%gridDim.x]--;            
            semaphore[(blockIdx.x)%gridDim.x]=2;
            for(int z=0;z<(int)threadNumber;z++)
            {
                if(maxBlockInside[z]>maxBlock[index])
                {
                    maxBlock[index]=maxBlockInside[z];
                    postionMaxBlock[index]=positionmaxBlockInside[z];
                }
            }
            break;
        }
        
    }

    cudaDeviceSynchronize();
    return;
} 

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            s1 - reference to sequence 1
            s2 - reference to sequence 2
            scorer - score structure (match, mismatch, gap-penalty)
            B - max number of blocks solving matrix
output: - alignment of sequence 1 and sequence 2
*/

void SmithWatermanGPU(std::string const& s1, std::string const& s2, double const B, Scorer scorer)
{

	std::string string_n(s1);
	std::string string_m(s2);

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
    
    //std::cout<<"k: "<<k<<"B: "<<B<<"m/n: "<<m/n<<"B/m/n: "<<B/(m/n)<<"blockNum_n: "<<blockNum_n<<" "<<"blockNum_m: "<<blockNum_m<<" blockNum: "<<blockNum<<std::endl;
    int N = (m)*(n);

	//here we define how much will there be blocks in m and n direction
    int blockSize_n = ceil((double)n/blockNum_n);
	int blockSize = (int)pow(blockSize_n,2);
    
	//std::cout<<"blockSize: "<<blockSize<<std::endl;
	//std::cout<<"blockNum_n: "<<blockNum_n<<" "<<"blockNum_m: "<<blockNum_m<<" blockNum: "<<blockNum<<std::endl;
	//std::cout<<"n: "<<n<<" "<<"m: "<<m<<std::endl;

    
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

    //std::cout<<"threadNumber: "<<threadNumber<<" threadNumber_m: "<<threadNumber_m<<" threadNumber_n; "<<threadNumber_n<<std::endl;
	//here we are padding strings so there are no elements that will be
	padding(string_m,string_n,blockNum_m*(int)ceil(pow(blockSize,0.5)),blockNum_n*(int)ceil(pow(blockSize,0.5)));

	m=string_m.length()+1;
	n=string_n.length()+1;	
   	//std::cout<<"n: "<<n<<" "<<"m: "<<m<<std::endl;
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

    float elapsed=0;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR( cudaEventRecord(start, 0));
    //CALCULATION
    //std::cout<<"Calculation started:"<<std::endl;
    
    kernelCallsKernel<<<blockNum, 1>>>(memory,m,n,m_orig,n_orig,N,x1, x2, blockNum,blockNum_m,blockNum_n,blockSize_n,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,semaphore,scorer,maxBlock,postionMaxBlock);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize (stop) );
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop) );
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));  

    int maxValue=0;
    int maxPosition=0;
    int currentValue;    
    for(int i=0;i<m;i++)
    {
	    for(int j=0;j<n;j++)
	    {
            currentValue = memory[i*(n)+j];
            if (currentValue > maxValue)
            {
                maxValue = currentValue;
                maxPosition = i*(n)+j;
            }
	        std::cout<<currentValue<<" ";  
	    }
	    std::cout<<std::endl;   
    }
	std::cout<<std::endl;
	//std::cout<<"blockNum "<<blockNum<<std::endl;
	/*for(int z=0;z<blockNum;z++)
	{
		std::cout<<maxBlock[z]<<" ";
		if(maxBlock[z]>maxValue)
		{
		    std::cout<<maxBlock[z]<<" ";
		    maxValue=maxBlock[z];
		    maxPosition=postionMaxBlock[z];
		}
	}
	std::cout<<std::endl;*/

	std::cout<<"Max value: "<<maxValue<<" ,position: "<<maxPosition<<std::endl;


    std::vector<std::tuple<char,char,char>> alig = pathReconstruction(memory,maxPosition,n,s2,s1);
    printAlignment(alig);
    std::cout<<"time: "<< elapsed <<std::endl;
    
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

	//memory freeing
	cudaFree(memory);
	cudaFree(M);
	cudaFree(semaphore);
	cudaFree(x1);
	cudaFree(x2);
	return;
}
