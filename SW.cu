
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    //CALCULATION
    //std::cout<<"Calculation started:"<<std::endl;
    
    kernelCallsKernel<<<blockNum, 1>>>(memory,m,n,m_orig,n_orig,N,x1, x2, blockNum,blockNum_m,blockNum_n,blockSize_n,blockSize_n,threadNumber,threadNumber_m,threadNumber_n,threadSize,threadSize_m,threadSize_n,semaphore,scorer,maxBlock,postionMaxBlock);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop) ;
    cudaEventElapsedTime(&elapsed, start, stop) ;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  

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
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
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


//CORRECT PROGRAM
//--------------------------------------------------------------------------------------------------------------------------


__global__ void threadLevel(int* memory, int m, int n, char *x1, char *x2, int BlockSize_n,int BlockSize_m, Scorer scorer,int*positionList,int* biggestValue,int* biggestPosition);
__global__ void kernelMain(int* memory,int m,int n,int numBlocks_m,int numBlocks_n,char *x1,char *x2,int *positionList,Scorer scorer,int BlockSize_n,int BlockSize_m, int MAXCORES,int* tempMax, int* tempPosition);

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory  - reference to memory
            i       - chunk row
            j       -chunk column
            BLockSize_n - size of block
            BLockSize_m - size of block
            numOfCores_n - number of kernels on x axis
            numOfCores_m - number of kernels on y axis
            arrayN      - pointer to initialisation vector on x axis
            arrayM      - pointer to initialisation vector on y axis
            MAXCORES    - max number of cores to use
*/
void initmemoryHSWchunk(int* memory,int i,int j,int BlockSize_n,int BlockSize_m,int numOfCores_n, int numOfCores_m, int *arrayN,int 
*arrayM, int MAXCORES)
{
	for(int k=0;k<BlockSize_m*numOfCores_m;k++)
	{
		for(int l=0;l<BlockSize_n*numOfCores_n;l++)
		{
			memory[k*BlockSize_n*numOfCores_n+l] = 0;
		}
	}
    for(int k=0;k<BlockSize_m*numOfCores_m;k++) //inicijalizacija stupca
    {
        memory[k*BlockSize_n*numOfCores_n] = arrayM[i*BlockSize_m*numOfCores_m+k];
    }
    for(int k=0;k<BlockSize_n*numOfCores_n;k++) //inicijalizacija retka
    {
        memory[k] = arrayN[j*BlockSize_n*MAXCORES+k];
    }
	return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory  - reference to memory
            i       - chunk row
            j       -chunk column
            BLockSize_n - size of block
            BLockSize_m - size of block
            numOfCores_n - number of kernels on x axis
            numOfCores_m - number of kernels on y axis
            arrayN      - pointer to initialisation vector on x axis
            arrayM      - pointer to initialisation vector on y axis
    Method saves last row and last column in a chunk
*/
void saveLastRowCol(int* memory,int i,int j,int BlockSize_n,int BlockSize_m,int numOfCores_n, int numOfCores_m,int* arrayN,int* arrayM)
{
	for(int k=0;k<BlockSize_m*numOfCores_m;k++)
	{
		for(int l=0;l<BlockSize_n*numOfCores_n;l++)
		{
			if(l==BlockSize_n*numOfCores_n-1) // zadnji stupac
			{
				arrayM[i*BlockSize_m*numOfCores_m+k]=memory[k*BlockSize_n*numOfCores_n-1];
			}
			else if(k==BlockSize_m*numOfCores_m-1) // zadnji redak
			{
				arrayN[j*BlockSize_n*numOfCores_n+l]=memory[k*BlockSize_n*numOfCores_n+l];
			}
		}
	}
	return;

}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            s1  - first string
            s2  - second string
            scorer - scorer object used to correctly score match and mismatch
    Main method. It prepares everithing for calculation. Does padding, calcualtes number of blocks to use, number of chunks 
    and iterates through chunks and calls MainKernel.
*/
void SmithWatermanPrep(std::string const& s1, std::string const& s2, Scorer scorer)
{    
    float elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

	std::string string_m(s1);
	std::string string_n(s2);

	//memory locations 
	int *memory;

	//every kernel will have 1024 threads
	int BlockSize_n = 32;
	int BlockSize_m = 32;

	int m = string_m.length();
	int n = string_n.length();

	float padm = ceil(float(m)/BlockSize_m)*BlockSize_m;
	float padn = ceil(float(n)/BlockSize_n)*BlockSize_n;

	std::cout<<"Size 1: "<<m<<" Size 2: "<<n<<" padding: "<<padm<<" "<<padn<<std::endl;

	padding(string_m,string_n,padm,padn);    

	m = string_m.length()+1;
	n = string_n.length()+1;

	std::cout<<"Size 1: "<<m<<" Size 2: "<<n<<std::endl;

	int numBlocks_m = float(m)/BlockSize_m;
	int numBlocks_n = float(n)/BlockSize_n;

	std::cout<<"NumBlock m: "<<numBlocks_m<<" NumBlock n: "<<numBlocks_n<<std::endl;

	int *positionList;

	char* x1 ;//= allocateMemory(string_m);
	char* x2 ;// = allocateMemory(string_n);

	///////////////////////////////////////////////////////////////////////////////
	int MAXCORES = 512;
	int numChunks_n = (int)ceil((float)numBlocks_n/(float)MAXCORES);
	int numChunks_m = (int)ceil((float)numBlocks_m/(float)MAXCORES);

    printf("numChunks_n = %d numChunks_m = %d\n",numChunks_n,numChunks_m);
	int numOfCores_n_last = numBlocks_n%MAXCORES;
	int numOfCores_m_last = numBlocks_m%MAXCORES;
	int numOfCores_n = 0;
	int numOfCores_m = 0;
	if(numOfCores_m_last==0){
		numOfCores_m_last=MAXCORES;
	}
	if(numOfCores_n_last==0){
		numOfCores_n_last=MAXCORES;
	}
	int *arrayN;//[n];
	int *arrayM;//[m];
	arrayN = (int*)malloc(n*sizeof(int));
	arrayM = (int*)malloc(m*sizeof(int));
	
	for(int i=0;i<n;i++) *(arrayN + i) = 0;
	for(int i=0;i<m;i++) *(arrayM + i) = 0;

	int maxValues[numChunks_n*numChunks_m];
	int maxPositions[numChunks_n*numChunks_m];
    int maxPosition = 0;
    int maxValue = 0;
	int *tempMax;
	int *tempPosition;
	std::string str1_temp,str2_temp;
	int numOfCores=0;

    cudaMallocManaged(&tempMax, 1*(sizeof(int)));
    cudaMallocManaged(&tempPosition, 1*(sizeof(int)));
	for(int i=0;i<numChunks_m;i++)
	{
		
		for(int j=0;j<numChunks_n;j++)
		{
            printf("Chunk number = %d,%d\n",i,j);
			if(j==numChunks_n-1)
			{
				if(i==numChunks_m-1)
				{
					numOfCores_n = numOfCores_n_last;
					numOfCores_m = numOfCores_m_last;
                    numOfCores=min(numOfCores_n,numOfCores_m);
				}
				else{
					numOfCores_n = numOfCores_n_last;
					numOfCores_m = MAXCORES;
					numOfCores=min(numOfCores_n,numOfCores_m);
				}	
			}
			else if (i==numChunks_m-1)
			{
				numOfCores_m = numOfCores_m_last;
				numOfCores_n = MAXCORES;
				numOfCores=min(numOfCores_n,numOfCores_m);
			}
			else
			{
				numOfCores_n = MAXCORES;
				numOfCores_m = MAXCORES;
				numOfCores=min(numOfCores_n,numOfCores_m);
			}
			cudaMallocManaged(&memory,((BlockSize_n*numOfCores_n)*(BlockSize_m*numOfCores_m)*sizeof(int)));	
			printf("numOfCores = %d, numOfCores_m = %d, numOfCores_n = %d\n",numOfCores,numOfCores_m,numOfCores_n);


			initmemoryHSWchunk(memory, i, j, BlockSize_n, BlockSize_m, numOfCores_n, numOfCores_m, arrayN, arrayM, numOfCores);//inicijalizacija dijela memorije
			cudaMallocManaged(&positionList, numOfCores*sizeof(int)); // inicijalizacija liste pomoću koje se sinkroniziraju jezgre

			initsemaphor<<<1, numOfCores>>>(positionList, numOfCores);
			cudaDeviceSynchronize();
	
			if(i==numChunks_m-1)
			{
                
				str1_temp = string_m.substr(i*BlockSize_m*MAXCORES,BlockSize_m*numOfCores_m_last);
			}
			else str1_temp = string_m.substr(i*BlockSize_m*MAXCORES,BlockSize_m*numOfCores_m);
			const char *cstr = str1_temp.c_str();
			cudaMallocManaged(&x1, str1_temp.length()*(sizeof(char)+1));
            strcpy(x1, cstr);   

			if(j==numChunks_n-1)
			{
				str2_temp = string_n.substr(j*BlockSize_n*MAXCORES,BlockSize_n*numOfCores_n_last);
			}
			else str2_temp = string_n.substr(j*BlockSize_n*MAXCORES,BlockSize_n*numOfCores_n);
			const char *cstr2 = str2_temp.c_str();
			cudaMallocManaged(&x2, str2_temp.length()*(sizeof(char)+1));
		    strcpy(x2, cstr2);   

			kernelMain<<<1,1>>>(memory,m,n,numBlocks_m,numBlocks_n,x1,x2,positionList,scorer,BlockSize_n,BlockSize_m, numOfCores, tempMax, tempPosition);
			cudaDeviceSynchronize();
			
			// spremaju se max vrijednost i pozicija max vrijednosti u trenutnom bloku
			maxValues[i*numChunks_n+j]=tempMax[0];
			maxPositions[i*numChunks_n+j]=tempPosition[0];
            if (tempMax[0] >= maxValue)
            {
                maxValue = tempMax[0];
                maxPosition = tempPosition[0];
            }
            int maxRowPosition = tempPosition[0]/n;
            int maxColumnPosition = tempPosition[0]%n;
			printf("maxVal: %d, maxPos: %d, Row: = %d, Column = %d\n",tempMax[0],tempPosition[0],maxRowPosition,maxColumnPosition);

			//moraju se spremiti vrijednosti s kojima ce se inicijalizirati matrica
			saveLastRowCol(memory,i,j,BlockSize_n,BlockSize_m,numOfCores_n,numOfCores_m,arrayN,arrayM);

			cudaFree(positionList);
			cudaFree(x1);
			cudaFree(x2);
			//cudaFree(memory);
		}		
	}
	
    cudaFree(tempMax);
    cudaFree(tempPosition);

    std::vector<std::tuple<char,char,char>> alig = pathReconstruction(memory,maxPosition,n,s1,s2);
    printAlignment(alig);

    //Stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop) ;
    cudaEventElapsedTime(&elapsed, start, stop) ;
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 

    std::cout<<"time: "<< elapsed <<std::endl;

    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);



	//Ispis memorije
	/*for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(2)<<memory[i*n+j]<<" ";
		}
		std::cout<<std::endl;
	}*/
	

	return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory  - reference to memory
            m       - size of string 1
            n       - size of string 2
            numBlocks_m - number of blocks on y axis
            numBlocks_n - number of blocks on x axis
            x1 - string 1
            x2 - string 2
            positionList      - pointer to positionsList
            scorer      - scored object used to score match and mismatch
            BlockSize_n     -size of block
            BlockSize_m     -size of block
            MAXCORES        -max number of cores to use
            tempMax         -pointer to maxValue found
            tempPosition    -pointer to position of maxValue found
*/

__global__ void kernelMain(int* memory,int m,int n,int numBlocks_m,int numBlocks_n,char *x1,char *x2,int* positionList,Scorer scorer, int BlockSize_n,int BlockSize_m, int MAXCORES,int* tempMax, int* tempPosition)
{
	int numBlocks;
	int nums =numBlocks_m+numBlocks_n-1;
    int *biggestValue = (int*)malloc((int)numBlocks_m*numBlocks_n*sizeof(int));
	int limit=0;
	int big = max(numBlocks_m,numBlocks_n);
	int small = min(numBlocks_m,numBlocks_n);
	int *positionListTemp;
	int iter = 0;
    
    int *biggestPosition = (int*)malloc((int)numBlocks_m*numBlocks_n*sizeof(int));

	for(int i=1;i<=nums;i++)
	{

		positionListTemp = (int*)malloc((int)min(numBlocks_m,numBlocks_n)*sizeof(int));
		numBlocks=min(numBlocks_m,min(i,numBlocks_n));
		if(i > big)
		{
			limit++;
			numBlocks = small-limit;
		}

		threadLevel<<<numBlocks,1024,2*BlockSize_n*BlockSize_m*sizeof(int)>>>(memory, m, n, x1, x2, BlockSize_n,BlockSize_m, scorer,positionList,biggestValue,biggestPosition);
		cudaDeviceSynchronize();

		iter = 0;
		for(int j=0;j<(numBlocks);j++)
		{

			if(positionList[j]%n==0) //First column
			{
				if(!((positionList[j]+BlockSize_n)%n==0)) //if it is first row and first column check if there is next element
				{
					positionListTemp[iter] = positionList[j] + BlockSize_n;
					iter++;
				}
				positionListTemp[iter] = positionList[j] + n*BlockSize_m;
				iter++;     
			}	
			else if((positionList[j]+BlockSize_n)%n==0) //Last column
			{
				continue;
			}
			else if((positionList[j]+n*BlockSize_m)>=(m-1)*n) //Last row
			{
				continue;
			}
			else
			{
				positionListTemp[iter] = positionList[j] + BlockSize_n;
				iter++;
			}
			
		}

		positionList = positionListTemp;
        //free(positionListTemp);
	}
    cudaDeviceSynchronize();
	int N_blocks = numBlocks_m*numBlocks_n;
    

	int tempMaxVal = 0;
	int tempMaxPosition = 0;
	for(int i= 0; i < N_blocks;i++)
	{
		int value = biggestValue[i];
		if (value >= tempMaxVal)
		{
			tempMaxVal = value;
            tempMaxPosition = biggestPosition[i];
        }
    }
    tempMax[0] = tempMaxVal;
    tempPosition[0] = tempMaxPosition;
    free(biggestValue);
	return;
}

/*
Authors: Franjo Matkovic, Dario Sitnik, Matej Crnac

Parameters:
    input:  
            memory  - reference to memory
            m       - size of string 1
            n       -size of string 2
            x1 - string 1
            x2 - string 2
            BlockSize_n     -size of block
            BlockSize_m     -size of block
            scorer      - scored object used to score match and mismatch
            positionList      - pointer to positionsList

            biggestValue         -pointer to maxValue found
            biggestPosition    -pointer to position of maxValue found
*/
__global__ void threadLevel(int* memory, int m, int n, char *x1, char *x2, int BlockSize_n,int BlockSize_m, Scorer scorer, int* positionList, int* biggestValue,int* biggestPosition)
{
	int index = positionList[blockIdx.x];
    int chacheindex = threadIdx.x;

	int threadNum = BlockSize_n*BlockSize_m;
	extern __shared__ int cache[];
	int *chacheMemory = cache;
	int *chachePosition = (int*)&chacheMemory[threadNum];
    
	int tempResult = 0;
	int tempPosition = 0;

	int simil, newScore;
	for(int i=0;i<BlockSize_n+BlockSize_m-1;i++)
	{
			
		if((threadIdx.x%BlockSize_n+threadIdx.x/BlockSize_m)==i)
		{
			if(!(((index + threadIdx.x%BlockSize_n < n) && threadIdx.x/BlockSize_m==0) || ((index%n==0) && (threadIdx.x%BlockSize_n==0))))
			{

				if(x1[(index/n + (threadIdx.x/BlockSize_m))-1]==x2[ index%n + (threadIdx.x%BlockSize_n)-1]) simil = 1;//score.m;
        		else simil = -3;//score.mm;

				newScore = (int)max(memory[(index/n + threadIdx.x/BlockSize_n - 1)*n + (index%n + threadIdx.x%BlockSize_n - 1)]+(int)simil,max(memory[(index/n + threadIdx.x/BlockSize_n - 1)*n + (index%n + threadIdx.x%BlockSize_n)]-(int)scorer.d,max(0,memory[(index/n + threadIdx.x/BlockSize_n)*n + (index%n + threadIdx.x%BlockSize_n - 1)]-(int)scorer.d)));

		       	memory[(index/n + threadIdx.x/BlockSize_m)*n + (index%n + threadIdx.x%BlockSize_n)] = newScore;
                tempResult = newScore;
                tempPosition = (index/n + threadIdx.x/BlockSize_m)*n + (index%n + threadIdx.x%BlockSize_n);
			}
		}
		__syncthreads();
		
	}
    
	
	chacheMemory[chacheindex] = tempResult;
	chachePosition[chacheindex] = tempPosition;

	__syncthreads();

	int i  = (BlockSize_n*BlockSize_m) / 2 ;
	while ( i!=0 )
	{

		if ( chacheindex < i )
		{
		    if (chacheMemory[chacheindex] <= chacheMemory[chacheindex + i])
		    {
			    chacheMemory[chacheindex] = chacheMemory[chacheindex + i];
			    chachePosition[chacheindex] = chachePosition[chacheindex + i];
            }
		}
		__syncthreads();
		i/=2 ;
	}
    
    if(chacheMemory[0] >= biggestValue[blockIdx.x])
    {
        
        biggestValue[blockIdx.x] = chacheMemory[0];

        biggestPosition[blockIdx.x] = chachePosition[0];
        
    }
    __syncthreads();
	return;
}
