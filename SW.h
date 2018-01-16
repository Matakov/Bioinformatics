#include<iostream>
#include<iomanip>
#include<sstream>
#include<functional>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include<map>
#include<time.h>
#include<iterator>
#include<string>
#include <cuda.h>
//#include "cuPrintf.cu"

//seqan
//#include <seqan/file.h>
//#include <seqan/sequence.h>

//for importing sequences
//#include <seqan/bam_io.h>

void add(int , float, float );

extern "C" void calculate(float *,float *, int );

//extern "C" void allocateMemory(std::string& , char* );
extern "C" char* allocateMemory(std::string& );
extern "C" float* allocateMatrixMemory(const std::string& ,const std::string& );

extern "C" void releaseMemory(char* );

//void initializeNWS(float** , double, float, float, float);
extern "C" void initialize(float** , double , float, float, float );

extern "C" void NWS(char* , char*, float* );

extern "C" float* allocateMatrixMemoryCPU(const std::string& ,const std::string& );
extern "C" float* initializeMemoryMatrixCPU(const std::string& ,const std::string& , double );

void NeedlemanWunschGPU(std::string const&, std::string const&, double const, double const,double (*)(char,char));
//__global__ void initmemoryHNW(double *,double const ,double const , double const , double const , double const );
//__global__ void initmemoryHSW(double *,double const ,double const , double const , double const , double const );
//__global__ void initmemoryAR(double *,double const ,double const , double const , double const , double const );

//void SmithWatermanGPU(std::string const& , std::string const& , double const , double const );
void SmithWatermanGPU(std::string const& , std::string const& , double const , double const , double const );
void SmithWatermanGPU_Basic(std::string const&, std::string const&, double const, double const);
