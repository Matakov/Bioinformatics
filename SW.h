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
//#include <cuda.h>
#include"utility.h"
//******************************************************************************
// PUBLIC
extern "C" char* allocateMemory(std::string& );

extern "C" float* allocateMatrixMemory(const std::string& ,const std::string& );

extern "C" void releaseMemory(char* );

extern "C" void initialize(float** , double , float, float, float );

extern "C" void NWS(char* , char*, float* );

extern "C" float* allocateMatrixMemoryCPU(const std::string& ,const std::string& );

extern "C" float* initializeMemoryMatrixCPU(const std::string& ,const std::string& , double );

void NeedlemanWunschGPU(std::string const&, std::string const&, double const, double const,double (*)(char,char));

void SmithWatermanGPU(std::string const& , std::string const& , double const, Scorer );

void SmithWatermanPrep(std::string const& , std::string const& , Scorer );
//******************************************************************************
