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

//seqan
#include <seqan/file.h>
#include <seqan/sequence.h>

//for importing sequences
#include <seqan/bam_io.h>

void add(int , float, float );

extern "C" void calculate(float *,float *, int );

extern "C" void allocateMemory(std::string& x, char* memory);
extern "C" void releaseMemory(char* memory);

