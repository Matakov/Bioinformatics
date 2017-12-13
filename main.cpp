#include "main.h"
#include "utility.h"
#include "SW.h"
#include "argparse.hpp"
#include <string>

int main(int argc, const char** argv)
{
	/*
	ArgumentParser parser;
	
	parser.addArgument("-f", "--files", 2, true);
	parser.addArgument("-m", "--mode");
	parser.addArgument("-o", "--output", true);
	parser.addFinalArgument("output");
	
	parser.parse(argc, argv);
	
	//std::string helper = parser.retrieve<std::string>("help"); Usage example
	
	std::vector<std::string> data;
	std::map<std::string,std::string> mapData;
	std::map<std::string,std::string>::iterator itr=mapData.begin();
	
	importFile(argv[1],data);

	importFile(argv[1],mapData);
	
	for (itr = mapData.begin(); itr != mapData.end(); ++itr)
	{
		std::cout<< itr->first <<  "\t" << itr->second<<std::endl;
	}
	std::cout<<std::endl;
	
	printVector(data);

	std::string check;
	mapToString(mapData, check);

	//std::cout<<check<<std::endl;
	*/
/*
	int N = 1<<20;
  	float *x, *y;
	calculate(x,y,N);

	char* mem;
	allocateMemory(check,mem);
	for(int i=0;i<check.length();++i)
	{
		std::cout<<mem[i];
	}
	releaseMemory(mem);
*/
	//test NM
	std::string a = "ATATTA";
	std::string b = "ATAT";
	double i,j,k,l,s;
	std::vector<double> vec;
	NeedlemanWunsch(a,b,2,sim,i,j,k,l,s,vec);
	std::cout<<a<<" : "<<b<<" "<<i<<" "<<j<<" "<<k<<" "<<l<<" S: "<<s<<std::endl;
	std::cout<<"Path: ";
	for(int z=0;z<vec.size();z++) std::cout<<vec[z]<<" ";
	std::cout<<std::endl;
	
	return 0;
}



