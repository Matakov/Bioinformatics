#include "main.h"
#include "utility.h"
#include "SW.h"
#include "argparse.hpp"
#include <string>

int main(int argc, char** argv)
{
	ArgumentParser parser;
	
	parser.addArgument("-f1", "--file1", 1, true);
	parser.addArgument("-f2", "--file2", 1, true);
	parser.addArgument("-m","--mode")
	parser.addArgument("-h", "--help");
	parser.addFinalArgument("-o","--output");
	
	parser.parse(argc, argv);
	
	string helper = parser.retrieve<string>("h");
	if(!helper.empty())
	{
		std::cout<<"Program should be called like this: $ program -f1 [input_file1] -f2 [input_file2]\nOptional arguments are -m [mode], -o [output] and -h as helper"<<std::endl;
		return -1;	
	}
	
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
	return 0;
}



