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

	int N = 1<<20;
  	float *x, *y;
	calculate(x,y,N);

	std::string check = "Hello World\n";
	char* mem;
	mem=allocateMemory(check);
	for(int i=0;i<check.length();++i)
	{
		std::cout<<mem[i];
	}
	releaseMemory(mem);

/*
	//test NM
	std::string a = "ATATTA";
	std::string b = "ATAT";
	double i,j,k,l,s;
	std::vector<char> vec;
	NeedlemanWunsch(b,b,2,sim,i,j,k,l,s,vec);
	//std::cout<<a<<" : "<<b<<" "<<i<<" "<<j<<" "<<k<<" "<<l<<" S: "<<s<<std::endl;
	std::cout<<"Path: ";
	for(int z=0;z<vec.size();z++) std::cout<<vec[z]<<" ";
	std::cout<<std::endl;
	printAlignment(b,a,vec);
*/
	float *memory;
	std::string a = "ATATTA";
	std::string b = "ATAT";
	//std::string a = "ATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTA";
	//std::string b = "ATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAATATTAAT";
	/*
	memory=allocateMatrixMemory(a,b);
	std::cout<<"Memory allocated: "<<memory<<std::endl;
	std::cout<<"Location of a memory: "<<&memory<<std::endl;
	initialize(&memory,-2,(a.length()+1),(b.length()+1),(int)(a.length()+1) * (b.length()+1));
	std::cout<<"Memory initialized"<<std::endl;
	std::cout<<(a.length()+1) * (b.length()+1)<<std::endl;
	std::cout<<"Memory address: "<<memory<<std::endl;
	
	for(int i=0;i<(a.length()+1) * (b.length()+1);i++)
	{
		std::cout<<*(memory)<<" ";
		if(i%(b.length()+1)==0)
		{
			std::cout<<std::endl;
		}
	}
	*/
	memory = initializeMemoryMatrixCPU(a,b,2);
	for(int i=0;i<(a.length()+1);i++)
	{
		for(int j=0;j<(b.length()+1);j++)
		{
			std::cout<<memory[i*(a.length()+1)+j]<<" ";
		}
		std::cout<<std::endl;
	}
	free(memory);
	return 0;
}



