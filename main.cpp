#include "main.h"
#include "utility.h"
#include "SW.h"

int main(int argc, char** argv)
{
	
	if(argc!=2)
	{
		std::cout<<"Program should be called like this: program input_file"<<std::endl;
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



