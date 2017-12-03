#include "main.h"
#include "utility.h"


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

	std::cout<<check<<std::endl;

	return 0;
}



