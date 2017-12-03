#include "main.h"
#include "utility.h"


int main(int argc, char** argv)
{
	
	if(argc!=2)
	{
		std::cout<<"Program should be called like this: program input_file"<<std::endl;	
	}
	
	std::vector<std::string> data;
	importFile(argv[1],data);
	
	for(int i=0;i<data.size();i++) std::cout<<data[i]<<std::endl;

	return 0;
}



