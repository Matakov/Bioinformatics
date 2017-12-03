#include "utility.h"


/*
Author: Franjo Matkovic

Input parameters: filename

Output parameters: read sequance
*/
void importFile(std::string filename, std::vector<std::string>& data){
	std::string line;
	std::ifstream myfile (filename);
	if (!myfile.is_open()) std::cout << "Unable to open file"<<std::endl; 
	
	while ( getline (myfile,line) )
	{
	data.push_back(line);		
	std::cout << line << '\n';
	}
	myfile.close();
	
	return;
}


