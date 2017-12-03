#include "utility.h"


/*
Authors: Franjo Matkovic

Input parameters: filename

Output parameters: read sequance in a vector
*/
void importFile(std::string filename, std::vector<std::string>& data){
	std::string line;
	std::ifstream myfile (filename);
	if (!myfile.is_open()) std::cout << "Unable to open file"<<std::endl; 
	
	while ( getline (myfile,line) )
	{
		data.push_back(line);		
		//std::cout << line << '\n';
	}
	myfile.close();
	
	return;
}

/*
Authors: Franjo Matkovic

Input parameters: filename

Output parameters: read file in a map
					-key: header
					-data: sequence							
*/
void importFile(std::string filename, std::map<std::string,std::string>& mapData)
{
	//bool Switch = false;
	std::string line;
	std::ifstream myfile (filename);
	std::vector<std::string> keys;
	std::vector<std::string> data;
	std::string seq="";
	if (!myfile.is_open()) std::cout << "Unable to open file"<<std::endl; 
	
	while ( getline (myfile,line) )
	{
		if(line.at(0)=='>')
		{
			keys.push_back(line);
			if(!seq.empty())
			{
				data.push_back(seq);
				seq.clear();
			}
		}
		else
		{
			seq+=line;
			//std::cout<<seq<<std::endl;
		}		
	}
	data.push_back(seq);
	
	myfile.close();

	for(int i=0;i<keys.size();i++)
	{
		mapData.insert(std::make_pair(keys[i], data[i]));
	}
	
	
	return;
}


void printVector(std::vector<std::string>& container)
{
	for(int i=0;i<container.size();i++)
	{
		std::cout<<container[i]<<std::endl;
	}
	//std::cout<<std::endl;
	return;
}


/*
Authors: Franjo Matkovic

concatenace all map values in a string
*/
void mapToString( const std::map<std::string,std::string>& myMap, std::string& vector)
{
	for (std::pair<std::string, std::string> element : myMap)
	{
		vector+=element.second;
	}
	return;
}


