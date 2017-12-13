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
//#include <seqan/file.h>
//#include <seqan/sequence.h>

//for importing sequences
//#include <seqan/bam_io.h>

void importFile(std::string, std::vector<std::string>&);
void importFile(std::string, std::map<std::string,std::string>&);

void printVector(std::vector<std::string>&);

void mapToString( const std::map<std::string,std::string>&, std::string&);

double maxFun(double , double ,double);

double sim(char , char );

void NeedlemanWunsch(std::string& , std::string& , double , double (*)(char,char), double& ,double& ,double& , double& ,double& ,std::vector<char>& );

void printAlignment(std::string const& ,std::string const& ,std::vector<char> const& );
