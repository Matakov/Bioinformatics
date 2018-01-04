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


void importFile(std::string, std::vector<std::string>&);
void importFile(std::string, std::map<std::string,std::string>&);

void printVector(std::vector<std::string>&);

void mapToString( const std::map<std::string,std::string>&, std::string&);

double maxFun(double , double ,double);

double sim(char , char );

void NeedlemanWunsch(std::string& , std::string& , double , double (*)(char,char), double& ,double& ,double& , double& ,double& ,std::vector<char>& );

void printAlignment(std::string const& ,std::string const& ,std::vector<char> const& );

void NWG(std::string& , std::string& , double , double (*)(char,char), double& ,int ,int ,int ,int ,double& ,double& , double& ,double& ,std::vector<char>& );

void NWS(std::string& , std::string& , double , double (*)(char,char), double& );

void NWH(std::string& , std::string& , double , double (*sim)(char,char), int ,int , double* , double* );

void padding(std::string&, std::string&, int, int);
