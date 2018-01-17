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
#include <ctime>

//******************************************************************************
// PUBLIC
typedef struct score {
    int m;
    int mm;
    int d;
    int e;
} Scorer;

void importFile(std::string, std::vector<std::string>&);
void importFile(std::string, std::map<std::string,std::string>&);

void printVector(std::vector<std::string>&);

void mapToString( const std::map<std::string,std::string>&, std::string&);
std::string getSequence(std::string);
void padding(std::string&, std::string&, int, int);
Scorer setScorer(int , int , int , int );


double maxFun(double , double ,double);

double sim(Scorer, char , char );

std::vector<std::tuple<char,char,char>> pathReconstruction(int* ,const int& , const int& ,const std::string& , const std::string& );

void printAlignment(std::string const& ,std::string const& ,std::vector<char> const& , int, int, int, int);
void printAlignment(std::vector<std::tuple<char,char,char>> vector);

void NeedlemanWunsch(std::string& , std::string& , double , double (*)(char,char), double& ,double& ,double& , double& ,double& ,std::vector<char>& ,Scorer);


void NWG(std::string& , std::string& , double , double (*)(char,char), double& ,int ,int ,int ,int ,double& ,double& , double& ,double& ,std::vector<char>& );

void SmithWaterman(std::string&, std::string&, double, double (*)(char,char), double&, double&, double&, double&, double&, std::vector<char>&,Scorer);


void Hirschberg(std::string&, std::string&, int, int, int, int, double (*)(char,char), int, int, std::vector<char>&);
void BHirschberg(std::string&, std::string&, int, int,  int, int, double (*)(char,char),int, int, int, std::vector<char>&);
void DHirschberg(std::string&, std::string&, int, int,  int, int, double (*)(char,char),int, int, int, int, int, std::vector<char>&);
void UHirschberg(std::string&, std::string&, int, int,  int, int, double (*)(char,char),int, int, int, int, int, double, std::vector<char>&);

void NWS(std::string& , std::string& , double , double (*)(char,char), double& );
void NWH(std::string& , std::string& , double , double (*sim)(char,char), int ,int , double* , double* );

//******************************************************************************
