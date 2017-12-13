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

/*
Authors: Franjo Matkovic

concatenace all map values in a string
*/
double sim(char a, char b)
{
	if(a==b) return 1;
	else return -3;
}


/*
Authors: Franjo Matkovic

concatenace all map values in a string
*/
double maxFun(double a, double b,double c)
{
	return std::max(std::max(a,b),c);
}

/*
Authors: Franjo Matkovic

concatenace all map values in a string
*/
void NeedlemanWunsch(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char), double& b1,double& e1,double& b2, double& e2,double& s,std::vector<char>& pe)
{
	//initialization
	int m=s1.length()+1;
	int n=s2.length()+1;
	//std::cout<<"Lengths: "<<m-1<<" "<<n-1<<std::endl;
	//opening and closing opening
	double d=penalty;
	double e=penalty;
	double en,f,h;
	//pi = 3, pd = 2, pa = 1, ps = 4;
	char pi = 'i';//double pi = 3; //insert
	char pd = 'd';//double pd = 2; //delete
	char pa = 'm';//double pa = 1; //match - mismatch
	char ps = 'e';//double ps = 4;
	double H[m][n];
	char M[m][n];//double M[m][n];
	double Gi[m][n];
	double Gd[m][n];
	double E[m][n];
	double F[m][n];

	//solving part
	//fill elements with zeros
	std::fill(Gi[0], Gi[0] + m * n, 0);
	std::fill(Gd[0], Gd[0] + m * n, 0);
	//std::fill(M[0], M[0] + m * n, 0);
	
	for(int i=1;i<m;i++)
	{
		H[i][0]=-(d+e*(i-1));
		M[i][0]=pd;
		E[i][0]= -1.0/0.0;//-inf
	}
	for(int i=1;i<n;i++)
	{
		H[0][i]=-(d+e*(i-1));
		M[0][i]=pi;
		F[0][i]= -1.0/0.0;//-inf
	}
	M[0][0]='e';//M[0][0]=ps;	
	H[0][0]=0;

	for(int i=1;i<m;i++)
	{
		for(int j=1;j<n;j++)
		{

			E[i][j] = std::max(E[i][j-1]-e,H[i][j-1]-d);
			F[i][j] = std::max(F[i-1][j]-e,H[i-1][j]-d);

			f = std::max(H[i-1][j]-d,F[i-1][j]-e);
			en = std::max(H[i][j-1]-d,E[i][j-1]-e);
			h = H[i-1][j-1] + sim(s1[i-1],s2[j-1]);
			if (f==(F[i-1][j]-e)) Gd[i][j] = Gd[i-1][j]+1;
			if (f==(E[i][j-1]-e)) Gi[i][j] = Gi[i][j-1]+1;
			H[i][j]=maxFun(f,en,h);
			if (H[i][j]==en) M[i][j]=pi;
			if (H[i][j]==f)	M[i][j]=pd;
			if (H[i][j]==h)	M[i][j]=pa;
		}
	}
	/*
	//print H
	std::cout<<"H: "<<std::endl;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<H[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	std::cout<<"M: "<<std::endl;	
	//print M
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<M[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	std::cout<<"Gi: "<<std::endl;
	//print Gi
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<Gi[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	//print Gd
	std::cout<<"Gd: "<<std::endl;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<Gd[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	//print E
	std::cout<<"E: "<<std::endl;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<E[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	//print F
	std::cout<<"F: "<<std::endl;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			std::cout<<std::setw(5)<<F[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
	*/
	//Reconstrucion
	int i=m-1;
	int j=n-1;
	std::vector<char> p;
	while(M[i][j]!=ps)
	{
		if(M[i][j]==pi)
		{
			p.insert(p.begin(),M[i][j]);//Gi[i][j]+1);
			j = j - 1;  //- Gi[i][j] - 1;
		}
		if(M[i][j]==pd)
		{
			p.insert(p.begin(),M[i][j]);//Gd[i][j]+1);
			i = i - 1; //- Gd[i][j] - 1;
		}
		if(M[i][j]==pa)
		{
			p.insert(p.begin(),M[i][j]);
			i--;
			j--;
		}
	}

	

	//Data to return
	b1=0;
	e1=0;
	b2=m-1;
	e2=n-1;
	s=H[m-1][n-1];
	pe=p;
	return;
}

void printAlignment(std::string const& s1,std::string const& s2,std::vector<char> const& vec)
{
	char pi = 'i';//double pi = 3; //insert
	char pd = 'd';//double pd = 2; //delete
	char pa = 'm';//double pa = 1; //match - mismatch
	char ps = 'e';//double ps = 4;

	std::string ps1="";
	std::string ps2="";
	int c1=0;
	int c2=0;
	for(int i=0;i<vec.size();i++)
	{
		//std::cout<<i<<" "<<vec[i]<<" "<<s1[i]<<" "<<s2[i]<<std::endl;
		if(vec[i]=='i')
		{
			ps2+=s2[c2];
			ps1+='-';
			c2++;
		}
		if(vec[i]=='d')
		{
			ps2+='-';
			ps1+=s1[c1];
			c1++;		
		}
		if(vec[i]=='m')
		{
			ps1+=s1[c1];
			ps2+=s2[c2];
			c1++;
			c2++;
		}
	}
	
	std::cout<<"Alignment: "<<std::endl;
	std::cout<<ps2<<"\n"<<ps1<<std::endl;
	return;
}
	
