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

void printMatrix(int **array, size_t rows, size_t cols)
{
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            std::cout<<std::setw(5)<<array[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    return;
}

/*
Authors: Franjo Matkovic

Needleman-Wunsch algorithm for string alignments
Input parameters:
        - string s1
        - string s2
        - penalty for gaps
        - function for evaluating match/mismatch
Output parameters:
        - b1 - index for beginning of alignemnt on sequence 1 
        - e1 - index for end of alignemnt on sequence 1 
        - b2 - index for beginning of alignemnt on sequence 2 
        - e2 - index for end of alignemnt on sequence 2 
        - s  - alignment score
        - pe  - path for alignment
*/
void NeedlemanWunsch(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char), double& b1,double& e1,double& b2, double& e2,double& s,std::vector<char>& pe)
{
	//initialization
	int m=s1.length()+1;
	int n=s2.length()+1;
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
			if (f==(F[i-1][j]-en)) Gd[i][j] = Gd[i-1][j]+1;
			if (f==(E[i][j-1]-en)) Gi[i][j] = Gi[i][j-1]+1;
			H[i][j]=maxFun(f,en,h);
			if (H[i][j]==en) M[i][j]=pi;
			if (H[i][j]==f)	M[i][j]=pd;
			if (H[i][j]==h)	M[i][j]=pa;
		}
	}
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
	e1=m-1;
	b2=0;
	e2=n-1;
	s=H[m-1][n-1];
	pe=p;
	return;
}

/*
Author: Dario Sitnik

Smith-Waterman algorithm for sequence alignments
Input parameters:
        - string s1
        - string s2
        - penalty for gaps
        - function for evaluating match/mismatch
Output parameters:
        - b1 - index for beginning of alignemnt on sequence 1 
        - e1 - index for end of alignemnt on sequence 1 
        - b2 - index for beginning of alignemnt on sequence 2 
        - e2 - index for end of alignemnt on sequence 2 
        - s  - alignment score
        - pe  - path for alignment
*/
void SmithWaterman(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char), double& b1,double& e1,double& b2, double& e2,double& s,std::vector<char>& pe)
{
    //initialization
    int m=s1.length()+1;
    int n=s2.length()+1;
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
        H[i][0]=0;
        M[i][0]=ps;
        E[i][0]= -1.0/0.0;//-inf
    }
    for(int i=1;i<n;i++)
    {
        H[0][i]=0;
        M[0][i]=ps;
        F[0][i]= -1.0/0.0;//-inf
    }
    M[0][0]=ps; 
    H[0][0]=0;
    s = -1.0/0.0;
    e1 = 0.0;
    e2 = 0.0;
    double temp;
    for(int i=1;i<m;i++)
    {
        for(int j=1;j<n;j++)
        {

            E[i][j] = std::max(E[i][j-1]-e,H[i][j-1]-d);
            F[i][j] = std::max(F[i-1][j]-e,H[i-1][j]-d);

            f = std::max(H[i-1][j]-d,F[i-1][j]-e);
            en = std::max(H[i][j-1]-d,E[i][j-1]-e);
            h = H[i-1][j-1] + sim(s1[i-1],s2[j-1]);
            if (f==(F[i-1][j]-en)) Gd[i][j] = Gd[i-1][j]+1;
            if (f==(E[i][j-1]-en)) Gi[i][j] = Gi[i][j-1]+1;
            temp = maxFun(f,en,h);
            H[i][j]=std::max(static_cast<double>(0),temp);
            if (H[i][j]==en) M[i][j]=pi;
            if (H[i][j]==f) M[i][j]=pd;
            if (H[i][j]==h) M[i][j]=pa;
            if (H[i][j]==0) M[i][j]=ps;
            if (H[i][j]>s){ 
                s=H[i][j];
                e1=i;
                e2=j;
            }

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
    //print M
    std::cout<<"M: "<<std::endl;    
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            std::cout<<std::setw(5)<<M[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    //Reconstruction
    int i=e1;
    int j=e2;
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
    b1=(double)i;
    b2=(double)j;
    e1=e1-1;
    e2=e2-1;
    s=H[(int)e1][(int)e2];
    pe=p;
    return;
}

void printAlignment(std::string const& s1,std::string const& s2,std::vector<char> const& vec,int b1, int e1, int b2, int e2)
{
	char pi = 'i';//double pi = 3; //insert
	char pd = 'd';//double pd = 2; //delete
	char pa = 'm';//double pa = 1; //match - mismatch
	char ps = 'e';//double ps = 4;

	std::string ps1="";
	std::string ps2="";
	int c1=0;
	int c2=0;
	

	std::string str1 = s1.substr(b1,e1);
	std::string str2 = s2.substr(b2,e2);

	for(int i=0;i<vec.size();i++)
	{
		if(vec[i]=='i')
		{
			ps2+=str2[c2];
			ps1+='-';
			c2++;
		}
		if(vec[i]=='d')
		{
			ps2+='-';
			ps1+=str1[c1];
			c1++;		
		}
		if(vec[i]=='m')
		{
			ps1+=str1[c1];
			ps2+=str2[c2];
			c1++;
			c2++;
		}
	}
	
	std::cout<<"Alignment: "<<std::endl;
	std::cout<<ps1<<"\n"<<ps2<<std::endl;
	return;
}
    

/*
Author: Dario Sitnik
helper function for finding neutral element sim(x,y)<=0
*/

char findPadChar(char a, char b, char c)
{
    std::string all("ATCG");
    all.erase(std::remove(all.begin(), all.end(), a), all.end());
    all.erase(std::remove(all.begin(), all.end(), b), all.end());
    if (c == 'N')
    {
        return (all[0]);  
    }
    else
    {
        all.erase(std::remove(all.begin(), all.end(), c), all.end());
        return (all[0]);  
    }
    
}


/*
Author: Dario Sitnik
sequences padding function to match block dimensions
*/
void padding(std::string& a, std::string& b, int M, int N)
{
    if (a.length() % M != 0)
        {
            if (b.length() % N != 0)
            {   
                char pa = findPadChar(a.back(),b.back(), 'N');
                char pb = findPadChar(a.back(),b.back(), pa);
                while (a.length() % M != 0)
                {
                    a += pa;
                }
                while (b.length() % N != 0)
                {
                    b += pb;
                }           
            }
            else 
            {
        
                char pa = findPadChar(a.back(),b.back(), 'N');
                while (a.length() % M != 0)
                {
                    a += pa;
                }           
            }
        }
    else
    {
        if (b.length() % N != 0)
        {
        std::cout<<a.back()<<" "<<b.back()<<std::endl;
            char pb = findPadChar(a.back(),b.back(), 'N');
        std::cout<<pb<<std::endl;
            while (b.length() % N != 0)
                {
                    b += pb;
                }
        }
    }
}




/*
Authors: Franjo Matkovic

Input parameters:
        - string s1
        - string s2
        - penalty for gaps
        - function for evaluating match/mismatch
        - gf1 - sequence 1 begins with a gap
        - gb1 - sequence 1 ends with a gap
        - gf2 - sequence 2 begins with a gap
        - gb2 - sequence 2 ends with a gap
Output parameters:
        - b1  - index for beginning of alignemnt on sequence 1 
        - e1  - index for end of alignemnt on sequence 1 
        - b2  - index for beginning of alignemnt on sequence 2 
        - e2  - index for end of alignemnt on sequence 2 
        - s   - alignment score
        - pe  - path for alignment
*/
void NWG(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char), int gf1,int gb1,int gf2,int gb2,double& b1, double& e1,double& b2, double& e2,double& s,std::vector<char>& pe)
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
		F[i][j] = std::max(H[i-1][j]-d,F[i-1][j]-e);
        	E[i][j] = std::max(H[i][j-1]-d,E[i][j-1]-e);
            

        	f = std::max(H[i-1][j]-d,F[i-1][j]-e);
        	en = std::max(H[i][j-1]-d,E[i][j-1]-e);
        	h = H[i-1][j-1] + sim(s1[i-1],s2[j-1]);
        	if (f==(F[i-1][j]-en)) Gd[i][j] = Gd[i-1][j]+1;
        	if (f==(E[i][j-1]-en)) Gi[i][j] = Gi[i][j-1]+1;
        	H[i][j]=maxFun(f,en,h);
        	//HWG
        	if (gf1 && j==1) H(i,j)=en;
        	if (gf2 && i==1) H(i,j)=f;
        	if (gb1 && j==n-1) H(i,j)=en;
        	if (gb2 && i==m-1) H(i,j)=f;
           
        	if (H[i][j]==en) M[i][j]=pi;
        	if (H[i][j]==f) M[i][j]=pd;
        	if (H[i][j]==h) M[i][j]=pa;
           
        }
    }


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
    e1=m-1;
    b2=0;
    e2=n-1;
    s=H[m-1][n-1];
    pe=p;
    return;
}
/*
Authors: Franjo Matkovic

Input parameters:
        - string s1
        - string s2
        - penalty for gaps
        - function for evaluating match/mismatch
        - gf1 - sequence 1 begins with a gap
        - gf2 - sequence 2 begins with a gap
Output parameters:
        - F   - alignment score
        - H  - path for alignment
*/
void NWH(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char),int gf1,int gf2, double* HR, double* FR)
{
    //initialization
    int m=s1.length()+1;
    int n=s2.length()+1;
    double g1=0;
    double g2=0;
    if (gf1) g1 = d-e;
    if (gf2) g2 = d-e;
    double en,f,h;
    double H[n];
    double E[n];
    double F[n];

    double m1,h1,e1;
    H[0]=0;
    F[0]=0; 
    for(int i=1;i<n;i++)
    {
        H[i]=-(d+e*(i-1))+g2;
        F[i]=-1.0/0.0;
    }
    for(int i=1;i<m;i++)
    {
        if(i==1) m1 = 0;
        else m1 =-(d+e*(i-2))+g1;
        h1 = -(d+e*(i-1))+g1;
        e1 = -1.0/0.0;
        for(int j=1;j<n;j++)
        {
            en = std::max(h1-d,e1-e);
            f = std::max(H[j]-d,F[j]-e);
            h = m1 + sim(s1(i-1),s2(j-1));
            h1 = maxFun(en,f,h);
            e1 = en;
            m1 = H[j];
            H[j] = h1;
            F[j] = f;
            
        }
    }
    //OUTPUT
    HR=H;
    FR=F;
    return;
}


/*
Input parameters:
        - string s1
        - string s2
        - penalty for gaps
        - function for evaluating match/mismatch
Output parameters:
        - s   - alignment score
        
*/
/*
void NWS(std::string& s1, std::string& s2, double penalty, double (*sim)(char,char), double& s)
{
    //initialization
    int m=s1.length()+1;
    int n=s2.length()+1;
    //std::cout<<"Lengths: "<<m-1<<" "<<n-1<<std::endl;
    //opening and closing opening
    double d=penalty;
    double e=penalty;
    double en,f,h;
    double H[n];
    double E[n];
    double F[n];

    double m1,h1,e1;
    H[0]=0;
    F[0]=0; 
    for(int i=0;i<n;i++)
    {
        H[i]=-(d+e*(i-1));
        F[i]=-1.0/0.0;
    }
    for(int i=0;i<m;i++)
    {
        if(i==1) m1 = 0;
        else m1 =-(d+e*(i-2));
        h1 = -(d+e*(i-1));
        e1 = -1.0/0.0;
        for(int j=0;j<n;j++)
        {
            en = std::max(h1-d,e1-e);
            f = std::max(H[j]-d,F[j]-e);
            h = m1 + sim(s1(i-1),s2(j-1));
            h1 = maxFun(en,f,h);
            e1 = en;
            m1 = H[j];
            H[j] = h1;
            F[j] = f;
            
        }
    }
    s = H[i];
    return;
}
*/




