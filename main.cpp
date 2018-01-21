#include "main.h"
#include "SW.h"
#include <string>

int main(int argc, const char** argv)
{
    //if(argc<3) {std::cout<<"Too few arguments!\nProgram needs to be called like this $./program [FILES FILES] [match] [mismatch] [gap-open] [gap-extend] [max-blocks]"<<std::endl;exit(1);}
    //if(argc>8) {std::cout<<"Too much arguments!\nProgram needs to be called like this $./program [FILES FILES] [match] [mismatch] [gap-open] [gap-extend] [max-blocks]"<<std::endl;exit(1);}
    int m,mm,d,e;
    double b;
    //Default values
    b = 100;//320;
    m = 1;
    mm = -3;
    d = 2;
    e = 2;
    /*
    std::string s1_name;
    std::string s2_name;

    for(int k=1;k<argc;k++)
    {
        if(k==1) s1_name=argv[k];
        if(k==2) s2_name=argv[k];   
        if(k==3) m=atoi(argv[k]);
        if(k==4) mm=atoi(argv[k]);
        if(k==5) d=atoi(argv[k]);
        if(k==6) e=atoi(argv[k]);
        if(k==7) b=atoi(argv[k]);  
    }
    std::cout<<argv[argc-1]<<std::endl;
    std::cout<<s1_name<<std::endl;
    std::cout<<s2_name<<std::endl;
    std::string s1 = getSequence(s1_name);
    std::string s2 = getSequence(s2_name);
    
    std::cout<<s1<<std::endl;
    std::cout<<s2<<std::endl;
    */
    std::string s1 = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATGGGGGGCCCCCGGGGGGGGGGGGGGGGGATATATATATATATATATATGGGGGGCCCCCGGGGGGGGGGGGGGGGGATATATATATATATATATATGGGGGGCCCCCGGGGGGGGGGGGGGGGG";
    std::string s2 = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT";
    Scorer scorer = setScorer(m,mm,d,e);
    //SmithWatermanGPU(s1,s2,b,scorer);   
    SmithWatermanPrep(s1,s2,scorer);

    return 0;
    
}
