#if !defined(laplace_INCLUDED)
#define laplace_INCLUDED
#include "MersenneTwister.h"
#include "math.h"



class Laplace{
private:
    MTRand mtr;
public:
    double get_laplacian_random_variable(double scale);

};

double Laplace::get_laplacian_random_variable(double scale){
    //loc for miu; scale for b
    double loc = 0;
//    mtr.seed(); //reseed
    double rnd = mtr.randExc();                       // real number in [0,1)
    int sign;


    double uniform = 0.5-rnd;    //real number in (-1/2, 1/2]
    if(uniform>=0){
        sign = 1;
    }else{
        sign = -1;
    }
    return loc-scale * sign * log(1 - 2.0 * fabs(uniform));
}
#endif


