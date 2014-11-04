#ifndef SBPOPS_H_INCLUDED
#define SBPOPS_H_INCLUDED

#include <petscksp.h>
#include <string>
#include "domain.hpp"
#include "debuggingFuncs.hpp"
#include <assert.h>
#include "spmat.hpp"

using namespace std;

class SbpOps
{

  private:

    const PetscInt    _order,_Ny,_Nz;
    const PetscReal   _dy,_dz;
    //~PetscScalar        *const _muArr;
    PetscScalar      *_muArr;
    Mat              *_mu;

    double _runTime;

    // map boundary conditions to rhs vector
    Mat _Hinvy_Izxe0y_Iz, _Hinvy_IzxeNy_Iz;
    Mat _Iy_HinvzxIy_e0z, _Iy_HinvzxIy_eNz;
    Mat _Hinvy_IzxBySy_IzTxe0y_Iz, _Hinvy_IzxBySy_IzTxeNy_Iz;

    // SBP factors
    PetscScalar *_HinvyArr,*_D1y,*_D1yint,*_D2y,*_SyArr;
    PetscScalar *_HinvzArr,*_D1z,*_D1zint,*_D2z,*_SzArr;
    PetscInt _Sylen,_Szlen;

    // Spmats holding 1D SBP operators (temporarily named with extraneous S's)
    // needed for all orders
    Spmat _Hy,_HyinvS,_D1yS,_D1yintS,_D2yS,_SyS,_Iy;
    Spmat _Hz,_HzinvS,_D1zS,_D1zintS,_D2zS,_SzS,_Iz;
    Spmat _C2y,_C2z; // only needed for 2nd order

    // only needed for 4th order
    Spmat _C3y,_C4y,_D3y,_D4y,_B3,_B4;
    Spmat _C3z,_C4z,_D3z,_D4z;

    // boundary conditions
    PetscScalar const _alphaF,_alphaR,_alphaS,_alphaD,_beta; // penalty terms

    // directory for matrix debugging
    string _debugFolder;


    PetscErrorCode computeDy_Iz();
    PetscErrorCode computeA();
    PetscErrorCode computeRhsFactors();
    //~PetscErrorCode sbpOpsMats(PetscInt N, Mat &D, Mat &D2);

    PetscErrorCode sbpSpmat(const PetscInt N,const PetscScalar scale,Spmat& H,Spmat& Hinv,Spmat& D1,
                 Spmat& D1int, Spmat& D2, Spmat& S);
    PetscErrorCode sbpSpmat4(const PetscInt N,const PetscScalar scale,
                Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4);

    PetscErrorCode sbpArrays(const PetscInt N,const PetscScalar scale,PetscScalar *Hinv,
                             PetscScalar *D1,PetscScalar *D1int,PetscScalar *D2,
                             PetscScalar *S,PetscInt *Slen);
    //~PetscErrorCode computeD2mu(Mat &D2mu);
    PetscErrorCode computeD2ymu(Mat &D2ymu);
    PetscErrorCode computeD2zmu(Mat &D2zmu);

    // disable default copy constructor and assignment operator
    SbpOps(const SbpOps & that);
    SbpOps& operator=( const SbpOps& rhs );

  public:

    Mat _A;
    Mat _Dy_Iz;
    Mat _Hinv;
    //~Vec _rhs;

    SbpOps(Domain&D);
    ~SbpOps();

    //~PetscErrorCode setSystem();
    PetscErrorCode setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcS,Vec &_bcD);
    PetscErrorCode computeHinv();

    // read/write commands
    PetscErrorCode loadOps(const std::string inputDir);
    PetscErrorCode writeOps(const std::string outputDir);

    // visualization
    //~PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N);
    //~PetscErrorCode viewSBP();


};

#endif
