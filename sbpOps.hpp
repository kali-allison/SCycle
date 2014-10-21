#ifndef SBPOPS_H_INCLUDED
#define SBPOPS_H_INCLUDED

#include <petscksp.h>
#include <string>
//~#include "userContext.h"
#include "domain.hpp"
#include "debuggingFuncs.hpp"
#include <assert.h>

using namespace std;

class SbpOps
{

  private:

    const PetscInt    _order,_Ny,_Nz;
    const PetscReal   _dy,_dz;
    //~PetscScalar        *const _muArr;
    PetscScalar        *_muArr;
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

    // boundary conditions
    PetscScalar const _alphaF,_alphaR,_alphaS,_alphaD,_beta; // penalty terms

    // directory for matrix debugging
    string _debugFolder;


    PetscErrorCode computeDy_Iz();
    PetscErrorCode computeA();
    PetscErrorCode computeRhsFactors();
    //~PetscErrorCode sbpOpsMats(PetscInt N, Mat &D, Mat &D2);

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
