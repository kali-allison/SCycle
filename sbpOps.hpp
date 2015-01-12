#ifndef SBPOPS_H_INCLUDED
#define SBPOPS_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <assert.h>
#include "domain.hpp"
#include "debuggingFuncs.hpp"
#include "spmat.hpp"

using namespace std;


/*
 * Container for matrices that are needed temporarily to construct
 * main operators. These include 1D SBP operators that are later mapped
 * to 2D, and the 2D factors that are used to enforce boundaries in the
 * A matrix.
 */
struct TempMats
    {
      const PetscInt    _order,_Ny,_Nz;
      const PetscReal   _dy,_dz;
      Mat              *_mu;

      Spmat _Hy,_D1y,_D1yint,_Iy;
      Spmat _Hz,_D1zint,_Iz;

      Mat _muxBySy_Iz;
      Mat _Hyinv_Iz;

      Mat _muxIy_BzSz;
      Mat _Iy_Hzinv;

      Mat _AL;
      Mat _AR;
      Mat _AT;
      Mat _AB;

      TempMats(const PetscInt order,const PetscInt Ny,const PetscScalar dy,const PetscInt Nz,const PetscScalar dz, Mat*mu);
      ~TempMats();

    private:
      // disable default copy constructor and assignment operator
      TempMats(const TempMats & that);
      TempMats& operator=( const TempMats& rhs );
  };


/*
 * Note: PETSc's ability to count matrix creation/destructions is off.
 * For every MATAXPY, the number of destructions increments by 1 more than
 * the number of creations. Thus, after satBoundaries() the number will be off by 4.
 */

class SbpOps
{

  protected:

    const PetscInt    _order,_Ny,_Nz;
    const PetscReal   _dy,_dz;
    PetscScalar      *_muArr;
    Mat              *_mu;

    double _runTime;

    // map boundary conditions to rhs vector
    Mat _rhsL,_rhsR,_rhsT,_rhsB;

    // boundary conditions
    //~PetscScalar const _alphaF,_alphaR,_alphaS,_alphaD,_beta; // penalty terms
    PetscScalar _alphaF,_alphaR,_alphaS,_alphaD,_beta; // penalty terms

    // directory for matrix debugging
    string _debugFolder;


    PetscErrorCode computeDy_Iz(const TempMats& tempMats);
    PetscErrorCode computeA(const TempMats& tempMats);
    PetscErrorCode satBoundaries(TempMats& tempMats);

    PetscErrorCode computeD2ymu(const TempMats& tempMats, Mat &D2ymu);
    PetscErrorCode computeD2zmu(const TempMats& tempMats, Mat &D2zmu);
    PetscErrorCode computeRymu(const TempMats& tempMats,Mat &Rymu);
    PetscErrorCode computeRzmu(const TempMats& tempMats,Mat &Rzmu);


    // disable default copy constructor and assignment operator
    SbpOps(const SbpOps & that);
    SbpOps& operator=( const SbpOps& rhs );

  public:

    Mat _A;
    Mat _Dy_Iz;
    Mat _H;

    SbpOps(Domain&D);
    ~SbpOps();

    //~PetscErrorCode setSystem();
    PetscErrorCode setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcS,Vec &_bcD);
    PetscErrorCode computeH(const TempMats& tempMats);

    // read/write commands
    PetscErrorCode loadOps(const std::string inputDir);
    PetscErrorCode writeOps(const std::string outputDir);

    // visualization
    //~PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N);
    //~PetscErrorCode viewSBP();


};

// functions to construct 1D sbp operators
PetscErrorCode sbpSpmat(const PetscInt order,const PetscInt N,const PetscScalar scale,
                        Spmat& H,Spmat& Hinv,Spmat& D1,Spmat& D1int, Spmat& S);
PetscErrorCode sbpSpmat2(const PetscInt N,const PetscScalar scale,Spmat& D2,Spmat& C2);
PetscErrorCode sbpSpmat4(const PetscInt N,const PetscScalar scale,
                         Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4);

#endif
