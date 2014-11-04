#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>

using namespace std;

class Domain
{
  public:

    const char * _file;

    // domain properties
    PetscInt     _order,_Ny,_Nz;
    PetscScalar  _Ly,_Lz,_dy,_dz,_Dc;

    // fault properties
    PetscScalar  _seisDepth;
    PetscScalar  _bAbove,_bBelow;
    PetscScalar  _sigma_N_val;

    // sedimentary basin properties
    std::string  _shearDistribution;
    PetscScalar  _muIn,_muOut;
    PetscScalar  _rhoIn,_rhoOut;
    PetscScalar  _csIn,_csOut;
    PetscScalar *_muArr,*_rhoArr,*_csArr;
    Mat          _mu;
    PetscScalar  _depth,_width;

    // linear solver settings
    std::string _linSolver;
    PetscScalar  _kspTol;

    // time integration settings
    std::string  _timeControlType,_timeIntegrator;
    PetscInt     _strideLength,_maxStepCount;
    PetscScalar  _initTime,_maxTime;
    PetscScalar  _minDeltaT,_maxDeltaT,_initDeltaT;
    PetscScalar  _atol;

    // other tolerances
    PetscScalar  _rootTol;

    // directory for output
    std::string  _outputDir;

    // values not loaded in input file
    PetscScalar  _f0,_v0,_vp;

    Domain(const char * file);
    Domain(const char *file,PetscInt Ny, PetscInt Nz);
    ~Domain();

    PetscErrorCode view(PetscMPIInt rank);
    PetscErrorCode write();


  private:

    // disable default copy constructor and assignment operator
    Domain(const Domain &that);
    Domain& operator=(const Domain &rhs);

    PetscErrorCode loadData(const char *file);
    PetscErrorCode setFields();


};

#endif
