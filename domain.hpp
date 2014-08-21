#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

class Domain
{
  public:

    const char * _file;

    PetscInt     _order,_Ny,_Nz;
    PetscScalar  _Ly,_Lz,_dy,_dz,_Dc;

    // sedimentary basin properties
    PetscScalar  _muIn,_muOut;
    PetscScalar  _rhoIn,_rhoOut;
    PetscScalar *_muArr,*_rhoArr;
    Mat          _mu;
    PetscScalar  _bAbove,_bBelow;
    PetscScalar  _depth,_width,_seisDepth;

    // time integration settings
    PetscInt     _strideLength,_maxStepCount;
    PetscScalar  _initTime,_maxTime;
    PetscScalar  _minDeltaT,_maxDeltaT,_initDeltaT;
    PetscScalar  _atol;

    // other tolerances
    PetscScalar  _kspTol,_rootTol;

    // directory for output
    std::string  _outputDir;

    // values not loaded in input file
    PetscScalar  _f0,_v0,_vp;

    Domain(const char * file);
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
