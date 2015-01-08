#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>

class Domain
{
  public:

    const char       *_file;
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _startBlock,_endBlock; // format for start/end of block of input parameters

    // domain properties
    PetscInt     _order,_Ny,_Nz;
    PetscScalar  _Ly,_Lz,_dy,_dz,_Dc;

    // fault properties
    PetscScalar  _seisDepth;
    PetscScalar  _aVal,_bAbove,_bBelow;
    PetscScalar  _sigma_N_val;

    // shear distribution properties
    std::string       _shearDistribution; // options: mms, constant, gradient, basin
    PetscScalar  _muVal,_rhoVal,_csVal; // if constant
    PetscScalar  _muIn,_muOut,_rhoIn,_rhoOut,_csIn,_csOut; // if basin
    PetscScalar  _depth,_width;
    PetscScalar *_muArr,*_rhoArr,*_csArr; // general data containers
    Mat          _mu;

    // viscosity for asthenosphere
    PetscScalar  _visc;


    // linear solver settings
    std::string _linSolver;
    PetscScalar  _kspTol;

    // time integration settings
    std::string       _timeControlType,_timeIntegrator;
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


    PetscErrorCode setFields();

    // load settings from input file
    PetscErrorCode loadData(const char *file);
    PetscErrorCode loadShearModulusSettings(std::ifstream& infile);

};

#endif
