#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <iostream>
#include <petscdmda.h>
#include <petscdm.h>
#include "genFuncs.hpp"


using namespace std;

class Domain
{
  public:

    const char       *_file;
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _startBlock,_endBlock; // format for start/end of block of input parameters

    // domain properties
    PetscInt     _order,_Ny,_Nz;
    PetscScalar  _Ly,_Lz,_dy,_dz;


    // boundary conditions
    std::string _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction

    // material distribution properties
    std::string  _inputDir; // directory to load shear modulus and normal stress from (if above is CVM)
    std::string  _zInputDir; // directory to load z from
    std::string  _yInputDir; // directory to load y from
    std::string  _shearDistribution, // options: mms, constant, gradient, basin, CVM
                 _problemType; // options: full, symmetric (only solve y>0 portion)
    PetscInt     _loadICs; // whether or not to look in inputDir for initial conditions
    // + side fields (always initiated)
    PetscScalar  _muValPlus,_rhoValPlus; // if constant
    PetscScalar  _muInPlus,_muOutPlus,_rhoInPlus,_rhoOutPlus; // if basin
    PetscScalar  _depth,_width;
    PetscScalar *_muArrPlus,*_csArrPlus,*_sigmaNArr; // general data containers
    // - side fields (sometimes initiated)
    PetscScalar  _muValMinus,_rhoValMinus; // if constant
    PetscScalar  _muInMinus,_muOutMinus,_rhoInMinus,_rhoOutMinus; // if basin
    PetscScalar *_muArrMinus,*_csArrMinus; // general data containers

    // for coordinate transform
    Vec   _q,_r,_y,_z; // q(y), r(z)
    PetscScalar _dq,_dr;



    // linear solver settings
    //~ std::string _linSolver; // type of linear solver used: MUMPSCHOLESKY, MUMPSLU, AMG
    std::string _sbpType; // matrix or matrix-free, compatible or fully compatible
    PetscScalar _bCoordTrans; // scalar for how aggressive the coordinate transform is
    //~ PetscScalar _kspTol; // tolerance for iterative solver

    // time integration settings
    std::string  _timeControlType,_timeIntegrator;
    PetscInt     _stride1D,_stride2D,_maxStepCount;
    PetscScalar  _initTime,_maxTime;
    PetscScalar  _minDeltaT,_maxDeltaT,_initDeltaT;
    PetscScalar  _atol;
    //~ std::vector<int> _timeIntInds; // indices of variables to be used in time integration
    std::vector<string> _timeIntInds; // keys of variables to be used in time integration

    // other tolerances

    // directory for output
    std::string  _outputDir;

    // values not loaded in input file
    //~ PetscScalar  _f0,_v0,_vL;
    PetscScalar  _vL;


    // DMDA for all vectors
    DM _da;
    PetscInt _yS,_yE,_zS,_zE; // Start and End indices for loops (does NOT include ghost points)
    Vec          _muVecP; // vector version of shear modulus
    Vec          _csVecP,_rhoVecP;
    Vec          _muVecM;

    Domain(const char * file);
    Domain(const char *file,PetscInt Ny, PetscInt Nz);
    ~Domain();

    PetscErrorCode view(PetscMPIInt rank);
    PetscErrorCode write();


  private:

    // disable default copy constructor and assignment operator
    Domain(const Domain &that);
    Domain& operator=(const Domain &rhs);


    PetscErrorCode loadFieldsFromFiles();
    PetscErrorCode setNormalStress();
    PetscErrorCode setFieldsPlus();
    PetscErrorCode setFieldsMinus();

    // load settings from input file
    PetscErrorCode loadData(const char *file);
    PetscErrorCode loadShearModSettings(ifstream& infile);
    //~PetscErrorCode loadVectorFromInputFile(const string& str,vector<double>&);

    // check input from file
    PetscErrorCode checkInput();

};

#endif
