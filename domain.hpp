#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <iostream>
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
    std::string  _shearDistribution, // options: mms, constant, gradient, basin, CVM
                 _problemType; // options: full, symmetric (only solve y>0 portion)
    std::string  _inputDir; // directory to load shear modulus and normal stress from (if above is CVM)
    // + side fields (always initiated)
    PetscScalar  _muValPlus,_rhoValPlus; // if constant
    PetscScalar  _muInPlus,_muOutPlus,_rhoInPlus,_rhoOutPlus; // if basin
    PetscScalar  _depth,_width;
    PetscScalar *_muArrPlus,*_csArrPlus,*_sigmaNArr; // general data containers
    Mat          _muP;
    // - side fields (sometimes initiated)
    PetscScalar  _muValMinus,_rhoValMinus; // if constant
    PetscScalar  _muInMinus,_muOutMinus,_rhoInMinus,_rhoOutMinus; // if basin
    PetscScalar *_muArrMinus,*_csArrMinus; // general data containers
    Mat          _muM;


    // viscosity distribtion properties for power law rheology
    Vec _A,_temp,_n;


    // linear solver settings
    std::string _linSolver;
    PetscScalar _kspTol;

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
    PetscScalar  _f0,_v0,_vL;

    Domain(const char * file);
    Domain(const char *file,PetscInt Ny, PetscInt Nz);
    ~Domain();

    PetscErrorCode view(PetscMPIInt rank);
    string vector2str(const vector<double> vec);
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
