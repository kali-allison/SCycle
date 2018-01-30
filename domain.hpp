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

    const char    *_file;
    std::string    _delim; // format is: var delim value (without the white space)
    std::string    _outputDir; // directory for output
    std::string    _bulkDeformationType; // options: linearElastic, powerLaw
    std::string    _problemType; // options: strikeSlip or iceStream
    std::string    _momentumBalanceType; // options: quasidynamic, dynamic, quasidynamic_and_dynamic, steadyStateIts
    std::string    _sbpType; // matrix or matrix-free, compatible or fully compatible
    int            _isMMS; // run MMS test or not
    int            _loadICs; // load conditions from input files
    std::string    _inputDir; // directory to load input files from

    // domain properties
    PetscInt     _order,_Ny,_Nz;
    PetscScalar  _Ly,_Lz;
    PetscScalar  _alphay,_alphaz;
    std::string  _yInputDir; // directory to load y from
    std::string  _zInputDir; // directory to load z from
    PetscScalar  _vL; // loading velocity

    // coordinate system
    Vec   _q,_r,_y,_z; // q(y), r(z)
    std::map <string, VecScatter>  _scatters;
    Vec _y0, _z0;
    PetscScalar _dq,_dr;
    PetscScalar _bCoordTrans; // scalar for how aggressive the coordinate transform is


    // DMDA for all vectors
    DM _da;
    PetscInt _yS,_yE,_zS,_zE; // Start and End indices for loops (does NOT include ghost points)

    Domain(const char * file);
    Domain(const char *file,PetscInt Ny, PetscInt Nz);
    ~Domain();


    PetscErrorCode view(PetscMPIInt rank);
    PetscErrorCode write();


  private:

    // disable default copy constructor and assignment operator
    Domain(const Domain &that);
    Domain& operator=(const Domain &rhs);

    PetscErrorCode loadData(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode setFields();
    PetscErrorCode setScatters();
};

#endif
