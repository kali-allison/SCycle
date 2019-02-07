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

/*
 * Class containing basic details of the domain and problem type which
 * must be either kept constant throughout all other classes in the program
 * (such as the size of the domain), or which are used to determine which
 * functions are called within main.cpp.
 *
 * All Vecs should be constructed with the same parallel structure as
 * this class's Vecs y0 (Vec of length Ny), z0 (Vec of length Nz), and y or z
 * (both Ny*Nz long), to ensure that Vec entries match up across classes.
 *
 */


using namespace std;

class Domain
{
  public:

    const char    *_file;
    std::string    _delim; // format is: var delim value (without the white space)
    std::string    _outputDir; // directory for output
    std::string    _bulkDeformationType; // options: linearElastic, powerLaw
    std::string    _momentumBalanceType; // options: quasidynamic, dynamic, quasidynamic_and_dynamic, steadyStateIts
    std::string    _sbpType; // matrix or matrix-free, compatible or fully compatible
    std::string    _operatorType; // matrix-based or matrix-free
    std::string    _sbpCompatibilityType; // compatible or fullyCompatible
    std::string    _gridSpacingType; // variableGridSpacing or constantGridSpacing
    int            _isMMS; // run MMS test or not
    int            _loadICs; // load conditions from input files
    std::string    _inputDir; // directory to load input files from

    // domain properties
    // order = order of accuracy for spatial derivatives
    // Ny = # points in y direction
    // Nz = # points in z direction
    PetscInt     _order,_Ny,_Nz;
    // Ly = domain size in y direction (km)
    // Lz = domain size in z direction (km)
    PetscScalar  _Ly,_Lz;
    std::string  _yInputDir; // directory to load y from
    std::string  _zInputDir; // directory to load z from
    PetscScalar  _vL; // loading velocity

    // coordinate system
    Vec   _q,_r,_y,_z; // q(y), r(z)
    PetscScalar _dq,_dr;  // spacing in q and r
    PetscScalar _bCoordTrans; // scalar for how aggressive the coordinate transform is

    // scatters to take values from body field(s) to 1D fields
    // naming convention for key (string): body2<boundary>, example: "body2L>"
    std::map <string, VecScatter>  _scatters;
    Vec _y0; // y = 0 vector, size Nz
    Vec _z0; // z = 0 vector, size Ny

    Domain(const char * file);  // constructor 1
    Domain(const char *file,PetscInt Ny, PetscInt Nz);  // constructor 2
    ~Domain(); // destructor


    PetscErrorCode view(PetscMPIInt rank);
    PetscErrorCode write();

  private:

    // disable default copy constructor and assignment operator
    Domain(const Domain &that);
    Domain& operator=(const Domain &rhs);

    // fuctions defined within the class in cpp file
    PetscErrorCode loadData(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode setFields();
    PetscErrorCode setScatters();
};

#endif
