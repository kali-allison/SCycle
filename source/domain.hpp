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
    string         _delim; // format is: var delim value (without the white space)
    string         _inputDir; // directory to load input files from
    string         _outputDir; // directory for output
    string         _bulkDeformationType; // options: linearElastic, powerLaw
    string         _momentumBalanceType; // options: quasidynamic, dynamic, quasidynamic_and_dynamic, steadyStateIts
    string         _sbpType; // matrix or matrix-free, compatible or fully compatible
    string         _operatorType; // matrix-based or matrix-free
    string         _sbpCompatibilityType; // compatible or fullyCompatible
    string         _gridSpacingType; // variableGridSpacing or constantGridSpacing
    int            _isMMS; // run MMS test or not
    int            _loadICs; // load conditions from input files

    // domain properties
    // order = order of accuracy for spatial derivatives
    // Ny = # points in y direction
    // Nz = # points in z direction
    PetscInt     _order,_Ny,_Nz;
    // Ly = domain size in y direction (km)
    // Lz = domain size in z direction (km)
    PetscScalar  _Ly,_Lz;
    string  _yInputDir; // directory to load y from
    string  _zInputDir; // directory to load z from
    PetscScalar  _vL; // loading velocity

    // coordinate system
    Vec   _q,_r,_y,_z,_y0,_z0; // q(y), r(z)
    PetscScalar _dq,_dr;  // spacing in q and r
    PetscScalar _bCoordTrans; // scalar for how aggressive the coordinate transform is

    // scatters to take values from body field(s) to 1D fields
    // naming convention for key (string): body2<boundary>, example: "body2L>"
    map<string, VecScatter> _scatters;
  
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
    PetscErrorCode setFields();  // set coordinate transforms
    // scatters indices of result vector to new vectors (e.g. displacement -> slip)
    PetscErrorCode setScatters();  
    PetscErrorCode testScatters();
};

#endif
