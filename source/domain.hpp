#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <iostream>
#include <petscviewerhdf5.h>
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
  string         _inputDir; // directory for optional input vectors
  string         _outputDir; // directory for output
  string         _bulkDeformationType; // options: linearElastic, powerLaw
  string         _momentumBalanceType; // options: quasidynamic, dynamic, quasidynamic_and_dynamic
  string         _systemEvolutionType; // options: steadyStateIts, transient (e.g. eq cycles or single eqs)
  string         _sbpType; // matrix or matrix-free, compatible or fully compatible
  string         _operatorType; // matrix-based or matrix-free
  string         _sbpCompatibilityType; // compatible or fullyCompatible
  string         _gridSpacingType; // variableGridSpacing or constantGridSpacing
  int            _isMMS; // run MMS test or not
  int            _computeGreensFunction_fault; // run computeGreensFunction_fault instead of earthquake cycle simulations
  int            _computeGreensFunction_offFault; // run computeGreensFunction_offFault instead of earthquake cycle simulations

  // domain properties
  PetscInt     _order; // accuracy of spatial operators
  PetscInt     _Ny,_Nz; // # of points in y and z directions
  PetscScalar  _Ly,_Lz; // (km) domain size in y and z directions
  PetscScalar  _vL; // loading velocity

  // coordinate system
  Vec   _q,_r,_y,_z,_y0,_z0; // q(y), r(z)
  PetscScalar _dq,_dr;  // spacing in q and r
  PetscScalar _bCoordTrans; // scalar for how aggressive the coordinate transform is

  // checkpoint enabling
  int             _saveChkpts, _restartFromChkpt,_restartFromChkptSS;
  PetscFileMode   _outputFileMode; // maybe change to outFileMode after getting rid of Yuyun's implementation (don't want to get them confused right now)
  PetscInt        _prevChkptTimeStep1D,_prevChkptTimeStep2D; // time step index of simulation data that corresponds to checkpoint data
  //~ PetscInt _ckpt, _ckptNumber, _interval;
  PetscFileMode   _outFileMode; // FILE_MODE_WRITE or FILE_MODE_APPEND

  // scatters to take values from body field(s) to 1D fields
  // naming convention for key (string): body2<boundary>, example: "body2L>"
  map<string, VecScatter> _scatters;

  Domain(const char * file);
  Domain(const char *file,PetscInt Ny, PetscInt Nz);
  ~Domain();

  PetscErrorCode view(PetscMPIInt rank);
  PetscErrorCode write(PetscViewer& viewer);
  PetscErrorCode writeHDF5(PetscViewer& viewer);
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);

private:

  // disable default copy constructor and assignment operator
  Domain(const Domain &that);
  Domain& operator=(const Domain &rhs);

  PetscErrorCode loadSettings(const char *file); // load settings from input file
  PetscErrorCode checkInput();
  PetscErrorCode allocateFields();
  PetscErrorCode setFields();
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();
  PetscErrorCode setScatters(); // generate scatters to move indices of one vector to a differently sized vector (e.g. displacement -> slip)
  PetscErrorCode testScatters();
};

#endif
