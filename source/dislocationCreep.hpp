#ifndef DISLOCATIONCREEP_H_INCLUDED
#define DISLOCATIONCREEP_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include <petscviewerhdf5.h>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "heatEquation.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"

using namespace std;

// computes effective viscosity for dislocation creep
// 1 / (effVisc) = A * exp(-QR/T) * fh2o^r * sdev^(n-1)
class DislocationCreep
{
private:
  // disable default copy constructor and assignment operator
  DislocationCreep(const DislocationCreep &that);
  DislocationCreep& operator=(const DislocationCreep &rhs);

  // load settings and set material parameters
  vector<double>  _AVals,_ADepths,_nVals,_nDepths,_QRVals,_QRDepths;
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode loadFieldsFromFiles(const string prefix);
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  string          _delim;
  string          _inputDir;
  string          _prefix; // appended to names of input and output files if multiple disl creep mechanisms desired
  const Vec      *_y,*_z;
  Vec             _A,_n,_QR;// prefactor, stress exponent,activation energy Q divided by gas constant
  Vec             _invEffVisc; // 1 / (effective viscosity)

  DislocationCreep(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim, const string prefix);
  ~DislocationCreep();
  PetscErrorCode guessInvEffVisc(const Vec& Temp, const double dg);
  PetscErrorCode computeInvEffVisc(const Vec& Temp, const Vec& sdev);
  PetscErrorCode writeContext(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(PetscViewer& viewer);
};

#endif
