#ifndef DIFFUSIONCREEP_H_INCLUDED
#define DIFFUSIONCREEP_H_INCLUDED

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


// computes effective viscosity for diffusion creep
// 1 / (effVisc) = A exp(-B/T) * fh2o^r sdev^n d^-m
class DiffusionCreep
{
private:
  // disable default copy constructor and assignment operator
  DiffusionCreep(const DiffusionCreep &that);
  DiffusionCreep& operator=(const DiffusionCreep &rhs);

  // load settings and set material parameters
  vector<double>  _AVals,_ADepths,_QRVals,_QRDepths,_nVals,_nDepths,_mVals,_mDepths;
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  string          _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  Vec             _A,_n,_QR,_m;// prefactor, stress exponent, fluid fugacity exponent, activation energy /R, grain size exponent
  Vec             _invEffVisc; // 1 / (effective viscosity)

  DiffusionCreep(Domain& D,const Vec& y, const Vec& z, const char *file, const string delim);
  ~DiffusionCreep();
  PetscErrorCode guessInvEffVisc(const Vec& Temp, const double dg,const Vec& grainSize);
  PetscErrorCode computeInvEffVisc(const Vec& Temp, const Vec& sdev,const Vec& grainSize);
  PetscErrorCode writeContext(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(PetscViewer& viewer);
};

#endif
