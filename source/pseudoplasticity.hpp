#ifndef PSEUDOPLASTICITY_H_INCLUDED
#define PSEUDOPLASTICITY_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include <petscviewerhdf5.h>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"

using namespace std;

// computes effective viscosity for pseudoplasticity
// 1 / (effVisc) = (yield stress) / (inelastic strain rate)
class Pseudoplasticity
{
private:
  // disable default copy constructor and assignment operator
  Pseudoplasticity(const Pseudoplasticity &that);
  Pseudoplasticity& operator=(const Pseudoplasticity &rhs);

  // load settings and set material parameters
  vector<double>  _yieldStressVals,_yieldStressDepths; // define yield stress
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  const string    _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  Vec             _yieldStress; // (MPa)
  Vec             _invEffVisc; // (GPa) eff. viscosity from plasticity

  Pseudoplasticity(Domain& D,const Vec& y, const Vec& z, const char *file, const string delim);
  ~Pseudoplasticity();
  PetscErrorCode guessInvEffVisc(const double dg);
  PetscErrorCode computeInvEffVisc(const Vec& dgdev);
  PetscErrorCode writeContext(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(PetscViewer& viewer);
};

#endif
