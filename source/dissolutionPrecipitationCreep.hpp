#ifndef DISSOLUTIONPRECIPITATION_H_INCLUDED
#define DISSOLUTIONPRECIPITATION_H_INCLUDED

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


// computes effective viscosity for dissolution-precipitation creep
// 1 / (effVisc) = B * fh2o^r * Vs * exp(3*Vs*sdev /(R*T)) d^-m / sdev
class DissolutionPrecipitationCreep
{
private:
  // disable default copy constructor and assignment operator
  DissolutionPrecipitationCreep(const DissolutionPrecipitationCreep &that);
  DissolutionPrecipitationCreep& operator=(const DissolutionPrecipitationCreep &rhs);

  // load settings and set material parameters
  vector<double>  _BVals,_BDepths,_DVals,_DDepths,_cVals,_cDepths,_VsVals,_VsDepths,_mVals,_mDepths;
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  string          _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  const float     _R; // (kJ/K/mol) gas constant
  Vec             _B,_D,_c,_Vs; // diffusion/shape, molar volume
  Vec             _m; // grain size exponent
  Vec             _invEffVisc; // 1 / (effective viscosity)

  DissolutionPrecipitationCreep(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim);
  ~DissolutionPrecipitationCreep();
  PetscErrorCode guessInvEffVisc(const Vec& Temp, const double dg,const Vec& grainSize, const Vec& WetDistribution);
  PetscErrorCode computeInvEffVisc(const Vec& Temp, const Vec& sdev,const Vec& grainSize, const Vec& WetDistribution);
  PetscErrorCode writeContext(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(PetscViewer& viewer);
};

#endif
