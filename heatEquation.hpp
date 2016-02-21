#ifndef HEATEQUATION_H_INCLUDED
#define HEATEQUATION_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"



/* Base class for a linear elastic material
 */
class HeatEquation
{
  private:
    // disable default copy constructor and assignment operator
    HeatEquation(const HeatEquation &that);
    HeatEquation& operator=(const HeatEquation &rhs);

  protected:

    std::string _heatFieldsDistribution;

    SbpOps* _sbpT;
    Vec     _k,_rho,_c,_h;  // thermal conductivity, density, heat capacity, heat generation
    PetscScalar *_kArr
    Mat          _kMat;

    Vec _bcT,_bcR,_bcB,_bcL; // boundary conditions


    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode setFields();
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);
    PetscErrorCode loadFieldsFromFiles();

    PetscErrorCode checkInput();     // check input from file

    PetscErrorCode computeSteadyStateTemp();


  public:

  Vec _T;

    HeatEquation(Domain&D);
    ~HeatEquation();

    // compute rate
    PetscErrorCode d_dt(const PetscScalar time,const Vec slipVel,
                                          it_vec dvarBegin,it_vec dvarEnd);
};





#endif
