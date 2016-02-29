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

    // domain dimensions etc
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;
    const PetscScalar    _kspTol;

    const char       *_file; // input file location
    std::string       _outputDir; // output file location
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _inputDir; // directory to load viscosity from
    std::string _heatFieldsDistribution;
    std::string  _kFile,_rhoFile,_hFile,_cFile; // names of each file within loadFromFile


    std::vector<double> _rhoVals,_rhoDepths,_kVals,_kDepths,_hVals,_hDepths,_cVals,_cDepths,_TVals,_TDepths;
    Vec     _k,_rho,_c,_h;  // thermal conductivity, density, heat capacity, heat generation
    PetscScalar *_kArr;
    Mat          _kMat;
    PetscViewer  _TV;

    SbpOps_fc* _sbpT;
    Vec _bcT,_bcR,_bcB,_bcL; // boundary conditions


    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode setFields();
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);
    PetscErrorCode loadFieldsFromFiles();

    PetscErrorCode checkInput();     // check input from file

    PetscErrorCode computeSteadyStateTemp();
    PetscErrorCode setBCs();


  public:

    Vec _T;

    HeatEquation(Domain&D);
    ~HeatEquation();

    PetscErrorCode getTemp(Vec& T); // return temperature
    PetscErrorCode setTemp(Vec& T); // set temperature

    // compute rate
    PetscErrorCode d_dt(const PetscScalar time,const Vec slipVel,const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt);

    PetscErrorCode writeContext(const string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount);
};





#endif
