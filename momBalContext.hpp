#ifndef MOMBALCONTEXT_HPP_INCLUDED
#define MOMBALCONTEXT_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>

/*
 * This abstract class defines an interface for momentum balance routines.
 */

class MomBalContext
{
  public:

    MomBalContext(){};
    virtual ~MomBalContext(){};

    // for steady state computations
    PetscErrorCode virtual getTauVisc(Vec& tauVisc, const PetscScalar ess_t) = 0; // compute initial tauVisc
    PetscErrorCode virtual updateSSa(map<string,Vec>& varSS) = 0; // update v, viscous strain rates, viscosity
    PetscErrorCode virtual updateSSb(map<string,Vec>& varSS) = 0; // does nothing for the linear elastic equations
    PetscErrorCode virtual initiateVarSS(map<string,Vec>& varSS) = 0; // put viscous strains etc in varSS
    PetscErrorCode virtual updateFieldsSS(map<string,Vec>& varSS, const PetscScalar ess_t) = 0;

    // time stepping function
    PetscErrorCode virtual initiateIntegrand_qs(const PetscScalar time,map<string,Vec>& varEx) = 0;
    PetscErrorCode virtual initiateIntegrand_dyn(const PetscScalar time,map<string,Vec>& varEx) = 0;
    PetscErrorCode virtual updateFields(const PetscScalar time,const map<string,Vec>& varEx) = 0;
    PetscErrorCode virtual updateTemperature(const Vec& T) = 0;
    PetscErrorCode virtual computeMaxTimeStep(PetscScalar& maxTimeStep) = 0;
    PetscErrorCode virtual getStresses(Vec& sxy, Vec& sxz, Vec& sdev) = 0;

    // methods for explicit time stepping
    PetscErrorCode virtual d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;
    PetscErrorCode virtual d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT) = 0;
    PetscErrorCode virtual d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;
    PetscErrorCode virtual d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;
    PetscErrorCode virtual debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage) = 0;

    // methods for implicit/explicit time stepping
    PetscErrorCode virtual d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt) = 0; // IMEX backward Euler

    PetscErrorCode virtual measureMMSError(const PetscScalar time) = 0;

    // IO commands
    PetscErrorCode virtual view(const double totRunTime) = 0;
    PetscErrorCode virtual writeContext(const std::string outputDir) = 0;
    PetscErrorCode virtual writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir) = 0; // write out 1D fields
    PetscErrorCode virtual writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir) = 0; // write out 2D fields

    PetscErrorCode virtual getRhoVec(Vec& rho){return 1;}; // compute initial tauVisc
    PetscErrorCode virtual updateU(map<string,Vec>& varEx){return 1;}; // update _u for vizualise
};


#endif
