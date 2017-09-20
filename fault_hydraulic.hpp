#ifndef FAULT_P_HPP_INCLUDED
#define FAULT_P_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "fault.hpp"


class SymmFault_Hydr: public SymmFault
{
  private:

    std::string _hydraulicCoupling,_hydraulicTimeIntType; // coupling and integration type (explicit vs implicit)

    // material properties
    Vec _n_p,_beta_p,_k_p,_eta_p,_rho_f;
    PetscScalar _g;

    // stress state
    Vec _sN; // TOTAL normal stress (not effective normal stress)

    // linear system
    std::string          _linSolver;
    KSP                  _ksp;
    PetscScalar          _kspTol;
    SbpOps               *_sbp;
    std::string           _sbpType;
    int                   _linSolveCount;
    Vec                   _bcL,_bcT,_bcB;


    // input fields
    std::vector<double>   _n_pVals,_n_pDepths,_beta_pVals,_beta_pDepths,_k_pVals,_k_pDepths;
    std::vector<double>   _eta_pVals,_eta_pDepths,_rho_fVals,_rho_fDepths;
    std::vector<double>   _pVals,_pDepths,_dpVals,_dpDepths;

    // run time monitoring
    double       _writeTime,_linSolveTime,_ptTime,_startTime,_miscTime;

    // disable default copy constructor and assignment operator
    SymmFault_Hydr(const SymmFault_Hydr & that);
    SymmFault_Hydr& operator=( const SymmFault_Hydr& rhs);

    PetscErrorCode computeVariableCoefficient(const Vec& p,Vec& coeff);
    PetscErrorCode computeInitialSteadyStatePressure(Domain& D);

  public:

    // pressure and perturbation from pressure
    Vec _p;

    SymmFault_Hydr(Domain& D, HeatEquation& He);
    ~SymmFault_Hydr();

    PetscErrorCode setFields(Domain& D);
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm);

    // explicit time integration
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // implicit time integration
    PetscErrorCode be(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);
    PetscErrorCode be_eqCycle(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);
    PetscErrorCode be_MMS(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);


    PetscErrorCode setSNEff(); // update effective normal stress to reflect new pore pressure

    // IO
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeContext();
    PetscErrorCode writeStep(const PetscInt stepCount, const PetscScalar time);
    //~ PetscErrorCode view();
};



#endif
