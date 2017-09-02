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

    // material properties
    Vec _n_p,_beta_p,_k_p,_eta_p,_rho_f;
    PetscScalar _g;

    // stress state
    Vec _sN; // TOTAL normal stress (not effective normal stress)

    std::vector<double>   _n_pVals,_n_pDepths,_beta_pVals,_beta_pDepths,_k_pVals,_k_pDepths;
    std::vector<double>   _eta_pVals,_eta_pDepths,_rho_fVals,_rho_fDepths;
    std::vector<double>   _pVals,_pDepths,_dpVals,_dpDepths;

    // IO viewers
    PetscViewer _pViewer,_dpViewer;

    // disable default copy constructor and assignment operator
    SymmFault_Hydr(const SymmFault_Hydr & that);
    SymmFault_Hydr& operator=( const SymmFault_Hydr& rhs);

  public:

    // pressure and perturbation from pressure
    Vec _p,_dp;

    SymmFault_Hydr(Domain& D, HeatEquation& He);
    ~SymmFault_Hydr();

    PetscErrorCode setFields(Domain& D);
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

    // method for implicit and explicit time stepping
    //~ // explicit:
    //~ PetscErrorCode d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sigmaxy,
      //~ const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt);

    //~ // implicitly solve for temperature using backward Euler
    //~ PetscErrorCode be(const PetscScalar time,const Vec slipVel,const Vec& tau,
      //~ const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);

    PetscErrorCode setSNEff(); // update effective normal stress to reflect new pore pressure

    // IO
    PetscErrorCode writeContext();
    PetscErrorCode writeStep(const PetscInt step);
    //~ PetscErrorCode view();
};



//~ #include "rootFinder.hpp"





#endif
