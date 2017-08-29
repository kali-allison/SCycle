#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "fault.hpp"
#include "heatEquation.hpp"
#include "rootFinderContext.hpp"


class Fault_p: public SymmFault
{
  private:

    Vec _n_p,_beta_p,_k_p,_eta_p,_rho_f;
    PetscScalar _g;

    // disable default copy constructor and assignment operator
    Fault_p(const Fault_p & that);
    Fault_p& operator=( const Fault_p& rhs);

  public:

    // pressure and perturbation from pressure
    Vec _p,_dp;

    Fault_p(Domain& D);
    ~Fault_p();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out,PetscScalar* J);
    PetscErrorCode d_dt(const_it_vec varBegin,it_vec dvarBegin);

    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};



#include "rootFinder.hpp"





#endif
