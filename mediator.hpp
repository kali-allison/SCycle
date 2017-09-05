#ifndef Mediator_H_INCLUDED
#define Mediator_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include <map>
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "fault_hydraulic.hpp"
#include "heatEquation.hpp"
#include "linearElastic.hpp"
#include "powerLaw.hpp"



/* Base class for a linear elastic material
 */
class Mediator: public IntegratorContextEx, public IntegratorContextImex
{
  private:
    // disable default copy constructor and assignment operator
    Mediator(const Mediator &that);
    Mediator& operator=(const Mediator &rhs);

    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool     _isMMS; // true if running mms test
    bool           _bcLTauQS; // true if spinning up viscoelastic problem from constant stress on left boundary
    std::string          _outputDir; // output data
    const PetscScalar    _vL;
    std::string _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _maxStepCount; // largest number of time steps
    PetscReal              _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;

    // set up integrated variables (_varEx, _varIm)
    PetscErrorCode initiateIntegrand();

    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

  public:
    OdeSolver           *_quadEx; // explicit time stepping
    OdeSolverImex       *_quadImex; // implicit time stepping

    Fault              *_fault;
    //~ SymmFault      *_fault;
    //~ SymmFault_Hydr *_fault;
    SymmLinearElastic  *_momBal; // solves momentum balance equation
    HeatEquation _he;


    Mediator(Domain&D);
    ~Mediator();

    PetscErrorCode integrate(); // will call OdeSolver method by same name

    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);
    PetscErrorCode d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);

    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm);

    // debugging and MMS tests
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar, const char *stage);
    PetscErrorCode measureMMSError();

};


#endif