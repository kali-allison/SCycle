#ifndef Mediator_H_INCLUDED
#define Mediator_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>

#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "integratorContextWave.hpp"
#include "momBalContext.hpp"

#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "odeSolver_WaveEq.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "linearElastic.hpp"
#include "powerLaw.hpp"



/* Base class for a linear elastic material
 */
class Mediator: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContextWave
{
private:
    // disable default copy constructor and assignment operator
    Mediator(const Mediator &that);
    Mediator& operator=(const Mediator &rhs);

    Domain *_D;

    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool           _isMMS; // true if running mms test
    bool                 _bcLTauQS; // true if spinning up viscoelastic problem from constant stress on left boundary
    std::string          _outputDir; // output data
    std::string          _inputDir; // input data
    const bool           _loadICs; // true if starting from a previous simulation
    const PetscScalar    _vL;
    std::string          _problemType; // options: quasidynamic, dynamic, quasidynamic_and_dynamic, steadyStateIts
    std::string          _momBalType; // "dynamic", "static"
    std::string          _bulkDeformationType; // constitutive law
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::string            _initialU; // gaussian
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration

    // steady state data
    std::map <string,Vec>  _varSS; // holds variables for steady state iteration
    PetscScalar            _fss_T,_fss_EffVisc; // damping coefficients, must be < 1


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;

    // set up integrated variables (_varEx, _varIm)


    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

  public:
    OdeSolver           *_quadEx; // explicit time stepping
    OdeSolverImex       *_quadImex; // implicit time stepping
    OdeSolver_WaveEq    *_quadWaveEq; // implicit time stepping

    Fault               *_fault;
    MomBalContext       *_momBal; // solves momentum balance equation
    HeatEquation        *_he;
    PressureEq          *_p;


    Mediator(Domain&D);
    ~Mediator();

    PetscErrorCode integrate(); // will call OdeSolver method by same name


    // dynamic wave equation functions
    PetscErrorCode initiateIntegrand_dyn();
    PetscErrorCode integrate_dyn();
    PetscErrorCode d_dt_WaveEq(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT);

    // adaptive time stepping functions for quasi-static problem
    PetscErrorCode initiateIntegrand_qs();
    PetscErrorCode integrate_qs();

    // to solve a steady-state problem
    std::map <string,PetscViewer>  _viewers;
    PetscErrorCode solveSS_pl(); // assume bcL is correct and do 1 linear solve
    PetscErrorCode solveSS_linEl(); // assume bcL is correct and do 1 linear solve, no effective viscosity
    PetscErrorCode solveSS_v2(); // iterate for eff visc etc
    PetscErrorCode writeSS(const int Ii);
    PetscErrorCode integrate_SS();
    PetscErrorCode initiateIntegrand_ss();

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
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    // debugging and MMS tests
    PetscErrorCode debug(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar, const char *stage);
    PetscErrorCode measureMMSError();

};


#endif
