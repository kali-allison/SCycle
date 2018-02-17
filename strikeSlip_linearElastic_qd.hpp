#ifndef STRIKESLIP_LINEARELASTIC_QD_H_INCLUDED
#define STRIKESLIP_LINEARELASTIC_QD_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <stdio.h>
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
#include "newFault.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "linearElastic.hpp"



/*
 * Mediator-level class for the simulation of earthquake cycles on a vertical strike-slip fault
 *  with linear elastic material properties.
 * Uses the quasi-dynamic approximation.
 */


class StrikeSlip_LinearElastic_qd: public IntegratorContextEx, public IntegratorContextImex
{
private:
    // disable default copy constructor and assignment operator
    StrikeSlip_LinearElastic_qd(const StrikeSlip_LinearElastic_qd &that);
    StrikeSlip_LinearElastic_qd& operator=(const StrikeSlip_LinearElastic_qd &rhs);

    Domain *_D;

    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool           _isMMS; // true if running mms test
    std::string          _outputDir; // output data
    std::string          _inputDir; // input data
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _vL;
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
    std::string          _stateLaw;
    int                  _guessSteadyStateICs; // 0 = no, 1 = yes

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds;// keys of variables to be used in time integration


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;


    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symm_fault, rigid_fault
    string              _bcRType,_bcTType,_bcLType,_bcBType;
    string              _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

    std::map <string,std::pair<PetscViewer,string> >  _viewers;


    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

  public:
    OdeSolver           *_quadEx; // explicit time stepping
    OdeSolverImex       *_quadImex; // implicit time stepping

    //~ Fault                      *_fault;
    NewFault_qd                *_fault;
    LinearElastic              *_material; // linear elastic off-fault material properties
    HeatEquation               *_he;
    PressureEq                 *_p;


    StrikeSlip_LinearElastic_qd(Domain&D);
    ~StrikeSlip_LinearElastic_qd();

    // estimating steady state conditions
    PetscErrorCode solveSS();
    PetscErrorCode solveSSb();


    // time stepping functions
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode initiateIntegrand();
    PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);


    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    // debugging and MMS tests
    PetscErrorCode measureMMSError();

};


#endif
