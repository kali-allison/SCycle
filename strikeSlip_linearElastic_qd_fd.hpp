#ifndef STRIKESLIP_LINEARELASTIC_QD_FD_H_INCLUDED
#define STRIKESLIP_LINEARELASTIC_QD_FD_H_INCLUDED

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
#include "integratorContext_WaveEq_Imex.hpp"

#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "odeSolver_WaveEq.hpp"
#include "odeSolver_WaveImex.hpp"
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



/*
 * Mediator-level class for the simulation of earthquake cycles on a vertical strike-slip fault
 *  with linear elastic material properties.
 * Uses the quasi-dynamic approximation.
 */


class strikeSlip_linearElastic_qd_fd: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContextWave, public IntegratorContext_WaveEq_Imex
{
private:
    // disable default copy constructor and assignment operator
    strikeSlip_linearElastic_qd_fd(const strikeSlip_linearElastic_qd_fd &that);
    strikeSlip_linearElastic_qd_fd& operator=(const strikeSlip_linearElastic_qd_fd &rhs);

    Domain *_D;

    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool           _isMMS; // true if running mms test
    std::string          _outputDir; // output data
    std::string          _inputDir; // input data
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _vL;
    std::string          _isFault; // "dynamic", "static"
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
    std::string          _stateLaw;
    int                  _guessSteadyStateICs; // 0 = no, 1 = yes

    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz;
    PetscScalar          _deltaT, _CFL;
    Vec                  *_y,*_z; // to handle variable grid spacing
    Vec                  _muVec, _rhoVec, _cs, _ay;
    Vec                  _Fhat, _savedU;
    Vec                  _alphay, _alphaz;

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    std::string            _initialConditions;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _stride1D_qd, _stride2D_qd, _stride1D_dyn, _stride2D_dyn, _stride1D_dyn_long, _stride2D_dyn_long;
    PetscInt               _withFhat;
    PetscInt               _maxStepCount_dyn, _maxStepCount_qd, _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime_dyn, _maxTime_qd, _minDeltaT,_maxDeltaT, _maxTime;
    bool                   _inDynamic, _firstCycle;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT, _dT;
    std::vector<string>    _timeIntInds;// keys of variables to be used in time integration
    std::string            _normType;

    PetscInt               _debug, _localStep, _startOnDynamic;

    // viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D, _whichRegime;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _propagateTime, _dynTime, _qdTime;

    bool         _allowed;
    PetscScalar  _triggerqd2d, _triggerd2qd, _limit_qd, _limit_dyn, _limit_stride_dyn;

    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symm_fault, rigid_fault
    string              _qd_bcRType,_qd_bcTType,_qd_bcLType,_qd_bcBType;
    string              _dyn_bcRType,_dyn_bcTType,_dyn_bcLType,_dyn_bcBType;
    string              _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;
    string              _mat_dyn_bcRType,_mat_dyn_bcTType,_mat_dyn_bcLType,_mat_dyn_bcBType;

    std::map <string,std::pair<PetscViewer,string> >  _viewers;


    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode parseBCs(); // parse boundary conditions
    PetscErrorCode computeTimeStep();
    PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz

  public:
    OdeSolver           *_quadEx_qd, *_quadEx_switch; // explicit time stepping
    OdeSolverImex       *_quadImex_qd, *_quadImex_switch; // implicit time stepping
    OdeSolver_WaveEq          *_quadWaveEx;
    OdeSolver_WaveEq_Imex          *_quadWaveImex;

    Fault_qd                   *_fault_qd;
    Fault_fd                   *_fault_dyn;
    LinearElastic              *_material; // linear elastic off-fault material properties
    HeatEquation               *_he;
    PressureEq                 *_p;


    strikeSlip_linearElastic_qd_fd(Domain&D);
    ~strikeSlip_linearElastic_qd_fd();

    // estimating steady state conditions
    PetscErrorCode solveSS();
    PetscErrorCode solveSSb();


    // time stepping functions
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode integrate_qd(); // will call OdeSolver method by same name
    PetscErrorCode integrate_dyn(); // will call OdeSolver method by same name
    PetscErrorCode initiateIntegrand_qd();
    PetscErrorCode initiateIntegrand_dyn();
    PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    bool check_switch(const Fault* _fault);
    PetscErrorCode reset_for_qd();

    // explicit time-stepping methods
    PetscErrorCode d_dt_qd(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_dyn(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt_qd(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);
    PetscErrorCode d_dt_dyn(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,map<string,Vec>& varImo);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);
    PetscErrorCode d_dt(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,map<string,Vec>& varImo);

    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode writeContext_dyn();

    PetscErrorCode view_dyn();

    PetscErrorCode timeMonitor_qd(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor_qd(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    PetscErrorCode timeMonitor_dyn(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor_dyn(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    // debugging and MMS tests
    PetscErrorCode measureMMSError();

};


#endif
