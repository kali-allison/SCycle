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
#include "integratorContext_WaveEq.hpp"
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
 * with linear elastic material properties.
 * Uses the quasi-dynamic approximation.
 */


class strikeSlip_linearElastic_qd_fd: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContext_WaveEq, public IntegratorContext_WaveEq_Imex
{
private:
    // disable default copy constructor and assignment operator
    strikeSlip_linearElastic_qd_fd(const strikeSlip_linearElastic_qd_fd &that);
    strikeSlip_linearElastic_qd_fd& operator=(const strikeSlip_linearElastic_qd_fd &rhs);

    Domain *_D;

    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    std::string          _outputDir; // output data
    std::string          _inputDir; // input data
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _vL;
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
    std::string          _stateLaw;
    int                  _guessSteadyStateICs; // 0 = no, 1 = yes
    PetscScalar          _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

    PetscInt             _cycleCount,_maxNumCycles;
    PetscScalar          _deltaT, _deltaT_fd, _CFL; // current time step size, time step for fully dynamic, CFL factor
    Vec                 *_y,*_z;
    Vec                  _ay;
    Vec                  _Fhat;
    Vec                  _alphay;
    bool                 _inDynamic,_allowed;
    PetscScalar          _trigger_qd2fd, _trigger_fd2qd, _limit_qd, _limit_dyn, _limit_stride_dyn;

    // time stepping data
    std::map <string,Vec>  _varFD,_varFDPrev; // holds variables for time step: n+1, n (current), n-1
    std::map <string,Vec>  _varQSEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    std::string            _initialConditions;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _stride1D_qd, _stride2D_qd, _stride1D_fd, _stride2D_fd, _stride1D_fd_end, _stride2D_fd_end;
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_minDeltaT,_maxDeltaT, _maxTime;

    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT, _dT;
    std::vector<string>    _timeIntInds;// keys of variables to be used in time integration
    std::string            _normType;


    // viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D, _regime1DV, _regime2DV;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _propagateTime, _dynTime, _qdTime;



    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symm_fault, rigid_fault
    string              _qd_bcRType,_qd_bcTType,_qd_bcLType,_qd_bcBType;
    string              _fd_bcRType,_fd_bcTType,_fd_bcLType,_fd_bcBType;
    string              _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;
    string              _mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType;

    std::map <string,std::pair<PetscViewer,string> >  _viewers;


    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode parseBCs(); // parse boundary conditions
    PetscErrorCode computeTimeStep();
    PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz

  public:
    OdeSolver                 *_quadEx_qd; // explicit time stepping
    OdeSolverImex             *_quadImex_qd; // implicit time stepping
    OdeSolver_WaveEq          *_quadWaveEx;
    OdeSolver_WaveEq_Imex     *_quadWaveImex;

    Fault_qd                   *_fault_qd;
    Fault_fd                   *_fault_fd;
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
    PetscErrorCode integrate_qd();
    PetscErrorCode integrate_fd();
    PetscErrorCode integrate_singleQDTimeStep(); // take 1 quasidynamic time step with deltaT = deltaT_fd
    PetscErrorCode initiateIntegrands(); // allocate space for vars, guess steady-state initial conditions
    PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // help with switching between fully dynamic and quasidynamic
    bool checkSwitchRegime(const Fault* _fault);
    PetscErrorCode prepare_qd2fd(); // switch from quasidynamic to fully dynamic
    PetscErrorCode prepare_fd2qd(); // switch from fully dynamic to quasidynamic

    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx); // quasidynamic

    PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, map<string,Vec>& var, map<string,Vec>& varPrev); // fully dynamic

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt); // quasidynamic

    PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev,
      map<string,Vec>& varIm, const map<string,Vec>& varImPrev); // fully dynamic


    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();

    // handles switching between quasidynamic and fully dynamic
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscScalar deltaT,const PetscInt stepCount,int& stopIntegration);

    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

};


#endif
