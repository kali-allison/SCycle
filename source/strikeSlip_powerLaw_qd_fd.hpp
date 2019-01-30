#ifndef STRIKESLIP_POWERLAW_QD_FD_H_INCLUDED
#define STRIKESLIP_POWERLAW_QD_FD_H_INCLUDED

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
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"
#include "fault.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "powerLaw.hpp"



/*
 * Mediator-level class for the simulation of earthquake cycles with power-law viscoelastic material properties.
 *
 */


class StrikeSlip_PowerLaw_qd_fd: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContext_WaveEq
{

  private:
    // disable default copy constructor and assignment operator
    StrikeSlip_PowerLaw_qd_fd(const StrikeSlip_PowerLaw_qd_fd &that);
    StrikeSlip_PowerLaw_qd_fd& operator=(const StrikeSlip_PowerLaw_qd_fd &rhs);

  public:

    Domain   *_D;
    Vec      *_y,*_z;

    // IO information
    std::string          _delim; // format is: var delim value (without the white space)

    // problem properties
    std::string          _outputDir; // output data
    std::string          _inputDir; // input data
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _vL;
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
    std::string          _stateLaw;
    int                  _guessSteadyStateICs; // 0 = no, 1 = yes
    std::string          _forcingType; // what body forcing term to include (i.e. iceStream)
    PetscScalar          _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

    PetscInt             _cycleCount,_maxNumCycles;
    PetscScalar          _deltaT, _deltaT_fd, _CFL; // current time step size, time step for fully dynamic, CFL factor
    Vec                  _ay;
    Vec                  _Fhat;
    Vec                  _alphay;
    bool                 _inDynamic,_allowed;
    PetscScalar          _trigger_qd2fd, _trigger_fd2qd, _limit_qd, _limit_fd, _limit_stride_fd;

    // time stepping data
    std::map <string,Vec>  _varFD,_varFDPrev; // holds variables for time step: n+1, n (current), n-1
    std::map <string,Vec>  _varQSEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    Vec                    _u0; // total displacement at start of fd
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // current stride
    PetscInt               _stride1D_qd, _stride2D_qd, _stride1D_fd, _stride2D_fd, _stride1D_fd_end, _stride2D_fd_end;
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                    _stepCount;
    PetscScalar            _timeStepTol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration
    std::vector<double>    _scale; // scale factor for entries in _timeIntInds
    std::string            _normType;


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime, _propagateTime, _dynTime, _qdTime;

    // viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D,_regime1DV,_regime2DV; // regime = 1 if fd, 0 if qd

    // forcing term for ice stream problem
    Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
    PetscScalar _forcingVal; // body force per unit volume (same in entire domain)


    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
    string              _qd_bcRType,_qd_bcTType,_qd_bcLType,_qd_bcBType;
    string              _fd_bcRType,_fd_bcTType,_fd_bcLType,_fd_bcBType;
    string              _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;
    string              _mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType;

    // for mapping from body fields to the fault
    VecScatter* _body2fault;

    OdeSolver                  *_quadEx; // explicit adaptive time stepping
    OdeSolverImex              *_quadImex; // IMEX adaptive time stepping
    OdeSolver_WaveEq           *_quadWaveEx; // explicit, constant time step, time stepping
    OdeSolver_WaveEq_Imex      *_quadWaveImex; // IMEX, constant time step, time stepping

    Fault_qd                   *_fault_qd;
    Fault_fd                   *_fault_fd;
    PowerLaw                   *_material; // power-law viscoelastic off-fault material properties
    HeatEquation               *_he;
    PressureEq                 *_p;


    StrikeSlip_PowerLaw_qd_fd(Domain&D);
    ~StrikeSlip_PowerLaw_qd_fd();

    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode parseBCs(); // parse boundary conditions
    PetscErrorCode computeTimeStep();
    PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz
    PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term

    // estimating steady state conditions
    // viewers:
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;
    std::map <string,Vec>                             _varSS; // holds variables for steady state iteration
    PetscScalar                                       _fss_T,_fss_EffVisc; // damping coefficients, must be < 1
    PetscScalar                                       _gss_t; // guess steady state strain rate
    PetscInt                 _maxSSIts_effVisc,_maxSSIts_tau,_maxSSIts_timesteps; // max iterations allowed
    PetscScalar              _atolSS_effVisc;


    // time integration functions
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode integrate_qd();
    PetscErrorCode integrate_fd();
    PetscErrorCode integrate_singleQDTimeStep(); // take 1 quasidynamic time step with deltaT = deltaT_fd
    PetscErrorCode initiateIntegrands(); // allocate space for vars, guess steady-state initial conditions
    PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode propagateWaves(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

    // help with switching between fully dynamic and quasidynamic
    bool checkSwitchRegime(const Fault* _fault);
    PetscErrorCode prepare_qd2fd(); // switch from quasidynamic to fully dynamic
    PetscErrorCode prepare_fd2qd(); // switch from fully dynamic to quasidynamic


    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx); // quasidynamic

    PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev); // fully dynamic

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt); // quasidynamic

    PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev,
      map<string,Vec>& varIm, const map<string,Vec>& varImPrev); // fully dynamic


    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscScalar deltaT,
      const PetscInt stepCount, int& stopIntegration);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

};


#endif
