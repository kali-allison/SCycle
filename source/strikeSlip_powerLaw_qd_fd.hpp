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
#include "problemContext.hpp"

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
#include "grainSizeEvolution.hpp"

using namespace std;

/*
 * Mediator-level class for the simulation of earthquake cycles with power-law viscoelastic material properties.
 *
 */


class StrikeSlip_PowerLaw_qd_fd: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContext_WaveEq, public ProblemContext
{

private:
  // disable default copy constructor and assignment operator
  StrikeSlip_PowerLaw_qd_fd(const StrikeSlip_PowerLaw_qd_fd &that);
  StrikeSlip_PowerLaw_qd_fd& operator=(const StrikeSlip_PowerLaw_qd_fd &rhs);

public:

  Domain   *_D;
  Vec      *_y,*_z;

  // IO information
  string          _delim; // format is: var delim value (without the white space)

  // problem properties
  string          _inputDir;
  string          _outputDir; // output data
  PetscScalar     _vL;
  string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
  string          _grainSizeEvCoupling,_grainSizeEvCouplingSS; // grain size evolution: no, uncoupled, coupled
  string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
  string          _stateLaw;
  int             _guessSteadyStateICs; // 0 = no, 1 = yes
  string          _forcingType; // what body forcing term to include (i.e. iceStream)
  PetscScalar     _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid
  int             _evolveTemperature,_evolveGrainSize;
  int             _computeSSTemperature,_computeSSGrainSize;

  PetscInt        _cycleCount,_maxNumCycles,_phaseCount;
  PetscScalar     _deltaT, _deltaT_fd, _CFL; // current time step size, time step for fully dynamic, CFL factor
  Vec             _ay;
  Vec             _Fhat;
  Vec             _alphay;
  bool            _inDynamic,_allowed;
  PetscScalar     _trigger_qd2fd, _trigger_fd2qd, _limit_qd, _limit_fd, _limit_stride_fd;

  // estimating steady state conditions
  map <string,Vec>   _varSS; // holds variables for steady state iteration
  Vec                _JjSSVec; // Vec containing current index (Ii) for steady state iteration
  PetscScalar        _fss_T,_fss_EffVisc,_fss_grainSize; // damping coefficients, must be < 1
  PetscScalar        _gss_t; // guess steady state strain rate
  PetscInt           _SS_index,_maxSSIts_effVisc,_maxSSIts_tot; // max iterations allowed
  PetscScalar        _atolSS_effVisc;

  // time stepping data
  map <string,Vec>  _varFD,_varFDPrev; // holds variables for time step: n+1, n (current), n-1
  map <string,Vec>  _varQSEx; // holds variables for explicit integration in time
  map <string,Vec>  _varIm; // holds variables for implicit integration in time
  Vec               _u0; // total displacement at start of fd
  string            _timeIntegrator,_timeControlType;
  PetscInt          _stride1D,_stride2D,_strideChkpt_qd,_strideChkpt_fd; // current stride
  PetscInt          _stride1D_qd, _stride2D_qd, _stride1D_fd, _stride2D_fd, _stride1D_fd_end, _stride2D_fd_end;
  PetscInt          _maxStepCount; // largest number of time steps
  PetscScalar       _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
  int               _stepCount;
  PetscScalar       _timeStepTol;
  PetscScalar       _initDeltaT;
  vector<string>    _timeIntInds; // indices of variables to be used in time integration
  vector<double>    _scale; // scale factor for entries in _timeIntInds
  string            _normType;
  PetscInt          _chkptTimeStep1D, _chkptTimeStep2D;

  // runtime data
  double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime, _propagateTime, _dynTime, _qdTime;

  // Vecs and viewers for output
  Vec               _time1DVec, _dtime1DVec,_time2DVec, _dtime2DVec, _regime1DVec, _regime2DVec; // Vecs to hold current time and time step for output
  PetscViewer       _viewer_context, _viewer1D, _viewer2D,_viewerSS,_viewer_chkpt;

  // forcing term for ice stream problem
  Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
  PetscScalar _forcingVal; // body force per unit volume (same in entire domain)


  // boundary conditions
  // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
  string     _qd_bcRType,_qd_bcTType,_qd_bcLType,_qd_bcBType;
  string     _fd_bcRType,_fd_bcTType,_fd_bcLType,_fd_bcBType;
  string     _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;
  string     _mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType;

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

  OdeSolver                 *_quadEx_qd; // explicit time stepping
  OdeSolverImex             *_quadImex_qd; // implicit time stepping
  OdeSolver_WaveEq          *_quadWaveEx; // explicit, constant time step, time stepping
  Fault_qd                  *_fault_qd;
  Fault_fd                  *_fault_fd;
  PowerLaw                  *_material; // power-law viscoelastic off-fault material properties
  HeatEquation              *_he;
  PressureEq                *_p;
  GrainSizeEvolution        *_grainDist;


  StrikeSlip_PowerLaw_qd_fd(Domain&D);
  ~StrikeSlip_PowerLaw_qd_fd();

  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode parseBCs(); // parse boundary conditions
  PetscErrorCode allocateFields();
  PetscErrorCode computeTimeStep();
  PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz
  PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term

  // steady-state iteration methods
  PetscErrorCode writeSS(const int Ii);
  PetscErrorCode guessTauSS(map<string,Vec>& varSS);
  PetscErrorCode solveSSb();
  PetscErrorCode integrateSS();
  PetscErrorCode initiateIntegrandSS();
  PetscErrorCode solveSS(const PetscInt Jj);
  PetscErrorCode setSSBCs();
  PetscErrorCode solveSSViscoelasticProblem(const PetscInt Jj); // iterate for effective viscosity
  PetscErrorCode solveSStau(const PetscInt Jj); // brute force for steady-state shear stress on fault
  PetscErrorCode solveSSHeatEquation(const PetscInt Jj); // brute force for steady-state temperature
  PetscErrorCode solveSSGrainSize(const PetscInt Jj); // solve for steady-state grain size distribution


  // time integration functions
  PetscErrorCode integrate(); // will call OdeSolver method by same name
  PetscErrorCode integrate_qd(int isFirstPhase);
  PetscErrorCode integrate_fd(int isFirstPhase);
  PetscErrorCode integrate_singleQDTimeStep(); // take 1 quasidynamic time step with deltaT = deltaT_fd
  PetscErrorCode initiateIntegrand(); // allocate space for vars, guess steady-state initial conditions
  PetscErrorCode initiateIntegrand_qd(); // allocate space for vars, guess steady-state initial conditions
  PetscErrorCode initiateIntegrand_fd(); // allocate space for vars
  PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
  PetscErrorCode propagateWaves(const PetscScalar time, const PetscScalar deltaT,
        map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

  // help with switching between fully dynamic and quasidynamic
  bool checkSwitchRegime(const Fault* _fault);
  PetscErrorCode prepare_qd2fd(); // switch from quasidynamic to fully dynamic
  PetscErrorCode prepare_fd2qd(); // switch from fully dynamic to quasidynamic

  // explicit time-stepping methods
  PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx); // quasidynamic

  PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev); // fully dynamic

  // methods for implicit/explicit time stepping
  PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt); // quasidynamic

  PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev, map<string,Vec>& varIm, const map<string,Vec>& varImPrev); // fully dynamic


  // IO functions
  PetscErrorCode view();
  PetscErrorCode writeContext();
  PetscErrorCode timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration);
  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode loadCheckpoint();
  PetscErrorCode writeCheckpoint();
  PetscErrorCode loadCheckpointSS();
  PetscErrorCode writeCheckpointSS();

};


#endif
