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
#include "linearElastic.hpp"

using namespace std;

/*
 * Mediator-level class for the simulation of earthquake cycles on a vertical strike-slip fault
 * with linear elastic material properties.
 * Uses the quasi-dynamic approximation.
 */


class StrikeSlip_LinearElastic_qd_fd: public IntegratorContextEx, public IntegratorContextImex, public IntegratorContext_WaveEq, public IntegratorContext_WaveEq_Imex, public ProblemContext
{
private:
  // disable default copy constructor and assignment operator
  StrikeSlip_LinearElastic_qd_fd(const StrikeSlip_LinearElastic_qd_fd &that);
  StrikeSlip_LinearElastic_qd_fd& operator=(const StrikeSlip_LinearElastic_qd_fd &rhs);

  Domain *_D;

  // IO information
  string       _delim; // format is: var delim value (without the white space)

  // problem properties
  string       _inputDir;
  string       _outputDir; // output data
  PetscScalar  _vL;
  string       _thermalCoupling,_heatEquationType; // thermomechanical coupling
  string       _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
  string       _stateLaw;
  int          _guessSteadyStateICs; // 0 = no, 1 = yes
  int          _computeSSMomBal; // 0 = no, 1 = yes, for momentum balance equation only
  string       _forcingType; // what body forcing term to include (i.e. iceStream)
  PetscScalar  _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid
  int          _evolveTemperature,_computeSSHeatEq;

  PetscInt     _cycleCount,_maxNumCycles,_phaseCount;
  PetscScalar  _deltaT, _deltaT_fd, _CFL; // current time step size, time step for fully dynamic, CFL factor
  Vec         *_y,*_z;
  Vec          _Req; // Req = eta_rad * slipVel / tauQSP, measure of magnitude of radiation damping term
  Vec          _ay;
  Vec          _alphay;
  bool         _inDynamic,_allowed;
  PetscScalar  _trigger_qd2fd, _trigger_fd2qd, _limit_qd, _limit_fd, _limit_stride_fd;

  // time stepping data
  map <string,Vec>  _varFD,_varFDPrev; // holds variables for time step: n+1, n (current), n-1
  map <string,Vec>  _varQSEx; // holds variables for explicit integration in time
  map <string,Vec>  _varIm; // holds variables for implicit integration in time
  Vec               _u0; // total displacement at start of fd
  string            _timeIntegrator,_timeControlType;
  PetscInt          _stride1D,_stride2D, _strideChkpt_qd, _strideChkpt_fd; // stride
  PetscInt          _stride1D_qd, _stride2D_qd, _stride1D_fd, _stride2D_fd, _stride1D_fd_end, _stride2D_fd_end;
  PetscInt          _maxStepCount; // largest number of time steps
  PetscScalar       _initTime,_currTime,_minDeltaT,_maxDeltaT, _maxTime;

  int               _stepCount;
  PetscScalar       _timeStepTol;
  PetscScalar       _initDeltaT;
  vector<string>    _timeIntInds;// keys of variables to be used in time integration
  vector<double>    _scale; // scale factor for entries in _timeIntInds
  string            _normType;
  PetscInt          _chkptTimeStep1D, _chkptTimeStep2D;

  Vec               _JjSSVec; // Vec containing current index (Ii) for steady state iteration


  // Vecs and viewers for output
  Vec               _time1DVec, _dtime1DVec,_time2DVec, _dtime2DVec, _regime1DVec, _regime2DVec; // Vecs to hold current time and time step for output
  PetscViewer       _viewer_context, _viewer1D, _viewer2D,_viewerSS,_viewer_chkpt;

  // runtime data
  double _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _propagateTime, _dynTime, _qdTime;

  // forcing term for ice stream problem
  Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
  PetscScalar _forcingVal; // body force per unit volume (same in entire domain)

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

  // boundary conditions
  // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
  string     _qd_bcRType,_qd_bcTType,_qd_bcLType,_qd_bcBType;
  string     _fd_bcRType,_fd_bcTType,_fd_bcLType,_fd_bcBType;
  string     _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;
  string     _mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType;

  map <string,pair<PetscViewer,string> >  _viewers;

  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode parseBCs();
  PetscErrorCode allocateFields();
  PetscErrorCode computeTimeStep();
  PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz
  PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term

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


  StrikeSlip_LinearElastic_qd_fd(Domain&D);
  ~StrikeSlip_LinearElastic_qd_fd();

  // estimating steady state conditions
  PetscErrorCode solveSS();
  PetscErrorCode solveSSb();


  // time stepping functions
  PetscErrorCode integrate(); // will call OdeSolver method by same name
  PetscErrorCode integrate_qd(int isFirstPhase);
  PetscErrorCode integrate_fd(int isFirstPhase);
  PetscErrorCode integrate_singleQDTimeStep(); // take 1 quasidynamic time step with deltaT = deltaT_fd
  PetscErrorCode initiateIntegrand(); // allocate space for vars, guess steady-state initial conditions
  PetscErrorCode initiateIntegrand_qd(); // allocate space for varQDEx and varIm, guess steady-state initial conditions
  PetscErrorCode initiateIntegrand_fd(); // allocate space for varFD
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
  PetscErrorCode timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount,int& stopIntegration);
  PetscErrorCode writeContext();
  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode writeSS(const int Ii);
  PetscErrorCode loadCheckpoint();
  PetscErrorCode writeCheckpoint();

};


#endif
