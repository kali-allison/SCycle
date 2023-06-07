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
#include "problemContext.hpp"

#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
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
 *  with linear elastic material properties.
 * Uses the quasi-dynamic approximation.
 */


// StrikeSlip_LinearElastic_qd is a derived class from IntegratorContextEx and IntegratorContextImex
class StrikeSlip_LinearElastic_qd: public IntegratorContextEx, public IntegratorContextImex, public ProblemContext
{
private:
  // disable default copy constructor and assignment operator
  StrikeSlip_LinearElastic_qd(const StrikeSlip_LinearElastic_qd &that);
  StrikeSlip_LinearElastic_qd& operator=(const StrikeSlip_LinearElastic_qd &rhs);

  Domain *_D;

  // IO information
  string       _delim; // format is: var delim value (without the white space)

  // problem properties
  const bool   _isMMS; // true if running mms test
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
  int           _evolveTemperature,_computeSSHeatEq;

  // time stepping data
  map <string,Vec>  _varEx; // holds variables for explicit integration in time
  map <string,Vec>  _varIm; // holds variables for implicit integration in time
  string            _timeIntegrator,_timeControlType;
  PetscInt          _stride1D,_stride2D, _strideChkpt; // # of time steps before writing out results
  PetscInt          _maxStepCount; // largest number of time steps
  PetscScalar       _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT,_deltaT;
  Vec               _time1DVec, _dtime1DVec,_time2DVec, _dtime2DVec; // Vecs to hold current time and time step for output
  int               _stepCount; // number of time steps at which results are written out
  PetscScalar       _timeStepTol;
  PetscScalar       _initDeltaT;
  vector<string>    _timeIntInds;// keys of variables to be used in time integration
  vector<double>    _scale; // scale factor for entries in _timeIntInds
  string            _normType;
  PetscInt          _chkptTimeStep1D, _chkptTimeStep2D;

  Vec               _JjSSVec; // Vec containing current index (Ii) for steady state iteration

  // runtime data
  double _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_totalRunTime, _miscTime;

  // viewers
  PetscViewer _viewer_context,_viewer1D,_viewer2D,_viewerSS,_viewer_chkpt;

  // forcing term for ice stream problem
  Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
  PetscScalar _forcingVal; // body force per unit volume (same in entire domain)


  // boundary conditions
  // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
  string  _bcRType,_bcTType,_bcLType,_bcBType;
  string  _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;
  //~ string  _bcRType_ss,_bcTType_ss,_bcLType_ss,_bcBType_ss; // steady-state type
  //~ string  _bcRType_trans,_bcTType_trans,_bcLType_trans,_bcBType_trans; // steady-state type

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

  // private member functions
  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode parseBCs(); // parse boundary conditions
  PetscErrorCode allocateFields();
  PetscErrorCode computeMinTimeStep(); // compute min allowed time step as dx / cs
  PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term

public:

  OdeSolver        *_quadEx; // explicit time stepping
  OdeSolverImex    *_quadImex; // implicit time stepping

  Fault_qd         *_fault;
  LinearElastic    *_material; // linear elastic off-fault material properties
  HeatEquation     *_he;
  PressureEq       *_p;

  // constructor and destructor
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
  PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);

  // IO functions
  PetscErrorCode view();
  PetscErrorCode writeContext();
  PetscErrorCode timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration);
  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time, PetscScalar deltaT);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time, PetscScalar deltaT);
  PetscErrorCode writeSS(const int Ii);

  // checkpointing functions
  PetscErrorCode loadCheckpoint();
  PetscErrorCode writeCheckpoint();

  // debugging and MMS tests
  PetscErrorCode measureMMSError();
};

#endif
