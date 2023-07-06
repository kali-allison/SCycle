#ifndef STRIKESLIP_POWERLAW_QD_H_INCLUDED
#define STRIKESLIP_POWERLAW_QD_H_INCLUDED

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
#include "powerLaw.hpp"
#include "grainSizeEvolution.hpp"

using namespace std;

/*
 * Mediator-level class for the simulation of earthquake cycles with power-law viscoelastic material properties.
 * Uses the quasi-dynamic approximation.
 */


class StrikeSlip_PowerLaw_qd: public IntegratorContextEx, public IntegratorContextImex, public ProblemContext
{

private:
  // disable default copy constructor and assignment operator
  StrikeSlip_PowerLaw_qd(const StrikeSlip_PowerLaw_qd &that);
  StrikeSlip_PowerLaw_qd& operator=(const StrikeSlip_PowerLaw_qd &rhs);

public:

  Domain *_D;

  // IO information
  string          _delim; // format is: var delim value (without the white space)
  string          _inputDir;
  string          _outputDir; // output data

  // problem properties
  int             _guessSteadyStateICs; // 0 = no, 1 = yes
  const bool      _isMMS; // true if running mms test
  string          _thermalCoupling; // thermomechanical coupling
  string          _grainSizeEvCoupling; // grain size evolution: no, uncoupled, coupled
  string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
  string          _stateLaw;
  string          _forcingType; // what body forcing term to include (i.e. iceStream)
  int             _evolveTemperature,_evolveGrainSize;
  int             _computeSSTemperature,_computeSSGrainSize;

  PetscScalar     _vL;
  PetscScalar     _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

  // time stepping data
  map <string,Vec>  _varEx; // holds variables for explicit integration in time
  map <string,Vec>  _varIm; // holds variables for implicit integration in time
  string            _timeIntegrator,_timeControlType;
  PetscInt          _stride1D,_stride2D, _strideChkpt; // stride
  PetscInt          _maxStepCount; // largest number of time steps
  PetscScalar       _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT,_deltaT;
  Vec               _time1DVec, _dtime1DVec,_time2DVec, _dtime2DVec; // Vecs to hold current time and time step for output
  int               _stepCount;
  PetscScalar       _timeStepTol;
  PetscScalar       _initDeltaT;
  vector<string>    _timeIntInds; // indices of variables to be used in time integration
  vector<double>    _scale; // scale factor for entries in _timeIntInds
  string            _normType;
  PetscInt          _chkptTimeStep1D, _chkptTimeStep2D;

  // estimating steady state conditions
  map <string,Vec>                       _varSS; // holds variables for steady state iteration
  Vec                                    _JjSSVec; // Vec containing current index (Ii) for steady state iteration
  PetscScalar                            _fss_T,_fss_EffVisc,_fss_grainSize; // damping coefficients, must be < 1
  PetscScalar                            _gss_t; // guess steady state strain rate
  PetscInt                               _SS_index,_maxSSIts_effVisc,_maxSSIts_tot; // max iterations allowed
  PetscScalar                            _atolSS_effVisc;
  PetscScalar                            _maxSSIts_time; // (s) max time during time integration phase


  // runtime data
  double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime;

  // viewers
  PetscViewer _viewer_context,_viewer1D,_viewer2D,_viewerSS,_viewer_chkpt;

  // forcing term for ice stream problem
  Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
  PetscScalar _forcingVal; // body force per unit volume (same in entire domain)

  // parameters for forced displacement top boundary condition
  // u(z=0) = (2/pi)*vL / bcT_A * atan(y / (2*pi*bcT_L) )
  PetscScalar _bcT_L; // (m/s) amplitude, (km) length scale


  // boundary conditions
  // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
  string              _bcRType,_bcTType,_bcLType,_bcBType;
  string              _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

  OdeSolver              *_quadEx; // explicit time stepping
  OdeSolverImex          *_quadImex; // implicit time stepping

  Fault_qd               *_fault;
  PowerLaw               *_material; // power-law viscoelastic off-fault material properties
  HeatEquation           *_he;
  PressureEq             *_p;
  GrainSizeEvolution     *_grainDist;


  StrikeSlip_PowerLaw_qd(Domain&D);
  ~StrikeSlip_PowerLaw_qd();

  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode parseBCs(); // parse boundary conditions
  PetscErrorCode allocateFields();
  PetscErrorCode computeMinTimeStep(); // compute min allowed time step as dx / cs
  PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term
  PetscErrorCode updateBCT_atan_v(); // bcT = atan term (velocity)
  PetscErrorCode updateBCT_atan_u(const PetscScalar time); // bcT = atan term (displacement)


  // steady-state iteration methods
  PetscErrorCode writeViscLoopSS(const int Ii); // for testing fixed point iteration method
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
  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time);
  PetscErrorCode writeCheckpoint();
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();

  // debugging and MMS tests
  PetscErrorCode measureMMSError();

};


#endif
