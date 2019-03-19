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


class StrikeSlip_PowerLaw_qd: public IntegratorContextEx, public IntegratorContextImex
{

private:
  // disable default copy constructor and assignment operator
  StrikeSlip_PowerLaw_qd(const StrikeSlip_PowerLaw_qd &that);
  StrikeSlip_PowerLaw_qd& operator=(const StrikeSlip_PowerLaw_qd &rhs);

public:

  Domain *_D;

  // IO information
  string          _delim; // format is: var delim value (without the white space)
  string          _outputDir; // output data

  // problem properties
  int             _guessSteadyStateICs; // 0 = no, 1 = yes
  const bool      _isMMS; // true if running mms test
  string          _thermalCoupling; // thermomechanical coupling
  string          _grainSizeEvCoupling; // grain size evolution: no, uncoupled, coupled (latter is only relevant if grain-size sensitive flow, such as diffusion creep, is used)
  string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
  string          _stateLaw;
  string          _forcingType; // what body forcing term to include (i.e. iceStream)
  string          _wLinearMaxwell; // if linear Maxwell, do not create a heat equation data member

  PetscScalar     _vL;
  PetscScalar     _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

  // time stepping data
  map <string,Vec>  _varEx; // holds variables for explicit integration in time
  map <string,Vec>  _varIm; // holds variables for implicit integration in time
  string            _timeIntegrator,_timeControlType;
  PetscInt          _stride1D,_stride2D; // stride
  PetscInt          _maxStepCount; // largest number of time steps
  PetscScalar       _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT,_deltaT;
  int               _stepCount;
  PetscScalar       _timeStepTol;
  PetscScalar       _initDeltaT;
  vector<string>    _timeIntInds; // indices of variables to be used in time integration
  vector<double>    _scale; // scale factor for entries in _timeIntInds
  string            _normType;


  // runtime data
  double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime;

  // checkpoint settings
  PetscInt _ckpt, _ckptNumber, _interval;

  // viewers
  PetscViewer _timeV1D,_dtimeV1D,_timeV2D;

  // forcing term for ice stream problem
  Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
  PetscScalar _forcingVal; // body force per unit volume (same in entire domain)


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
  PetscErrorCode constructIceStreamForcingTerm(); // ice stream forcing term

  // estimating steady state conditions
  // viewers:
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  //~ map <string,PetscViewer>  _viewers;
  map <string,pair<PetscViewer,string> >  _viewers;
  map <string,Vec>                             _varSS; // holds variables for steady state iteration
  PetscScalar                                       _fss_T,_fss_EffVisc,_fss_grainSize; // damping coefficients, must be < 1
  PetscScalar                                       _gss_t; // guess steady state strain rate
  PetscInt                 _maxSSIts_effVisc,_maxSSIts_tot,_maxSSIts_timesteps; // max iterations allowed
  PetscScalar              _atolSS_effVisc;

  PetscErrorCode writeSS(const int Ii, const string outputDir);
  PetscErrorCode computeSSEffVisc();
  PetscErrorCode guessTauSS(map<string,Vec>& varSS);
  PetscErrorCode solveSSb();
  PetscErrorCode integrateSS();
  PetscErrorCode solveSS(const PetscInt Jj, const string baseOutDir);
  PetscErrorCode setSSBCs();
  PetscErrorCode solveSSViscoelasticProblem(const PetscInt Jj, const string baseOutDir); // iterate for effective viscosity
  PetscErrorCode solveSStau(const PetscInt Jj, const string outputDir); // brute force for steady-state shear stress on fault
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
  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time, const string outputDir);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time, const string outputDir);

  // debugging and MMS tests
  PetscErrorCode measureMMSError();

};


#endif
