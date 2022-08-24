#ifndef STRIKESLIP_LINEARELASTIC_FD_H_INCLUDED
#define STRIKESLIP_LINEARELASTIC_FD_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>

#include "problemContext.hpp"
#include "integratorContext_WaveEq.hpp"

#include "odeSolver_WaveEq.hpp"
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
 * Mediator-level class for the simulation of earthquake a single fully
 * dynamic earthquake with linear elastic off-fault material properties.
 */


class StrikeSlip_LinearElastic_fd: public IntegratorContext_WaveEq, public ProblemContext
{
private:

  // disable default copy constructor and assignment operator
  StrikeSlip_LinearElastic_fd(const StrikeSlip_LinearElastic_fd &that);
  StrikeSlip_LinearElastic_fd& operator=(const StrikeSlip_LinearElastic_fd &rhs);

  Domain *_D;
  // IO information
  string       _delim; // format is: var delim value (without the white space)

  // problem properties
  const bool      _isMMS; // true if running mms test

  const PetscInt  _order,_Ny,_Nz;
  PetscScalar     _Ly,_Lz;
  PetscScalar     _deltaT, _CFL;
  Vec            *_y,*_z; // to handle variable grid spacing

  Vec             _mu, _rho, _cs, _ay;
  Vec             _alphay;
  string          _inputDir;
  string          _outputDir; // output data
  PetscScalar     _vL;
  string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
  string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
  string          _initialConditions;
  int             _guessSteadyStateICs; // 0 = no, 1 = yes
  PetscScalar     _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

  // time stepping data
  map <string,Vec>  _var,_varPrev; // holds variables for time step: n (current), n-1
  string            _timeIntegrator,_timeControlType;
  PetscInt          _maxStepCount; // largest number of time steps
  PetscInt          _stride1D,_stride2D; // stride
  PetscScalar       _initTime,_currTime,_maxTime;
  Vec               _time1DVec, _dtime1DVec,_time2DVec, _dtime2DVec; // Vecs to hold current time and time step for output
  int               _stepCount;
  vector<string>    _timeIntInds;// keys of variables to be used in time integration

  // viewers
  PetscViewer _viewer_context,_viewer1D,_viewer2D;

  // runtime data
  double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _propagateTime;

  // boundary conditions
  // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
  string       _bcRType,_bcTType,_bcLType,_bcBType;
  string       _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode computeTimeStep();
  PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

public:

  OdeSolver_WaveEq      *_quadWaveEx;
  Fault_fd              *_fault;
  LinearElastic         *_material; // linear elastic off-fault material properties


  StrikeSlip_LinearElastic_fd(Domain&D);
  ~StrikeSlip_LinearElastic_fd();

  // time stepping functions
  PetscErrorCode integrate(); // will call OdeSolver method by same name
  PetscErrorCode initiateIntegrand();
  PetscErrorCode propagateWaves(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

  // explicit time-stepping methods
  PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

  // IO functions
  PetscErrorCode view();
  PetscErrorCode writeContext();
  PetscErrorCode timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration);

  PetscErrorCode writeStep1D(PetscInt stepCount, PetscScalar time,const string outputDir);
  PetscErrorCode writeStep2D(PetscInt stepCount, PetscScalar time,const string outputDir);


};


#endif
