#ifndef STRIKESLIP_LINEARELASTIC_FD_H_INCLUDED
#define STRIKESLIP_LINEARELASTIC_FD_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>

#include "integratorContext_WaveEq.hpp"
#include "odeSolver_WaveEq.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "linearElastic.hpp"



/*
 * Mediator-level class for the simulation of earthquake a single fully
 * dynamic earthquake with linear elastic off-fault material properties.
 */


class strikeSlip_linearElastic_fd: public IntegratorContext_WaveEq
{
  private:

    // disable default copy constructor and assignment operator
    strikeSlip_linearElastic_fd(const strikeSlip_linearElastic_fd &that);
    strikeSlip_linearElastic_fd& operator=(const strikeSlip_linearElastic_fd &rhs);

    Domain *_D;
    // IO information
    std::string       _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool           _isMMS; // true if running mms test

    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz;
    PetscScalar          _deltaT, _CFL;
    Vec                 *_y,*_z; // to handle variable grid spacing

    Vec                  _muVec, _rhoVec, _cs, _ay;
    Vec                  _alphay;
    std::string          _outputDir; // output data
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _vL;
    std::string          _thermalCoupling,_heatEquationType; // thermomechanical coupling
    std::string          _hydraulicCoupling,_hydraulicTimeIntType; // coupling to hydraulic fault
    std::string          _initialConditions, _inputDir;
    int                  _guessSteadyStateICs; // 0 = no, 1 = yes
    PetscScalar          _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid

    // time stepping data
    std::map <string,Vec>  _var,_varPrev; // holds variables for time step: n (current), n-1
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _maxStepCount; // largest number of time steps
    PetscInt               _stride1D,_stride2D; // stride
    PetscScalar            _initTime,_currTime,_maxTime;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _yCenterU, _zCenterU, _yStdU, _zStdU, _ampU;
    std::vector<string>    _timeIntInds;// keys of variables to be used in time integration

    //viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _propagateTime;

    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
    string              _bcRType,_bcTType,_bcLType,_bcBType;
    string              _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode computeTimeStep();
    PetscErrorCode computePenaltyVectors(); // computes alphay and alphaz

    // for mapping from body fields to the fault
    VecScatter* _body2fault;

  public:

    OdeSolver_WaveEq          *_quadWaveEx;
    Fault_fd                   *_fault;
    LinearElastic              *_material; // linear elastic off-fault material properties


    strikeSlip_linearElastic_fd(Domain&D);
    ~strikeSlip_linearElastic_fd();

    // time stepping functions
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode initiateIntegrand();
    PetscErrorCode propagateWaves(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);

    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscScalar deltaT,
      const PetscInt stepCount, int& stopIntegration);

    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);


};


#endif
