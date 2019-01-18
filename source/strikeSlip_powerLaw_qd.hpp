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
    std::string          _delim; // format is: var delim value (without the white space)

    // problem properties
    const bool           _isMMS; // true if running mms test
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
    std::string          _viscosityType; // options: power-law, linearMaxwell
    // if linear Maxwell, do not create a heat equation data member

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT,_deltaT;
    int                    _stepCount;
    PetscScalar            _timeStepTol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration
    std::vector<double>    _scale; // scale factor for entries in _timeIntInds
    std::string            _normType;


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime;

    // viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D;

    // forcing term for ice stream problem
    Vec _forcingTerm, _forcingTermPlain; // body forcing term, copy of body forcing term for output
    PetscScalar _forcingVal; // body force per unit volume (same in entire domain)


    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symmFault, rigidFault
    std::string              _bcRType,_bcTType,_bcLType,_bcBType;
    std::string              _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

    // for mapping from body fields to the fault
    VecScatter* _body2fault;

    OdeSolver              *_quadEx; // explicit time stepping
    OdeSolverImex          *_quadImex; // implicit time stepping

    Fault_qd               *_fault;
    PowerLaw               *_material; // power-law viscoelastic off-fault material properties
    HeatEquation           *_he;
    PressureEq             *_p;


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
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;
    std::map <string,Vec>                             _varSS; // holds variables for steady state iteration
    PetscScalar                                       _fss_T,_fss_EffVisc; // damping coefficients, must be < 1
    PetscScalar                                       _gss_t; // guess steady state strain rate
    PetscInt                 _maxSSIts_effVisc,_maxSSIts_tau,_maxSSIts_timesteps; // max iterations allowed
    PetscScalar              _atolSS_effVisc;

    PetscErrorCode writeSS(const int Ii, const std::string outputDir);
    PetscErrorCode computeSSEffVisc();
    PetscErrorCode guessTauSS(map<string,Vec>& varSS);
    PetscErrorCode solveSSb();
    PetscErrorCode integrateSS();
    PetscErrorCode solveSS();
    PetscErrorCode setSSBCs();
    PetscErrorCode solveSSViscoelasticProblem();


    // time stepping functions
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode initiateIntegrand();
    PetscErrorCode solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // explicit time-stepping methods
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);


    // IO functions
    PetscErrorCode view();
    PetscErrorCode writeContext();
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscScalar deltaT,
      const PetscInt stepCount, int& stopIntegration);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

    // debugging and MMS tests
    PetscErrorCode measureMMSError();

};


#endif
