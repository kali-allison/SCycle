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
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "newFault.hpp"
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

    // time stepping data
    std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // stride
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;


    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symm_fault, rigid_fault
    std::string              _bcRType,_bcTType,_bcLType,_bcBType;
    std::string              _mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType;

    OdeSolver           *_quadEx; // explicit time stepping
    OdeSolverImex       *_quadImex; // implicit time stepping

    //~ Fault                      *_fault;
    NewFault_qd                *_fault;
    PowerLaw                   *_material; // power-law viscoelastic off-fault material properties
    HeatEquation               *_he;
    PressureEq                 *_p;


    StrikeSlip_PowerLaw_qd(Domain&D);
    ~StrikeSlip_PowerLaw_qd();

    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();

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
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration);

    // debugging and MMS tests
    PetscErrorCode measureMMSError();

};


#endif
