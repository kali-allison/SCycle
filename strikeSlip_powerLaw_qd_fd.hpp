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

#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "powerLaw.hpp"



/*
 * Mediator-level class for the simulation of earthquake cycles with power-law viscoelastic material properties.
 *
 */


class StrikeSlip_PowerLaw_qd_fd: public IntegratorContextEx, public IntegratorContextImex
{

  private:
    // disable default copy constructor and assignment operator
    StrikeSlip_PowerLaw_qd_fd(const StrikeSlip_PowerLaw_qd_fd &that);
    StrikeSlip_PowerLaw_qd_fd& operator=(const StrikeSlip_PowerLaw_qd_fd &rhs);

  public:

    Domain *_D;

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

    PetscInt             _cycleCount,_maxNumCycles;
    PetscScalar          _deltaT, _deltaT_fd, _CFL; // current time step size, time step for fully dynamic, CFL factor
    Vec                  _ay;
    Vec                  _Fhat;
    Vec                  _alphay;
    bool                 _inDynamic;

    // time stepping data
    std::map <string,Vec>  _varQSEx; // holds variables for explicit integration in time
    std::map <string,Vec>  _varIm; // holds variables for implicit integration in time
    std::string            _timeIntegrator,_timeControlType;
    PetscInt               _stride1D,_stride2D; // current stride
    PetscInt               _stride1D_qd, _stride2D_qd, _stride1D_fd, _stride2D_fd, _stride1D_fd_end, _stride2D_fd_end;
    PetscInt               _maxStepCount; // largest number of time steps
    PetscScalar            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT,_dT;
    int                    _stepCount;
    PetscScalar            _atol;
    PetscScalar            _initDeltaT;
    std::vector<string>    _timeIntInds; // indices of variables to be used in time integration
    std::string            _normType;


    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime,_startIntegrateTime;

    // viewers
    PetscViewer      _timeV1D,_dtimeV1D,_timeV2D;


    // boundary conditions
    // Options: freeSurface, tau, outgoingCharacteristics, remoteLoading, symm_fault, rigid_fault
    std::string              _bcRType,_bcTType,_bcLType,_bcBType;
    std::string              _mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType;

    OdeSolver               *_quadEx; // explicit time stepping
    OdeSolverImex           *_quadImex; // implicit time stepping

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

};


#endif
