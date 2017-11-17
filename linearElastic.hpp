#ifndef LINEARELASTIC_H_INCLUDED
#define LINEARELASTIC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include <map>

#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "momBalContext.hpp"

#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "heatEquation.hpp"



// Class for a linear elastic material
class LinearElastic : public MomBalContext
//~ class LinearElastic
{
  private:
    // disable default copy constructor and assignment operator
    //~ LinearElastic(const LinearElastic &that);
    //~ LinearElastic& operator=(const LinearElastic &rhs);

    // initialize data members
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadFieldsFromFiles();
    PetscErrorCode setInitialSlip(Vec& out);
    PetscErrorCode setUpSBPContext(Domain& D);
    PetscErrorCode setupKSP(SbpOps* sbp,KSP& ksp,PC& pc);

    PetscErrorCode computeShearStress();
    PetscErrorCode setSurfDisp();

    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSBoundaryConditions(const double time);

  public:

    // domain properties
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // output data
    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz,_dy,_dz;
    Vec                  *_y,*_z; // to handle variable grid spacing
    const bool           _isMMS; // true if running mms test
    const bool           _loadICs; // true if starting from a previous simulation
    std::string          _momBalType; // "dynamic", "static"
    bool                 _bcLTauQS; // true if left boundary is traction
    PetscScalar          _currTime;
    PetscInt             _stepCount;
    PetscScalar          _vL; // loading velocity

    // off-fault material fields
    Vec                  _muVec, _rhoVec, _cs, _ay;
    PetscScalar          _muVal,_rhoVal; // if constant
    Vec                  _bcRShift,_surfDisp;
    Vec                  _rhs,_u,_sxy,_sxz;

    // linear system data
    std::string          _linSolver;
    KSP                  _ksp;
    PC                   _pc;
    PetscScalar          _kspTol;
    SbpOps              *_sbp;
    std::string          _sbpType;

    // thermomechanical coupling
    std::string   _thermalCoupling;

    // viewers
    PetscViewer      _timeV1D,_timeV2D;
    std::map <string,PetscViewer>  _viewers;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
    PetscInt     _linSolveCount;

  //~ public:

    // boundary conditions
    string               _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction
    Vec                  _bcT,_bcR,_bcB,_bcL;

    LinearElastic(Domain&D,HeatEquation& he);
    ~LinearElastic();

    // for steady state computations
    PetscErrorCode getTauVisc(Vec& tauVisc, const PetscScalar ess_t); // compute initial tauVisc
    PetscErrorCode getRhoVec(Vec& rho); // compute initial tauVisc
    PetscErrorCode updateSSa(map<string,Vec>& varSS); // update v, viscous strain rates, viscosity
    PetscErrorCode updateSSb(map<string,Vec>& varSS); // does nothing for the linear elastic equations
    PetscErrorCode initiateVarSS(map<string,Vec>& varSS); // put viscous strains etc in varSS

    // time stepping function
    PetscErrorCode initiateIntegrand_qs(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode initiateIntegrand_dyn(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode updateTemperature(const Vec& T);
    PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep);
    PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscReal _deltaT);
    PetscErrorCode d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt); // IMEX backward Euler

    PetscErrorCode measureMMSError(const PetscScalar time);

    // IO commands
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeContext();
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time); // write out 1D fields
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time); // write out 2D fields


    // MMS functions
    static double zzmms_f(const double y,const double z);
    static double zzmms_f_y(const double y,const double z);
    static double zzmms_f_yy(const double y,const double z);
    static double zzmms_f_z(const double y,const double z);
    static double zzmms_f_zz(const double y,const double z);
    static double zzmms_g(const double t);
    static double zzmms_g_t(const double t);

    static double zzmms_mu(const double y,const double z);
    static double zzmms_mu_y(const double y,const double z);
    static double zzmms_mu_z(const double y,const double z);

    static double zzmms_uA(const double y,const double z,const double t);
    static double zzmms_uA_y(const double y,const double z,const double t);
    static double zzmms_uA_yy(const double y,const double z,const double t);
    static double zzmms_uA_z(const double y,const double z,const double t);
    static double zzmms_uA_zz(const double y,const double z,const double t);
    static double zzmms_uA_t(const double y,const double z,const double t);

    static double zzmms_sigmaxy(const double y,const double z,const double t);
    static double zzmms_uSource(const double y,const double z,const double t);


    // 1D
    static double zzmms_f1D(const double y);// helper function for uA
    static double zzmms_f_y1D(const double y);
    static double zzmms_f_yy1D(const double y);

    static double zzmms_uA1D(const double y,const double t);
    static double zzmms_uA_y1D(const double y,const double t);
    static double zzmms_uA_yy1D(const double y,const double t);
    static double zzmms_uA_z1D(const double y,const double t);
    static double zzmms_uA_zz1D(const double y,const double t);
    static double zzmms_uA_t1D(const double y,const double t);

    static double zzmms_mu1D(const double y);
    static double zzmms_mu_y1D(const double y);
    static double zzmms_mu_z1D(const double z);

    static double zzmms_sigmaxy1D(const double y,const double t);
    static double zzmms_uSource1D(const double y,const double t);
};


#endif
