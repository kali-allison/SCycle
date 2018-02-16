#ifndef HEATEQUATION_H_INCLUDED
#define HEATEQUATION_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "odeSolver.hpp"
#include "odeSolverImex.hpp"



/* Base class for the heat equation
 */
class HeatEquation
{
  private:
    // disable default copy constructor and assignment operator
    HeatEquation(const HeatEquation &that);
    HeatEquation& operator=(const HeatEquation &rhs);

  public:

    // domain dimensions etc
    Domain *_D;
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;
    Vec                 *_y,*_z; // to handle variable grid spacing
    std::string          _heatEquationType;
    int                  _isMMS;
    PetscScalar          _initTime,_initDeltaT;

    // IO information
    const char       *_file; // input ASCII file location
    std::string       _outputDir; // output file location
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _inputDir; // directory to load fields from

    // material parameters
    std::string       _kFile,_rhoFile,_hFile,_cFile; // names of each file within loadFromFile
    std::vector<double>  _rhoVals,_rhoDepths,_kVals,_kDepths,_hVals,_hDepths,_cVals,_cDepths,_TVals,_TDepths;

    // heat fluxes
    Vec  _surfaceHeatFlux,_heatFlux; // surface and total heat flux

    // viewers:
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;
    PetscViewer          _timeV; // time output viewer

    // which factors to include
    std::string          _wViscShearHeating,_wFrictionalHeating;

    // linear system data
    std::string          _sbpType;
    SbpOps*              _sbpT;
    Vec                  _bcT,_bcR,_bcB,_bcL; // boundary conditions
    std::string          _linSolver;
    PetscScalar          _kspTol;
    KSP                  _ksp;
    PC                   _pc;
    Mat                  _I,_rcInv,_B,_pcMat; // intermediates for Backward Euler
    Mat                  _D2ath;

    // finite width shear zone
    Mat                  _MapV; // maps slip velocity to full size vector for scaling Gw
    Vec                  _Gw,_omega; // Green's function for shear heating, d/dt strain in shear zone, total heat
    PetscScalar          _w; // width of shear zone (m)

    // runtime data
    double               _linSolveTime,_factorTime,_beTime,_writeTime,_miscTime;
    PetscInt             _linSolveCount;


    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode setFields();
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);
    PetscErrorCode loadFieldsFromFiles();
    PetscErrorCode checkInput();     // check input from file


    PetscErrorCode constructMapV();
    PetscErrorCode computeInitialSteadyStateTemp();
    PetscErrorCode setUpSteadyStateProblem();
    PetscErrorCode setUpTransientProblem();
    PetscErrorCode setBCsforBE();
    PetscErrorCode computeViscousShearHeating(Vec& Qvisc,const Vec& sdev, const Vec& dgxy, const Vec& dgxz);
    PetscErrorCode computeFrictionalShearHeating(const Vec& tau, const Vec& slipVel);
    PetscErrorCode setupKSP(SbpOps* sbp,const PetscScalar dt);
    PetscErrorCode setupKSP_SS(SbpOps* sbp);
    PetscErrorCode computeHeatFlux();


    Vec _dT; // actually change in temperature
    Vec _Tamb; // initial temperature
    Vec _k,_rho,_c,_h,_Q;  // thermal conductivity, density, heat capacity, radioactive heat generation

    HeatEquation(Domain& D);
    ~HeatEquation();

    PetscErrorCode getTemp(Vec& T); // return total temperature
    PetscErrorCode setTemp(Vec& T); // set temperature

    PetscErrorCode computeSteadyStateTemp(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T);
    PetscErrorCode initiateVarSS(map<string,Vec>& varSS);
    PetscErrorCode updateSS(map<string,Vec>& varSS);

    // compute rate
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& _varIm);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm);
    PetscErrorCode d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt);

    // implicitly solve for temperature using backward Euler
    PetscErrorCode be(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sdev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);
    PetscErrorCode be_transient(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sdev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);
    PetscErrorCode be_steadyState(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sdev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);

    PetscErrorCode d_dt_mms(const PetscScalar time,const Vec& T, Vec& dTdt);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode be_steadyStateMMS(const PetscScalar time,const Vec slipVel,const Vec& tau,
      const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt);

    // IO commands
    PetscErrorCode view();
    PetscErrorCode writeDomain(const std::string outputDir);
    PetscErrorCode writeContext(const std::string outputDir);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

    // MMS functions
    PetscErrorCode measureMMSError(const PetscScalar time);
    PetscErrorCode setMMSBoundaryConditions(const double time,
        std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType);

    static double zzmms_rho(const double y,const double z);
    static double zzmms_c(const double y,const double z);
    static double zzmms_h(const double y,const double z);

    static double zzmms_k(const double y,const double z);
    static double zzmms_k_y(const double y,const double z);
    static double zzmms_k_z(const double y,const double z);


    static double zzmms_f(const double y,const double z);
    static double zzmms_f_y(const double y,const double z);
    static double zzmms_f_yy(const double y,const double z);
    static double zzmms_f_z(const double y,const double z);
    static double zzmms_f_zz(const double y,const double z);

    static double zzmms_g(const double t);
    static double zzmms_g_t(const double t);
    static double zzmms_T(const double y,const double z,const double t);
    static double zzmms_T_y(const double y,const double z,const double t);
    static double zzmms_T_yy(const double y,const double z,const double t);
    static double zzmms_T_z(const double y,const double z,const double t);
    static double zzmms_T_zz(const double y,const double z,const double t);
    static double zzmms_T_t(const double y,const double z,const double t);
    static double zzmms_SSTsource(const double y,const double z,const double t);

    static double zzmms_dT(const double y,const double z,const double t);
    static double zzmms_dT_y(const double y,const double z,const double t);
    static double zzmms_dT_yy(const double y,const double z,const double t);
    static double zzmms_dT_z(const double y,const double z,const double t);
    static double zzmms_dT_zz(const double y,const double z,const double t);
    static double zzmms_dT_t(const double y,const double z,const double t);
    static double zzmms_SSdTsource(const double y,const double z,const double t);
};





#endif
