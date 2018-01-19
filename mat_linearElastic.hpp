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
class Mat_LinearElastic_qd
{
  private:
    // disable default copy constructor and assignment operator
    Mat_LinearElastic_qd(const Mat_LinearElastic_qd &that);
    Mat_LinearElastic_qd& operator=(const Mat_LinearElastic_qd &rhs);

  public:


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

    // domain properties
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // output data
    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz,_dy,_dz;
    Vec                  *_y,*_z; // to handle variable grid spacing
    const bool           _isMMS; // true if running mms test
    const bool           _loadICs; // true if starting from a previous simulation
    PetscScalar          _currTime;
    PetscInt             _stepCount;

    // off-fault material fields
    Vec                  _muVec, _rhoVec, _cs;
    PetscScalar          _muVal,_rhoVal; // if constant
    Vec                  _bcRShift,_surfDisp;
    Vec                  _rhs,_u,_sxy,_sxz,_sdev;
    int                  _computeSxz,_computeSdev; // 0 = no, 1 = yes

    // linear system data
    std::string          _linSolver;
    KSP                  _ksp;
    PC                   _pc;
    PetscScalar          _kspTol;
    SbpOps              *_sbp;
    std::string          _sbpType;

    // viewers
    PetscViewer      _timeV1D,_timeV2D;
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
    PetscInt     _linSolveCount;

    // boundary conditions
    string               _bcTType,_bcRType,_bcBType,_bcLType; // options: Dirichlet, Neumann
    Vec                  _bcT,_bcR,_bcB,_bcL;

    // constructors and destructors
    Mat_LinearElastic_qd(Domain&D,std::string bcTRtype,std::string bcTTtype,std::string bcTLtype,std::string bcTBtype);
    ~Mat_LinearElastic_qd();

    // time stepping function
    PetscErrorCode initiateIntegrand_qs(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);
    PetscErrorCode computeStresses();
    PetscErrorCode computeSDev();

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt); // IMEX backward Euler

    PetscErrorCode measureMMSError(const PetscScalar time);

    // IO commands
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeContext(const std::string outputDir);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time); // write out 1D fields
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time); // write out 2D fields
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

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