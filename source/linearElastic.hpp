#ifndef LINEARELASTIC_H_INCLUDED
#define LINEARELASTIC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include <map>

#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"


// Class for a linear elastic material
class LinearElastic
{
  private:
    // disable default copy constructor and assignment operator
    LinearElastic(const LinearElastic &that);
    LinearElastic& operator=(const LinearElastic &rhs);

  public:
    // domain properties
    Domain              *_D; // shallow copy of domain
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // output data
    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz,_dy,_dz;
    Vec                  *_y,*_z; // to handle variable grid spacing
    const bool           _isMMS; // true if running mms test
    const bool           _loadICs; // true if starting from a previous simulation
    PetscInt             _stepCount;

    // off-fault material fields
    Vec                  _mu, _rho, _cs;
    std::vector<double>   _muVals,_muDepths,_rhoVals,_rhoDepths;
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
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;

    // runtime data
    double       _writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _matrixTime;
    PetscInt     _linSolveCount;

    // checkpoint data
    PetscInt _ckpt, _ckptNumber, _interval;
  
    // boundary conditions
    string               _bcRType,_bcTType,_bcLType,_bcBType; // options: Dirichlet, Neumann
    Vec                  _bcR,_bcT,_bcL,_bcB;

    // constructors and destructors
    LinearElastic(Domain&D,std::string bcRTtype,std::string bcTTtype,std::string bcLType,std::string bcBType);
    ~LinearElastic();


    // allocate and initialize data members
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadFieldsFromFiles();
    PetscErrorCode setUpSBPContext();
    PetscErrorCode setupKSP(SbpOps* sbp,KSP& ksp,PC& pc,Mat& A);

    // time stepping function
    PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);
    PetscErrorCode computeStresses();
    PetscErrorCode computeSDev();
    PetscErrorCode setSurfDisp();
    PetscErrorCode setRHS();
    PetscErrorCode computeU();
    PetscErrorCode changeBCTypes(std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype);

    // IO commands
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeContext(const std::string outputDir);
    // write out 1D fields
    PetscErrorCode writeStep1D(const PetscInt stepCount, const std::string outputDir);
    // write out 2D fields
    PetscErrorCode writeStep2D(const PetscInt stepCount, const std::string outputDir);

    // MMS functions
    PetscErrorCode setMMSInitialConditions(const double time);
    PetscErrorCode setMMSBoundaryConditions(const double time);
    PetscErrorCode measureMMSError(const PetscScalar time);
    PetscErrorCode addRHS_MMSSource(const PetscScalar time,Vec& rhs);

    // 2D
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
