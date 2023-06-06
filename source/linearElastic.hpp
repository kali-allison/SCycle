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

using namespace std;


class KeepKSPCount
{
  public:
    int _myKspItNumTot,_myKspItNumStep;

    KeepKSPCount(int startIt);
    ~KeepKSPCount();
};


PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx);

// Class for a linear elastic material
class LinearElastic
{
private:
  // disable default copy constructor and assignment operator
  LinearElastic(const LinearElastic &that);
  LinearElastic& operator=(const LinearElastic &rhs);

public:
  // domain properties
  Domain         *_D; // shallow copy of domain
  string          _delim; // format is: var delim value (without the white space)
  string          _inputDir;
  string          _outputDir;  // output data
  const PetscInt  _order,_Ny,_Nz;
  PetscScalar     _Ly,_Lz,_dy,_dz;
  Vec            *_y,*_z,*_y0,*_z0; // to handle variable grid spacing
  const bool      _isMMS; // true if running mms test

  // off-fault material fields
  Vec             _mu, _rho, _cs;
  vector<double>  _muVals,_muDepths,_rhoVals,_rhoDepths;
  Vec             _surfDisp,_bcRShift,_bcTShift,_bcBShift;
  Vec             _rhs,_u,_sxy,_sxz,_sdev;
  int             _computeSxz,_computeSdev; // 0 = no, 1 = yes

  // linear system data
  string          _linSolverSS,_linSolverTrans; // linear solver algorithm for steady-state, transient problem
  KSP             _ksp;
  PC              _pc;
  PetscScalar     _kspTol,_akspTol,_rkspTol;
  SbpOps         *_sbp;
  string          _sbpType;
  PetscInt        _kspItNum;
  KeepKSPCount    _myKspCtx;
  PetscInt        _pcIluFill;

  // viewers for 1D and 2D fields
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  //~ map <string,pair<PetscViewer,string> >  _viewers1D;
  map <string,pair<PetscViewer,string> >  _viewers2D;
  PetscViewer _viewer1D_hdf5;
  PetscViewer _viewer2D_hdf5;

  // runtime data
  double   _writeTime,_linSolveTime,_factorTime,_startTime,_miscTime, _matrixTime;
  PetscInt _linSolveCount;

  // boundary conditions
  string _bcRType,_bcTType,_bcLType,_bcBType; // options: Dirichlet, Neumann
  Vec    _bcR,_bcT,_bcL,_bcB;

  // constructors and destructors
  LinearElastic(Domain&D,string bcRTtype,string bcTTtype,string bcLType,string bcBType);
  ~LinearElastic();

  // allocate and initialize data members
  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();
  PetscErrorCode allocateFields(); // allocate space for member fields
  PetscErrorCode setMaterialParameters();
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode setUpSBPContext();
  PetscErrorCode setupKSP(KSP& ksp,PC& pc,Mat& A,std::string& linSolver);
  //~ static PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy);

  // time stepping function
  PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);
  PetscErrorCode computeStresses();
  PetscErrorCode computeSDev();
  PetscErrorCode setSurfDisp();
  PetscErrorCode setRHS();
  PetscErrorCode computeU();
  PetscErrorCode changeBCTypes(string bcRTtype,string bcTTtype,string bcLTtype,string bcBTtype);

  // IO functions
  PetscErrorCode view(const double totRunTime);
  PetscErrorCode writeDomain();
  PetscErrorCode writeContext(const string outputDir, PetscViewer& viewer);
  PetscErrorCode writeStep1D(PetscViewer& viewer);
  PetscErrorCode writeStep2D(PetscViewer& viewer);

  // checkpointing functions
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);

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
