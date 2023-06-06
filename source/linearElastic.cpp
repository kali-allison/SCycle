#include "linearElastic.hpp"

#define FILENAME "linearElastic.cpp"

using namespace std;


KeepKSPCount::KeepKSPCount(int startIt)
  : _myKspItNumTot(startIt),_myKspItNumStep(startIt)
{}

KeepKSPCount::~KeepKSPCount()
{}

/* ------------------------------------------------------------- */
/*
   MyKSPMonitor - This is a user-defined routine for monitoring
   the KSP iterative solvers.

   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     ctx - optional user-defined monitor context for keeping track of iteration number
*/
PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscErrorCode ierr = 0;
  KeepKSPCount * usrctx = static_cast<KeepKSPCount *> (ctx);
  usrctx->_myKspItNumTot++;
  usrctx->_myKspItNumStep++;
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"  total iteration %D:  iteration %D KSP Residual norm %14.12e \n",usrctx->_myKspItNum,n,rnorm);CHKERRQ(ierr);
  return ierr;
}


// construct class object
LinearElastic::LinearElastic(Domain&D,string bcRTtype,string bcTTtype,string bcLTtype,string bcBTtype)
  : _D(&D),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
    _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
    _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),_y0(&D._y0),_z0(&D._z0),
    _isMMS(D._isMMS),
    _mu(NULL),_rho(NULL),_cs(NULL),_surfDisp(NULL),_bcRShift(NULL),_bcTShift(NULL),_bcBShift(NULL),
    _rhs(NULL),_u(NULL),_sxy(NULL),_sxz(NULL),_computeSxz(0),_computeSdev(0),
    _linSolverSS("MUMPSCHOLESKY"),_linSolverTrans("MUMPSCHOLESKY"),_ksp(NULL),_pc(NULL),_kspTol(1e-10),
    _sbp(NULL),_kspItNum(0),_myKspCtx(0),_pcIluFill(0.),
    _viewer1D_hdf5(NULL),_viewer2D_hdf5(NULL),
    _writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
    _miscTime(0), _matrixTime(0), _linSolveCount(0),
    _bcRType(bcRTtype),_bcTType(bcTTtype),_bcLType(bcLTtype),_bcBType(bcBTtype),
    _bcR(NULL),_bcT(NULL),_bcL(NULL),_bcB(NULL)
{
  #if VERBOSE > 1
    string funcName = "LinearElastic::LinearElastic()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // load and set parameters
  loadSettings(D._file);
  checkInput();
  allocateFields();
  setMaterialParameters();
  if (_D->_restartFromChkpt) { loadCheckpoint(); }
  else { loadFieldsFromFiles(); }

  double startMatrix = MPI_Wtime();
  setUpSBPContext(); // set up matrix operators
  _matrixTime += MPI_Wtime() - startMatrix;


  setSurfDisp();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// destructor
LinearElastic::~LinearElastic()
{
  #if VERBOSE > 1
    string funcName = "LinearElastic::~LinearElastic()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);
  VecDestroy(&_bcRShift);
  VecDestroy(&_bcTShift);
  VecDestroy(&_bcBShift);

  // body fields
  VecDestroy(&_rho);
  VecDestroy(&_cs);
  VecDestroy(&_mu);
  VecDestroy(&_rho);
  VecDestroy(&_cs);
  VecDestroy(&_rhs);
  VecDestroy(&_u);
  VecDestroy(&_sxy);
  VecDestroy(&_sxz);
  VecDestroy(&_surfDisp);

  KSPDestroy(&_ksp);

  delete _sbp;
  _sbp = NULL;

  //~ for (map<string,pair<PetscViewer,string> >::iterator it=_viewers1D.begin(); it !=_viewers1D.end(); it++) {
    //~ PetscViewerDestroy(&_viewers1D[it->first].first);
  //~ }
  for (map<string,pair<PetscViewer,string> >::iterator it=_viewers2D.begin(); it !=_viewers2D.end(); it++) {
    PetscViewerDestroy(&_viewers2D[it->first].first);
  }

  PetscViewerDestroy(&_viewer1D_hdf5);
  PetscViewerDestroy(&_viewer2D_hdf5);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// loads settings from the input text file
PetscErrorCode LinearElastic::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ifstream infile( file );
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);
    rhs = "";
    if (line.length() > (pos + _delim.length())) {
      rhs = line.substr(pos+_delim.length(),line.npos);
    }
    rhsFull = rhs; // everything after _delim

    pos = rhs.find(" "); // interpret everything after the appearance of a space on the line as a comment
    rhs = rhs.substr(0,pos); // rhs is everything starting at location 0 and spans pos characters

    if (var.compare("linSolverSS")==0) { _linSolverSS = rhs; }
    else if (var.compare("linSolverTrans")==0) { _linSolverTrans = rhs; }
    else if (var.compare("kspTol")==0) { _kspTol = atof( (rhs).c_str() ); }
    else if (var.compare("akspTol")==0) { _akspTol = atof( (rhs).c_str() ); }
    else if (var.compare("rkspTol")==0) { _rkspTol = atof( (rhs).c_str() ); }
    else if (var.compare("pcIluFill")==0) { _pcIluFill = atoi( (rhs).c_str() ); }

    else if (var.compare("muVals")==0) { loadVectorFromInputFile(rhsFull,_muVals); }
    else if (var.compare("muDepths")==0) { loadVectorFromInputFile(rhsFull,_muDepths); }
    else if (var.compare("rhoVals")==0) { loadVectorFromInputFile(rhsFull,_rhoVals); }
    else if (var.compare("rhoDepths")==0) { loadVectorFromInputFile(rhsFull,_rhoDepths); }

    // switches for computing extra stresses
    else if (var.compare("momBal_computeSxz")==0) { _computeSxz = atof( rhs.c_str() ); }
    else if (var.compare("momBal_computeSdev")==0) { _computeSdev = atof( rhs.c_str() ); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode LinearElastic::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::checkInput()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_linSolverSS.compare("MUMPSCHOLESKY") == 0 ||
         _linSolverSS.compare("MUMPSLU") == 0 ||
         _linSolverSS.compare("CG_PCBJacobi_SubCholesky") == 0 ||
         _linSolverSS.compare("CG_PCBJacobi_SubILU") == 0 ||
         _linSolverSS.compare("CG_PCAMG") == 0 ||
         _linSolverSS.compare("AMG") == 0 );

  assert(_linSolverTrans.compare("MUMPSCHOLESKY") == 0 ||
         _linSolverTrans.compare("MUMPSLU") == 0 ||
         _linSolverTrans.compare("CG_PCBJacobi_SubCholesky") == 0 ||
         _linSolverTrans.compare("CG_PCBJacobi_SubILU") == 0 ||
         _linSolverTrans.compare("CG_PCAMG") == 0 ||
         _linSolverTrans.compare("AMG") == 0 );


  assert(_muVals.size() == _muDepths.size());
  assert(_muVals.size() != 0);
  assert(_rhoVals.size() == _rhoDepths.size());
  assert(_rhoVals.size() != 0);

  if (_computeSdev == 1) { _computeSxz = 1; }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}




/*
 * Set up the Krylov Subspace and Preconditioner (KSP) environment. A
 * table of options available through PETSc and linked external packages
 * is available at
 * http://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html.
 *
 * The methods implemented here are:
 *     Algorithm             Package           input file syntax
 * algebraic multigrid       HYPRE                AMG
 * direct LU                 MUMPS                MUMPSLU
 * direct Cholesky           MUMPS                MUMPSCHOLESKY
 *
 * A list of options for each algorithm that can be set can be obtained
 * by running the code with the argument main <input file> -help and
 * searching through the output for "Preconditioner (PC) options" and
 * "Krylov Method (KSP) options".
 *
 * To view convergence information, including number of iterations, use
 * the command line argument: -ksp_converged_reason.
 *
 * For information regarding HYPRE's solver options, especially the
 * preconditioner options, use the User manual online.
 * Use -ksp_view.
 */

PetscErrorCode LinearElastic::setupKSP(KSP& ksp,PC& pc,Mat& A,std::string& linSolver)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // create linear solver context
  ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);

  // set operators, here the matrix that defines the linear system also serves as the preconditioning matrix
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);

  // algebraic multigrid from HYPRE
  if (linSolver == "AMG") {
    ierr = KSPSetType(ksp,KSPRICHARDSON);                               CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);                                     CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
  }

  // direct LU from MUMPS
#if defined(PETSC_HAVE_MUMPS)
  else if (linSolver == "MUMPSLU") {
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);                                          CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }

  // direct Cholesky (RR^T) from MUMPS
  else if (linSolver == "MUMPSCHOLESKY") {
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);                                    CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }
#endif

  // preconditioned conjugate gradient, using AMG as preconditioner
  else if (linSolver == "CG_PCAMG") {
    ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);        CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);                                      CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);                                               CHKERRQ(ierr);
  }
  // preconditioned conjugate gradient, using block Jacobi preconditioner
  else if (linSolver == "CG_PCBJacobi_SubCholesky") {
    ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_rkspTol,_akspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCBJACOBI);                                     CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);                                               CHKERRQ(ierr);

    // now set solver for each block
    // Extract the array of KSP contexts for the local blocks
    PetscInt       nlocal,first,ii;
    KSP            *subksp; /* array of local KSP contexts on this processor */
    PC             subpc;
    ierr = PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);CHKERRQ(ierr);

    // Loop over the local blocks, setting various KSP options for each block.
    for (ii=0; ii<nlocal; ii++) {
      ierr = KSPGetPC(subksp[ii],&subpc);                               CHKERRQ(ierr);
      //~ ierr = PCSetType(subpc,PCILU);                                    CHKERRQ(ierr);
      //~ ierr = PCFactorSetLevels(subpc,_pcIluFill);                       CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCCHOLESKY);                               CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    }
    ierr = KSPSetUp(ksp);                                               CHKERRQ(ierr);
  }
  else if (linSolver == "CG_PCBJacobi_SubILU") {
    ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,&MyKSPMonitor,(void*)&_myKspCtx,0);                      CHKERRQ(ierr);
    ierr = KSPCGSetType(ksp,KSP_CG_SYMMETRIC);                           CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_rkspTol,_akspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCBJACOBI);                                     CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);                                               CHKERRQ(ierr);

    // now set solver for each block
    // Extract the array of KSP contexts for the local blocks
    PetscInt       nlocal,first,ii;
    KSP            *subksp; /* array of local KSP contexts on this processor */
    PC             subpc;
    ierr = PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);CHKERRQ(ierr);

    // Loop over the local blocks, setting various KSP options for each block.
    for (ii=0; ii<nlocal; ii++) {
      ierr = KSPGetPC(subksp[ii],&subpc);                               CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCICC);                                    CHKERRQ(ierr);
      ierr = PCFactorSetLevels(subpc,_pcIluFill);                       CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(subksp[ii],PETSC_TRUE);          CHKERRQ(ierr);
      //~ PetscPrintf(PETSC_COMM_WORLD,"picIluFill = %i\n",_pcIluFill);
    }
    //~ ierr = KSPSetUp(ksp);                                               CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);                                               CHKERRQ(ierr);
  }

  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0);
  }
  // perform computation of preconditioners now, rather than on first use
  //~ ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  //~ ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// allocate space for member fields
PetscErrorCode LinearElastic::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    // boundary conditions
  VecDuplicate(_D->_y0,&_bcL);
  PetscObjectSetName((PetscObject) _bcL, "bcL");
  VecSet(_bcL,0.0);

  VecDuplicate(_bcL,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "bcRShift");
  VecSet(_bcRShift,0.0);
  VecDuplicate(_bcL,&_bcR); PetscObjectSetName((PetscObject) _bcR, "bcR");
  VecSet(_bcR,0.);

  VecDuplicate(_D->_z0,&_bcT);
  PetscObjectSetName((PetscObject) _bcT, "bcT");
  VecSet(_bcT,0.0);
  VecDuplicate(_bcT,&_bcTShift); PetscObjectSetName((PetscObject) _bcTShift, "bcTShift");
  VecSet(_bcTShift,0.0);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "bcB");
  VecSet(_bcB,0.0);
  VecDuplicate(_bcB,&_bcBShift); PetscObjectSetName((PetscObject) _bcBShift, "bcBShift");
  VecSet(_bcBShift,0.0);


  // other fieds
  VecDuplicate(*_z,&_rhs); VecSet(_rhs,0.0); PetscObjectSetName((PetscObject) _rhs, "rhs");
  VecDuplicate(*_z,&_mu); PetscObjectSetName((PetscObject) _mu, "mu");
  VecDuplicate(*_z,&_rho); PetscObjectSetName((PetscObject) _rho, "rho");
  VecDuplicate(*_z,&_cs); PetscObjectSetName((PetscObject) _cs, "cs");
  VecDuplicate(_rhs,&_u); VecSet(_u,0.0); PetscObjectSetName((PetscObject) _u, "u");
  VecDuplicate(_rhs,&_sxy); VecSet(_sxy,0.0); PetscObjectSetName((PetscObject) _sxy, "sxy");
  if (_computeSxz) {
    VecDuplicate(_rhs,&_sxz); VecSet(_sxz,0.0);
    PetscObjectSetName((PetscObject) _sxz, "sxz");
    }
  else { _sxz = NULL; }
  if (_computeSdev) {
    VecDuplicate(_rhs,&_sdev); VecSet(_sdev,0.0);
    PetscObjectSetName((PetscObject) _sxz, "sxz");
  }
  else { _sdev = NULL; }
  VecDuplicate(_bcT,&_surfDisp); PetscObjectSetName((PetscObject) _surfDisp, "surfDisp");

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

return ierr;
}


// set off-fault material properties
PetscErrorCode LinearElastic::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = setVec(_mu,*_y,_muVals,_muDepths);CHKERRQ(ierr);
  ierr = setVec(_rho,*_z,_rhoVals,_rhoDepths);CHKERRQ(ierr);
  VecPointwiseDivide(_cs, _mu, _rho);
  VecSqrtAbs(_cs);

  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_mu,zzmms_mu1D,*_y); }
    else { mapToVec(_mu,zzmms_mu,*_y,*_z); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


PetscErrorCode LinearElastic::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::loadFieldsFromFiles";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = loadVecFromInputFile(_bcL,_inputDir,"momBal_bcL"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"momBal_bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.);
  ierr = loadVecFromInputFile(_mu,_inputDir,"momBal_mu"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_rho,_inputDir,"momBal_rho"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_cs,_inputDir,"momBal_cs"); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// load data from a checkpoint
PetscErrorCode LinearElastic::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);

  ierr = VecLoad(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_cs, viewer);                                          CHKERRQ(ierr);

  ierr = VecLoad(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcRShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);

  ierr = VecLoad(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_sxy,viewer);                                          CHKERRQ(ierr);
  if (_computeSxz) {
    ierr = VecLoad(_sxy,viewer);                                        CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// load data from a checkpoint
PetscErrorCode LinearElastic::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::loadCheckpointSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // load saved checkpoint data
  PetscViewer viewer;

  string fileName = _outputDir + "data_context.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = VecLoad(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_cs, viewer);                                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);

  fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);

  ierr = VecLoad(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcRShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);

  ierr = VecLoad(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_sxy,viewer);                                          CHKERRQ(ierr);
  if (_computeSxz) {
    ierr = VecLoad(_sxy,viewer);                                        CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set up SBP operators
PetscErrorCode LinearElastic::setUpSBPContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setUpSBPContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _sbp;
  KSPDestroy(&_ksp);

  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    _sbp = new SbpOps_m_constGrid(_order,_Ny,_Nz,_Ly,_Lz,_mu);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    _sbp = new SbpOps_m_varGrid(_order,_Ny,_Nz,_Ly,_Lz,_mu);
    if (_Ny > 1 && _Nz > 1) { _sbp->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbp->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbp->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setCompatibilityType(_D->_sbpCompatibilityType);
  _sbp->setBCTypes(_bcRType,_bcTType,_bcLType,_bcBType);
  _sbp->setMultiplyByH(1);
  _sbp->setLaplaceType("yz");
  _sbp->setDeleteIntermediateFields(1);
  _sbp->computeMatrices(); // actually create the matrices

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// solve momentum balance equation for displacement vector u
PetscErrorCode LinearElastic::computeU()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::computeU";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ // if u is very close to 0, choose initial guess of all 1s
  //~ Vec u;
  //~ VecDuplicate(_u,&u);
  //~ VecCopy(_u,u);
  //~ PetscScalar norm2U;
  //~ VecNorm(_u,NORM_2,&norm2U);
  //~ if (norm2U < 1e-8) {
    //~ VecSet(_u,1.0);
  //~ }

  // scale rhs by norm2(rhs)
  //~ PetscScalar scaleRhs;
  //~ VecNorm(_rhs,NORM_2,&scaleRhs);
  //~ scaleRhs = abs(scaleRhs);
  //~ if (scaleRhs > 1e-7) {
      //~ VecScale(_rhs,1.0/scaleRhs);
      //~ VecScale(_u,1.0/scaleRhs);
  //~ }

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u); CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  //~ // scale u to account for rhs scaling
  //~ if (scaleRhs > 1e-7) {
    //~ VecScale(_u,scaleRhs);
  //~ }


  // print number of iterations required to converge
  PetscInt itNum = 0;
  KSPGetIterationNumber(_ksp,&itNum);
  _kspItNum += itNum;
  //~ PetscPrintf(PETSC_COMM_WORLD,"itNum = %i\n",itNum);

  ierr = setSurfDisp();
  //~ VecDestroy(&u);

  // // force solution to be accurate to debug MMS test
  // ierr = mapToVec(_u,zzmms_uA,*_y,*_z,time); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set the right-hand side vector for linear solve
PetscErrorCode LinearElastic::setRHS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setRHS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(_rhs,0.0);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// change boundary condition types and reset linear solver
PetscErrorCode LinearElastic::changeBCTypes(string bcRTtype,string bcTTtype,string bcLTtype,string bcBTtype)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::changeBCTypes";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // destroy current KSP context to reset
  KSPDestroy(&_ksp);
  _sbp->changeBCTypes(bcRTtype,bcTTtype,bcLTtype,bcBTtype);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set up surface displacement
PetscErrorCode LinearElastic::setSurfDisp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setSurfDisp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // extract surface displacement from u
  VecScatterBegin(_D->_scatters["body2T"], _u, _surfDisp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2T"], _u, _surfDisp, INSERT_VALUES, SCATTER_FORWARD);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// view runtime summary
PetscErrorCode LinearElastic::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Linear Elastic Runtime Summary:\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   linear solver algorithm: %s\n",_linSolverTrans.c_str()); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent creating matrices (s): %g\n",_matrixTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   PETSc number of iterations for linear system solve (s): %i\n",_kspItNum); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   SCycle number of iterations for linear system solve (s): %i\n",_myKspCtx._myKspItNumTot); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent creating matrices: %g\n",_matrixTime/totRunTime*100.); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  //~ ierr = KSPView(_ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  return ierr;
}


// write out momentum balance context, such as linear solve settings, boundary conditions
PetscErrorCode LinearElastic::writeDomain()
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;

  #if VERBOSE > 1
    string funcName = "LinearElastic::writeDomain";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // write out scalar info in output directory text file
  string str = _outputDir + "momBal.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolverSS = %s\n",_linSolverSS.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"linSolverTrans = %s\n",_linSolverTrans.c_str());CHKERRQ(ierr);
  //~ ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rkspTol = %.15e\n",_rkspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"akspTol = %.15e\n",_akspTol);CHKERRQ(ierr);

  // boundary conditions
  ierr = PetscViewerASCIIPrintf(viewer,"bcR_type = %s\n",_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcT_type = %s\n",_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcL_type = %s\n",_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcB_type = %s\n",_bcBType.c_str());CHKERRQ(ierr);

  // free viewer memory
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// write out momentum balance context, such as linear solve settings, boundary conditions
PetscErrorCode LinearElastic::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "LinearElastic::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  writeDomain();

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = VecView(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_cs, viewer);                                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// writes out fields of length Ny or Nz at each time step
PetscErrorCode LinearElastic::writeStep1D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = VecView(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcRShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcTShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcBShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// writes out fields of length Ny*Nz. These take up much more hard drive space, and more runtime to output, so are separate from the 1D fields
PetscErrorCode LinearElastic::writeStep2D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::writeStep2D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                  CHKERRQ(ierr);
  ierr = VecView(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_sxy,viewer);                                          CHKERRQ(ierr);
  if (_computeSxz) {
    ierr = VecView(_sxy,viewer);                                        CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// writes out fields of length Ny or Nz at each time step
PetscErrorCode LinearElastic::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);

  ierr = VecView(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_cs, viewer);                                          CHKERRQ(ierr);

  ierr = VecView(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcRShift,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcT,viewer);                                          CHKERRQ(ierr);

  ierr = VecView(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_sxy,viewer);                                          CHKERRQ(ierr);
  if (_computeSxz) {
    ierr = VecView(_sxy,viewer);                                        CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// explicit time stepping to compute stresses
PetscErrorCode LinearElastic::computeStresses()
{
  PetscErrorCode ierr = 0;
    #if VERBOSE > 1
    string funcName = "LinearElastic::computeStresses()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // solve for shear stress
  ierr = _sbp->muxDy(_u,_sxy); CHKERRQ(ierr);

  // if compute sigma_xz
  if (_computeSxz) {
    ierr = _sbp->muxDz(_u,_sxz); CHKERRQ(ierr);
  }

  // if compute deviatoric stress
  if (_computeSdev) {
    ierr = computeSDev(); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  return ierr;
}


// computes sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode LinearElastic::computeSDev()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::computeSDev()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  // deviatoric stress: part 1/3
  VecPointwiseMult(_sdev,_sxy,_sxy);

  if (_Nz > 1) {
    // deviatoric stress: part 2/3
    Vec temp;
    VecDuplicate(_sxz,&temp);
    VecPointwiseMult(temp,_sxz,_sxz);
    VecAXPY(_sdev,1.0,temp);
    VecDestroy(&temp);
  }

  // deviatoric stress: part 3/3
  VecSqrtAbs(_sdev);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  return ierr;
}


// set stress pointers to calculated values
PetscErrorCode LinearElastic::getStresses(Vec& sxy, Vec& sxz, Vec& sdev)
{
  sxy = _sxy;
  sxz = _sxz;
  sdev = _sdev;
  return 0;
}


// set boundary conditions for MMS test
PetscErrorCode LinearElastic::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setMMSBoundaryConditions";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time); CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);


  if (_Nz == 1) {
    Ii = Istart;

    // left boundary
    y = 0;
    // uAnal(y=0,z), Dirichlet boundary condition
    if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); }
    // sigma_xy = mu * (du/dy), Neumann boundary condition
    else if (!_bcLType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); }
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES); CHKERRQ(ierr);

    // right boundary
    y = _Ly;
    // uAnal(y=Ly,z)
    if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); }
    // sigma_xy = mu * (du/dy)
    else if (!_bcRType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); }
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES); CHKERRQ(ierr);
  }

  else {
    for (Ii = Istart; Ii < Iend; Ii++) {
      ierr = VecGetValues(*_y0,1,&Ii,&z); CHKERRQ(ierr);
      //~ z = _dz * Ii;

      // left boundary
      y = 0;
      // uAnal(y=0,z)
      if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); }
      // sigma_xy = mu * d/dy u
      else if (!_bcLType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); }
      ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      // right boundary
      y = _Ly;
      // uAnal(y=Ly,z)
      if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); }
      // sigma_xy = mu * d/dy u
      else if (!_bcRType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); }
      ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  // assemble boundary vectors
  ierr = VecAssemblyBegin(_bcL); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcR); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcL); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR); CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(*_y,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii = Istart; Ii < Iend; Ii++) {
    if (Ii % _Nz == 0) {
      //~ y = _dy * Ii;
      ierr = VecGetValues(*_y,1,&Ii,&y);CHKERRQ(ierr);
      PetscInt Jj = Ii / _Nz;

      // top boundary
      z = 0;
      // uAnal(y, z = 0)
      if (!_bcTType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); }
      else if (!_bcTType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time)); }
      ierr = VecSetValues(_bcT,1,&Jj,&v,INSERT_VALUES); CHKERRQ(ierr);

      // bottom boundary
      z = _Lz;
      // uAnal(y, z = Lz)
      if (!_bcBType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); }
      else if (!_bcBType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_z(y,z,time); }
      ierr = VecSetValues(_bcB,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  // assemble top and bottom boundary vectors
  ierr = VecAssemblyBegin(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcB);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// compute source term for MMS test and add it to rhs vector
PetscErrorCode LinearElastic::addRHS_MMSSource(const PetscScalar time,Vec& rhs)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::addRHS_MMSSource";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time); CHKERRQ(ierr);
  #endif

  Vec source,Hxsource;
  VecDuplicate(_u,&source);
  VecDuplicate(_u,&Hxsource);

  if (_Nz == 1) {
    mapToVec(source,zzmms_uSource1D,*_y,time);
  }
  else {
    mapToVec(source,zzmms_uSource,*_y,*_z,time);
  }
  ierr = _sbp->H(source,Hxsource);

  if (_D->_gridSpacingType.compare("variableGridSpacing") == 0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }

  ierr = VecAXPY(_rhs,1.0,Hxsource); CHKERRQ(ierr); // rhs = rhs + H*source

  // free memory
  VecDestroy(&source);
  VecDestroy(&Hxsource);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set MMS initial conditions
PetscErrorCode LinearElastic::setMMSInitialConditions(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::setMMSInitialConditions";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str()); CHKERRQ(ierr);
  #endif

  Vec source,Hxsource;
  VecDuplicate(_u,&source);
  VecDuplicate(_u,&Hxsource);

  if (_Nz == 1) {
    mapToVec(source,zzmms_uSource1D,*_y,time);
  }
  else {
    mapToVec(source,zzmms_uSource,*_y,*_z,time);
  }
  writeVec(source,_outputDir + "mms_uSource");
  ierr = _sbp->H(source,Hxsource); CHKERRQ(ierr);

  if (_D->_gridSpacingType.compare("variableGridSpacing") == 0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }

  // set rhs, including body source term
  VecSet(_bcRShift,0.0);
  ierr = setMMSBoundaryConditions(time); CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB); CHKERRQ(ierr);
  ierr = VecAXPY(_rhs,1.0,Hxsource); CHKERRQ(ierr); // rhs = rhs + H*source

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u); CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  _sbp->muxDy(_u,_sxy);

  // free memory
  VecDestroy(&source);
  VecDestroy(&Hxsource);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif

  return ierr;
}


// get MMS error to measure convergence
PetscErrorCode LinearElastic::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;

  // measure error between analytical and numerical solution
  Vec uA, sigmaxyA;
  VecDuplicate(_u,&uA);
  VecDuplicate(_u,&sigmaxyA);

  if (_Nz == 1) {
    mapToVec(uA,zzmms_uA1D,*_y,time);
    mapToVec(sigmaxyA,zzmms_sigmaxy1D,*_y,time);
  }
  else {
    mapToVec(uA,zzmms_uA,*_y,*_z,time);
    mapToVec(sigmaxyA,zzmms_sigmaxy,*_y,*_z,time);
  }

  double err2uA = computeNormDiff_2(_u,uA);
  double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  writeVec(uA,_outputDir+"uA");
  // writeVec(_bcL,_outputDir+"mms_u_bcL");
  // writeVec(_bcR,_outputDir+"mms_u_bcR");
  // writeVec(_bcT,_outputDir+"mms_u_bcT");
  // writeVec(_bcB,_outputDir+"mms_u_bcB");

  // Mat H; _sbp->getH(H);
  // double err2uA = computeNormDiff_Mat(H,_u,uA);
  // double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2uA,log2(err2uA),err2sigmaxy,log2(err2sigmaxy));

  // free memory
  VecDestroy(&uA);
  VecDestroy(&sigmaxyA);

  return ierr;
}


// MMS functions
// helper function for uA
double LinearElastic::zzmms_f(const double y,const double z) {
  return cos(y)*sin(z);
}

double LinearElastic::zzmms_f_y(const double y,const double z) {
  return -sin(y)*sin(z);
}

double LinearElastic::zzmms_f_yy(const double y,const double z) {
  return -cos(y)*sin(z);
}

double LinearElastic::zzmms_f_z(const double y,const double z) {
  return cos(y)*cos(z);
}

double LinearElastic::zzmms_f_zz(const double y,const double z) {
  return -cos(y)*sin(z);
}

double LinearElastic::zzmms_g(const double t) {
  return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9);
}

double LinearElastic::zzmms_g_t(const double t) {
  return (-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9);
}

double LinearElastic::zzmms_uA(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g(t);
}

double LinearElastic::zzmms_uA_y(const double y,const double z,const double t) {
  return zzmms_f_y(y,z)*zzmms_g(t);
}

double LinearElastic::zzmms_uA_yy(const double y,const double z,const double t) {
  return zzmms_f_yy(y,z)*zzmms_g(t);
}

double LinearElastic::zzmms_uA_z(const double y,const double z,const double t) {
  return zzmms_f_z(y,z)*zzmms_g(t);
}

double LinearElastic::zzmms_uA_zz(const double y,const double z,const double t) {
  return zzmms_f_zz(y,z)*zzmms_g(t);
}

double LinearElastic::zzmms_uA_t(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g_t(t);
}

double LinearElastic::zzmms_mu(const double y,const double z) {
  return sin(y)*sin(z) + 30;
}

double LinearElastic::zzmms_mu_y(const double y,const double z) {
  return cos(y)*sin(z);
}

double LinearElastic::zzmms_mu_z(const double y,const double z) {
  return sin(y)*cos(z);
}

double LinearElastic::zzmms_sigmaxy(const double y,const double z,const double t) {
  return zzmms_mu(y,z)*zzmms_uA_y(y,z,t);
}

double LinearElastic::zzmms_uSource(const double y,const double z,const double t) {
  PetscScalar mu = zzmms_mu(y,z);
  PetscScalar mu_y = zzmms_mu_y(y,z);
  PetscScalar mu_z = zzmms_mu_z(y,z);
  PetscScalar u_y = zzmms_uA_y(y,z,t);
  PetscScalar u_yy = zzmms_uA_yy(y,z,t);
  PetscScalar u_z = zzmms_uA_z(y,z,t);
  PetscScalar u_zz = zzmms_uA_zz(y,z,t);
  return mu*(u_yy + u_zz) + mu_y*u_y + mu_z*u_z;
}


// 1D
// helper function for uA
double LinearElastic::zzmms_f1D(const double y) {
  return cos(y) + 2;
}

double LinearElastic::zzmms_f_y1D(const double y) {
  return -sin(y);
}

double LinearElastic::zzmms_f_yy1D(const double y) {
  return -cos(y);
}

// double LinearElastic::zzmms_f_z1D(const double y) { return 0; }
// double LinearElastic::zzmms_f_zz1D(const double y) { return 0; }

double LinearElastic::zzmms_uA1D(const double y,const double t) {
  return zzmms_f1D(y)*exp(-t);
}

double LinearElastic::zzmms_uA_y1D(const double y,const double t) {
  return zzmms_f_y1D(y)*exp(-t);
}

double LinearElastic::zzmms_uA_yy1D(const double y,const double t) {
  return zzmms_f_yy1D(y)*exp(-t);
}

double LinearElastic::zzmms_uA_z1D(const double y,const double t) {
  return 0;
}

double LinearElastic::zzmms_uA_zz1D(const double y,const double t) {
  return 0;
}

double LinearElastic::zzmms_uA_t1D(const double y,const double t) {
  return -zzmms_f1D(y)*exp(-t);
}

double LinearElastic::zzmms_mu1D(const double y) {
  return sin(y) + 2.0;
}

double LinearElastic::zzmms_mu_y1D(const double y) {
  return cos(y);
}

// double LinearElastic::zzmms_mu_z1D(const double y) { return 0; }

double LinearElastic::zzmms_sigmaxy1D(const double y,const double t) {
  return zzmms_mu1D(y)*zzmms_uA_y1D(y,t);
}

double LinearElastic::zzmms_uSource1D(const double y,const double t) {
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar u_y = zzmms_uA_y1D(y,t);
  PetscScalar u_yy = zzmms_uA_yy1D(y,t);
  PetscScalar u_zz = zzmms_uA_zz1D(y,t);

  return mu*(u_yy + u_zz) + mu_y*u_y;
}
