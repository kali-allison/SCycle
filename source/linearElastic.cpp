#include "linearElastic.hpp"

#define FILENAME "linearElastic.cpp"

using namespace std;


LinearElastic::LinearElastic(Domain&D,std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype)
: _D(&D),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_loadICs(D._loadICs),
  _stepCount(0),
  _muVec(NULL),_rhoVec(NULL),_cs(NULL),
  _bcRShift(NULL),_surfDisp(NULL),
  _rhs(NULL),_u(NULL),_sxy(NULL),_sxz(NULL),_computeSxz(0),_computeSdev(0),
  _linSolver("MUMPSCHOLESKY"),_ksp(NULL),_pc(NULL),
  _kspTol(1e-10),
  _sbp(NULL),_sbpType(D._sbpType),
  _timeV1D(NULL),_timeV2D(NULL),
  _writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0), _matrixTime(0), _linSolveCount(0),
  _bcRType(bcRTtype),_bcTType(bcTTtype),_bcLType(bcLTtype),_bcBType(bcBTtype),
  _bcR(NULL),_bcT(NULL),_bcL(NULL),_bcB(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::LinearElastic()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  allocateFields();
  setMaterialParameters();

  double startMatrix = MPI_Wtime();
  setUpSBPContext(); // set up matrix operators
  _matrixTime += MPI_Wtime() - startMatrix;

  //~ if (_inputDir.compare("unspecified") != 0) {
    loadFieldsFromFiles(); // load from previous simulation
  //~ }

  setSurfDisp();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


LinearElastic::~LinearElastic()
{
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::~LinearElastic()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);
  VecDestroy(&_bcRShift);

  // body fields
  VecDestroy(&_rhoVec);
  VecDestroy(&_cs);
  VecDestroy(&_muVec);
  VecDestroy(&_rhoVec);
  VecDestroy(&_cs);
  VecDestroy(&_rhs);
  VecDestroy(&_u);
  VecDestroy(&_sxy);
  VecDestroy(&_sxz);
  VecDestroy(&_surfDisp);

  KSPDestroy(&_ksp);

  delete _sbp; _sbp = NULL;

  // destroy viewers
  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_timeV2D);
  for (map<string,std::pair<PetscViewer,string> >::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first].first);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// loads settings from the input text file
PetscErrorCode LinearElastic::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "LinearElastic::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  ifstream infile( file );
  string line,var;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);


    if (var.compare("linSolver")==0) {
      _linSolver = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("kspTol")==0) {
      _kspTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }


    else if (var.compare("muVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_muVals);
    }
    else if (var.compare("muDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_muDepths);
    }
    else if (var.compare("rhoVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoVals);
    }
    else if (var.compare("rhoDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoDepths);
    }

    // switches for computing extra stresses
    else if (var.compare("momBal_computeSxz")==0) {
      _computeSxz = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("momBal_computeSdev")==0) {
      _computeSdev = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

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
    std::string funcName = "LinearElastic::checkInput()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_linSolver.compare("MUMPSCHOLESKY") == 0 ||
         _linSolver.compare("MUMPSLU") == 0 ||
         _linSolver.compare("PCG") == 0 ||
         _linSolver.compare("AMG") == 0 ||
         _linSolver.compare("CG") == 0 );

  if (_linSolver.compare("CG")==0 || _linSolver.compare("AMG")==0) {
    assert(_kspTol >= 1e-14);
  }

  assert(_muVals.size() == _muDepths.size() );
  assert(_muVals.size() != 0 );
  assert(_rhoVals.size() == _rhoDepths.size() );
  assert(_rhoVals.size() != 0 );

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
 * A list of options for each algorithm that can be set can be optained
 * by running the code with the argument main <input file> -help and
 * searching through the output for "Preconditioner (PC) options" and
 * "Krylov Method (KSP) options".
 *
 * To view convergence information, including number of iterations, use
 * the command line argument: -ksp_converged_reason.
 *
 * For information regarding HYPRE's solver options, especially the
 * preconditioner options, use the User manual online. Also, use -ksp_view.
 */
PetscErrorCode LinearElastic::setupKSP(SbpOps* sbp,KSP& ksp,PC& pc,Mat& A)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);

  if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(ksp,KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE); CHKERRQ(ierr); // necessary for solving steady state power law
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE); CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg"); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE); CHKERRQ(ierr);

    //~ PetscOptionsSetValue(NULL,"-pc_hypre_boomeramg_agg_nl 1");
  }
  else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    // use direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU); CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS); CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc); CHKERRQ(ierr);
  }
  else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY); CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS); CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc); CHKERRQ(ierr);
  }
  else if (_linSolver.compare("CG")==0) { // conjugate gradient
    ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE); CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE); CHKERRQ(ierr);
  }
    else if (_linSolver.compare("CG")==0) { // CG solver
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    PCSetType(pc,PCHYPRE);
    PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);

  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0);
  }

  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);

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
    std::string funcName = "LinearElastic::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecDuplicate(_D->_y0,&_bcL);
  PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);

  VecDuplicate(_bcL,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "bcRPShift");
  VecSet(_bcRShift,0.0);
  VecDuplicate(_bcL,&_bcR); PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.);

  VecDuplicate(_D->_z0,&_bcT);
  PetscObjectSetName((PetscObject) _bcT, "_bcT");
  VecSet(_bcT,0.0);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "_bcB");
  VecSet(_bcB,0.0);


  // other fieds
  VecDuplicate(*_z,&_rhs); VecSet(_rhs,0.0);
  VecDuplicate(*_z,&_muVec);
  VecDuplicate(*_z,&_rhoVec);
  VecDuplicate(*_z,&_cs);
  VecDuplicate(_rhs,&_u); VecSet(_u,0.0);
  VecDuplicate(_rhs,&_sxy); VecSet(_sxy,0.0);
  if (_computeSxz) { VecDuplicate(_rhs,&_sxz); VecSet(_sxz,0.0); }
  else { _sxz = NULL; }
  if (_computeSdev) { VecDuplicate(_rhs,&_sdev); VecSet(_sdev,0.0); }
  else { _sdev = NULL; }
  VecDuplicate(_bcT,&_surfDisp); PetscObjectSetName((PetscObject) _surfDisp, "_surfDisp");

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
    std::string funcName = "LinearElastic::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = setVec(_muVec,*_y,_muVals,_muDepths);CHKERRQ(ierr);
  ierr = setVec(_rhoVec,*_z,_rhoVals,_rhoDepths);CHKERRQ(ierr);
  VecPointwiseDivide(_cs, _muVec, _rhoVec);
  VecSqrtAbs(_cs);

  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_muVec,zzmms_mu1D,*_y); }
    else { mapToVec(_muVec,zzmms_mu,*_y,*_z); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}





// parse input file and load values into data members
PetscErrorCode LinearElastic::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  std::string funcName = "LinearElastic::loadFieldsFromFiles";
  PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
#endif

  // load bcL
  ierr = loadVecFromInputFile(_bcL,_inputDir,"momBal_bcL"); CHKERRQ(ierr);

  // load bcR
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"momBal_bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.);

  // load u
  ierr = loadVecFromInputFile(_u,_inputDir,"u"); CHKERRQ(ierr);

  // load shear modulus
  ierr = loadVecFromInputFile(_muVec,_inputDir,"mu"); CHKERRQ(ierr);

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
    std::string funcName = "LinearElastic::setUpSBPContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _sbp;
  KSPDestroy(&_ksp);


  if (_sbpType.compare("mc")==0) {
    _sbp = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
    if (_Ny > 1 && _Nz > 1) { _sbp->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbp->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbp->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setBCTypes(_bcRType,_bcTType,_bcLType,_bcBType);
  _sbp->setMultiplyByH(1);
  _sbp->setDeleteIntermediateFields(1);
  _sbp->computeMatrices(); // actually create the matrices

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// solve momentum balance equation for u
PetscErrorCode LinearElastic::computeU()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::computeU";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  ierr = setSurfDisp();

  // force solution to be accurate to debug MMS test
  //~ ierr = mapToVec(_u,zzmms_uA,*_y,*_z,time); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::setRHS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::computeU";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(_rhs,0.);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::changeBCTypes(std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::changeBCTypes";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  KSPDestroy(&_ksp);
  _sbp->changeBCTypes(bcRTtype,bcTTtype,bcLTtype,bcBTtype);
  Mat A; _sbp->getA(A);
  setupKSP(_sbp,_ksp,_pc,A);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::setSurfDisp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::setSurfDisp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // extract surface displacement from u
  VecScatterBegin(_D->_scatters["body2T"], _u, _surfDisp, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2T"], _u, _surfDisp, INSERT_VALUES, SCATTER_FORWARD);

  //~ PetscInt    Ii,Istart,Iend,y;
  //~ PetscScalar u;
  //~ ierr = VecGetOwnershipRange(_u,&Istart,&Iend);
  //~ for (Ii=Istart;Ii<Iend;Ii++) {
    //~ y = Ii / _Nz;
    //~ if (Ii % _Nz == 0) {
      //~ ierr = VecGetValues(_u,1,&Ii,&u);CHKERRQ(ierr);
      //~ ierr = VecSetValue(_surfDisp,y,u,INSERT_VALUES);CHKERRQ(ierr);
    //~ }
  //~ }
  //~ ierr = VecAssemblyBegin(_surfDisp);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(_surfDisp);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Linear Elastic Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent creating matrices (s): %g\n",_matrixTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent creating matrices: %g\n",_matrixTime/totRunTime*100.);CHKERRQ(ierr);

  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   misc time (s): %g\n",_miscTime);CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% misc time: %g\n",_miscTime/_integrateTime);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  //~ierr = KSPView(_ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode LinearElastic::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer    viewer;

  // write out scalar info
  std::string str = outputDir + "momBal_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);

  // boundary conditions
  ierr = PetscViewerASCIIPrintf(viewer,"bcR_type = %s\n",_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcT_type = %s\n",_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcL_type = %s\n",_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcB_type = %s\n",_bcBType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


  ierr = writeVec(_muVec,outputDir + "momBal_mu"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();
  _stepCount = stepCount;

  if (_timeV1D==NULL) {
    //~ ierr = _sbp->writeOps(outputDir + "ops_u_"); CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = io_initiateWriteAppend(_viewers, "surfDisp", _surfDisp, outputDir + "surfDisp"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcL", _bcL, outputDir + "momBal_bcL"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcR", _bcR, outputDir + "momBal_bcR"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcB", _bcB, outputDir + "momBal_bcB"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcT", _bcT, outputDir + "momBal_bcT"); CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_surfDisp,_viewers["surfDisp"].first); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["bcL"].first); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["bcR"].first); CHKERRQ(ierr);
    ierr = VecView(_bcB,_viewers["bcB"].first); CHKERRQ(ierr);
    ierr = VecView(_bcT,_viewers["bcT"].first); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode LinearElastic::writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::writeStep2D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();


  if (_timeV2D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = io_initiateWriteAppend(_viewers, "u", _u, outputDir + "u"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "sxy", _sxy, outputDir + "sxy"); CHKERRQ(ierr);
    if (_computeSxz) { ierr = io_initiateWriteAppend(_viewers, "sxz", _sxz, outputDir + "sxz"); CHKERRQ(ierr); }
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_u,_viewers["u"].first); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["sxy"].first); CHKERRQ(ierr);
    if (_computeSxz) { ierr = VecView(_sxz,_viewers["sxz"].first); CHKERRQ(ierr); }
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// explicit time stepping
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

  if (_computeSxz) { ierr = _sbp->muxDz(_u,_sxz); CHKERRQ(ierr); }
  if (_computeSdev) { ierr = computeSDev(); CHKERRQ(ierr); }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// computes sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode LinearElastic::computeSDev()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::computeSDev()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode LinearElastic::getStresses(Vec& sxy, Vec& sxz, Vec& sdev)
{
  sxy = _sxy;
  sxz = _sxz;
  sdev = _sdev;
  return 0;
}


PetscErrorCode LinearElastic::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "LinearElastic::setMMSBoundaryConditions";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);
  if (_Nz == 1) {
    Ii = Istart;
    y = 0;
    if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  else {
    for(Ii=Istart;Ii<Iend;Ii++) {
      ierr = VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
      //~ z = _dz * Ii;
      y = 0;
      if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=0,z)
      else if (!_bcLType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      y = _Ly;
      if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=Ly,z)
      else if (!_bcRType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ly = %f, y = %f, z = %f, bcR = %f\n",_Ly,y,z,v);
    }
  }
  ierr = VecAssemblyBegin(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR);CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(*_y,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    if (Ii % _Nz == 0) {
      //~ y = _dy * Ii;
      ierr = VecGetValues(*_y,1,&Ii,&y);CHKERRQ(ierr);
      PetscInt Jj = Ii / _Nz;

      z = 0;
      if (!_bcTType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y,z=0)
      else if (!_bcTType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time)); }
      ierr = VecSetValues(_bcT,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);

      z = _Lz;
      if (!_bcBType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y,z=Lz)
      else if (!_bcBType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_z(y,z,time); }
      ierr = VecSetValues(_bcB,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcB);CHKERRQ(ierr);

  //~ writeVec(_bcL,_outputDir+"mms_bcL");
  //~ writeVec(_bcR,_outputDir+"mms_bcR");
  //~ writeVec(_bcT,_outputDir+"mms_bcT");
  //~ writeVec(_bcB,_outputDir+"mms_bcB");

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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif


  Vec source,Hxsource;
  VecDuplicate(_u,&source);
  VecDuplicate(_u,&Hxsource);
  if (_Nz==1) { mapToVec(source,zzmms_uSource1D,*_y,time); }
  else { mapToVec(source,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbp->H(source,Hxsource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  VecDestroy(&source);

  ierr = VecAXPY(_rhs,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::setMMSInitialConditions(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::setMMSInitialConditions";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  Vec source,Hxsource;
  VecDuplicate(_u,&source);
  VecDuplicate(_u,&Hxsource);

  if (_Nz == 1) { mapToVec(source,zzmms_uSource1D,*_y,time); }
  else { mapToVec(source,zzmms_uSource,*_y,*_z,time); }
  writeVec(source,_outputDir + "mms_uSource");
  ierr = _sbp->H(source,Hxsource); CHKERRQ(ierr);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  VecDestroy(&source);


  // set rhs, including body source term
  VecSet(_bcRShift,0.0);
  ierr = setMMSBoundaryConditions(time); CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(_rhs,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  _sbp->muxDy(_u,_sxy);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;

  // measure error between analytical and numerical solution
  Vec uA;
  VecDuplicate(_u,&uA);
  if (_Nz == 1) { mapToVec(uA,zzmms_uA1D,*_y,time); }
  else { mapToVec(uA,zzmms_uA,*_y,*_z,time); }

  Vec sigmaxyA;
  VecDuplicate(_u,&sigmaxyA);
  //~ mapToVec(sigmaxyA,zzmms_sigmaxy,_Nz,_dy,_dz,_currTime);
    if (_Nz == 1) { mapToVec(sigmaxyA,zzmms_sigmaxy1D,*_y,time); }
  else { mapToVec(sigmaxyA,zzmms_sigmaxy,*_y,*_z,time); }


  double err2uA = computeNormDiff_2(_u,uA);
  double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  //~ std::str = _outputDir = "uA";
  writeVec(uA,_outputDir+"uA");
  //~ writeVec(_bcL,_outputDir+"mms_u_bcL");
  //~ writeVec(_bcR,_outputDir+"mms_u_bcR");
  //~ writeVec(_bcT,_outputDir+"mms_u_bcT");
  //~ writeVec(_bcB,_outputDir+"mms_u_bcB");

  //~ Mat H; _sbp->getH(H);
  //~ double err2uA = computeNormDiff_Mat(H,_u,uA);
  //~ double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2uA,log2(err2uA),err2sigmaxy,log2(err2sigmaxy));

  return ierr;
}


// MMS functions
double LinearElastic::zzmms_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double LinearElastic::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double LinearElastic::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double LinearElastic::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double LinearElastic::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }

double LinearElastic::zzmms_g(const double t) { return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9); }
double LinearElastic::zzmms_g_t(const double t) {
  return (-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9);
}

double LinearElastic::zzmms_uA(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double LinearElastic::zzmms_uA_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double LinearElastic::zzmms_uA_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double LinearElastic::zzmms_uA_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double LinearElastic::zzmms_uA_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
//~ double LinearElastic::zzmms_uA_t(const double y,const double z,const double t) {
  //~ return zzmms_f(y,z)*((-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9));
//~ }
double LinearElastic::zzmms_uA_t(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g_t(t);
}

double LinearElastic::zzmms_mu(const double y,const double z) { return sin(y)*sin(z) + 30; }
double LinearElastic::zzmms_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double LinearElastic::zzmms_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double LinearElastic::zzmms_sigmaxy(const double y,const double z,const double t)
{ return zzmms_mu(y,z)*zzmms_uA_y(y,z,t); }

double LinearElastic::zzmms_uSource(const double y,const double z,const double t)
{
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
double LinearElastic::zzmms_f1D(const double y) { return cos(y) + 2; } // helper function for uA
double LinearElastic::zzmms_f_y1D(const double y) { return -sin(y); }
double LinearElastic::zzmms_f_yy1D(const double y) { return -cos(y); }
//~ double LinearElastic::zzmms_f_z1D(const double y) { return 0; }
//~ double LinearElastic::zzmms_f_zz1D(const double y) { return 0; }

double LinearElastic::zzmms_uA1D(const double y,const double t) { return zzmms_f1D(y)*exp(-t); }
double LinearElastic::zzmms_uA_y1D(const double y,const double t) { return zzmms_f_y1D(y)*exp(-t); }
double LinearElastic::zzmms_uA_yy1D(const double y,const double t) { return zzmms_f_yy1D(y)*exp(-t); }
double LinearElastic::zzmms_uA_z1D(const double y,const double t) { return 0; }
double LinearElastic::zzmms_uA_zz1D(const double y,const double t) { return 0; }
double LinearElastic::zzmms_uA_t1D(const double y,const double t) { return -zzmms_f1D(y)*exp(-t); }

double LinearElastic::zzmms_mu1D(const double y) { return sin(y) + 2.0; }
double LinearElastic::zzmms_mu_y1D(const double y) { return cos(y); }
//~ double LinearElastic::zzmms_mu_z1D(const double y) { return 0; }

double LinearElastic::zzmms_sigmaxy1D(const double y,const double t) { return zzmms_mu1D(y)*zzmms_uA_y1D(y,t); }
double LinearElastic::zzmms_uSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar u_y = zzmms_uA_y1D(y,t);
  PetscScalar u_yy = zzmms_uA_yy1D(y,t);
  PetscScalar u_zz = zzmms_uA_zz1D(y,t);
  return mu*(u_yy + u_zz) + mu_y*u_y;
}


