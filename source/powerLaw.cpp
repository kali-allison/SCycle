#include "powerLaw.hpp"

#define FILENAME "powerLaw.cpp"


PowerLaw::PowerLaw(Domain& D,HeatEquation& he,std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
: _D(&D),_file(D._file),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_loadICs(D._loadICs),
  _stepCount(0),
  _muVec(NULL),_rhoVec(NULL),_cs(NULL),
  _viscDistribution("effectiveVisc"),
  _A(NULL),_n(NULL),_QR(NULL),_T(NULL),_effVisc(NULL),_effViscCap(1e30),
  _linSolver("unspecified"),_ksp(NULL),_pc(NULL),
  _kspTol(1e-10),
  _sbp(NULL),_sbpType(D._sbpType),
  _B(NULL),_C(NULL),
  _sbp_eta(NULL),_ksp_eta(NULL),_pc_eta(NULL),_ssEffViscScale(1),
  _timeV1D(NULL),_timeV2D(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),_linSolveCount(0),
  _u(NULL),_sxy(NULL),_sxz(NULL),_sdev(NULL),
  _gxy(NULL),_dgxy(NULL),
  _gxz(NULL),_dgxz(NULL),
  _gTxy(NULL),_gTxz(NULL),
  _bcRType(bcRType),_bcTType(bcTType),_bcLType(bcLType),_bcBType(bcBType),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  allocateFields(); // initialize fields
  he.getTemp(_T);
  setMaterialParameters();

  // set up matrix operators and KSP environment
  setUpSBPContext(D); // set up matrix operators
  initializeMomBalMats();

  //~ computeTotalStrains();
  //~ computeStresses();
  //~ computeViscosity(_effViscCap);

  //~ if (_inputDir.compare("unspecified") != 0) {
    loadFieldsFromFiles(); // load from previous simulation
    computeTotalStrains();
    computeStresses();
    computeViscosity(_effViscCap);
  //~ }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PowerLaw::~PowerLaw()
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::~PowerLaw";
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
  VecDestroy(&_rhs);
  VecDestroy(&_u);
  VecDestroy(&_sxy);
  VecDestroy(&_sxz);
  VecDestroy(&_surfDisp);

  KSPDestroy(&_ksp);
  KSPDestroy(&_ksp_eta);

  delete _sbp; _sbp = NULL;
  delete _sbp_eta; _sbp_eta = NULL;

  // destroy viewers
  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_timeV2D);
  for (map<string,std::pair<PetscViewer,string> >::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first].first);
  }

  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_T);
  VecDestroy(&_QR);
  VecDestroy(&_effVisc);
  VecDestroy(&_sdev);
  VecDestroy(&_gTxy);
  VecDestroy(&_gTxz);
  VecDestroy(&_gxy);
  VecDestroy(&_gxz);
  VecDestroy(&_dgxy);
  VecDestroy(&_dgxz);

  PetscViewerDestroy(&_timeV2D);

  MatDestroy(&_B);
  MatDestroy(&_C);

  KSPDestroy(&_ksp_eta);
  delete _sbp_eta; _sbp = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// loads settings from the input text file
PetscErrorCode PowerLaw::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "PowerLaw::loadSettings()";
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

    // interpret everything after the appearance of a space on the line as a comment
    pos = rhs.find(" ");
    rhs = rhs.substr(0,pos);

    if (var.compare("linSolver")==0) { _linSolver = rhs; }
    else if (var.compare("kspTol")==0) { _kspTol = atof( rhs.c_str() ); }
    else if (var.compare("muVals")==0) { loadVectorFromInputFile(rhsFull,_muVals); }
    else if (var.compare("muDepths")==0) { loadVectorFromInputFile(rhsFull,_muDepths); }
    else if (var.compare("rhoVals")==0) { loadVectorFromInputFile(rhsFull,_rhoVals); }
    else if (var.compare("rhoDepths")==0) { loadVectorFromInputFile(rhsFull,_rhoDepths); }

    // viscosity
    else if (var.compare("viscDistribution")==0) {
      _viscDistribution = rhs.c_str();
    }

    // if values are set by a vector
    else if (var.compare("AVals")==0) { loadVectorFromInputFile(rhsFull,_AVals); }
    else if (var.compare("ADepths")==0) { loadVectorFromInputFile(rhsFull,_ADepths); }
    else if (var.compare("BVals")==0) { loadVectorFromInputFile(rhsFull,_BVals); }
    else if (var.compare("BDepths")==0) { loadVectorFromInputFile(rhsFull,_BDepths); }
    else if (var.compare("nVals")==0) { loadVectorFromInputFile(rhsFull,_nVals); }
    else if (var.compare("nDepths")==0) { loadVectorFromInputFile(rhsFull,_nDepths); }

    // cap on viscosity
    else if (var.compare("maxEffVisc")==0) { _effViscCap = atof( rhs.c_str() ); }
    else if (var.compare("ssEffViscScale")==0) { _ssEffViscScale = atof( rhs.c_str() ); }

  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode PowerLaw::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_linSolver.compare("MUMPSCHOLESKY") == 0 ||
         _linSolver.compare("MUMPSLU") == 0 ||
         _linSolver.compare("PCG") == 0 ||
         _linSolver.compare("AMG") == 0 );

  if (_linSolver.compare("PCG")==0 || _linSolver.compare("AMG")==0) {
    assert(_kspTol >= 1e-14);
  }

  assert(_AVals.size() == _ADepths.size() );
  assert(_BVals.size() == _BDepths.size() );
  assert(_nVals.size() == _nDepths.size() );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// allocate space for member fields
PetscErrorCode PowerLaw::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcL);
  VecSetSizes(_bcL,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcL);
  PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);

  VecDuplicate(_bcL,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "bcRPShift");
  VecSet(_bcRShift,0.0);
  VecDuplicate(_bcL,&_bcR); PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.);


  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcT);
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
  VecDuplicate(_bcT,&_surfDisp); PetscObjectSetName((PetscObject) _surfDisp, "_surfDisp");
  ierr = VecDuplicate(_u,&_A);CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&_QR);CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&_n);CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&_T);CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&_effVisc);CHKERRQ(ierr);


  // allocate space for stress and strain vectors
  VecDuplicate(_u,&_sxz); VecSet(_sxz,0.0);
  VecDuplicate(_u,&_sdev); VecSet(_sdev,0.0);

  VecDuplicate(_u,&_gxy);
  PetscObjectSetName((PetscObject) _gxy, "_gxy");
  VecSet(_gxy,0.0);
  VecDuplicate(_u,&_dgxy);
  PetscObjectSetName((PetscObject) _dgxy, "_dgxy");
  VecSet(_dgxy,0.0);

  VecDuplicate(_u,&_gxz);
  PetscObjectSetName((PetscObject) _gxz, "_gxz");
  VecSet(_gxz,0.0);
  VecDuplicate(_u,&_dgxz);
  PetscObjectSetName((PetscObject) _dgxz, "_dgxz");
  VecSet(_dgxz,0.0);

  VecDuplicate(_u,&_gTxy); VecSet(_gTxy,0.0);
  VecDuplicate(_u,&_gTxz); VecSet(_gTxz,0.0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// set off-fault material properties
PetscErrorCode PowerLaw::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set each field using it's vals and depths std::vectors
  ierr = setVec(_muVec,*_z,_muVals,_muDepths);        CHKERRQ(ierr);
  ierr = setVec(_rhoVec,*_z,_rhoVals,_rhoDepths);  CHKERRQ(ierr);
  ierr = setVec(_A,*_z,_AVals,_ADepths);           CHKERRQ(ierr);
  ierr = setVec(_QR,*_z,_BVals,_BDepths);        CHKERRQ(ierr);
  ierr = setVec(_n,*_z,_nVals,_nDepths);         CHKERRQ(ierr);
  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_A,zzmms_A1D,*_y); }
    else { mapToVec(_A,zzmms_A,*_y,*_z); }
    if (_Nz == 1) { mapToVec(_QR,zzmms_B1D,*_y); }
    else { mapToVec(_QR,zzmms_B,*_y,*_z); }
    if (_Nz == 1) { mapToVec(_n,zzmms_n1D,*_y); }
    else { mapToVec(_n,zzmms_n,*_y,*_z); }
    if (_Nz == 1) { mapToVec(_muVec,zzmms_mu1D,*_y); }
    else { mapToVec(_muVec,zzmms_mu,*_y,*_z); }
  }
  if (_viscDistribution.compare("loadFromFile")==0) { loadEffViscFromFiles(); }

  VecPointwiseDivide(_cs, _muVec, _rhoVec);
  VecSqrtAbs(_cs);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}


//parse input file and load values into data members
PetscErrorCode PowerLaw::loadEffViscFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::loadEffViscFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_effVisc,_inputDir,"EffVisc"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_A,_inputDir,"A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"B"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"n"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

//parse input file and load values into data members
PetscErrorCode PowerLaw::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // load u
  ierr = loadVecFromInputFile(_u,_inputDir,"u"); CHKERRQ(ierr);

  // load shear modulus
  ierr = loadVecFromInputFile(_muVec,_inputDir,"mu"); CHKERRQ(ierr);

  // load power law parameters
  ierr = loadVecFromInputFile(_A,_inputDir,"momBal_A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"momBal_QR"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"momBal_n"); CHKERRQ(ierr);

  // load bcL and bcR
  ierr = loadVecFromInputFile(_bcL,_inputDir,"momBal_bcL"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"momBal_bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.0);

  // load viscous strains
  ierr = loadVecFromInputFile(_gxy,_inputDir,"GVxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_gxz,_inputDir,"GVxz"); CHKERRQ(ierr);

  // load stresses
  ierr = loadVecFromInputFile(_sxy,_inputDir,"Sxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_sxz,_inputDir,"Sxz"); CHKERRQ(ierr);

  // load effective viscosity
  ierr = loadVecFromInputFile(_effVisc,_inputDir,"EffVisc"); CHKERRQ(ierr);



  // load temperature
  ierr = loadVecFromInputFile(_T,_inputDir,"T"); CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set up SBP operators
PetscErrorCode PowerLaw::setUpSBPContext(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setUpSBPContext";
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
  _sbp->computeMatrices(); // actually create the matrices


  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  Mat A; _sbp->getA(A);
  setupKSP(A,_ksp,_pc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
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
PetscErrorCode PowerLaw::setupKSP(Mat& A,KSP& ksp,PC& pc)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(ksp,KSPRICHARDSON);                               CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);                                     CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
    //~ PetscOptionsSetValue(NULL,"-pc_hypre_boomeramg_agg_nl 1");
  }
  else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);                                          CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr);
  }
  else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);                                    CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr);
  }
  else if (_linSolver.compare("PCG")==0) { // preconditioned conjugate gradient
    ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n"); CHKERRQ(ierr);
    assert(0);
  }

  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(ksp);                                        CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp);                                                 CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// set up KSP for steady state iterations
PetscErrorCode PowerLaw::setupKSP_SSIts(Mat& A,KSP& ksp,PC& pc)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setupKSP_SSIts";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    //~ // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    //~ ierr = KSPSetType(ksp,KSPRICHARDSON);                               CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    //~ ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    //~ ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    //~ ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    //~ ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    //~ ierr = PCFactorSetLevels(pc,4);                                     CHKERRQ(ierr);
    //~ ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
  //~ }
  //~ else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    //~ ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    //~ ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    //~ ierr = PCSetType(pc,PCLU);                                          CHKERRQ(ierr);
    //~ ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr);
    //~ ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr);
  //~ }
  //~ else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);                                    CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr);
  //~ }
  //~ else if (_linSolver.compare("PCG")==0) { // preconditioned conjugate gradient
    //~ ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
    //~ ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    //~ ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    //~ ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);                   CHKERRQ(ierr);
  //~ }
  //~ else {
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n"); CHKERRQ(ierr);
    //~ assert(0);
  //~ }

  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(ksp);                                        CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp);                                                 CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute B and C
// B = H*Dy*mu + SAT terms
// C = H*Dz*mu + SAT terms
PetscErrorCode PowerLaw::initializeMomBalMats()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initializeMomBalMats";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // get necessary matrix factors
  Mat Dy,Dz;
  Mat Hyinv,Hzinv,Hy,Hz,H;
  Mat muqy,murz,mu;
  Mat E0y,ENy,E0z,ENz;
  Mat J,Jinv,qy,rz,yq,zr;
  _sbp->getDs(Dy,Dz);
  _sbp->getHinvs(Hyinv,Hzinv);
  _sbp->getHs(Hy,Hz);
  _sbp->getH(H);
  _sbp->getHs(Hy,Hz);
  _sbp->getMus(mu,muqy,murz);
  _sbp->getEs(E0y,ENy,E0z,ENz);

  // helpful factor qyxrzxH = qy * rz * H, and yqxzrxH = yq * zr * H
  Mat yqxHy,zrxHz,yqxzrxH;
  if (_sbpType.compare("mfc_coordTrans")==0) {
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMatMatMult(yq,zr,H,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxzrxH); CHKERRQ(ierr);
    //~ ierr = MatMatMult(yqxzrxH,Hzinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy); CHKERRQ(ierr);
    ierr = MatMatMult(yq,Hy,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy); CHKERRQ(ierr);
    //~ ierr = MatMatMult(yqxzrxH,Hyinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&zrxHz); CHKERRQ(ierr);
    ierr = MatMatMult(zr,Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&zrxHz); CHKERRQ(ierr);
  }
  else {
    ierr = MatMatMult(H,Hzinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy);
    ierr = MatMatMult(H,Hyinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&zrxHz);
    ierr = MatDuplicate(H,MAT_COPY_VALUES,&yqxzrxH); CHKERRQ(ierr);
  }


  // B = (yq*zr*H) * Dy*mu + SAT
  ierr = MatMatMatMult(yqxzrxH,Dy,mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_B); CHKERRQ(ierr);

  // add B_SAT: 0 if Dirichlet, zr*H*mu*Hyinv*EXy if von Neumann
  if (_bcLType.compare("Neumann")==0) {
    Mat B_SATL;
    ierr = MatMatMatMult(zrxHz,mu,E0y,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_SATL); CHKERRQ(ierr);
    ierr = MatAXPY(_B,1.,B_SATL,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    MatDestroy(&B_SATL);
  }
  if (_bcRType.compare("Neumann")==0) {
    Mat B_SATR;
    ierr = MatMatMatMult(zrxHz,mu,ENy,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_SATR); CHKERRQ(ierr);
    ierr = MatAXPY(_B,-1.,B_SATR,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    MatDestroy(&B_SATR);
  }

  // C = (yq*zr*H) * Dz*mu + SAT
  ierr = MatMatMatMult(yqxzrxH,Dz,mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_C); CHKERRQ(ierr);

  // add C_SAT: 0 if Dirichlet, yq*H*mu*Hzinv*EXz if von Neumann
  if (_bcTType.compare("Neumann")==0) {
    Mat C_SATL;
    ierr = MatMatMatMult(yqxHy,mu,E0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C_SATL); CHKERRQ(ierr);
    ierr = MatAXPY(_C,1.,C_SATL,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    MatDestroy(&C_SATL);
  }
  if (_bcBType.compare("Neumann")==0) {
    Mat C_SATR;
    ierr = MatMatMatMult(yqxHy,mu,ENz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C_SATR); CHKERRQ(ierr);
    ierr = MatAXPY(_C,-1.,C_SATR,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    MatDestroy(&C_SATR);
  }

  MatDestroy(&yqxHy);
  MatDestroy(&zrxHz);
  MatDestroy(&yqxzrxH);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// for steady state computations
// compute initial tauVisc (from guess at effective viscosity)
PetscErrorCode PowerLaw::getTauVisc(Vec& tauVisc, const PetscScalar ess_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::getTauVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // get steady state effective viscosity
  guessSteadyStateEffVisc(ess_t);

  // use effective viscosity to compute strength of off-fault material
  if (tauVisc == NULL) { VecDuplicate(_bcL,&tauVisc); }

  // extract viscosity on fault and use it to compute viscous strength, tauVisc
  VecScatterBegin(_D->_scatters["body2L"], _effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], _effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
  VecScale(tauVisc,ess_t);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::setMMSInitialConditions(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setMMSInitialConditions()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
  #endif

  if (_Nz == 1) { mapToVec(_gxy,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gxy,zzmms_gxy,*_y,*_z,time); }
  if (_Nz == 1) { VecSet(_gxz,0.0); }
  else { mapToVec(_gxz,zzmms_gxz,*_y,*_z,time); }

  // set material properties
  if (_Nz == 1) { mapToVec(_muVec,zzmms_mu1D,*_y); }
  else { mapToVec(_muVec,zzmms_mu,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_A,zzmms_A1D,*_y); }
  else { mapToVec(_A,zzmms_A,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_QR,zzmms_B1D,*_y); }
  else { mapToVec(_QR,zzmms_B,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_n,zzmms_n1D,*_y); }
  else { mapToVec(_n,zzmms_n,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_T,zzmms_T1D,*_y); }
  else { mapToVec(_T,zzmms_T,*_y,*_z); }

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcL,_bcR,_bcT, and _bcB
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_u,&viscSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&viscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxviscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&uSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxuSource); CHKERRQ(ierr);

  ierr = computeViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,zzmms_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,zzmms_gSource,*_y,*_z,time); }
  ierr = _sbp->H(viscSourceMMS,HxviscSourceMMS); CHKERRQ(ierr);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,zzmms_uSource1D,*_y,time); }
  else { mapToVec(uSource,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbp->H(uSource,HxuSource); CHKERRQ(ierr);
  VecDestroy(&uSource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,viscSource); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxviscSourceMMS); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxuSource); CHKERRQ(ierr);
  }

  ierr = VecAXPY(_rhs,1.0,viscSource); CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhs,1.0,HxviscSourceMMS); CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhs,1.0,HxuSource); CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&viscSource);
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u); CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp(); CHKERRQ(ierr);

  // set stresses
  ierr = computeTotalStrains(); CHKERRQ(ierr);
  ierr = computeStresses(); CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::forceMMSSolutions_u(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::forceMMSSolutions_u";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
  #endif

  // force u to be correct
  ierr = mapToVec(_u,zzmms_uA,*_y,*_z,time); CHKERRQ(ierr);

  // force viscous strains to be correct
  if (_Nz == 1) { mapToVec(_gxy,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gxy,zzmms_gxy,*_y,*_z,time); }
  if (_Nz == 1) { mapToVec(_gxz,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gxz,zzmms_gxz,*_y,*_z,time); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute source term for MMS test and add it to rhs vector
PetscErrorCode PowerLaw::addRHS_MMSSource(const PetscScalar time,Vec& rhs)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::addRHS_MMSSource";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_u,&viscSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&viscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxviscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&uSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxuSource); CHKERRQ(ierr);

  if (_Nz == 1) { mapToVec(viscSourceMMS,zzmms_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,zzmms_gSource,*_y,*_z,time); }
  ierr = _sbp->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,zzmms_uSource1D,*_y,time); }
  else { mapToVec(uSource,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbp->H(uSource,HxuSource);
  VecDestroy(&uSource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxviscSourceMMS); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxuSource); CHKERRQ(ierr);
  }

  ierr = VecAXPY(_rhs,1.0,HxviscSourceMMS); CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhs,1.0,HxuSource); CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute source terms for MMS test and add to viscous strain rates
PetscErrorCode PowerLaw::addViscStrainRates_MMSSource(const PetscScalar time,Vec& gVxy_t,Vec& gVxz_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::addViscStrainRates_MMSSource";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(_u,&source);
  if (_Nz == 1) { mapToVec(source,zzmms_pl_gxy_t_source1D,*_y,time); }
  else { mapToVec(source,zzmms_pl_gxy_t_source,*_y,*_z,time); }
  VecAXPY(gVxy_t,1.0,source);
  if (_Nz == 1) { mapToVec(source,zzmms_pl_gxz_t_source1D,*_y,time); }
  else { mapToVec(source,zzmms_pl_gxz_t_source,*_y,*_z,time); }
  VecAXPY(gVxz_t,1.0,source);
  VecDestroy(&source);

  // force rates to be correct
  //~ mapToVec(gVxy_t,zzmms_gxy_t,*_y,*_z,time);
  //~ mapToVec(gVxz_t,zzmms_gxz_t,*_y,*_z,time);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// limited by Maxwell time
PetscErrorCode PowerLaw::computeMaxTimeStep(PetscScalar& maxTimeStep)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeMaxTimeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec Tmax;
  VecDuplicate(_u,&Tmax);
  VecSet(Tmax,0.0);
  VecPointwiseDivide(Tmax,_effVisc,_muVec);
  PetscScalar min_Tmax;
  VecMin(Tmax,NULL,&min_Tmax);

  maxTimeStep = min_Tmax;

  VecDestroy(&Tmax);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // add deep copies of viscous strains to integrated variables, stored in _var
  if (varEx.find("gVxy") != varEx.end() ) { VecCopy(_gxy,varEx["gVxy"]); }
  else { Vec vargxyP; VecDuplicate(_u,&vargxyP); VecCopy(_gxy,vargxyP); varEx["gVxy"] = vargxyP; }

  if (varEx.find("gVxz") != varEx.end() ) { VecCopy(_gxz,varEx["gVxz"]); }
  else { Vec vargxzP; VecDuplicate(_u,&vargxzP); VecCopy(_gxz,vargxzP); varEx["gVxz"] = vargxzP; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // if integrating viscous strains in time
  VecCopy(varEx.find("gVxy")->second,_gxy);
  VecCopy(varEx.find("gVxz")->second,_gxz);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::updateTemperature()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(T,_T);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::setRHS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setRHS()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(_rhs,0.);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// solve momentum balance equation for u
PetscErrorCode PowerLaw::computeU()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::computeU";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  ierr = setSurfDisp();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::changeBCTypes(std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::changeBCTypes()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _sbp->changeBCTypes(bcRTtype,bcTTtype,bcLTtype,bcBTtype);
  KSPDestroy(&_ksp);
  Mat A; _sbp->getA(A);
  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP(A,_ksp,_pc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::computeViscStrainSourceTerms(Vec& out,Vec& gxy, Vec& gxz)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscStrainSourceTerms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz) + SAT
  ierr = MatMult(_B,_gxy,out); CHKERRQ(ierr);
  if (_Nz > 1) {
    ierr = MatMultAdd(_C,_gxz,out,out); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode PowerLaw::computeViscosity(const PetscScalar viscCap)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscosity";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // compute effective viscosity
  PetscScalar const *sigmadev,*A,*B,*n,*T=0;
  PetscScalar *effVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArrayRead(_sdev,&sigmadev);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&B);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(_T,&T);
  VecGetArray(_effVisc,&effVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar invEffVisc = 1e3 * A[Jj]*pow(sigmadev[Jj],n[Jj]-1.0)*exp(-B[Jj]/T[Jj]) + 1./viscCap;
    effVisc[Jj] = 1.0/invEffVisc;

    assert(~isnan(effVisc[Jj]));
    assert(~isinf(effVisc[Jj]));
    Jj++;
  }
  VecRestoreArrayRead(_sdev,&sigmadev);
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&B);
  VecRestoreArrayRead(_n,&n);
  VecRestoreArrayRead(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode PowerLaw::computeViscStrainRates(const PetscScalar time,const Vec& gVxy, const Vec& gVxz,
  Vec& gVxy_t, Vec& gVxz_t)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_gTxy,&SAT);
  ierr = computeViscousStrainRateSAT(_u,_bcL,_bcR,SAT); CHKERRQ(ierr);

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  VecPointwiseMult(gVxy_t,_muVec,SAT);
  VecAXPY(gVxy_t,1.0,_sxy);
  VecPointwiseDivide(gVxy_t,gVxy_t,_effVisc);

  if (_Nz > 1) {
    VecCopy(_sxz,gVxz_t);
    VecPointwiseDivide(gVxz_t,gVxz_t,_effVisc);
  }

  VecDestroy(&SAT);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode PowerLaw::computeViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscousStrainRateSAT";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecSet(out,0.0);

  Vec GL, GR, temp1;
  VecDuplicate(u,&GL); VecSet(GL,0.0);
  VecDuplicate(u,&GR); VecSet(GR,0.0);
  VecDuplicate(u,&temp1); VecSet(temp1,0.0);

  // left Dirichlet boundary
  if (_bcLType.compare("Dirichlet")==0) {
    ierr = _sbp->HyinvxE0y(u,temp1);CHKERRQ(ierr);
    ierr = _sbp->Hyinvxe0y(gL,GL);CHKERRQ(ierr);
    VecAXPY(out,1.0,temp1);
    VecAXPY(out,-1.0,GL);
  }

  // right Dirichlet boundary
  if (_bcRType.compare("Dirichlet")==0) {
    VecSet(temp1,0.0);
    ierr = _sbp->HyinvxENy(u,temp1);CHKERRQ(ierr);
    ierr = _sbp->HyinvxeNy(gR,GR);CHKERRQ(ierr);
    VecAXPY(out,-1.0,temp1);
    VecAXPY(out,1.0,GR);
  }

  VecDestroy(&GL);
  VecDestroy(&GR);
  VecDestroy(&temp1);

  // include effects of coordinate transform
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gxy,&temp1);
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,out,temp1);
    VecCopy(temp1,out);
    VecDestroy(&temp1);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes gTxy and gTxz
PetscErrorCode PowerLaw::computeTotalStrains()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeTotalStrains";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  _sbp->Dy(_u,_gTxy);
  if (_Nz > 1) {
    _sbp->Dz(_u,_gTxz);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes sigmaxy, sigmaxz, and sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::computeStresses()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeStresses";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(_gTxy,_sxy);
  VecAXPY(_sxy,-1.0,_gxy);
  VecPointwiseMult(_sxy,_sxy,_muVec);

  if (_Nz > 1) {
    VecCopy(_gTxz,_sxz);
    VecAXPY(_sxz,-1.0,_gxz);
    VecPointwiseMult(_sxz,_sxz,_muVec);
  }

  computeSDev();

  // force stresses to be correct for MMS test
  //~ mapToVec(_sxy,zzmms_pl_sigmaxy,*_y,*_z,time);
  //~ mapToVec(_sxz,zzmms_pl_sigmaxz,*_y,*_z,time);
  //~ mapToVec(_sdev,zzmms_sdev,*_y,*_z,time);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::computeSDev()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeStresses";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

PetscErrorCode PowerLaw::getStresses(Vec& sxy, Vec& sxz, Vec& sdev)
{
  sxy = _sxy;
  sxz = _sxz;
  sdev = _sdev;
  return 0;
}

PetscErrorCode PowerLaw::setSurfDisp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setSurfDisp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif


  PetscInt    Ii,Istart,Iend,y;
  PetscScalar u;
  ierr = VecGetOwnershipRange(_u,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    //~ z = Ii-_Nz*(Ii/_Nz);
    y = Ii / _Nz;
    if (Ii % _Nz == 0) {
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ii = %i, y = %i, z = %i\n",Ii,y,z);
      ierr = VecGetValues(_u,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDisp,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDisp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDisp);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



//======================================================================
// Steady state functions
//======================================================================



// inititialize effective viscosity
PetscErrorCode PowerLaw::guessSteadyStateEffVisc(const PetscScalar strainRate)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::guessSteadyStateEffVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar s=0.;
  PetscScalar *A,*B,*n,*T,*effVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArray(_A,&A);
  VecGetArray(_QR,&B);
  VecGetArray(_n,&n);
  VecGetArray(_T,&T);
  VecGetArray(_effVisc,&effVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    s = pow( strainRate/ (A[Jj]*exp(-B[Jj]/T[Jj]) ), 1.0/n[Jj] + 1.0/_effViscCap);
    effVisc[Jj] =  s/strainRate * 1e-3; // (GPa s)  in terms of strain rate
    Jj++;
  }
  VecRestoreArray(_A,&A);
  VecRestoreArray(_QR,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute Bss and Css
PetscErrorCode PowerLaw::initializeSSMatrices(std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initializeSSMatrices";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up SBP operators
  if (_sbpType.compare("mc")==0) {
    _sbp_eta = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp_eta = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp_eta = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
    _sbp_eta->setGrid(_y,_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp_eta->setBCTypes(bcRType,bcTType,bcLType,bcBType);
  _sbp_eta->setMultiplyByH(1);
  _sbp_eta->computeMatrices(); // actually create the matrices

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::setSSRHS(map<string,Vec>& varSS,std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setSSRHS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(_rhs,0.);

  delete _sbp_eta;
  initializeSSMatrices(bcRType,bcTType,bcLType,bcBType);

  ierr = _sbp_eta->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// put viscous strains etc in varSS
PetscErrorCode PowerLaw::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initiateVarSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  //~ VecDuplicate(_u,&varSS["v"]); VecSet(varSS["v"],0.);
  //~ VecDuplicate(_u,&varSS["gVxy_t"] ); VecSet(varSS["gVxy_t"] ,0.);
  //~ VecDuplicate(_u,&varSS["gVxz_t"]); VecSet(varSS["gVxz_t"],0.);
  varSS["effVisc"] = _effVisc;
  varSS["sDev"] = _sdev;
  varSS["sxy"] = _sxy; // included so it'll be written out
  varSS["sxz"] = _sxz; // included so it'll be written out
  varSS["u"] = _u; // included so it'll be written out
  varSS["gxy"] = _gxy; // included so it'll be written out
  varSS["gxz"] = _gxz; // included so it'll be written out

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


// solve for steady-state v, viscous strain rates
PetscErrorCode PowerLaw::updateSSa(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::updateSSa";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up linear system
  //~ VecScale(_rhs,_ssEffViscScale); // scale rhs to improve condition number of linear system
  Mat A;
  _sbp_eta->getA(A);
  KSPDestroy(&_ksp_eta);
  KSPCreate(PETSC_COMM_WORLD,&_ksp_eta);
  setupKSP_SSIts(A,_ksp_eta,_pc_eta);

  // solve for steady-state velocity
  ierr = KSPSolve(_ksp_eta,_rhs,varSS["v"]);CHKERRQ(ierr);

  // update viscous strain rates
  _sbp_eta->Dy(varSS["v"],varSS["gVxy_t"]);
  _sbp_eta->Dz(varSS["v"],varSS["gVxz_t"]);

  // update stresses
  ierr = VecPointwiseMult(_sxy,_effVisc,varSS["gVxy_t"]);
  ierr = VecPointwiseMult(_sxz,_effVisc,varSS["gVxz_t"]);
  ierr = computeSDev(); CHKERRQ(ierr); // deviatoric stress

  // update effective viscosity
  ierr = computeViscosity(_effViscCap); CHKERRQ(ierr); // new viscosity


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

// solve for steady-state u, gVxy
PetscErrorCode PowerLaw::updateSSb(map<string,Vec>& varSS,const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::updateSSb";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // compute u
  //~ PetscScalar time = 1.0;
  VecCopy(varSS["v"],_u);
  VecScale(_u,time);

  // make u always positive
  PetscScalar minVal = 0;
  VecMin(_u,NULL,&minVal);
  if (minVal < 0) {
    minVal = abs(minVal);
    Vec temp;
    VecDuplicate(_u,&temp);
    VecSet(temp,minVal);
    VecAXPY(_u,1.,temp);
    VecDestroy(&temp);
  }


  PetscScalar *mu,*gVxy_t,*gVxz_t,*gxy,*gxz,*sxy,*sxz=0;
  PetscInt Istart, Iend;
  VecGetOwnershipRange(_sxy,&Istart,&Iend);
  VecGetArray(_muVec,&mu);
  VecGetArray(_sxy,&sxy);
  VecGetArray(_sxz,&sxz);
  VecGetArray(_gxy,&gxy);
  VecGetArray(_gxz,&gxz);
  VecGetArray(varSS["gVxy_t"],&gVxy_t);
  VecGetArray(varSS["gVxz_t"],&gVxz_t);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar gVxy0 = -sxy[Jj]/mu[Jj];
    PetscScalar gVxz0 = -sxz[Jj]/mu[Jj];
    gxy[Jj] = gVxy_t[Jj] * time + gVxy0;
    gxz[Jj] = gVxz_t[Jj] * time + gVxz0;
    Jj++;
  }
  VecRestoreArray(_muVec,&mu);
  VecRestoreArray(_sxy,&sxy);
  VecRestoreArray(_sxz,&sxz);
  VecRestoreArray(_gxy,&gxy);
  VecRestoreArray(_gxz,&gxz);
  VecRestoreArray(varSS["gVxy_t"],&gVxy_t);
  VecRestoreArray(varSS["gVxz_t"],&gVxz_t);

  //~ _viewers["SS_gVxy"] = initiateViewer(_outputDir + "SS_gVxy");
  //~ ierr = VecView(_gxy,_viewers["SS_gVxy"]); CHKERRQ(ierr);
  //~ _viewers["SS_gVxz"] = initiateViewer(_outputDir + "SS_gVxz");
  //~ ierr = VecView(_gxz,_viewers["SS_gVxz"]); CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setVecFromVectors";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    //~ z = _dz*(Ii-_Nz*(Ii/_Nz));
    VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
    //~PetscPrintf(PETSC_COMM_WORLD,"1: Ii = %i, z = %g\n",Ii,z);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
        z0 = depths[0+ind];
        z1 = depths[0+ind+1];
        v0 = vals[0+ind];
        v1 = vals[0+ind+1];
        if (z>=z0 && z<=z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }
        ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}




//======================================================================
// IO functions
//======================================================================

// Save all scalar fields to text file named pl_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode PowerLaw::writeDomain(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = outputDir + "momBal_context.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"viscDistribution = %s\n",_viscDistribution.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"effViscCap = %.15e\n",_effViscCap);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ssEffViscScale = %.15e\n",_ssEffViscScale);CHKERRQ(ierr);


  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  writeDomain(outputDir);

  ierr = writeVec(_muVec,outputDir + "momBal_mu"); CHKERRQ(ierr);
  ierr = writeVec(_A,outputDir + "momBal_A"); CHKERRQ(ierr);
  ierr = writeVec(_QR,outputDir + "momBal_QR"); CHKERRQ(ierr);
  ierr = writeVec(_n,outputDir + "momBal_n"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep1D(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();
  _stepCount = stepCount;

  if (stepCount == 0) {
    //~ ierr = _sbp->writeOps(outputDir + "ops_u_"); CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = io_initiateWriteAppend(_viewers, "surfDisp", _surfDisp, outputDir + "surfDisp"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcL", _bcL, outputDir + "momBal_bcL"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcR", _bcR, outputDir + "momBal_bcR"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcRShift", _bcRShift, outputDir + "momBal_bcRShift"); CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_surfDisp,_viewers["surfDisp"].first); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["bcL"].first); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["bcR"].first); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep2D(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (stepCount == 0) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = io_initiateWriteAppend(_viewers, "u", _u, outputDir + "u"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "sxy", _sxy, outputDir + "sxy"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "sxz", _sxz, outputDir + "sxz"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gTxy", _gTxy, outputDir + "gTxy"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gTxz", _gTxz, outputDir + "gTxz"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gxy", _gxy, outputDir + "gxy"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gxz", _gxz, outputDir + "gxz"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "effVisc", _effVisc, outputDir + "effVisc"); CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_u,_viewers["u"].first); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["sxy"].first); CHKERRQ(ierr);
    ierr = VecView(_gTxy,_viewers["gTxy"].first); CHKERRQ(ierr);
    ierr = VecView(_gxy,_viewers["gxy"].first); CHKERRQ(ierr);
    ierr = VecView(_effVisc,_viewers["effVisc"].first); CHKERRQ(ierr);
    ierr = VecView(_gTxz,_viewers["gTxz"].first); CHKERRQ(ierr);
    ierr = VecView(_gxz,_viewers["gxz"].first); CHKERRQ(ierr);
    ierr = VecView(_sxz,_viewers["sxz"].first); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Power Law Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Ny = %i, Nz = %i\n",_Ny,_Nz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   solver algorithm = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.);CHKERRQ(ierr);

  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   misc time (s): %g\n",_miscTime);CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% misc time: %g\n",_miscTime/_integrateTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}



// MMS functions



PetscErrorCode PowerLaw::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  string funcName = "PowerLaw::setMMSBoundaryConditions";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);
  if (_Nz == 1) {
    Ii = Istart;
    y = 0;
    if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("Neumann")) { v = zzmms_mu1D(y) * (zzmms_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("Neumann")) { v = zzmms_mu1D(y) * (zzmms_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  else {
    for(Ii=Istart;Ii<Iend;Ii++) {
      //~ z = _dz * Ii;
      ierr = VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);

      y = 0;
      if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=0,z)
      else if (!_bcLType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_y(y,z,time)- zzmms_gxy(y,z,time));} // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      y = _Ly;
      if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=Ly,z)
      else if (!_bcRType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_y(y,z,time)- zzmms_gxy(y,z,time)); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
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
    else if (!_bcTType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time) - zzmms_gxz(y,z,time)); }
    //~ else if (!_bcTType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcT,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);

    z = _Lz;
    if (!_bcBType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y,z=Lz)
    else if (!_bcBType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time) - zzmms_gxz(y,z,time));}
    //~ else if (!_bcBType.compare("Neumann")) { v = zzmms_mu(y,z) * (zzmms_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcB,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcT); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcB); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcT); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcB); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;

  // measure error between analytical and numerical solution
  Vec uA,gxyA,gxzA;
  VecDuplicate(_u,&uA);
  VecDuplicate(_u,&gxyA);
  VecDuplicate(_u,&gxzA);

  if (_Nz == 1) { mapToVec(uA,zzmms_uA1D,*_y,time); }
  else { mapToVec(uA,zzmms_uA,*_y,*_z,time); }
    if (_Nz == 1) { mapToVec(gxyA,zzmms_gxy1D,*_y,time); }
  else { mapToVec(gxyA,zzmms_gxy,*_y,*_z,time); }
  if (_Nz == 1) { mapToVec(gxzA,zzmms_gxy1D,*_y,time); }
  else { mapToVec(gxzA,zzmms_gxz,*_y,*_z,time); }

  writeVec(uA,_outputDir+"mms_uA");
  writeVec(gxyA,_outputDir+"mms_gxyA");
  writeVec(gxzA,_outputDir+"mms_gxzA");
  writeVec(_bcL,_outputDir+"mms_bcL");
  writeVec(_bcR,_outputDir+"mms_bcR");
  writeVec(_bcT,_outputDir+"mms_bcT");
  writeVec(_bcB,_outputDir+"mms_bcB");

  double err2u = computeNormDiff_2(_u,uA);
  double err2epsxy = computeNormDiff_2(_gxy,gxyA);
  double err2epsxz = computeNormDiff_2(_gxz,gxzA);

  PetscPrintf(PETSC_COMM_WORLD,"%i %3i %.4e %.4e % .15e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u),err2epsxy,log2(err2epsxy),err2epsxz,log2(err2epsxz));

  VecDestroy(&uA);
  VecDestroy(&gxyA);
  VecDestroy(&gxzA);
  return ierr;
}


double PowerLaw::zzmms_sigmaxz(const double y,const double z, const double t)
{ return zzmms_mu(y,z)*zzmms_uA_z(y,z,t); }
double PowerLaw::zzmms_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double PowerLaw::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double PowerLaw::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double PowerLaw::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double PowerLaw::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }

double PowerLaw::zzmms_g(const double t) { return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9); }
double PowerLaw::zzmms_g_t(const double t) {
  return (-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9);
}

double PowerLaw::zzmms_uA(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double PowerLaw::zzmms_uA_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double PowerLaw::zzmms_uA_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double PowerLaw::zzmms_uA_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double PowerLaw::zzmms_uA_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
//~ double PowerLaw::zzmms_uA_t(const double y,const double z,const double t) {
  //~ return zzmms_f(y,z)*((-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9));
//~ }
double PowerLaw::zzmms_uA_t(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g_t(t);
}

double PowerLaw::zzmms_mu(const double y,const double z) { return sin(y)*sin(z) + 30; }
double PowerLaw::zzmms_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double PowerLaw::zzmms_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double PowerLaw::zzmms_sigmaxy(const double y,const double z,const double t)
{ return zzmms_mu(y,z)*zzmms_uA_y(y,z,t); }

double PowerLaw::zzmms_uSource(const double y,const double z,const double t)
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
double PowerLaw::zzmms_f1D(const double y) { return cos(y) + 2; } // helper function for uA
double PowerLaw::zzmms_f_y1D(const double y) { return -sin(y); }
double PowerLaw::zzmms_f_yy1D(const double y) { return -cos(y); }
//~ double PowerLaw::zzmms_f_z1D(const double y) { return 0; }
//~ double PowerLaw::zzmms_f_zz1D(const double y) { return 0; }

double PowerLaw::zzmms_uA1D(const double y,const double t) { return zzmms_f1D(y)*exp(-t); }
double PowerLaw::zzmms_uA_y1D(const double y,const double t) { return zzmms_f_y1D(y)*exp(-t); }
double PowerLaw::zzmms_uA_yy1D(const double y,const double t) { return zzmms_f_yy1D(y)*exp(-t); }
double PowerLaw::zzmms_uA_z1D(const double y,const double t) { return 0; }
double PowerLaw::zzmms_uA_zz1D(const double y,const double t) { return 0; }
double PowerLaw::zzmms_uA_t1D(const double y,const double t) { return -zzmms_f1D(y)*exp(-t); }

double PowerLaw::zzmms_mu1D(const double y) { return sin(y) + 2.0; }
double PowerLaw::zzmms_mu_y1D(const double y) { return cos(y); }
//~ double PowerLaw::zzmms_mu_z1D(const double y) { return 0; }

double PowerLaw::zzmms_sigmaxy1D(const double y,const double t) { return zzmms_mu1D(y)*zzmms_uA_y1D(y,t); }
double PowerLaw::zzmms_uSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar u_y = zzmms_uA_y1D(y,t);
  PetscScalar u_yy = zzmms_uA_yy1D(y,t);
  PetscScalar u_zz = zzmms_uA_zz1D(y,t);
  return mu*(u_yy + u_zz) + mu_y*u_y;
}




// power-law specific MMS functions
double PowerLaw::zzmms_visc(const double y,const double z) { return cos(y)*cos(z) + 2e10; }
double PowerLaw::zzmms_invVisc(const double y,const double z) { return 1.0/zzmms_visc(y,z); }
double PowerLaw::zzmms_invVisc_y(const double y,const double z)
{ return sin(y)*cos(z)/pow( cos(y)*cos(z)+2e10, 2.0); }
double PowerLaw::zzmms_invVisc_z(const double y,const double z)
{ return cos(y)*sin(z)/pow( cos(y)*cos(z)+2e10 ,2.0); }

double PowerLaw::zzmms_gxy(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fy = zzmms_f_y(y,z);
  //~ return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fy/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fy/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double PowerLaw::zzmms_gxy_y(const double y,const double z,const double t)
{
  //~return 0.5 * zzmms_uA_yy(y,z,t);
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double Ay = zzmms_mu_y(y,z)*zzmms_invVisc(y,z) + zzmms_mu(y,z)*zzmms_invVisc_y(y,z);
  double fy = zzmms_f_y(y,z);
  double fyy = zzmms_f_yy(y,z);

  double T1 = 60, T2 = 3e7, T3 = 3e9;
  double d1 = T1*A-1, d2 = T2*A-1, d3 = T3*A-1;
  double out1 = -pow(T1,2.0)*A*Ay*fy/pow(d1,2.0)*(exp(-t/T1)-exp(-A*t))  + T1*fy*Ay/d1 *(exp(-t/T1)-exp(-A*t))
      +T1*A*Ay*fy*exp(-A*t)*t/d1 + T1*A*fyy/d1*(exp(-t/T1)-exp(-A*t));
  double out2 = pow(T2,2.0)*A*Ay*fy/pow(d2,2.0)*(exp(-t/T2)-exp(-A*t)) - T2*fy*Ay/d2 *(exp(-t/T2)-exp(-A*t))
       -T2*A*Ay*fy*exp(-A*t)*t/d2 - T2*A*fyy/d2*(exp(-t/T2)-exp(-A*t));
  double out3 = -pow(T3,2.0)*A*Ay*fy/pow(d3,2.0)*(exp(-t/T3)-exp(-A*t))  + T3*fy*Ay/d3 *(exp(-t/T3)-exp(-A*t))
       +T3*A*Ay*fy*exp(-A*t)*t/d3 + T3*A*fyy/d3*(exp(-t/T3)-exp(-A*t));
  return out1 + out2 + out3;

}
double PowerLaw::zzmms_gxy_t(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fy = zzmms_f_y(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*((-1.0/T1)*exp(-t/T1)+A*exp(-A*t))
       - T2*A*fy/(T2*A-1)*((-1.0/T2)*exp(-t/T2)+A*exp(-A*t))
       + T3*A*fy/(T3*A-1)*((-1.0/T3)*exp(-t/T3)+A*exp(-A*t));
}

double PowerLaw::zzmms_gxz(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fz = zzmms_f_z(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fz/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fz/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fz/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double PowerLaw::zzmms_gxz_z(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double Az = zzmms_mu_z(y,z)*zzmms_invVisc(y,z) + zzmms_mu(y,z)*zzmms_invVisc_z(y,z);
  double fz = zzmms_f_z(y,z);
  double fzz = zzmms_f_zz(y,z);
  //~ double den = A-1.0, B = exp(-t)-exp(-A*t);
  //~ return t*A*Az*fz*exp(-A*t)/den - A*fz*Az*B/pow(den,2.0) + fz*Az*B/den + A*fzz*B/den;

  double T1 = 60, T2 = 3e7, T3 = 3e9;
  double d1 = T1*A-1, d2 = T2*A-1, d3 = T3*A-1;
  double out1 = -pow(T1,2.0)*A*Az*fz/pow(d1,2.0)*(exp(-t/T1)-exp(-A*t))  + T1*fz*Az/d1 *(exp(-t/T1)-exp(-A*t))
      +T1*A*Az*fz*exp(-A*t)*t/d1 + T1*A*fzz/d1*(exp(-t/T1)-exp(-A*t));
  double out2 = pow(T2,2.0)*A*Az*fz/pow(d2,2.0)*(exp(-t/T2)-exp(-A*t)) - T2*fz*Az/d2 *(exp(-t/T2)-exp(-A*t))
       -T2*A*Az*fz*exp(-A*t)*t/d2 - T2*A*fzz/d2*(exp(-t/T2)-exp(-A*t));
  double out3 = -pow(T3,2.0)*A*Az*fz/pow(d3,2.0)*(exp(-t/T3)-exp(-A*t))  + T3*fz*Az/d3 *(exp(-t/T3)-exp(-A*t))
       +T3*A*Az*fz*exp(-A*t)*t/d3 + T3*A*fzz/d3*(exp(-t/T3)-exp(-A*t));
  return out1 + out2 + out3;
}
double PowerLaw::zzmms_gxz_t(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fz = zzmms_f_z(y,z);
  //~ return A*fz/(A-1.0)*(-exp(-t) + A*exp(-A*t));
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fz/(T1*A-1)*((-1.0/T1)*exp(-t/T1)+A*exp(-A*t))
       - T2*A*fz/(T2*A-1)*((-1.0/T2)*exp(-t/T2)+A*exp(-A*t))
       + T3*A*fz/(T3*A-1)*((-1.0/T3)*exp(-t/T3)+A*exp(-A*t));
}

// source terms for viscous strain rates
double PowerLaw::zzmms_max_gxy_t_source(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double uy = zzmms_uA_y(y,z,t);
  double g = zzmms_gxy(y,z,t);

  return zzmms_gxy_t(y,z,t) - A*(uy - g);
}
double PowerLaw::zzmms_max_gxz_t_source(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double uz = zzmms_uA_z(y,z,t);
  double g = zzmms_gxz(y,z,t);

  return zzmms_gxz_t(y,z,t) - A*(uz - g);
}

double PowerLaw::zzmms_gSource(const double y,const double z,const double t)
{
  PetscScalar mu = zzmms_mu(y,z);
  PetscScalar mu_y = zzmms_mu_y(y,z);
  PetscScalar mu_z = zzmms_mu_z(y,z);
  PetscScalar gxy = zzmms_gxy(y,z,t);
  PetscScalar gxz = zzmms_gxz(y,z,t);
  PetscScalar gxy_y = zzmms_gxy_y(y,z,t);
  PetscScalar gxz_z = zzmms_gxz_z(y,z,t);
  return -mu*(gxy_y + gxz_z) - mu_y*gxy - mu_z*gxz; // full answer
}

double PowerLaw::zzmms_A(const double y,const double z) { return cos(y)*cos(z) + 398; }
double PowerLaw::zzmms_B(const double y,const double z) { return sin(y)*sin(z) + 4.28e4; }
double PowerLaw::zzmms_T(const double y,const double z) { return sin(y)*cos(z) + 800; }
double PowerLaw::zzmms_n(const double y,const double z) { return cos(y)*sin(z) + 3.0; }
double PowerLaw::zzmms_pl_sigmaxy(const double y,const double z,const double t) { return zzmms_mu(y,z)*(zzmms_uA_y(y,z,t) - zzmms_gxy(y,z,t)); }
double PowerLaw::zzmms_pl_sigmaxz(const double y,const double z, const double t) { return zzmms_mu(y,z)*(zzmms_uA_z(y,z,t) - zzmms_gxz(y,z,t)); }
double PowerLaw::zzmms_sdev(const double y,const double z,const double t)
{
  return sqrt( pow(zzmms_pl_sigmaxy(y,z,t),2.0) + pow(zzmms_pl_sigmaxz(y,z,t),2.0) );
}


// source terms for viscous strain rates
double PowerLaw::zzmms_pl_gxy_t_source(const double y,const double z,const double t)
{
  double A = zzmms_A(y,z);
  double B = zzmms_B(y,z);
  double n = zzmms_n(y,z);
  double T = zzmms_T(y,z);
  double sigmadev = zzmms_sdev(y,z,t) * 1.0;
  double sigmaxy = zzmms_pl_sigmaxy(y,z,t);
  double effVisc = 1.0/( A*pow(sigmadev,n-1.0)*exp(-B/T) ) * 1e-3;
  double v = sigmaxy/effVisc;

  return zzmms_gxy_t(y,z,t) - v;
}
double PowerLaw::zzmms_pl_gxz_t_source(const double y,const double z,const double t)
{
  double A = zzmms_A(y,z);
  double B = zzmms_B(y,z);
  double n = zzmms_n(y,z);
  double T = zzmms_T(y,z);
  double sigmadev = zzmms_sdev(y,z,t);
  double sigmaxz = zzmms_pl_sigmaxz(y,z,t);
  double effVisc = 1.0/( A*pow(sigmadev,n-1.0)*exp(-B/T) ) * 1e-3;
  double v = sigmaxz/effVisc;

  return zzmms_gxz_t(y,z,t) - v;
}


double PowerLaw::zzmms_visc1D(const double y) { return cos(y) + 20.0; }
double PowerLaw::zzmms_invVisc1D(const double y) { return 1.0/(cos(y) + 20.0); }
double PowerLaw::zzmms_invVisc_y1D(const double y) { return sin(y)/pow( cos(y)+20.0, 2.0); }
double PowerLaw::zzmms_invVisc_z1D(const double y) { return 0; }

double PowerLaw::zzmms_gxy1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double fy = zzmms_f_y1D(y);
  return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
}
double PowerLaw::zzmms_gxy_y1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double Ay = zzmms_mu_y1D(y)*zzmms_invVisc1D(y) + zzmms_mu1D(y)*zzmms_invVisc_y1D(y);
  double fy = zzmms_f_y1D(y);
  double fyy = zzmms_f_yy1D(y);
  double den = A-1.0, B = exp(-t)-exp(-A*t);
  return t*A*Ay*fy*exp(-A*t)/den - A*fy*Ay*B/pow(den,2.0) + fy*Ay*B/den + A*fyy*B/den;
}
double PowerLaw::zzmms_gxy_t1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double fy = zzmms_f_y1D(y);
  return A*fy*(-exp(-t) + A*exp(-A*t))/(A-1.0);
}

double PowerLaw::zzmms_gSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar gxy = zzmms_gxy1D(y,t);
  PetscScalar gxy_y = zzmms_gxy_y1D(y,t);
  return -mu*gxy_y - mu_y*gxy;
}



// specific to power law
double PowerLaw::zzmms_A1D(const double y) { return cos(y) + 1e-9; }
double PowerLaw::zzmms_B1D(const double y) { return sin(y) + 1.44e4; }
double PowerLaw::zzmms_T1D(const double y) { return sin(y) + 600; }
double PowerLaw::zzmms_n1D(const double y) { return cos(y) + 3.0; }
double PowerLaw::zzmms_pl_sigmaxy1D(const double y,const double t)
{ return zzmms_mu1D(y)*(zzmms_uA_y1D(y,t) - zzmms_gxy1D(y,t)); }
double PowerLaw::zzmms_pl_sigmaxz1D(const double y,const double t) { return 0; }
double PowerLaw::zzmms_sdev1D(const double y,const double t)
{ return sqrt( pow(zzmms_pl_sigmaxy1D(y,t),2.0)); }


// source terms for viscous strain rates
double PowerLaw::zzmms_pl_gxy_t_source1D(const double y,const double t)
{
  double A = zzmms_A1D(y);
  double B = zzmms_B1D(y);
  double n = zzmms_n1D(y);
  double T = zzmms_T1D(y);
  double sigmadev = zzmms_sdev1D(y,t);
  double sigmaxy = zzmms_pl_sigmaxy1D(y,t);
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxy*1e-3;

  return zzmms_gxy_t1D(y,t) - v;
}
double PowerLaw::zzmms_pl_gxz_t_source1D(const double y,const double t)
{
  double A = zzmms_A1D(y);
  double B = zzmms_B1D(y);
  double n = zzmms_n1D(y);
  double T = zzmms_T1D(y);
  double sigmadev = zzmms_sdev1D(y,t);
  double sigmaxz = zzmms_pl_sigmaxz1D(y,t);
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxz*1e-3;

  //~ return zzmms_gxz_t1D(y,t) - v;
  return  - v;
}

