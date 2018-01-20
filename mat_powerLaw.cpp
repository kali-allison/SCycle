#include "mat_powerLaw.hpp"

#define FILENAME "mat_powerLaw.cpp"


Mat_PowerLaw::Mat_PowerLaw(Domain& D,HeatEquation& he,std::string bcRTtype,std::string bcTTtype,std::string bcLType,std::string bcBType)
: _file(D._file),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_loadICs(D._loadICs),
  _currTime(D._initTime),_stepCount(0),
  _muVec(NULL),_rhoVec(NULL),_cs(NULL),_muVal(30.0),_rhoVal(3.0),
  _viscDistribution("unspecified"),_AFile("unspecified"),_BFile("unspecified"),_nFile("unspecified"),
  _A(NULL),_n(NULL),_QR(NULL),_T(NULL),_effVisc(NULL),_effViscCap(1e30),
  _linSolver("unspecified"),_ksp(NULL),_pc(NULL),
  _kspTol(1e-10),
  _sbp(NULL),_sbpType(D._sbpType),
  _B(NULL),_C(NULL),_A2(NULL),
  _sbp_eta(NULL),_ksp_eta(NULL),_pc_eta(NULL),_ssEffViscScale(1e-15),
  _timeV1D(NULL),_timeV2D(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),_linSolveCount(0),
  _u(NULL),_sxy(NULL),_sxz(NULL),_sdev(NULL),
  _gxy(NULL),_dgxy(NULL),
  _gxz(NULL),_dgxz(NULL),
  _gTxy(NULL),_gTxz(NULL),
  _bcRType(bcRTtype),_bcTType(bcTTtype),_bcLType(bcLTtype),_bcBType(bcBTtype),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::Mat_PowerLaw";
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

  if (_inputDir.compare("unspecified") != 0) {
    loadFieldsFromFiles(); // load from previous simulation
  }

  computeTotalStrains(_currTime);
  computeStresses(_currTime);
  computeViscosity(_effViscCap);

  if (_isMMS) { setMMSInitialConditions(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

Mat_PowerLaw::~Mat_PowerLaw()
{
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::~Mat_PowerLaw";
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
PetscErrorCode Mat_PowerLaw::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::loadSettings()";
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

    else if (var.compare("linSolver")==0) {
      _momBalType = line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("muPlus")==0) {
      _muVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("rhoPlus")==0) {
      _rhoVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // viscosity
    else if (var.compare("viscDistribution")==0) {
      _viscDistribution = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // names of each field's source file
    else if (var.compare("AFile")==0) {
      _AFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("BFile")==0) {
      _BFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("nFile")==0) {
      _nFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // if values are set by a vector
    else if (var.compare("AVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_AVals);
    }
    else if (var.compare("ADepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_ADepths);
    }
    else if (var.compare("BVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BVals);
    }
    else if (var.compare("BDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BDepths);
    }
    else if (var.compare("nVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nVals);
    }
    else if (var.compare("nDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nDepths);
    }

    // cap on viscosity
    else if (var.compare("maxEffVisc")==0) {
      _effViscCap = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("ssEffViscScale")==0) {
      _ssEffViscScale = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode Mat_PowerLaw::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::checkInput";
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

  assert(_viscDistribution.compare("layered")==0 ||
      _viscDistribution.compare("mms")==0 ||
      _viscDistribution.compare("loadFromFile")==0 ||
      _viscDistribution.compare("effectiveVisc")==0 );

  if (_viscDistribution.compare("loadFromFile")==0) { assert(_inputDir.compare("unspecified")); }

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
PetscErrorCode Mat_PowerLaw::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::allocateFields";
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
  VecSet(_bcR,_vL*_currTime/2.0);


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

  ierr = VecDuplicate(_u,&_effViscTemp);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// set off-fault material properties
PetscErrorCode Mat_PowerLaw::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecSet(_muVec,_muVal);
  VecSet(_rhoVec,_rhoVal);
  VecPointwiseDivide(_cs, _muVec, _rhoVec);

  PetscScalar *cs;
  VecGetArray(_cs,&cs);
  VecSqrtAbs(_cs);

  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_muVec,zzmms_mu1D,*_y); }
    else { mapToVec(_muVec,zzmms_mu,*_y,*_z); }
  }


  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_A,_AVals[0]);
    VecSet(_QR,_BVals[0]);
    VecSet(_n,_nVals[0]);
  }
  else {
    if (_viscDistribution.compare("mms")==0) {
      if (_Nz == 1) { mapToVec(_A,zzmms_A1D,*_y); }
      else { mapToVec(_A,zzmms_A,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_QR,zzmms_B1D,*_y); }
      else { mapToVec(_QR,zzmms_B,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_n,zzmms_n1D,*_y); }
      else { mapToVec(_n,zzmms_n,*_y,*_z); }
    }
    else if (_viscDistribution.compare("loadFromFile")==0) { loadEffViscFromFiles(); }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_QR,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_n,_nVals,_nDepths);CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}


//parse input file and load values into data members
PetscErrorCode Mat_PowerLaw::loadEffViscFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::loadEffViscFromFiles()";
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
PetscErrorCode Mat_PowerLaw::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // load u
  ierr = loadVecFromInputFile(_u,_inputDir,"u"); CHKERRQ(ierr);

  // load shear modulus
  ierr = loadVecFromInputFile(_muVec,_inputDir,"mu"); CHKERRQ(ierr);

  // load bcL and bcR
  ierr = loadVecFromInputFile(_bcL,_inputDir,"bcL"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.0);

  // load viscous strains
  ierr = loadVecFromInputFile(_gxy,_inputDir,"Gxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_gxz,_inputDir,"Gxz"); CHKERRQ(ierr);

  // load stresses
  ierr = loadVecFromInputFile(_sxy,_inputDir,"Sxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_sxz,_inputDir,"Sxz"); CHKERRQ(ierr);

  // load effective viscosity
  ierr = loadVecFromInputFile(_effVisc,_inputDir,"EffVisc"); CHKERRQ(ierr);

  // load temperature
  ierr = loadVecFromInputFile(_T,_inputDir,"T"); CHKERRQ(ierr);

  // load power law parameters
  ierr = loadVecFromInputFile(_A,_inputDir,"A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"B"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"n"); CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set up SBP operators
PetscErrorCode Mat_PowerLaw::setUpSBPContext(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::setUpSBPContext";
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
PetscErrorCode Mat_PowerLaw::setupKSP(Mat& A,KSP& ksp,PC& pc)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr); // necessary for solving steady state power law
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

    //~ PetscOptionsSetValue(NULL,"-pc_hypre_boomeramg_agg_nl 1");
  }
  else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCLU);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
  }
  else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCCHOLESKY);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
  }
  else if (_linSolver.compare("PCG")==0) { // preconditioned conjugate gradient
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0);
  }

  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute B and C
// B = H*Dy*mu + SAT terms
// C = H*Dz*mu + SAT terms
PetscErrorCode Mat_PowerLaw::initializeMomBalMats()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::initializeMomBalMats";
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

// compute Bss and Css
PetscErrorCode Mat_PowerLaw::initializeSSMatrices()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::initializeSSMatrices";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up SBP operators
  std::string bcRType = "Dirichlet";
  std::string bcTType = "Neumann";
  std::string bcLType = "Neumann";
  std::string bcBType = "Neumann";
  if (_sbpType.compare("mc")==0) {
    _sbp_eta = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp_eta = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    //~ _sbp_eta = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
    VecCopy(_effVisc,_effViscTemp); VecScale(_effViscTemp,_ssEffViscScale);
    _sbp_eta = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_effViscTemp);
    _sbp_eta->setGrid(_y,_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp_eta->setBCTypes(bcRType,bcTType,bcLType,bcBType);
  _sbp_eta->setMultiplyByH(1);
  _sbp_eta->computeMatrices(); // actually create the matrices

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// inititialize effective viscosity
PetscErrorCode Mat_PowerLaw::guessSteadyStateEffVisc(const PetscScalar strainRate)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::guessSteadyStateEffVisc";
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
    s = pow( strainRate/ (A[Jj]*exp(-B[Jj]/T[Jj]) ), 1.0/n[Jj] );
    effVisc[Jj] =  s/strainRate * 1e-3; // (GPa s)  in terms of strain rate
    Jj++;
  }
  VecRestoreArray(_A,&A);
  VecRestoreArray(_QR,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// for steady state computations
// compute initial tauVisc (from guess at effective viscosity)
PetscErrorCode Mat_PowerLaw::getTauVisc(Vec& tauVisc, const PetscScalar ess_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::getTauVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // get steady state effective viscosity
  guessSteadyStateEffVisc(ess_t);

  // use effective viscosity to compute strength of off-fault material
  if (tauVisc == NULL) { VecDuplicate(_bcL,&tauVisc); }

  // first get viscosity just on fault
  PetscInt Istart,Iend;
  PetscScalar v = 0;
  ierr = VecGetOwnershipRange(_effVisc,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_Nz) {
      ierr = VecGetValues(_effVisc,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(tauVisc,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(tauVisc);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(tauVisc);CHKERRQ(ierr);

  VecScale(tauVisc,ess_t);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode Mat_PowerLaw::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::setMMSInitialConditions()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
  #endif

  PetscScalar time = _currTime;
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

  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,zzmms_gSource1D,*_y,_currTime); }
  else { mapToVec(viscSourceMMS,zzmms_gSource,*_y,*_z,_currTime); }
  ierr = _sbp->H(viscSourceMMS,HxviscSourceMMS); CHKERRQ(ierr);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,zzmms_uSource1D,*_y,_currTime); }
  else { mapToVec(uSource,zzmms_uSource,*_y,*_z,_currTime); }
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
  ierr = computeTotalStrains(time); CHKERRQ(ierr);
  ierr = computeStresses(time); CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// limited by Maxwell time
PetscErrorCode Mat_PowerLaw::computeMaxTimeStep(PetscScalar& maxTimeStep)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeMaxTimeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec Tmax;
  VecDuplicate(_u,&Tmax);
  VecSet(Tmax,0.0);
  VecPointwiseDivide(Tmax,_effVisc,_muVec);
  PetscScalar min_Tmax;
  VecMin(Tmax,NULL,&min_Tmax);

  maxTimeStep = 0.9 * min_Tmax;

  VecDestroy(&Tmax);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode Mat_PowerLaw::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) { setMMSInitialConditions(); }

  // add viscous strain to integrated variables, stored in _var
  Vec vargxyP; VecDuplicate(_u,&vargxyP); VecCopy(_gxy,vargxyP);
  Vec vargxzP; VecDuplicate(_u,&vargxzP); VecCopy(_gxz,vargxzP);
  varEx["gVxy"] = vargxyP;
  varEx["gVxz"] = vargxzP;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Mat_PowerLaw::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::updateFields()";
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

PetscErrorCode Mat_PowerLaw::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::updateTemperature()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(T,_T);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Mat_PowerLaw::setRHS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::setRHS()";
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
PetscErrorCode Mat_PowerLaw::computeU()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::computeU";
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

PetscErrorCode Mat_PowerLaw::changeBCTypes(std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mat_PowerLaw::changeBCTypes()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _sbp->changeBCTypes(bcRTtype,bcTTtype,bcLTtype,bcBTtype);
  KSPDestroy(&_ksp);
  Mat A; _sbp->getA(A);
  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP(A,_sbp,_ksp,_pc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode Mat_PowerLaw::computViscStrainSourceTerms(Vec& out,Vec& gxy, Vec& gxz)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computViscStrainSourceTerms";
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


PetscErrorCode Mat_PowerLaw::computeViscosity(const PetscScalar viscCap)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeViscosity";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // compute effective viscosity
  PetscScalar const *sigmadev,*A,*B,*n,*T=0;
  PetscScalar *effVisc=0;
  PetscScalar *effViscTemp=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArrayRead(_sdev,&sigmadev);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&B);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(_T,&T);
  VecGetArray(_effVisc,&effVisc);
  VecGetArray(_effViscTemp,&effViscTemp);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar invEffVisc = 1e3 * A[Jj]*pow(sigmadev[Jj],n[Jj]-1.0)*exp(-B[Jj]/T[Jj]) + 1./viscCap;
    effVisc[Jj] = 1.0/invEffVisc;

    effViscTemp[Jj] = _ssEffViscScale * 1.0 / (A[Jj]*pow(sigmadev[Jj],n[Jj]-1.0)*exp(-B[Jj]/T[Jj]) + 1./viscCap);

    //~ effVisc[Jj] = 1e-3 / ( A[Jj]*pow(sigmadev[Jj],n[Jj]-1.0)*exp(-B[Jj]/T[Jj]) ) ;
    //~ effVisc[Jj] = min(effVisc[Jj],viscCap);

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
  VecRestoreArray(_effViscTemp,&effViscTemp);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode Mat_PowerLaw::computeViscStrainRates(const PetscScalar time,const Vec& gVxy, const Vec& gVxz,
  Vec& gVxy_t, Vec& gVxz_t)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeViscStrainRates";
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


PetscErrorCode Mat_PowerLaw::computeViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeViscousStrainRateSAT";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecSet(out,0.0);

  Vec GL, GR,temp1;
  VecDuplicate(u,&GL); VecSet(GL,0.0);
  VecDuplicate(u,&GR); VecSet(GR,0.0);
  VecDuplicate(u,&temp1); VecSet(temp1,0.0);

  // left displacement boundary
  if (_bcLTauQS==0) {
    ierr = _sbp->HyinvxE0y(u,temp1);CHKERRQ(ierr);
    ierr = _sbp->Hyinvxe0y(gL,GL);CHKERRQ(ierr);
    VecAXPY(out,1.0,temp1);
    VecAXPY(out,-1.0,GL);
  }

  // right displacement boundary
  VecSet(temp1,0.0);
  ierr = _sbp->HyinvxENy(u,temp1);CHKERRQ(ierr);
  ierr = _sbp->HyinvxeNy(gR,GR);CHKERRQ(ierr);
  VecAXPY(out,-1.0,temp1);
  VecAXPY(out,1.0,GR);

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
PetscErrorCode Mat_PowerLaw::computeTotalStrains()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeTotalStrains";
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
PetscErrorCode Mat_PowerLaw::computeStresses()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeStresses";
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

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode Mat_PowerLaw::computeSDev()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::computeStresses";
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

PetscErrorCode Mat_PowerLaw::getStresses(Vec& sxy, Vec& sxz, Vec& sdev)
{
  sxy = _sxy;
  sxz = _sxz;
  sdev = _sdev;
  return 0;
}

PetscErrorCode Mat_PowerLaw::setSurfDisp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::setSurfDisp";
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
// IO functions
//======================================================================

// Save all scalar fields to text file named pl_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode Mat_PowerLaw::writeDomain(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = outputDir + "pl_context.txt";
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
  ierr = PetscViewerASCIIPrintf(viewer,"effViscCapSS = %.15e\n",_ssEffViscScale);CHKERRQ(ierr);


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

PetscErrorCode Mat_PowerLaw::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::writeContext";
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


PetscErrorCode Mat_PowerLaw::writeStep1D(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();
  _stepCount = stepCount;

  if (stepCount == 0) {
    ierr = _sbp->writeOps(outputDir + "ops_u_"); CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = io_initiateWriteAppend(_viewers, "surfDisp", _surfDisp, outputDir + "surfDisp"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcL", _bcL, outputDir + "bcL"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "bcR", _bcR, outputDir + "bcR"); CHKERRQ(ierr);
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


PetscErrorCode Mat_PowerLaw::writeStep2D(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Mat_PowerLaw::writeStep2D";
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

PetscErrorCode Mat_PowerLaw::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Power Law Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Ny = %i, Nz = %i\n",_Ny,_Nz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   solver algorithm = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
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



PetscErrorCode Mat_PowerLaw::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  string funcName = "Mat_PowerLaw::setMMSBoundaryConditions";
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

PetscErrorCode Mat_PowerLaw::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  _currTime = time;

  // measure error between analytical and numerical solution
  Vec uA,gxyA,gxzA;
  VecDuplicate(_u,&uA);
  VecDuplicate(_u,&gxyA);
  VecDuplicate(_u,&gxzA);

  if (_Nz == 1) { mapToVec(uA,zzmms_uA1D,*_y,_currTime); }
  else { mapToVec(uA,zzmms_uA,*_y,*_z,_currTime); }
    if (_Nz == 1) { mapToVec(gxyA,zzmms_gxy1D,*_y,_currTime); }
  else { mapToVec(gxyA,zzmms_gxy,*_y,*_z,_currTime); }
  if (_Nz == 1) { mapToVec(gxzA,zzmms_gxy1D,*_y,_currTime); }
  else { mapToVec(gxzA,zzmms_gxz,*_y,*_z,_currTime); }

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


double Mat_PowerLaw::zzmms_sigmaxz(const double y,const double z, const double t)
{ return zzmms_mu(y,z)*zzmms_uA_z(y,z,t); }
double Mat_PowerLaw::zzmms_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double Mat_PowerLaw::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double Mat_PowerLaw::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double Mat_PowerLaw::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double Mat_PowerLaw::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }

double Mat_PowerLaw::zzmms_g(const double t) { return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9); }
double Mat_PowerLaw::zzmms_g_t(const double t) {
  return (-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9);
}

double Mat_PowerLaw::zzmms_uA(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double Mat_PowerLaw::zzmms_uA_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double Mat_PowerLaw::zzmms_uA_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double Mat_PowerLaw::zzmms_uA_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double Mat_PowerLaw::zzmms_uA_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
//~ double Mat_PowerLaw::zzmms_uA_t(const double y,const double z,const double t) {
  //~ return zzmms_f(y,z)*((-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9));
//~ }
double Mat_PowerLaw::zzmms_uA_t(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g_t(t);
}

double Mat_PowerLaw::zzmms_mu(const double y,const double z) { return sin(y)*sin(z) + 30; }
double Mat_PowerLaw::zzmms_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double Mat_PowerLaw::zzmms_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double Mat_PowerLaw::zzmms_sigmaxy(const double y,const double z,const double t)
{ return zzmms_mu(y,z)*zzmms_uA_y(y,z,t); }

double Mat_PowerLaw::zzmms_uSource(const double y,const double z,const double t)
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
double Mat_PowerLaw::zzmms_f1D(const double y) { return cos(y) + 2; } // helper function for uA
double Mat_PowerLaw::zzmms_f_y1D(const double y) { return -sin(y); }
double Mat_PowerLaw::zzmms_f_yy1D(const double y) { return -cos(y); }
//~ double Mat_PowerLaw::zzmms_f_z1D(const double y) { return 0; }
//~ double Mat_PowerLaw::zzmms_f_zz1D(const double y) { return 0; }

double Mat_PowerLaw::zzmms_uA1D(const double y,const double t) { return zzmms_f1D(y)*exp(-t); }
double Mat_PowerLaw::zzmms_uA_y1D(const double y,const double t) { return zzmms_f_y1D(y)*exp(-t); }
double Mat_PowerLaw::zzmms_uA_yy1D(const double y,const double t) { return zzmms_f_yy1D(y)*exp(-t); }
double Mat_PowerLaw::zzmms_uA_z1D(const double y,const double t) { return 0; }
double Mat_PowerLaw::zzmms_uA_zz1D(const double y,const double t) { return 0; }
double Mat_PowerLaw::zzmms_uA_t1D(const double y,const double t) { return -zzmms_f1D(y)*exp(-t); }

double Mat_PowerLaw::zzmms_mu1D(const double y) { return sin(y) + 2.0; }
double Mat_PowerLaw::zzmms_mu_y1D(const double y) { return cos(y); }
//~ double Mat_PowerLaw::zzmms_mu_z1D(const double y) { return 0; }

double Mat_PowerLaw::zzmms_sigmaxy1D(const double y,const double t) { return zzmms_mu1D(y)*zzmms_uA_y1D(y,t); }
double Mat_PowerLaw::zzmms_uSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar u_y = zzmms_uA_y1D(y,t);
  PetscScalar u_yy = zzmms_uA_yy1D(y,t);
  PetscScalar u_zz = zzmms_uA_zz1D(y,t);
  return mu*(u_yy + u_zz) + mu_y*u_y;
}




// power-law specific MMS functions
double Mat_PowerLaw::zzmms_visc(const double y,const double z) { return cos(y)*cos(z) + 2e10; }
double Mat_PowerLaw::zzmms_invVisc(const double y,const double z) { return 1.0/zzmms_visc(y,z); }
double Mat_PowerLaw::zzmms_invVisc_y(const double y,const double z)
{ return sin(y)*cos(z)/pow( cos(y)*cos(z)+2e10, 2.0); }
double Mat_PowerLaw::zzmms_invVisc_z(const double y,const double z)
{ return cos(y)*sin(z)/pow( cos(y)*cos(z)+2e10 ,2.0); }

double Mat_PowerLaw::zzmms_gxy(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fy = zzmms_f_y(y,z);
  //~ return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fy/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fy/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double Mat_PowerLaw::zzmms_gxy_y(const double y,const double z,const double t)
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
double Mat_PowerLaw::zzmms_gxy_t(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fy = zzmms_f_y(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*((-1.0/T1)*exp(-t/T1)+A*exp(-A*t))
       - T2*A*fy/(T2*A-1)*((-1.0/T2)*exp(-t/T2)+A*exp(-A*t))
       + T3*A*fy/(T3*A-1)*((-1.0/T3)*exp(-t/T3)+A*exp(-A*t));
}

double Mat_PowerLaw::zzmms_gxz(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double fz = zzmms_f_z(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fz/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fz/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fz/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double Mat_PowerLaw::zzmms_gxz_z(const double y,const double z,const double t)
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
double Mat_PowerLaw::zzmms_gxz_t(const double y,const double z,const double t)
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
double Mat_PowerLaw::zzmms_max_gxy_t_source(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double uy = zzmms_uA_y(y,z,t);
  double g = zzmms_gxy(y,z,t);

  return zzmms_gxy_t(y,z,t) - A*(uy - g);
}
double Mat_PowerLaw::zzmms_max_gxz_t_source(const double y,const double z,const double t)
{
  double A = zzmms_mu(y,z)*zzmms_invVisc(y,z);
  double uz = zzmms_uA_z(y,z,t);
  double g = zzmms_gxz(y,z,t);

  return zzmms_gxz_t(y,z,t) - A*(uz - g);
}

double Mat_PowerLaw::zzmms_gSource(const double y,const double z,const double t)
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

double Mat_PowerLaw::zzmms_A(const double y,const double z) { return cos(y)*cos(z) + 398; }
double Mat_PowerLaw::zzmms_B(const double y,const double z) { return sin(y)*sin(z) + 4.28e4; }
double Mat_PowerLaw::zzmms_T(const double y,const double z) { return sin(y)*cos(z) + 800; }
double Mat_PowerLaw::zzmms_n(const double y,const double z) { return cos(y)*sin(z) + 3.0; }
double Mat_PowerLaw::zzmms_pl_sigmaxy(const double y,const double z,const double t) { return zzmms_mu(y,z)*(zzmms_uA_y(y,z,t) - zzmms_gxy(y,z,t)); }
double Mat_PowerLaw::zzmms_pl_sigmaxz(const double y,const double z, const double t) { return zzmms_mu(y,z)*(zzmms_uA_z(y,z,t) - zzmms_gxz(y,z,t)); }
double Mat_PowerLaw::zzmms_sdev(const double y,const double z,const double t)
{
  return sqrt( pow(zzmms_pl_sigmaxy(y,z,t),2.0) + pow(zzmms_pl_sigmaxz(y,z,t),2.0) );
}


// source terms for viscous strain rates
double Mat_PowerLaw::zzmms_pl_gxy_t_source(const double y,const double z,const double t)
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
double Mat_PowerLaw::zzmms_pl_gxz_t_source(const double y,const double z,const double t)
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


double Mat_PowerLaw::zzmms_visc1D(const double y) { return cos(y) + 20.0; }
double Mat_PowerLaw::zzmms_invVisc1D(const double y) { return 1.0/(cos(y) + 20.0); }
double Mat_PowerLaw::zzmms_invVisc_y1D(const double y) { return sin(y)/pow( cos(y)+20.0, 2.0); }
double Mat_PowerLaw::zzmms_invVisc_z1D(const double y) { return 0; }

double Mat_PowerLaw::zzmms_gxy1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double fy = zzmms_f_y1D(y);
  return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
}
double Mat_PowerLaw::zzmms_gxy_y1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double Ay = zzmms_mu_y1D(y)*zzmms_invVisc1D(y) + zzmms_mu1D(y)*zzmms_invVisc_y1D(y);
  double fy = zzmms_f_y1D(y);
  double fyy = zzmms_f_yy1D(y);
  double den = A-1.0, B = exp(-t)-exp(-A*t);
  return t*A*Ay*fy*exp(-A*t)/den - A*fy*Ay*B/pow(den,2.0) + fy*Ay*B/den + A*fyy*B/den;
}
double Mat_PowerLaw::zzmms_gxy_t1D(const double y,const double t)
{
  double A = zzmms_mu1D(y)*zzmms_invVisc1D(y);
  double fy = zzmms_f_y1D(y);
  return A*fy*(-exp(-t) + A*exp(-A*t))/(A-1.0);
}

double Mat_PowerLaw::zzmms_gSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar gxy = zzmms_gxy1D(y,t);
  PetscScalar gxy_y = zzmms_gxy_y1D(y,t);
  return -mu*gxy_y - mu_y*gxy;
}



// specific to power law
double Mat_PowerLaw::zzmms_A1D(const double y) { return cos(y) + 1e-9; }
double Mat_PowerLaw::zzmms_B1D(const double y) { return sin(y) + 1.44e4; }
double Mat_PowerLaw::zzmms_T1D(const double y) { return sin(y) + 600; }
double Mat_PowerLaw::zzmms_n1D(const double y) { return cos(y) + 3.0; }
double Mat_PowerLaw::zzmms_pl_sigmaxy1D(const double y,const double t)
{ return zzmms_mu1D(y)*(zzmms_uA_y1D(y,t) - zzmms_gxy1D(y,t)); }
double Mat_PowerLaw::zzmms_pl_sigmaxz1D(const double y,const double t) { return 0; }
double Mat_PowerLaw::zzmms_sdev1D(const double y,const double t)
{ return sqrt( pow(zzmms_pl_sigmaxy1D(y,t),2.0)); }


// source terms for viscous strain rates
double Mat_PowerLaw::zzmms_pl_gxy_t_source1D(const double y,const double t)
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
double Mat_PowerLaw::zzmms_pl_gxz_t_source1D(const double y,const double t)
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
