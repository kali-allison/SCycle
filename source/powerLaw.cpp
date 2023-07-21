#include "powerLaw.hpp"

#define FILENAME "powerLaw.cpp"

using namespace std;


//======================================================================
// power-law rheology class

PowerLaw::PowerLaw(Domain& D,std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
: _D(&D),_file(D._file),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_Ly(D._Ly),_Lz(D._Lz),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_wPlasticity("no"),_wDissPrecCreep("no"),_wDislCreep("yes"),_wDiffCreep("no"),_wLinearMaxwell("no"),_wDislCreep2("no"),
  _plastic(NULL),_dp(NULL),_disl(NULL),_disl2(NULL),_diff(NULL),
  _mu(NULL),_rho(NULL),_cs(NULL),_effVisc(NULL),_T(NULL),_grainSize(NULL),_wetDist(NULL),_effViscCap(1e16),
  _u(NULL),_surfDisp(NULL),_sxy(NULL),_sxz(NULL),_sdev(NULL),
  _gTxy(NULL),_gVxy(NULL),_dgVxy(NULL),_gTxz(NULL),_gVxz(NULL),_dgVxz(NULL),_dgVdev(NULL),_dgVdev_disl(NULL),
  _linSolverSS("MUMPSCHOLESKY"),_linSolverTrans("MUMPSCHOLESKY"),_bcRType(bcRType),_bcTType(bcTType),_bcLType(bcLType),_bcBType(bcBType),
  _rhs(NULL),_bcR(NULL),_bcT(NULL),_bcL(NULL),_bcB(NULL),_bcRShift(NULL),_bcTShift(NULL),
  _ksp(NULL),_pc(NULL),_kspTol(1e-10),_sbp(NULL),_B(NULL),_C(NULL),
  _sbp_eta(NULL),_ksp_eta(NULL),_pc_eta(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),_miscTime(0),_linSolveCount(0)
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  allocateFields(); // initialize fields
  setMaterialParameters();

  // set up deformation mechanisms
  if (_wPlasticity == "yes") {
    _plastic = new Pseudoplasticity(D, *_y,*_z,_file,_delim);
  }
  if (_wDissPrecCreep == "yes") {
    _dp = new DissolutionPrecipitationCreep(D, *_y,*_z,_file,_delim);
  }
  if (_wDislCreep == "yes") {
    _disl = new DislocationCreep(D, *_y,*_z,_file,_delim,"");
  }
  if (_wDislCreep2 == "yes") {
    _disl2 = new DislocationCreep(D, *_y,*_z,_file,_delim,"2");
  }
  if (_wDiffCreep == "yes") {
    _diff = new DiffusionCreep(D, *_y,*_z,_file,_delim);
  }

  if (_D->_restartFromChkpt) {
    loadCheckpoint();
  }
  else if (_D->_restartFromChkptSS) {
    loadCheckpointSS();
  }
  else { loadFieldsFromFiles(); }

  // set up matrix operators and KSP environment
  setUpSBPContext(D); // set up matrix operators
  initializeMomBalMats();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PowerLaw::~PowerLaw()
{
  #if VERBOSE > 1
    string funcName = "PowerLaw::~PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _plastic; _plastic = NULL;
  delete _dp; _dp = NULL;
  delete _disl; _disl = NULL;
  delete _disl2; _disl = NULL;
  delete _diff; _diff = NULL;

  // boundary conditions
  VecDestroy(&_rhs);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcL);
  VecDestroy(&_bcB);
  VecDestroy(&_bcRShift);
  VecDestroy(&_bcTShift);

  // body fields
  VecDestroy(&_mu);
  VecDestroy(&_rho);
  VecDestroy(&_cs);
  VecDestroy(&_effVisc);
  VecDestroy(&_T); VecDestroy(&_grainSize);
  VecDestroy(&_u); VecDestroy(&_surfDisp);
  VecDestroy(&_sxy); VecDestroy(&_sxz); VecDestroy(&_sdev);
  VecDestroy(&_gTxy); VecDestroy(&_gVxy); VecDestroy(&_dgVxy);
  VecDestroy(&_gTxz); VecDestroy(&_gVxz); VecDestroy(&_dgVxz);
  VecDestroy(&_dgVdev); VecDestroy(&_dgVdev_disl);

  // linear system
  KSPDestroy(&_ksp);
  KSPDestroy(&_ksp_eta);
  MatDestroy(&_B);
  MatDestroy(&_C);
  delete _sbp; _sbp = NULL;
  delete _sbp_eta; _sbp_eta = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode PowerLaw::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "PowerLaw::loadSettings()";
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

    if (var.compare("linSolverSS")==0) { _linSolverSS = rhs; }
    if (var.compare("linSolverTrans")==0) { _linSolverTrans = rhs; }
    else if (var.compare("kspTol")==0) { _kspTol = atof( rhs.c_str() ); }
    else if (var.compare("muVals")==0) { loadVectorFromInputFile(rhsFull,_muVals); }
    else if (var.compare("muDepths")==0) { loadVectorFromInputFile(rhsFull,_muDepths); }
    else if (var.compare("rhoVals")==0) { loadVectorFromInputFile(rhsFull,_rhoVals); }
    else if (var.compare("rhoDepths")==0) { loadVectorFromInputFile(rhsFull,_rhoDepths); }

    // deformation style
    else if (var.compare("wPlasticity")==0) { _wPlasticity = rhs.c_str(); }
    else if (var.compare("wDissPrecCreep")==0) { _wDissPrecCreep = rhs.c_str(); }
    else if (var.compare("wDislCreep")==0) { _wDislCreep = rhs.c_str(); }
    else if (var.compare("wDislCreep2")==0) { _wDislCreep2 = rhs.c_str(); }
    else if (var.compare("wDiffCreep")==0) { _wDiffCreep = rhs.c_str(); }
    else if (var.compare("wLinearMaxwell")==0) { _wLinearMaxwell = rhs.c_str(); }

    // linear Maxwell viscosity
    else if (var.compare("effViscVals_lm")==0) { loadVectorFromInputFile(rhsFull,_effViscVals_lm); }
    else if (var.compare("effViscDepths_lm")==0) { loadVectorFromInputFile(rhsFull,_effViscDepths_lm); }

    // temperature and grain size
    else if (var.compare("TVals")==0) { loadVectorFromInputFile(rhsFull,_TVals); }
    else if (var.compare("TDepths")==0) { loadVectorFromInputFile(rhsFull,_TDepths); }
    else if (var.compare("grainSizeEv_grainSizeVals")==0) { loadVectorFromInputFile(rhsFull,_grainSizeVals); }
    else if (var.compare("grainSizeEv_grainSizeDepths")==0) { loadVectorFromInputFile(rhsFull,_grainSizeDepths); }

    // cap on viscosity
    else if (var.compare("maxEffVisc")==0) { _effViscCap = atof( rhs.c_str() ); }

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
    string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_wPlasticity.compare("yes") == 0 || _wPlasticity.compare("no") == 0 );
  assert(_wDissPrecCreep.compare("yes") == 0 || _wDissPrecCreep.compare("no") == 0 );
  assert(_wDislCreep.compare("yes") == 0 || _wDislCreep.compare("no") == 0 );
  assert(_wDislCreep2.compare("yes") == 0 || _wDislCreep2.compare("no") == 0 );
  assert(_wDiffCreep.compare("yes") == 0 || _wDiffCreep.compare("no") == 0 );
  assert(_wLinearMaxwell.compare("yes") == 0 || _wLinearMaxwell.compare("no") == 0 );

// ensure wDislCreep2 only yes if wDislCreep is also yes
PetscPrintf(PETSC_COMM_WORLD,"wDislCreep = %s\n",_wDislCreep.c_str());
PetscPrintf(PETSC_COMM_WORLD,"wDislCreep2 = %s\n",_wDislCreep2.c_str());
if (_wDislCreep=="no" && _wDislCreep2=="yes") {
    assert(0);
  }

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

  if (_wLinearMaxwell == "yes") {
    assert(_effViscVals_lm.size() >= 2);
    assert(_effViscVals_lm.size() == _effViscDepths_lm.size() );
    _wPlasticity = "no";
    _wDissPrecCreep = "no";
    _wDiffCreep = "no";
    _wDislCreep = "no";
  }
  if (_wDiffCreep.compare("yes") == 0 || _wDissPrecCreep.compare("yes") == 0) {
    assert(_grainSizeVals.size() >= 2);
    assert(_grainSizeVals.size() == _grainSizeDepths.size() );
  }

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
    string funcName = "PowerLaw::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecDuplicate(_D->_y0,&_bcR);   VecSet(_bcR,0.0);      PetscObjectSetName((PetscObject) _bcR, "bcR");
  VecDuplicate(_bcR,&_bcRShift); VecSet(_bcRShift,0.0); PetscObjectSetName((PetscObject) _bcRShift, "bcRShift");
  VecDuplicate(_bcR,&_bcL);      VecSet(_bcL,0.);       PetscObjectSetName((PetscObject) _bcL, "bcL");
  VecDuplicate(_D->_z0,&_bcT);   VecSet(_bcT,0.0);      PetscObjectSetName((PetscObject) _bcT, "bcT");
  VecDuplicate(_bcT,&_bcB);      VecSet(_bcB,0.0);      PetscObjectSetName((PetscObject) _bcB, "bcB");
  VecDuplicate(_bcT,&_bcTShift); VecSet(_bcTShift,0.0); PetscObjectSetName((PetscObject) _bcTShift, "bcTShift");


  // other fields
  VecDuplicate(*_z,&_rhs);       VecSet(_rhs,0.0);      PetscObjectSetName((PetscObject) _rhs, "rhs");
  VecDuplicate(_rhs,&_mu);       VecSet(_mu,0.0);       PetscObjectSetName((PetscObject) _mu, "mu");
  VecDuplicate(_rhs,&_rho);      VecSet(_rho,0.0);      PetscObjectSetName((PetscObject) _rho, "rho");
  VecDuplicate(_rhs,&_cs);       VecSet(_cs,0.0);       PetscObjectSetName((PetscObject) _cs, "cs");
  VecDuplicate(_rhs,&_u);        VecSet(_u,0.0);        PetscObjectSetName((PetscObject) _u, "u");
  VecDuplicate(_bcT,&_surfDisp); VecSet(_surfDisp,0.0); PetscObjectSetName((PetscObject) _surfDisp, "surfDisp");
  VecDuplicate(_u,&_effVisc);    VecSet(_effVisc,0.0);  PetscObjectSetName((PetscObject) _effVisc, "effVisc");
  VecDuplicate(_u,&_wetDist);    VecSet(_wetDist,0.0);  PetscObjectSetName((PetscObject) _wetDist, "wetDist");


  // allocate space for stress and strain vectors
  VecDuplicate(_rhs,&_sxy);       VecSet(_sxy,0.0);         PetscObjectSetName((PetscObject) _sxy, "sxy");
  VecDuplicate(_u,&_sxz);         VecSet(_sxz,0.0);         PetscObjectSetName((PetscObject) _sxz, "sxz");
  VecDuplicate(_u,&_sdev);        VecSet(_sdev,0.0);        PetscObjectSetName((PetscObject) _sdev, "sdev");
  VecDuplicate(_u,&_gVxy);        VecSet(_gVxy,0.0);        PetscObjectSetName((PetscObject) _gVxy, "gVxy");
  VecDuplicate(_u,&_dgVxy);       VecSet(_dgVxy,0.0);       PetscObjectSetName((PetscObject) _dgVxy, "dgVxy");
  VecDuplicate(_u,&_gVxz);        VecSet(_gVxz,0.0);        PetscObjectSetName((PetscObject) _gVxz, "gVxz");
  VecDuplicate(_u,&_dgVxz);       VecSet(_dgVxz,0.0);       PetscObjectSetName((PetscObject) _dgVxz, "dgVxz");
  VecDuplicate(_u,&_gTxy);        VecSet(_gTxy,0.0);        PetscObjectSetName((PetscObject) _gTxy, "gTxy");
  VecDuplicate(_u,&_gTxz);        VecSet(_gTxz,0.0);        PetscObjectSetName((PetscObject) _gTxz, "gTxz");
  VecDuplicate(_u,&_dgVdev);      VecSet(_dgVdev,0.0);      PetscObjectSetName((PetscObject) _dgVdev, "dgVdev");
  VecDuplicate(_u,&_dgVdev_disl); VecSet(_dgVdev_disl,0.0); PetscObjectSetName((PetscObject) _dgVdev_disl, "dgVdev_disl");

  VecDuplicate(_u,&_T);           VecSet(_T,0.0);           PetscObjectSetName((PetscObject) _T, "T");
  VecDuplicate(_u,&_grainSize);   VecSet(_grainSize,0.0);   PetscObjectSetName((PetscObject) _grainSize, "grainSize");

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
    string funcName = "PowerLaw::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set each field using it's vals and depths vectors
  ierr = setVec(_mu,*_z,_muVals,_muDepths);                             CHKERRQ(ierr);
  ierr = setVec(_rho,*_z,_rhoVals,_rhoDepths);                          CHKERRQ(ierr);
  if (_wDiffCreep.compare("yes")==0 || _wDissPrecCreep.compare("yes")==0) {
    ierr = setVec(_grainSize,*_z,_grainSizeVals,_grainSizeDepths);      CHKERRQ(ierr);
  }

  if (_wLinearMaxwell.compare("yes")==0) {
    setVec(_effVisc,*_z,_effViscVals_lm,_effViscDepths_lm);

    PetscScalar *v;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_effVisc,&Istart,&Iend);
    VecGetArray(_effVisc,&v);
    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      v[Jj] = pow(10.0,v[Jj]);
      Jj++;
    }
    VecRestoreArray(_effVisc,&v);

  }
  else {
    ierr = setVec(_T,*_z,_TVals,_TDepths);                              CHKERRQ(ierr);
  }

  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_mu,zzmms_mu1D,*_y); }
    else { mapToVec(_mu,zzmms_mu,*_y,*_z); }
  }

  VecPointwiseDivide(_cs, _mu, _rho);
  VecSqrtAbs(_cs);

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
    string funcName = "PowerLaw::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // load bcL and bcR
  ierr = loadVecFromInputFile(_bcL,_inputDir,"momBal_bcL"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"momBal_bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.0);

  ierr = loadVecFromInputFile(_u,_inputDir,"u"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_mu,_inputDir,"mu"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_rho,_inputDir,"rho"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_effVisc,_inputDir,"effVisc"); CHKERRQ(ierr);

  ierr = loadVecFromInputFile(_T,_inputDir,"T"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_grainSize,_inputDir,"grainSizeEv_d"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_wetDist,_inputDir,"pl_wetDist"); CHKERRQ(ierr);

  ierr = loadVecFromInputFile(_gVxy,_inputDir,"GVxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_gVxz,_inputDir,"GVxz"); CHKERRQ(ierr);

  ierr = loadVecFromInputFile(_sxy,_inputDir,"Sxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_sxz,_inputDir,"Sxz"); CHKERRQ(ierr);
  computeSDev();

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
    string funcName = "PowerLaw::setUpSBPContext";
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
  _sbp->setDeleteIntermediateFields(0);
  _sbp->computeMatrices(); // actually create the matrices

  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  Mat A; _sbp->getA(A);
  //~ setupKSP(_ksp,_pc,A);

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
PetscErrorCode PowerLaw::setupKSP(KSP& ksp,PC& pc,Mat& A,std::string& linSolver)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
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
  else if (linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
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
  else if (linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);                                  CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                  CHKERRQ(ierr);
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
  else if (linSolver.compare("CG_PCAMG")==0) { // preconditioned conjugate gradient
    ierr = KSPSetType(ksp,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);                                    CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    //~ ierr = PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);        CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
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
    string funcName = "PowerLaw::initializeMomBalMats";
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
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMatMatMult(yq,zr,H,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxzrxH); CHKERRQ(ierr);
    ierr = MatMatMult(yq,Hy,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy); CHKERRQ(ierr);
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

  if (_Nz == 1) { mapToVec(_gVxy,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gVxy,zzmms_gxy,*_y,*_z,time); }
  if (_Nz == 1) { VecSet(_gVxz,0.0); }
  else { mapToVec(_gVxz,zzmms_gxz,*_y,*_z,time); }

  // set material properties
  if (_Nz == 1) { mapToVec(_mu,zzmms_mu1D,*_y); }
  else { mapToVec(_mu,zzmms_mu,*_y,*_z); }

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcL,_bcR,_bcT, and _bcB
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_u,&viscSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&viscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxviscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&uSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxuSource); CHKERRQ(ierr);

  ierr = computeViscStrainSourceTerms(viscSource);CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,zzmms_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,zzmms_gSource,*_y,*_z,time); }
  ierr = _sbp->H(viscSourceMMS,HxviscSourceMMS); CHKERRQ(ierr);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,zzmms_uSource1D,*_y,time); }
  else { mapToVec(uSource,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbp->H(uSource,HxuSource); CHKERRQ(ierr);
  VecDestroy(&uSource);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
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
  if (_Nz == 1) { mapToVec(_gVxy,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gVxy,zzmms_gxy,*_y,*_z,time); }
  if (_Nz == 1) { mapToVec(_gVxz,zzmms_gxy1D,*_y,time); }
  else { mapToVec(_gVxz,zzmms_gxz,*_y,*_z,time); }

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
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  Vec Tmax;
  ierr = VecDuplicate(_u,&Tmax); CHKERRQ(ierr);
  ierr = VecSet(Tmax,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Tmax,_effVisc,_mu); CHKERRQ(ierr);
  PetscScalar min_Tmax;
  ierr = VecMin(Tmax,NULL,&min_Tmax); CHKERRQ(ierr);

  maxTimeStep = min_Tmax;

  VecDestroy(&Tmax);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // add deep copies of viscous strains to integrated variables, stored in _var
  if (varEx.find("gVxy") != varEx.end() ) { VecCopy(_gVxy,varEx["gVxy"]); }
  else { Vec vargxyP; VecDuplicate(_u,&vargxyP); VecCopy(_gVxy,vargxyP); varEx["gVxy"] = vargxyP; }

  if (varEx.find("gVxz") != varEx.end() ) { VecCopy(_gVxz,varEx["gVxz"]); }
  else { Vec vargxzP; VecDuplicate(_u,&vargxzP); VecCopy(_gVxz,vargxzP); varEx["gVxz"] = vargxzP; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // if integrating viscous strains in time
  VecCopy(varEx.find("gVxy")->second,_gVxy);
  VecCopy(varEx.find("gVxz")->second,_gVxz);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::updateTemperature()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(T,_T);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::updateGrainSize(const Vec& grainSize)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::updateGrainSize()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(grainSize,_grainSize);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::setRHS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setRHS()";
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
    string funcName = "PowerLaw::computeU";
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

PetscErrorCode PowerLaw::changeBCTypes(string bcRTtype,string bcTTtype,string bcLTtype,string bcBTtype)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::changeBCTypes()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _sbp->changeBCTypes(bcRTtype,bcTTtype,bcLTtype,bcBTtype);
  KSPDestroy(&_ksp);
  Mat A;
  _sbp->getA(A);
  KSPCreate(PETSC_COMM_WORLD,&_ksp);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::computeViscStrainSourceTerms(Vec& out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscStrainSourceTerms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz) + SAT
  ierr = MatMult(_B,_gVxy,out); CHKERRQ(ierr);
  if (_Nz > 1) {
    ierr = MatMultAdd(_C,_gVxz,out,out); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::computeViscosity(const PetscScalar viscCap)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscosity";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_wLinearMaxwell.compare("yes")==0) {
    return ierr;
  }

  // estimate 1 / (effective viscosity) based on strain rate
  if (_wPlasticity.compare("yes")==0) { _plastic->computeInvEffVisc(_dgVdev); }
  if (_wDissPrecCreep.compare("yes")==0) { _dp->computeInvEffVisc(_T,_sdev,_grainSize,_wetDist); }
  if (_wDislCreep.compare("yes")==0) { _disl->computeInvEffVisc(_T,_sdev); }
  if (_wDislCreep2.compare("yes")==0) { _disl2->computeInvEffVisc(_T,_sdev); }
  if (_wDiffCreep.compare("yes")==0) { _diff->computeInvEffVisc(_T,_sdev,_grainSize); }

  // 1 / effVisc = 1/(plastic eff visc) + 1/(disl eff visc) + 1/(diff eff visc) + 1/(max eff visc)
  VecSet(_effVisc,1.0/_effViscCap);
  if (_wPlasticity.compare("yes")==0) { VecAXPY(_effVisc,1.0,_plastic->_invEffVisc); }
  if (_wDissPrecCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_dp->_invEffVisc); }
  if (_wDislCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_disl->_invEffVisc); }
  if (_wDislCreep2.compare("yes")==0) { VecAXPY(_effVisc,1.0,_disl2->_invEffVisc); }
  if (_wDiffCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_diff->_invEffVisc); }
  VecReciprocal(_effVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::computeViscStrainRates(const PetscScalar time)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_gTxy,&SAT);
  ierr = computeViscousStrainRateSAT(_u,_bcL,_bcR,SAT); CHKERRQ(ierr);

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  //~ VecSet(SAT,0.); // !!! Just for testing purposes
  VecSet(_dgVxy,0.);
  VecPointwiseMult(_dgVxy,_mu,SAT);
  VecAXPY(_dgVxy,1.0,_sxy);
  VecPointwiseDivide(_dgVxy,_dgVxy,_effVisc);

  if (_Nz > 1) {
    VecCopy(_sxz,_dgVxz);
    VecPointwiseDivide(_dgVxz,_dgVxz,_effVisc);
  }

  VecDestroy(&SAT);

  // compute 2nd invariant of viscous strain rate
  computeDevViscStrainRates();


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
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
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gVxy,&temp1);
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,out,temp1);
    VecCopy(temp1,out);
    VecDestroy(&temp1);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

// computes gTxy and gTxz
PetscErrorCode PowerLaw::computeTotalStrains()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeTotalStrains";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_Ny > 1) {
    _sbp->Dy(_u,_gTxy);
  }
  if (_Nz > 1) {
    _sbp->Dz(_u,_gTxz);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// computes sigmaxy, sigmaxz, and sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::computeStresses()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeStresses";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(_gTxy,_sxy);
  VecAXPY(_sxy,-1.0,_gVxy);
  VecPointwiseMult(_sxy,_sxy,_mu);

  if (_Nz > 1) {
    VecCopy(_gTxz,_sxz);
    VecAXPY(_sxz,-1.0,_gVxz);
    VecPointwiseMult(_sxz,_sxz,_mu);
  }

  computeSDev();

  // force stresses to be correct for MMS test
  //~ mapToVec(_sxy,zzmms_pl_sigmaxy,*_y,*_z,time);
  //~ mapToVec(_sxz,zzmms_pl_sigmaxz,*_y,*_z,time);
  //~ mapToVec(_sdev,zzmms_sdev,*_y,*_z,time);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// computes sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::computeSDev()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeSDev";
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
  return ierr;
}

// computes the 2nd invariant of the deviatoric strain rate
// dgdev = sqrt(dgxy^2 + dgxz^2)
PetscErrorCode PowerLaw::computeDevViscStrainRates()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeDevViscStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  {
    // compute deviatoric strain rate
    PetscScalar const *dgVxy,*dgVxz;
    PetscScalar *dgVdev;
    VecGetArrayRead(_dgVxy,&dgVxy);
    VecGetArrayRead(_dgVxz,&dgVxz);
    VecGetArray(_dgVdev,&dgVdev);
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_dgVdev,&Istart,&Iend);
    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      dgVdev[Jj] = sqrt(dgVxy[Jj]*dgVxy[Jj] + dgVxz[Jj]*dgVxz[Jj]);
      Jj++;
    }
    VecRestoreArrayRead(_dgVxy,&dgVxy);
    VecRestoreArrayRead(_dgVxz,&dgVxz);
    VecRestoreArray(_dgVdev,&dgVdev);
  }

  // compute deviatoric strain rate from dislocation creep only
  if (_wDislCreep.compare("yes")==0) {
    PetscScalar const *invVisc_disl,*sxy,*sxz;
    PetscScalar *dgVdev;
    VecGetArrayRead(_disl->_invEffVisc,&invVisc_disl);
    VecGetArrayRead(_sxy,&sxy);
    VecGetArrayRead(_sxz,&sxz);
    VecGetArray(_dgVdev_disl,&dgVdev);
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_dgVdev_disl,&Istart,&Iend);
    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      PetscScalar dgVxy = sxy[Jj] * invVisc_disl[Jj];
      PetscScalar dgVxz = sxz[Jj] * invVisc_disl[Jj];
      dgVdev[Jj] = sqrt(dgVxy*dgVxy + dgVxz*dgVxz);
      Jj++;
    }
    VecRestoreArrayRead(_disl->_invEffVisc,&invVisc_disl);
    VecRestoreArrayRead(_sxy,&sxy);
    VecRestoreArrayRead(_sxz,&sxz);
    VecRestoreArray(_dgVdev_disl,&dgVdev);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
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

// inititialize effective viscosity based on estimated strain rate
// strainRate is in 1e-3 s^-1
// let A = A exp(-Q/RT)
// dg = A s^n -> s = (dg/A)^(1/n)
// dg = s / v -> v = s/dg OR 1/v = dg/s
PetscErrorCode PowerLaw::guessSteadyStateEffVisc(const PetscScalar strainRate)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::guessSteadyStateEffVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_wLinearMaxwell.compare("yes")==0) {
    return ierr;
  }


  // estimate 1 / (effective viscosity) based on strain rate
  if (_wPlasticity.compare("yes")==0) { _plastic->guessInvEffVisc(strainRate); }
  //~ if (_wDissPrecCreep.compare("yes")==0) { assert(0); } // this requires a nonlinear solve, and may not be wanted
  if (_wDislCreep.compare("yes")==0) { _disl->guessInvEffVisc(_T,strainRate); }
  if (_wDiffCreep.compare("yes")==0) { _diff->guessInvEffVisc(_T,strainRate,_grainSize); }

  // 1 / effVisc = 1/(plastic eff visc) + 1/(disl eff visc) + 1/(diff eff visc) + 1/(max eff visc)
  VecSet(_effVisc,1.0/_effViscCap);
  if (_wPlasticity.compare("yes")==0) { VecAXPY(_effVisc,1.0,_plastic->_invEffVisc); }
  //~ if (_wDissPrecCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_disl->_invEffVisc); }
  if (_wDislCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_disl->_invEffVisc); }
  if (_wDiffCreep.compare("yes")==0) { VecAXPY(_effVisc,1.0,_diff->_invEffVisc); }
  VecReciprocal(_effVisc);

  ierr = loadVecFromInputFile(_effVisc,_inputDir,"effVisc"); CHKERRQ(ierr);
  if (_wPlasticity.compare("yes")==0) {
    ierr = loadVecFromInputFile(_plastic->_invEffVisc,_inputDir,"plasticity_invEffVisc"); CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {
    ierr = loadVecFromInputFile(_disl->_invEffVisc,_inputDir,"disl_invEffVisc"); CHKERRQ(ierr);
  }
  if (_wDiffCreep.compare("yes")==0) {
    ierr = loadVecFromInputFile(_diff->_invEffVisc,_inputDir,"diff_invEffVisc"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute Bss and Css
PetscErrorCode PowerLaw::initializeSSMatrices(string bcRType,string bcTType,string bcLType,string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::initializeSSMatrices";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up SBP operators
  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    _sbp_eta = new SbpOps_m_constGrid(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    _sbp_eta = new SbpOps_m_varGrid(_order,_Ny,_Nz,_Ly,_Lz,_effVisc);
    if (_Ny > 1 && _Nz > 1) { _sbp_eta->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbp_eta->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbp_eta->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp_eta->setCompatibilityType(_D->_sbpCompatibilityType);
  _sbp_eta->setBCTypes(bcRType,bcTType,bcLType,bcBType);
  _sbp_eta->setMultiplyByH(1);
  _sbp_eta->setLaplaceType("yz");
  _sbp_eta->setDeleteIntermediateFields(0);
  _sbp_eta->computeMatrices(); // actually create the matrices

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::setSSRHS(map<string,Vec>& varSS,string bcRType,string bcTType,string bcLType,string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setSSRHS";
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


// solve for steady-state v, viscous strain rates
PetscErrorCode PowerLaw::updateSSa(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::updateSSa";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up linear system
  Mat A;
  _sbp_eta->getA(A);
  KSPDestroy(&_ksp_eta);
  KSPCreate(PETSC_COMM_WORLD,&_ksp_eta);
  setupKSP(_ksp_eta,_pc_eta,A,_linSolverSS);

  // solve for steady-state velocity
  ierr = KSPSolve(_ksp_eta,_rhs,varSS["v"]);CHKERRQ(ierr);
  ierr = changeAnyIsnan(varSS["v"], "varSS[v]",_D->_vL);CHKERRQ(ierr);

  // update viscous strain rates
  _sbp_eta->Dy(varSS["v"],varSS["dgVxy"]);
  _sbp_eta->Dz(varSS["v"],varSS["dgVxz"]);
  VecCopy(varSS["dgVxy"],_dgVxy);
  VecCopy(varSS["dgVxz"],_dgVxz);
  ierr = anyIsnan(varSS["dgVxy"], "varSS[dgVxy]");CHKERRQ(ierr);
  ierr = anyIsnan(varSS["dgVxz"], "varSS[dgVxz]");CHKERRQ(ierr);

  // update stresses
  ierr = VecPointwiseMult(_sxy,_effVisc,varSS["dgVxy"]);
  ierr = VecPointwiseMult(_sxz,_effVisc,varSS["dgVxz"]);
  ierr = anyIsnan(_sxy, "sxy");CHKERRQ(ierr);
  ierr = anyIsnan(_sxz, "sxz");CHKERRQ(ierr);

  ierr = computeSDev(); CHKERRQ(ierr); // deviatoric stress
  ierr = computeDevViscStrainRates(); CHKERRQ(ierr); // deviatoric strain rate

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
    string funcName = "PowerLaw::updateSSb";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // compute u
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


  PetscScalar *gxy=0,*gxz=0;
  const PetscScalar *mu,*gVxy_t,*gVxz_t,*sxy,*sxz;
  PetscInt Istart, Iend;
  VecGetOwnershipRange(_sxy,&Istart,&Iend);
  VecGetArrayRead(_mu,&mu);
  VecGetArrayRead(_sxy,&sxy);
  VecGetArrayRead(_sxz,&sxz);
  VecGetArrayRead(varSS["dgVxy"],&gVxy_t);
  VecGetArrayRead(varSS["dgVxz"],&gVxz_t);
  VecGetArray(_gVxy,&gxy);
  VecGetArray(_gVxz,&gxz);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar gVxy0 = -sxy[Jj]/mu[Jj];
    PetscScalar gVxz0 = -sxz[Jj]/mu[Jj];
    gxy[Jj] = gVxy_t[Jj] * time + gVxy0;
    gxz[Jj] = gVxz_t[Jj] * time + gVxz0;
    assert(!std::isnan(mu[Jj]));
    assert(!std::isnan(time));
    assert(!std::isnan(sxy[Jj]));
    assert(!std::isnan(sxz[Jj]));
    assert(!std::isnan(gVxy0));
    assert(!std::isnan(gVxz0));
    assert(!std::isnan(gVxy_t[Jj]));
    assert(!std::isnan(gVxz_t[Jj]));
    Jj++;
  }
  VecRestoreArrayRead(_mu,&mu);
  VecRestoreArrayRead(_sxy,&sxy);
  VecRestoreArrayRead(_sxz,&sxz);
  VecRestoreArrayRead(varSS["dgVxy"],&gVxy_t);
  VecRestoreArrayRead(varSS["dgVxz"],&gVxz_t);
  VecRestoreArray(_gVxy,&gxy);
  VecRestoreArray(_gVxz,&gxz);


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
PetscErrorCode PowerLaw::writeDomain(const string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  string str = outputDir + "momBal.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolverSS = %s\n",_linSolverSS.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"linSolverTrans = %s\n",_linSolverTrans.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"wPlasticity = %s\n",_wPlasticity.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"wDissPrecCreep = %s\n",_wDissPrecCreep.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"wDislCreep = %s\n",_wDislCreep.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"wDiffCreep = %s\n",_wDiffCreep.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"effViscCap = %.15e\n",_effViscCap);CHKERRQ(ierr);


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

PetscErrorCode PowerLaw::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  writeDomain(outputDir);

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = VecView(_mu,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_rho,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cs,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_wetDist,viewer);                                      CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  if (_wPlasticity.compare("yes")==0) {_plastic->writeContext(viewer); }
  if (_wDissPrecCreep.compare("yes")==0) {
    _dp->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");  CHKERRQ(ierr);
    ierr = VecView(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {_disl->writeContext(viewer); }
  if (_wDislCreep2.compare("yes")==0) {_disl2->writeContext(viewer); }
  if (_wDiffCreep.compare("yes")==0) {
    _diff->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");  CHKERRQ(ierr);
    ierr = VecView(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep1D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                     CHKERRQ(ierr);

  ierr = VecView(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcRShift, viewer);                                    CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::writeStep2D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                     CHKERRQ(ierr);

  ierr = VecView(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_sxy,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_sxz,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_sdev,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gTxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gTxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gVxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gVxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_dgVxy,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_dgVxz,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_effVisc,viewer);                                      CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  if (_wDissPrecCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");CHKERRQ(ierr);
    ierr = VecView(_dp->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep");CHKERRQ(ierr);
    ierr = VecView(_dgVdev_disl,viewer);                                CHKERRQ(ierr);
    ierr = VecView(_disl->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep2.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep2");CHKERRQ(ierr);
    ierr = VecView(_disl2->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDiffCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");  CHKERRQ(ierr);
    ierr = VecView(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = VecView(_diff->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                    CHKERRQ(ierr);

  ierr = VecView(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcRShift, viewer);                                    CHKERRQ(ierr);

  ierr = VecView(_mu,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_rho,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cs,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_T,viewer);                                            CHKERRQ(ierr);

  ierr = VecView(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_sxy,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_sxz,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_sdev,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gTxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gTxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gVxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_gVxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_dgVxy,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_dgVxz,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_effVisc,viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_wetDist,viewer);                                      CHKERRQ(ierr);

  if (_wPlasticity.compare("yes")==0) {_plastic->writeContext(viewer); }
  if (_wDissPrecCreep.compare("yes")==0) {
    _dp->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");CHKERRQ(ierr);
    ierr = VecView(_dp->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {
    _disl->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep");CHKERRQ(ierr);
    ierr = VecView(_dgVdev_disl,viewer);                                CHKERRQ(ierr);
    ierr = VecView(_disl->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep2.compare("yes")==0) {
    _disl2->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep2");CHKERRQ(ierr);
    ierr = VecView(_disl2->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDiffCreep.compare("yes")==0) {
    _diff->writeContext(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");          CHKERRQ(ierr);
    ierr = VecView(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = VecView(_diff->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
double startTime = MPI_Wtime();

  string fileName = _outputDir + "checkpoint.h5";

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                    CHKERRQ(ierr);

  ierr = VecLoad(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcRShift, viewer);                                    CHKERRQ(ierr);

  ierr = VecLoad(_mu,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_rho,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cs,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_wetDist,viewer);                                      CHKERRQ(ierr);

  ierr = VecLoad(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_sxy,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_sxz,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_sdev,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gTxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gTxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gVxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gVxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_dgVxy,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_dgVxz,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,viewer);                                      CHKERRQ(ierr);

  if (_wPlasticity.compare("yes")==0) {_plastic->loadCheckpoint(viewer); }
  if (_wDissPrecCreep.compare("yes")==0) {
    _dp->loadCheckpoint(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");CHKERRQ(ierr);
    ierr = VecLoad(_dp->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {
    _disl->loadCheckpoint(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep");CHKERRQ(ierr);
    ierr = VecLoad(_dgVdev_disl,viewer);                                CHKERRQ(ierr);
    ierr = VecLoad(_disl->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep2.compare("yes")==0) {
    _disl2->loadCheckpoint(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep2");CHKERRQ(ierr);
    ierr = VecLoad(_disl2->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDiffCreep.compare("yes")==0) {
    _diff->loadCheckpoint(viewer);
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");          CHKERRQ(ierr);
    ierr = VecLoad(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = VecLoad(_diff->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }


  ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeCheckpointSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
double startTime = MPI_Wtime();

  PetscViewer viewer;

  string fileName = _outputDir + "data_context.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);
  ierr = VecLoad(_mu,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_rho,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cs,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_wetDist,viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);
  if (_wPlasticity.compare("yes")==0) {_plastic->loadCheckpoint(viewer); }
  if (_wDissPrecCreep.compare("yes")==0) { _dp->loadCheckpoint(viewer); }
  if (_wDislCreep.compare("yes")==0) { _disl->loadCheckpoint(viewer); }
  if (_wDislCreep2.compare("yes")==0) { _disl2->loadCheckpoint(viewer); }
  if (_wDiffCreep.compare("yes")==0) { _diff->loadCheckpoint(viewer); }
  PetscViewerDestroy(&viewer);

  fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                   CHKERRQ(ierr);

  ierr = VecLoad(_surfDisp,viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcRShift, viewer);                                    CHKERRQ(ierr);

  ierr = VecLoad(_u,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_sxy,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_sxz,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_sdev,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gTxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gTxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gVxy,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_gVxz,viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_dgVxy,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_dgVxz,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,viewer);                                      CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = VecLoad(_T,viewer);                                            CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  if (_wDissPrecCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");  CHKERRQ(ierr);
    ierr = VecLoad(_dp->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep");CHKERRQ(ierr);
    ierr = VecLoad(_dgVdev_disl,viewer);                                CHKERRQ(ierr);
    ierr = VecLoad(_disl->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDislCreep2.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dislocationCreep2");CHKERRQ(ierr);
    ierr = VecLoad(_disl->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }
  if (_wDiffCreep.compare("yes")==0) {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");  CHKERRQ(ierr);
    ierr = VecLoad(_grainSize,viewer);                                  CHKERRQ(ierr);
    ierr = VecLoad(_diff->_invEffVisc,viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Power Law Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   linear solver steady-state algorithm: %s\n",_linSolverSS.c_str()); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   linear solver transient algorithm: %s\n",_linSolverTrans.c_str()); CHKERRQ(ierr);
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
  double err2epsxy = computeNormDiff_2(_gVxy,gxyA);
  double err2epsxz = computeNormDiff_2(_gVxz,gxzA);

  PetscPrintf(PETSC_COMM_WORLD,"%i %3i %.4e % .15e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,err2u,log2(err2u),err2epsxy,log2(err2epsxy),err2epsxz,log2(err2epsxz));

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

