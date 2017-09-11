#include "linearElastic.hpp"

#define FILENAME "linearElastic.cpp"

using namespace std;


LinearElastic::LinearElastic(Domain&D,Vec& tau)
: _delim(D._delim),_inputDir(D._inputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_loadICs(D._loadICs),
  _bcLTauQS(0),_currTime(D._initTime),_stepCount(0),
  _outputDir(D._outputDir),
  _vL(D._vL),
  _muVecP(NULL),_muValPlus(30.0),_rhoValPlus(3.0),
  _bcRPShift(NULL),_surfDispPlus(NULL),
  _rhsP(NULL),_uP(NULL),_sxy(NULL),_sxz(NULL),
  _linSolver("unspecified"),_kspP(NULL),_pcP(NULL),
  _kspTol(1e-10),
  _sbpP(NULL),_sbpType(D._sbpType),
  _thermalCoupling("no"),_heatEquationType("transient"),_he(D),_T(NULL),_tempViewer(NULL),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _timeV1D(NULL),_timeV2D(NULL),_surfDispPlusViewer(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),_linSolveCount(0),
  _bcRPlusV(NULL),_bcRPShiftV(NULL),_bcLPlusV(NULL),
  _uPV(NULL),_uAnalV(NULL),_rhsPlusV(NULL),_sxyPV(NULL),
  _bcTType("Neumann"),_bcRType("Dirichlet"),_bcBType("Neumann"),_bcLType("Dirichlet"),
  _bcTP(NULL),_bcRP(NULL),_bcBP(NULL),_bcLP(NULL),_quadEx(NULL),_quadImex(NULL),
  _tLast(0),_uPPrev(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"\nStarting LinearElastic::LinearElastic in linearElastic.cpp.\n");
#endif

  loadSettings(D._file);
  checkInput();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::LinearElastic in linearElastic.cpp.\n\n");
#endif
}


LinearElastic::~LinearElastic()
{
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::~LinearElastic()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecDestroy(&_bcLP);
  VecDestroy(&_bcRP);
  VecDestroy(&_bcTP);
  VecDestroy(&_bcBP);
  VecDestroy(&_bcRPShift);

  // body fields
  VecDestroy(&_rhsP);
  VecDestroy(&_uP);
  VecDestroy(&_sxy);
  VecDestroy(&_sxz);
  VecDestroy(&_surfDispPlus);
  VecDestroy(&_T);

  KSPDestroy(&_kspP);

  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_timeV2D);
  PetscViewerDestroy(&_surfDispPlusViewer);
  PetscViewerDestroy(&_uPV);
  PetscViewerDestroy(&_tempViewer);

  delete _sbpP; _sbpP = NULL;

  PetscViewerDestroy(&_bcRPlusV);
  PetscViewerDestroy(&_bcRPShiftV);
  PetscViewerDestroy(&_bcLPlusV);
  PetscViewerDestroy(&_uAnalV);
  PetscViewerDestroy(&_rhsPlusV);
  PetscViewerDestroy(&_sxyPV);

  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_timeV2D);


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

    else if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("bcLTauQS")==0) {
      _bcLTauQS = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    else if (var.compare("hydraulicCoupling")==0) {
      _hydraulicCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hydraulicTimeIntType")==0) {
      _hydraulicTimeIntType = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    if (var.compare("muPlus")==0) {
      _muValPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("rhoPlus")==0) {
      _rhoValPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::checkInput in linearelastic.cpp.\n");CHKERRQ(ierr);
  #endif

  assert(_linSolver.compare("MUMPSCHOLESKY") == 0 ||
         _linSolver.compare("MUMPSLU") == 0 ||
         _linSolver.compare("PCG") == 0 ||
         _linSolver.compare("AMG") == 0 );

  if (_linSolver.compare("PCG")==0 || _linSolver.compare("AMG")==0) {
    assert(_kspTol >= 1e-14);
  }

  #if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::checkInput in linearelastic.cpp.\n");CHKERRQ(ierr);
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
PetscErrorCode LinearElastic::setupKSP(SbpOps* sbp,KSP& ksp,PC& pc)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::setupKSP in linearElastic.cpp\n");CHKERRQ(ierr);
#endif

  Mat A;
  sbp->getA(A);

  if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

    //~ PetscOptionsSetValue(NULL,"-pc_hypre_boomeramg_agg_nl 1");
  }
  else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    // use direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCLU);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
  }
  else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCCHOLESKY);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::setupKSP in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}




// limited by Maxwell time
PetscErrorCode SymmLinearElastic::computeMaxTimeStep(PetscScalar& maxTimeStep)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeMaxTimeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  maxTimeStep = 1e14; // impose no limit

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}










//======================================================================
// Symmetric LinearElastic Functions
//======================================================================

SymmLinearElastic::SymmLinearElastic(Domain&D, Vec& tau)
: LinearElastic(D,tau)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"\n\nStarting SymmLinearElastic::SymmLinearElastic in linearElastic.cpp.\n");
#endif


  allocateFields();
  setMaterialParameters();
  if (_loadICs==1) { loadFieldsFromFiles(); } // load from previous simulation
  else { setInitialConds(D,tau); } // guess at steady-state configuration
  setUpSBPContext(D); // set up matrix operators

  _sbpP->muxDy(_uP,_sxy); // initialize for shear stress

  if (_isMMS) { setMMSInitialConditions(); }

  setSurfDisp();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::SymmLinearElastic in linearElastic.cpp.\n\n\n");
#endif
}

SymmLinearElastic::~SymmLinearElastic()
{
  VecDestroy(&_bcRPShift);
};

// allocate space for member fields
PetscErrorCode SymmLinearElastic::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcLP);
  VecSetSizes(_bcLP,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcLP);
  PetscObjectSetName((PetscObject) _bcLP, "_bcLP");
  VecSet(_bcLP,0.0);

  VecDuplicate(_bcLP,&_bcRPShift); PetscObjectSetName((PetscObject) _bcRPShift, "bcRPShift");
  VecSet(_bcRPShift,0.0);
  VecDuplicate(_bcLP,&_bcRP); PetscObjectSetName((PetscObject) _bcRP, "_bcRP");
  VecSet(_bcRP,_vL*_currTime/2.0);


  VecCreate(PETSC_COMM_WORLD,&_bcTP);
  VecSetSizes(_bcTP,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcTP);
  PetscObjectSetName((PetscObject) _bcTP, "_bcTP");
  VecSet(_bcTP,0.0);

  VecDuplicate(_bcTP,&_bcBP); PetscObjectSetName((PetscObject) _bcBP, "_bcBP");
  VecSet(_bcBP,0.0);


  // other fieds
  VecDuplicate(*_z,&_rhsP);
  VecDuplicate(*_z,&_muVecP);
  VecDuplicate(_rhsP,&_uP);
  VecDuplicate(_rhsP,&_sxy); VecSet(_sxy,0.0);
  VecDuplicate(_rhsP,&_T); _he.getTemp(_T);
  VecDuplicate(_bcTP,&_surfDispPlus); PetscObjectSetName((PetscObject) _surfDispPlus, "_surfDisp");

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
#endif
  return ierr;
}

// set off-fault material properties
PetscErrorCode SymmLinearElastic::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecSet(_muVecP,_muValPlus);

  if (_isMMS) {
    if (_Nz == 1) { mapToVec(_muVecP,zzmms_mu1D,*_y); }
    else { mapToVec(_muVecP,zzmms_mu,*_y,*_z); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}





// parse input file and load values into data members
PetscErrorCode SymmLinearElastic::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  std::string funcName = "SymmLinearElastic::loadFieldsFromFiles";
  PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
#endif

  PetscViewer inv; // input viewer

  // load bcL
  string vecSourceFile = _inputDir + "bcL";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcLP,inv);CHKERRQ(ierr);

  // load bcR
  vecSourceFile = _inputDir + "bcR";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcRPShift,inv);CHKERRQ(ierr);

  // load u
  vecSourceFile = _inputDir + "u";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_uP,inv);CHKERRQ(ierr);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
#endif
  return ierr;
}

// try to speed up spin up by starting closer to steady state
PetscErrorCode SymmLinearElastic::setInitialConds(Domain& D,Vec& tauRS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::setInitialConds";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (tauRS == NULL) { return ierr;}

  delete _sbpP;
  KSPDestroy(&_kspP);

  // set up SBP operators
  //~ string bcT,string bcR,string bcB, string bcL
  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";
  if (_sbpType.compare("mc")==0) {
    _sbpP = new SbpOps_c(D,_Ny,_Nz,_muVecP,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpP = new SbpOps_fc(D,_Ny,_Nz,_muVecP,bcTType,bcRType,bcBType,bcLType,"yz"); // to spin up viscoelastic
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpP = new SbpOps_fc_coordTrans(D,_Ny,_Nz,_muVecP,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  KSPCreate(PETSC_COMM_WORLD,&_kspP);
  setupKSP(_sbpP,_kspP,_pcP);


  // set up boundary conditions
  VecSet(_bcRP,0.0);
  VecSet(_bcLP,0.0);
  PetscInt Istart, Iend;
  VecGetOwnershipRange(_bcLP,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    //~ PetscScalar tauRS = _fault->getTauSS(Ii); // rate-and-state strength
    PetscScalar tauRSV = 0;
    ierr = VecGetValues(tauRS,1,&Ii,&tauRSV);CHKERRQ(ierr);
    VecSetValue(_bcLP,Ii,tauRSV,INSERT_VALUES);
  }
  VecAssemblyBegin(_bcLP); VecAssemblyEnd(_bcLP);

  _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  KSPDestroy(&_kspP);
  delete _sbpP;
  _sbpP = NULL;

  // extract boundary condition information from u
  Vec uL;
  VecDuplicate(_bcLP,&uL);
  PetscScalar minVal = 0;
  VecMin(_uP,NULL,&minVal);
  PetscScalar v = 0.0;
  ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    // put left boundary info into fault slip vector
    if ( Ii < _Nz ) {
      ierr = VecGetValues(_uP,1,&Ii,&v);CHKERRQ(ierr);
      v = v + abs(minVal);
      //~ ierr = VecSetValues(_fault->_slip,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValues(uL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }

    // put right boundary data into bcR
    if ( Ii > (_Ny*_Nz - _Nz - 1) ) {
      PetscInt zI =  Ii - (_Ny*_Nz - _Nz);
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ny*Nz = %i, Ii = %i, zI = %i\n",_Ny*_Nz,Ii,zI);
      ierr = VecGetValues(_uP,1,&Ii,&v);CHKERRQ(ierr);
      v = v + abs(minVal);
      ierr = VecSetValues(_bcRPShift,1,&zI,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcRPShift);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRPShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uL);CHKERRQ(ierr);
  VecCopy(_bcRPShift,_bcRP);

  if (!_bcLTauQS) {
    VecCopy(uL,_bcLP);
  }
  VecDestroy(&uL);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// compute bcL from u so that fault's initial conditions can be set from it
PetscErrorCode SymmLinearElastic::setInitialSlip(Vec& out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::setInitialSlip";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  if (!_loadICs) {
    PetscScalar minVal = 0;
    VecMin(_uP,NULL,&minVal);
    ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart;Ii<Iend;Ii++) {
      if (Ii<_Nz) {
        ierr = VecGetValues(_uP,1,&Ii,&v);CHKERRQ(ierr);
        v = 2. * (v + abs(minVal));
        ierr = VecSetValues(out,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(out);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(out);CHKERRQ(ierr);
  }
  else {
    // load slip
    PetscViewer inv; // in viewer
    std::string vecSourceFile = _inputDir + "slip";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecLoad(out,inv);CHKERRQ(ierr);
  }


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// try to speed up spin up by starting closer to steady state
PetscErrorCode SymmLinearElastic::setUpSBPContext(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::setUpSBPContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _sbpP;
  KSPDestroy(&_kspP);

  // set up SBP operators
  //~ string bcT,string bcR,string bcB, string bcL
  _bcTType = "Neumann";
  _bcBType = "Neumann";
  _bcRType = "Dirichlet";
  _bcLType = "Dirichlet";
  if (_bcLTauQS==1) { _bcLType = "Neumann"; }

  // for MMS tests
  //~ _bcTType = "Dirichlet";
  //~ _bcBType = "Dirichlet";
  //~ _bcRType = "Dirichlet";
  //~ _bcLType = "Dirichlet";

  if (_sbpType.compare("mc")==0) {
    _sbpP = new SbpOps_c(D,_Ny,_Nz,_muVecP,_bcTType,_bcRType,_bcBType,_bcLType,"yz");
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpP = new SbpOps_fc(D,_Ny,_Nz,_muVecP,_bcTType,_bcRType,_bcBType,_bcLType,"yz");
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpP = new SbpOps_fc_coordTrans(D,_Ny,_Nz,_muVecP,_bcTType,_bcRType,_bcBType,_bcLType,"yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  KSPCreate(PETSC_COMM_WORLD,&_kspP);
  setupKSP(_sbpP,_kspP,_pcP);


  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


PetscErrorCode SymmLinearElastic::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
#endif


  PetscInt    Ii,Istart,Iend,y;
  PetscScalar u;
  ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    //~ z = Ii-_Nz*(Ii/_Nz);
    y = Ii / _Nz;
    if (Ii % _Nz == 0) {
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ii = %i, y = %i, z = %i\n",Ii,y,z);
      ierr = VecGetValues(_uP,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispPlus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispPlus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmLinearElastic::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Linear Elastic Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent solving linear system: %g\n",_linSolveTime/totRunTime*100.);CHKERRQ(ierr);

  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   misc time (s): %g\n",_miscTime);CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% misc time: %g\n",_miscTime/_integrateTime);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  //~ierr = KSPView(_kspP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode SymmLinearElastic::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = _sbpP->writeOps(_outputDir + "ops_u_"); CHKERRQ(ierr);

  PetscViewer    viewer;

  // write out scalar info
  std::string str = _outputDir + "linEl_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);

  // hydraulic fault type
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);
  if (_hydraulicCoupling.compare("coupled") == 0 || _hydraulicCoupling.compare("uncoupled") == 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"hydraulicTimeIntType = %s\n",_hydraulicTimeIntType.c_str());CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


  ierr = writeVec(_muVecP,_outputDir + "_muPlus"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode SymmLinearElastic::writeStep1D(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::writeStep1D";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();
  _stepCount++;

  if (_timeV1D==NULL) {
    ierr = _sbpP->writeOps(_outputDir + "ops_u_");CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    // boundary conditions
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),FILE_MODE_WRITE,
                                 &_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
                                   FILE_MODE_APPEND,&_bcRPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
              FILE_MODE_WRITE,&_bcLPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcLP,_bcLPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcLPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
                                   FILE_MODE_APPEND,&_bcLPlusV);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_bcLP,_bcLPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



PetscErrorCode SymmLinearElastic::writeStep2D(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::writeStep2D";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();


  if (_timeV2D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);

    // output body fields
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
              FILE_MODE_WRITE,&_uPV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
                                   FILE_MODE_APPEND,&_uPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
              FILE_MODE_WRITE,&_sxyPV);CHKERRQ(ierr);
    ierr = VecView(_sxy,_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
                                   FILE_MODE_APPEND,&_sxyPV);CHKERRQ(ierr);

    if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"T").c_str(),
              FILE_MODE_WRITE,&_tempViewer);CHKERRQ(ierr);
    ierr = VecView(_T,_tempViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_tempViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"T").c_str(),
                                   FILE_MODE_APPEND,&_tempViewer);CHKERRQ(ierr);
    }
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);

    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = VecView(_sxy,_sxyPV);CHKERRQ(ierr);
  }


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode SymmLinearElastic::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set slip based on uP
  Vec slip;
  VecDuplicate(_bcLP,&slip);
  setInitialSlip(slip);
  varEx["slip"] = slip;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmLinearElastic::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmLinearElastic::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _currTime = time;
  if (_bcLTauQS==0) { // var holds slip, bcL is displacement at y=0+
    ierr = VecCopy(varEx.find("slip")->second,_bcLP);CHKERRQ(ierr);
    ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr);
  } // else do nothing
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmLinearElastic::getSigmaDev(Vec& sdev)
{
  return 0;
}

// explicit time stepping
PetscErrorCode SymmLinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  if (_isMMS) {
    ierr = d_dt_mms(time,varEx,dvarEx);CHKERRQ(ierr);
  }
  else {
    ierr = d_dt_eqCycle(time,varEx,dvarEx);CHKERRQ(ierr);
  }
  return ierr;
}


// explicit time stepping
PetscErrorCode SymmLinearElastic::d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif

  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = _sbpP->muxDy(_uP,_sxy); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode SymmLinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  ierr = d_dt_eqCycle(time,varEx,dvarEx);CHKERRQ(ierr);
/*
  if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    Vec stressxzP,tau;
    VecDuplicate(_uP,&stressxzP);
    ierr = _sbpP->muxDz(_uP,stressxzP); CHKERRQ(ierr);
    VecDuplicate(_fault->_tauQSP,&tau);
    _fault->getTau(tau);

    _fault->setTemp(_T);
    //~ ierr = _he.be(time,*(dvarBegin+2),tau,NULL,NULL,
      //~ NULL,*varBeginIm,*varBeginImo,dt);CHKERRQ(ierr);
    ierr = _he.be(time,dvarEx.find("slip")->second,tau,NULL,NULL,
      NULL,varIm.find("Temp")->second,varImo.find("Temp")->second,dt);CHKERRQ(ierr);
    VecDestroy(&stressxzP);
    VecDestroy(&tau);
    // arguments:
    // time, slipVel, txy, sigmadev, dgxy, dgxz, T, dTdt

    _he.getTemp(_T);
  }
  else {
    //~ ierr = VecSet(*varBeginIm,0.0);CHKERRQ(ierr);
    ierr = VecSet(varImo.find("Temp")->second,0.0);CHKERRQ(ierr);
  }*/


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmLinearElastic::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLinearElastic::d_dt_mms in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif

  Vec source,Hxsource;
  VecDuplicate(_uP,&source);
  VecDuplicate(_uP,&Hxsource);
  if (_Nz==1) { mapToVec(source,zzmms_uSource1D,*_y,time); }
  else { mapToVec(source,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbpP->H(source,Hxsource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  VecDestroy(&source);


  // set rhs, including body source term
  setMMSBoundaryConditions(time); // modifies _bcLP,_bcRP,_bcTP, and _bcBP
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);
  ierr = VecAXPY(_rhsP,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  _sbpP->muxDy(_uP,_sxy);

  // force u to be correct
  //~ ierr = mapToVec(_uP,zzmms_uA,*_y,*_z,time); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::d_dt_mms in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmLinearElastic::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::setMMSBoundaryConditions";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcLP,&Istart,&Iend);CHKERRQ(ierr);
  if (_Nz == 1) {
    Ii = Istart;
    y = 0;
    if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA1D(y,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("Neumann")) { v = zzmms_mu1D(y) * zzmms_uA_y1D(y,time); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  else {
    for(Ii=Istart;Ii<Iend;Ii++) {
      ierr = VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
      //~ z = _dz * Ii;
      y = 0;
      if (!_bcLType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=0,z)
      else if (!_bcLType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      y = _Ly;
      if (!_bcRType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y=Ly,z)
      else if (!_bcRType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_y(y,z,time); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ly = %f, y = %f, z = %f, bcR = %f\n",_Ly,y,z,v);
    }
  }
  ierr = VecAssemblyBegin(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRP);CHKERRQ(ierr);

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
      ierr = VecSetValues(_bcTP,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);

      z = _Lz;
      if (!_bcBType.compare("Dirichlet")) { v = zzmms_uA(y,z,time); } // uAnal(y,z=Lz)
      else if (!_bcBType.compare("Neumann")) { v = zzmms_mu(y,z) * zzmms_uA_z(y,z,time); }
      ierr = VecSetValues(_bcBP,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcBP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcBP);CHKERRQ(ierr);

  //~ writeVec(_bcLP,_outputDir+"mms_bcL");
  //~ writeVec(_bcRP,_outputDir+"mms_bcR");
  //~ writeVec(_bcTP,_outputDir+"mms_bcT");
  //~ writeVec(_bcBP,_outputDir+"mms_bcB");

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}

PetscErrorCode SymmLinearElastic::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::setMMSInitialConditions";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif


  PetscScalar time = _currTime;

  Vec source,Hxsource;
  VecDuplicate(_uP,&source);
  VecDuplicate(_uP,&Hxsource);

  if (_Nz == 1) { mapToVec(source,zzmms_uSource1D,*_y,_currTime); }
  else { mapToVec(source,zzmms_uSource,*_y,*_z,_currTime); }
  //~ ierr = mapToVec(source,zzmms_uSource,_Nz,_dy,_dz,time); CHKERRQ(ierr);
  writeVec(source,_outputDir + "mms_uSource");
  ierr = _sbpP->H(source,Hxsource); CHKERRQ(ierr);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  VecDestroy(&source);


  // set rhs, including body source term
  VecSet(_bcRPShift,0.0);
  ierr = setMMSBoundaryConditions(time); CHKERRQ(ierr);
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);
  ierr = VecAXPY(_rhsP,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  writeVec(_uP,"data/mms_uuu");
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  _sbpP->muxDy(_uP,_sxy);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}



// Outputs data at each time step.
PetscErrorCode SymmLinearElastic::debug(const PetscReal time,const PetscInt stepCount,
                         const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;

  //~PetscScalar k = _muArrPlus[0]/2/_Ly;

  ierr= VecGetOwnershipRange(varEx.find("psi"),&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(varEx.find("psi"),1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr = VecGetValues(varEx.find("slip"),1,&Istart,&uVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(dvarEx.find("psi"),&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(dvarEx.find("psi"),1,&Istart,&dQVal);CHKERRQ(ierr);
  ierr = VecGetValues(dvarEx.find("slip"),1,&Istart,&velVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcRP,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRP,1,&Istart,&bcRval);CHKERRQ(ierr);

  ierr = VecGetValues(_fault->_tauQSP,1,&Istart,&tauQS);CHKERRQ(ierr);

  if (stepCount == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s | %-15s %-15s %-15s | %-15s %-15s %-16s | %-15s\n",
                       "Step","Stage","bcR","D","Q","tauQS","V","dQ","time");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ",stepCount,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",bcRval,uVal,psiVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",tauQS,velVal,dQVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e\n",time);CHKERRQ(ierr);


  //~VecView(_fault->_tauQSP,PETSC_VIEWER_STDOUT_WORLD);
#endif
  return ierr;
}

PetscErrorCode SymmLinearElastic::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  _currTime = time;

  // measure error between analytical and numerical solution
  Vec uA;
  VecDuplicate(_uP,&uA);
  if (_Nz == 1) { mapToVec(uA,zzmms_uA1D,*_y,_currTime); }
  else { mapToVec(uA,zzmms_uA,*_y,*_z,_currTime); }

  Vec sigmaxyA;
  VecDuplicate(_uP,&sigmaxyA);
  //~ mapToVec(sigmaxyA,zzmms_sigmaxy,_Nz,_dy,_dz,_currTime);
    if (_Nz == 1) { mapToVec(sigmaxyA,zzmms_sigmaxy1D,*_y,_currTime); }
  else { mapToVec(sigmaxyA,zzmms_sigmaxy,*_y,*_z,_currTime); }


  double err2uA = computeNormDiff_2(_uP,uA);
  double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  //~ std::str = _outputDir = "uA";
  writeVec(uA,_outputDir+"uA");
  //~ writeVec(_bcLP,_outputDir+"mms_u_bcL");
  //~ writeVec(_bcRP,_outputDir+"mms_u_bcR");
  //~ writeVec(_bcTP,_outputDir+"mms_u_bcT");
  //~ writeVec(_bcBP,_outputDir+"mms_u_bcB");

  //~ Mat H; _sbpP->getH(H);
  //~ double err2uA = computeNormDiff_Mat(H,_uP,uA);
  //~ double err2sigmaxy = computeNormDiff_2(_sxy,sigmaxyA);

  PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2uA,log2(err2uA),err2sigmaxy,log2(err2sigmaxy));

  return ierr;
}


// MMS functions
double SymmLinearElastic::zzmms_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double SymmLinearElastic::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double SymmLinearElastic::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double SymmLinearElastic::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double SymmLinearElastic::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }

double SymmLinearElastic::zzmms_g(const double t) { return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9); }
double SymmLinearElastic::zzmms_g_t(const double t) {
  return (-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9);
}

double SymmLinearElastic::zzmms_uA(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double SymmLinearElastic::zzmms_uA_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double SymmLinearElastic::zzmms_uA_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double SymmLinearElastic::zzmms_uA_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double SymmLinearElastic::zzmms_uA_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
//~ double SymmLinearElastic::zzmms_uA_t(const double y,const double z,const double t) {
  //~ return zzmms_f(y,z)*((-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9));
//~ }
double SymmLinearElastic::zzmms_uA_t(const double y,const double z,const double t) {
  return zzmms_f(y,z)*zzmms_g_t(t);
}

double SymmLinearElastic::zzmms_mu(const double y,const double z) { return sin(y)*sin(z) + 30; }
double SymmLinearElastic::zzmms_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double SymmLinearElastic::zzmms_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double SymmLinearElastic::zzmms_sigmaxy(const double y,const double z,const double t)
{ return zzmms_mu(y,z)*zzmms_uA_y(y,z,t); }

double SymmLinearElastic::zzmms_uSource(const double y,const double z,const double t)
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
double SymmLinearElastic::zzmms_f1D(const double y) { return cos(y) + 2; } // helper function for uA
double SymmLinearElastic::zzmms_f_y1D(const double y) { return -sin(y); }
double SymmLinearElastic::zzmms_f_yy1D(const double y) { return -cos(y); }
//~ double SymmLinearElastic::zzmms_f_z1D(const double y) { return 0; }
//~ double SymmLinearElastic::zzmms_f_zz1D(const double y) { return 0; }

double SymmLinearElastic::zzmms_uA1D(const double y,const double t) { return zzmms_f1D(y)*exp(-t); }
double SymmLinearElastic::zzmms_uA_y1D(const double y,const double t) { return zzmms_f_y1D(y)*exp(-t); }
double SymmLinearElastic::zzmms_uA_yy1D(const double y,const double t) { return zzmms_f_yy1D(y)*exp(-t); }
double SymmLinearElastic::zzmms_uA_z1D(const double y,const double t) { return 0; }
double SymmLinearElastic::zzmms_uA_zz1D(const double y,const double t) { return 0; }
double SymmLinearElastic::zzmms_uA_t1D(const double y,const double t) { return -zzmms_f1D(y)*exp(-t); }

double SymmLinearElastic::zzmms_mu1D(const double y) { return sin(y) + 2.0; }
double SymmLinearElastic::zzmms_mu_y1D(const double y) { return cos(y); }
//~ double SymmLinearElastic::zzmms_mu_z1D(const double y) { return 0; }

double SymmLinearElastic::zzmms_sigmaxy1D(const double y,const double t) { return zzmms_mu1D(y)*zzmms_uA_y1D(y,t); }
double SymmLinearElastic::zzmms_uSource1D(const double y,const double t)
{
  PetscScalar mu = zzmms_mu1D(y);
  PetscScalar mu_y = zzmms_mu_y1D(y);
  PetscScalar u_y = zzmms_uA_y1D(y,t);
  PetscScalar u_yy = zzmms_uA_yy1D(y,t);
  PetscScalar u_zz = zzmms_uA_zz1D(y,t);
  return mu*(u_yy + u_zz) + mu_y*u_y;
}


/*
//================= Full LinearElastic (+ and - sides) Functions =========
FullLinearElastic::FullLinearElastic(Domain&D)
: LinearElastic(D),
  _bcLMShift(NULL),_surfDispMinus(NULL),
  _rhsM(NULL),_uM(NULL),_sigma_xyMinus(NULL),
  _kspM(NULL),_pcMinus(NULL),_sbpM(NULL),
  _surfDispMinusViewer(NULL),_bcTMinus(NULL),_bcRMinus(NULL),_bcBMinus(NULL),_bcLMinus(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::FullLinearElastic in linearElastic.cpp.\n");
#endif

  _sbpM = new SbpOps_c(D,D._muVecM,"Neumann","Dirichlet","Neumann","Dirichlet","yx");

  _fault = new  FullFault(D,_he);

  // initialize y<0 boundary conditions
  VecDuplicate(_bcLP,&_bcLMShift);
  PetscObjectSetName((PetscObject) _bcLMShift, "_bcLMShift");
  setShifts(); // set position of boundary from steady sliding
  VecAXPY(_bcRP,1.0,_bcRPShift);


  // fault initial displacement on minus side
  VecDuplicate(_bcLP,&_bcRMinus); PetscObjectSetName((PetscObject) _bcRMinus, "_bcRMinus");
  VecSet(_bcRMinus,0.0);

  // remote displacement on - side
  VecDuplicate(_bcLP,&_bcLMinus); PetscObjectSetName((PetscObject) _bcLMinus, "bcLMinus");
  VecSet(_bcLMinus,-_vL*_initTime/2.0);
  VecAXPY(_bcLMinus,1.0,_bcLMShift);

  VecDuplicate(_bcTP,&_bcTMinus); PetscObjectSetName((PetscObject) _bcTMinus, "bcTMinus");
  VecSet(_bcTMinus,0.0);
  VecDuplicate(_bcBP,&_bcBMinus); PetscObjectSetName((PetscObject) _bcBMinus, "bcBMinus");
  VecSet(_bcBMinus,0.0);

  // initialize and allocate memory for body fields
  double startTime;

  VecDuplicate(_rhsP,&_uP);
  VecDuplicate(_rhsP,&_sxy);

  VecCreate(PETSC_COMM_WORLD,&_rhsM);
  VecSetSizes(_rhsM,PETSC_DECIDE,_Ny*_Nz);
  VecSetFromOptions(_rhsM);
  VecDuplicate(_rhsM,&_uM);
  VecDuplicate(_rhsM,&_sigma_xyMinus);


  // initialize KSP for y<0
  KSPCreate(PETSC_COMM_WORLD,&_kspM);
  startTime = MPI_Wtime();
  setupKSP(_sbpM,_kspM,_pcMinus);
  _factorTime += MPI_Wtime() - startTime;


  // solve for displacement and shear stress in y<0
  _sbpM->setRhs(_rhsM,_bcLMinus,_bcRMinus,_bcTMinus,_bcBMinus);
  KSPSolve(_kspM,_rhsM,_uM);
  //~MatMult(_sbpM->_muxDy_Iz,_uM,_sigma_xyMinus);
  _sbpM->muxDy(_uM,_sigma_xyMinus);

  // solve for displacement and shear stress in y>0
  _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);
  startTime = MPI_Wtime();
  KSPSolve(_kspP,_rhsP,_uP);
  _factorTime += MPI_Wtime() - startTime;
  //~MatMult(_sbpP->_muxDy_Iz,_uP,_sxy);
  _sbpP->muxDy(_uP,_sxy);



  // set up fault
  _fault->setTauQS(_sxy,_sigma_xyMinus);
  _fault->setFaultDisp(_bcLP,_bcRMinus);
  _fault->computeVel();

  _var.push_back(_fault->_psi);
  _var.push_back(_fault->_uP);
  _var.push_back(_fault->_uM);


  VecDuplicate(_bcTMinus,&_surfDispMinus); PetscObjectSetName((PetscObject) _surfDispMinus, "_surfDispMinus");
  setSurfDisp(); // extract surface displacement from displacement fields

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::FullLinearElastic in linearElastic.cpp.\n");
#endif
}

FullLinearElastic::~FullLinearElastic()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::~FullLinearElastic in linearElastic.cpp.\n");
#endif

  // boundary conditions: minus side
  VecDestroy(&_bcLMinus);
  VecDestroy(&_bcLMShift);
  VecDestroy(&_bcRMinus);
  VecDestroy(&_bcTMinus);
  VecDestroy(&_bcBMinus);

  // body fields
  VecDestroy(&_rhsM);
  VecDestroy(&_uM);
  VecDestroy(&_sigma_xyMinus);

  VecDestroy(&_surfDispMinus);

  KSPDestroy(&_kspM);


  PetscViewerDestroy(&_surfDispMinusViewer);
  //~ PetscViewerDestroy(&_bcRMinusV);
  //~ PetscViewerDestroy(&_bcRMShiftV);
  //~ PetscViewerDestroy(&_bcLMinusV);
  //~ PetscViewerDestroy(&_uMV);
  //~ PetscViewerDestroy(&_rhsMV);
  //~ PetscViewerDestroy(&_stressxyMV);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::~FullLinearElastic in linearElastic.cpp.\n");
#endif
}

PetscErrorCode FullLinearElastic::setShifts()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::setShifts in linearElastic.cpp\n");CHKERRQ(ierr);
#endif


  PetscInt Ii,Istart,Iend;
  PetscScalar v,bcRshift = 0;
  ierr = VecGetOwnershipRange(_bcRPShift,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = _fault->getTauSS(Ii);
    v = 0;
    //~v = 0.8*v;
    //~ bcRshift = v*_Ly/_muArrPlus[_Ny*_Nz-_Nz+Ii]; // use last values of muArr

    ierr = VecSetValue(_bcRPShift,Ii,bcRshift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcRPShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRPShift);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(_bcLMShift,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = _fault->getTauSS(Ii);
     //~v = 0;
     v = 0.8*v;
    bcRshift = -v*_Ly/_muArrMinus[_Ny*_Nz-_Nz+Ii]; // use last values of muArr
    ierr = VecSetValue(_bcLMShift,Ii,bcRshift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcLMShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcLMShift);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::setShifts in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLinearElastic::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
#endif

  PetscInt    Ii,Istart,Iend;
  PetscScalar u,y,z;
  ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uP,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispPlus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispPlus);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(_uM,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uM,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispMinus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispMinus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispMinus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLinearElastic::writeStep1D()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::writeStep1D in linearElastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    //~ierr = _sbpP->writeOps(_outputDir+"plus_");CHKERRQ(ierr);
    //~if (_problemType.compare("full")==0) { ierr = _sbpM->writeOps(_outputDir+"minus_");CHKERRQ(ierr); }
    ierr = _fault->writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispMinus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispMinus,_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispMinus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispMinusViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispMinus,_surfDispMinusViewer);CHKERRQ(ierr);
  }
  ierr = _fault->writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::writeStep in linearElastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLinearElastic::writeStep2D()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::writeStep2D in linearElastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();


  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::writeStep in linearElastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode FullLinearElastic::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::integrate in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  //~ ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::integrate in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}*/
/*
PetscErrorCode FullLinearElastic::setU()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::setU in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif


  PetscInt       Ii,Istart,Iend;
  PetscScalar    bcLPlus,bcRPlus,bcLMinus,bcRMinus,v;

  ierr = VecGetOwnershipRange(_bcRP,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcLP,1,&Istart,&bcLPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRP,1,&Istart,&bcRPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcLMinus,1,&Istart,&bcLMinus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRMinus,1,&Istart,&bcRMinus);CHKERRQ(ierr);


  for (Ii=Istart;Ii<Iend;Ii++) {
    v = bcLPlus + Ii*(bcRPlus-bcLPlus)/_Ny;//bcL:(bcR-bcL)/(dom.Ny-1):bcR
    ierr = VecSetValues(_uP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    v = bcLMinus + Ii*(bcRMinus - bcLMinus)/_Ny;
    ierr = VecSetValues(_uM,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_uP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_uM);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_uP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_uM);CHKERRQ(ierr);


  //~PetscPrintf(PETSC_COMM_WORLD,"_uP = \n");
  //~printVec(_uP);
  //~PetscPrintf(PETSC_COMM_WORLD,"_uM = \n");
  //~printVec(_uM);
  //~assert(0>1);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::setU in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
return ierr;
}


PetscErrorCode FullLinearElastic::setSigmaxy()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::setSigmaxy in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif


  PetscInt       Istart,Iend;
  PetscScalar    bcLPlus,bcRPlus,bcLMinus,bcRMinus,muP,muM;

  ierr = VecGetOwnershipRange(_bcRP,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcLP,1,&Istart,&bcLPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRP,1,&Istart,&bcRPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcLMinus,1,&Istart,&bcLMinus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRMinus,1,&Istart,&bcRMinus);CHKERRQ(ierr);
  ierr = VecGetValues(_muVecP,1,&Istart,&muP);CHKERRQ(ierr);
  muM = muP; // !!! NOT RIGHT

  ierr = VecSet(_sigma_xyMinus,muM*(bcRMinus - bcLMinus)/_Ly);CHKERRQ(ierr);
  ierr = VecSet(_sxy,muP*(bcRPlus - bcLPlus)/_Ly);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::setSigmaxy in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
return ierr;
}

PetscErrorCode FullLinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  // update boundaries: + side
  //~ ierr = VecCopy(*(varBegin+1),_bcLP);CHKERRQ(ierr);
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  // update boundaries: - side
  //~ ierr = VecCopy(*(varBegin+2),_bcRMinus);CHKERRQ(ierr);
  ierr = VecSet(_bcLMinus,-_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcLMinus,1.0,_bcLMShift);CHKERRQ(ierr);

   // solve for displacement: + side
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;


  // solve for displacement: - side
  ierr = _sbpM->setRhs(_rhsM,_bcLMinus,_bcRMinus,_bcTMinus,_bcBMinus);CHKERRQ(ierr);
  startTime = MPI_Wtime();
  ierr = KSPSolve(_kspM,_rhsM,_uM);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // set _uP, _uM analytically
  //~setU();


  // solve for shear stress
  //~ierr = MatMult(_sbpM->_muxDy_Iz,_uM,_sigma_xyMinus);CHKERRQ(ierr);
  //~ierr = MatMult(_sbpP->_muxDy_Iz,_uP,_sxy);CHKERRQ(ierr);
  _sbpM->muxDy(_uM,_sigma_xyMinus);
  _sbpP->muxDy(_uP,_sxy);



  // set shear stresses analytically
  //~setSigmaxy();

  //~PetscPrintf(PETSC_COMM_WORLD,"_sigma_xyMinus = \n");
  //~printVec(_sigma_xyMinus);
  //~PetscPrintf(PETSC_COMM_WORLD,"_sxy = \n");
  //~printVec(_sxy);
  //~assert(0>1);

  ierr = _fault->setTauQS(_sxy,_sigma_xyMinus);CHKERRQ(ierr);
  //~ ierr = _fault->d_dt(varBegin,dvarBegin);

  ierr = setSurfDisp();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}
*/

/*
// implicit/explicit time stepping
PetscErrorCode FullLinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}
*/

/*
// Outputs data at each time step.
PetscErrorCode FullLinearElastic::debug(const PetscReal time,const PetscInt stepCount,
                         const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage)
{
  PetscErrorCode ierr = 0;
#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRPlus,bcLMinus,uMinus,uPlus,psi,velMinus,velPlus,dPsi,
                 tauQSPlus,tauQSMinus;
#if ODEPRINT > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    y>0 |  %.9e  %.9e %.9e  %.9e \n",
              bcRPlus,uPlus,tauQSPlus,velPlus);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"    y<0 | %.9e %.9e %.9e %.9e \n",
              bcLMinus,uMinus,tauQSMinus,velMinus);CHKERRQ(ierr);
#endif
#endif
  return ierr;
}
*/

/*
PetscErrorCode FullLinearElastic::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // measure error between uAnal and _uP (the numerical solution)
  //~Vec diff;
  //~ierr = VecDuplicate(_uP,&diff);CHKERRQ(ierr);
  //~ierr = VecWAXPY(diff,-1.0,_uP,_uAnal);CHKERRQ(ierr);
  //~PetscScalar err;
  //~ierr = VecNorm(diff,NORM_2,&err);CHKERRQ(ierr);
  //~err = err/sqrt(_Ny*_Nz);

  //~double err = computeNormDiff_Mat(_sbpP->_H,_uP,_uAnal);
  double err = 1e8;

  PetscPrintf(PETSC_COMM_WORLD,"Ny = %i, dy = %e MMS err = %e, log2(err) = %e\n",_Ny,_dy,err,log2(err));

  return ierr;
}
*/



