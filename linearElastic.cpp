#include "linearElastic.hpp"

#define FILENAME "linearElastic.cpp"

using namespace std;


LinearElastic::LinearElastic(Domain&D,Vec& tau)
: _delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _isMMS(D._isMMS),_loadICs(D._loadICs),
  _bcLTauQS(0),_currTime(D._initTime),_stepCount(0),
  _vL(D._vL),
  _muVec(NULL),_rhoVec(NULL),_cs(NULL),_ay(NULL),_muVal(30.0),_rhoVal(3.0),
  _bcRShift(NULL),_surfDisp(NULL),
  _rhs(NULL),_u(NULL),_sxy(NULL),_sxz(NULL),
  _linSolver("unspecified"),_ksp(NULL),_pc(NULL),
  _kspTol(1e-10),
  _sbp(NULL),_sbpType(D._sbpType),
  _thermalCoupling("no"),
  _timeV1D(NULL),_timeV2D(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),_linSolveCount(0),
  _bcTType("Neumann"),_bcRType("Dirichlet"),_bcBType("Neumann"),_bcLType("Dirichlet"),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"\nStarting LinearElastic::LinearElastic in linearElastic.cpp.\n");
#endif

  loadSettings(D._file);
  checkInput();
  allocateFields();
  setMaterialParameters();
    // define boundary condition types
  _bcTType = "Neumann";
  _bcBType = "Neumann";
  _bcRType = "Dirichlet";
  _bcLType = "Dirichlet";
  if (_bcLTauQS==1) { _bcLType = "Neumann"; }
  if (_timeIntegrator.compare("WaveEq")==0){_bcLType = "Neumann";_bcRType = "Neumann";}

  // for MMS tests
  _bcTType = "Dirichlet";
  _bcBType = "Dirichlet";
  _bcRType = "Dirichlet";
  _bcLType = "Dirichlet";

  //~ setInitialConds(D,tau); // guess at steady-state configuration
  //~ if (_loadICs==1) {  } // load from previous simulation
  if (_inputDir.compare("unspecified") != 0) {
    loadFieldsFromFiles(); // load from previous simulation
  }


  setSurfDisp();

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
  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);
  VecDestroy(&_bcRShift);

  // body fields
  VecDestroy(&_muVec);
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
  for (map<string,PetscViewer>::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first]);
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

    else if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("bcLTauQS")==0) {
      _bcLTauQS = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    else if (var.compare("muPlus")==0) {
      _muVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("rhoPlus")==0) {
      _rhoVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr); // necessary for solving steady state power law
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
PetscErrorCode LinearElastic::computeMaxTimeStep(PetscScalar& maxTimeStep)
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



// allocate space for member fields
PetscErrorCode LinearElastic::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::allocateFields";
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

  VecSet(_muVec,_muVal);
  VecSet(_rhoVec,_rhoVal);
  VecPointwiseMult(_cs, _muVec, _rhoVec);

  PetscScalar *cs;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_cs,&Istart,&Iend);
  VecGetArray(_cs,&cs);
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
  ierr = loadVecFromInputFile(_bcL,_inputDir,"bcL"); CHKERRQ(ierr);

  // load bcR
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"bcR"); CHKERRQ(ierr);
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




// for steady state computations
// compute initial tauVisc (from guess at effective viscosity)
PetscErrorCode LinearElastic::getTauVisc(Vec& tauVisc, const PetscScalar ess_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::getTauVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (tauVisc == NULL) { VecDuplicate(_bcL,&tauVisc); }
  VecSet(tauVisc,1e20); // just needs to be really, really big


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::initiateVarSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // do nothing

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute steady state u, bcs
PetscErrorCode LinearElastic::updateSSa(Domain& D,map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::updateSSa";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif



  delete _sbp;
  KSPDestroy(&_ksp);

  // set up SBP operators
  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";
  if (_sbpType.compare("mc")==0) {
    _sbp = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_muVec);
    _sbp->setGrid(_y,_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setBCTypes(bcRType,bcTType,bcLType,bcBType);
  if (_timeIntegrator.compare("WaveEq")!=0){ _sbp->setMultiplyByH(1); }
  _sbp->computeMatrices(); // actually create the matrices

  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP(_sbp,_ksp,_pc);

  // set up boundary conditions
  VecSet(_bcR,0.0);
  VecCopy(varSS["tau"],_bcL);

  // compute uss that satisfies tau at left boundary
  _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  KSPDestroy(&_ksp);
  delete _sbp;
  _sbp = NULL;

  // extract boundary condition information from u
  Vec uL;
  PetscInt Istart,Iend;
  VecDuplicate(_bcL,&uL);
  PetscScalar minVal = 0;
  VecMin(_u,NULL,&minVal);
  PetscScalar v = 0.0;
  ierr = VecGetOwnershipRange(_u,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    // extract left boundary info for bcL
    if ( Ii < _Nz ) {
      ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
      v = v + abs(minVal);
      ierr = VecSetValues(uL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }

    // put right boundary data into bcR
    if ( Ii > (_Ny*_Nz - _Nz - 1) ) {
      PetscInt zI =  Ii - (_Ny*_Nz - _Nz);
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ny*Nz = %i, Ii = %i, zI = %i\n",_Ny*_Nz,Ii,zI);
      ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
      v = v + abs(minVal);
      ierr = VecSetValues(_bcRShift,1,&zI,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcRShift);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uL);CHKERRQ(ierr);
  VecCopy(_bcRShift,_bcR);

  if (!_bcLTauQS) {
    VecCopy(uL,_bcL);
  }

  VecDestroy(&uL);
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::updateSSb(Domain& D,map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::updateSSb";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // do nothing, this is only needed for the power-law problem

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// try to speed up spin up by starting closer to steady state
PetscErrorCode LinearElastic::prepareForIntegration(Domain& D, std::string _timeIntegrator)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::setInitialConds";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  setUpSBPContext(D, _timeIntegrator); // set up matrix operators

  if (_timeIntegrator.compare("WaveEq")!=0){
    _sbp->muxDy(_u,_sxy); // initialize for shear stress
    if (_isMMS) { setMMSInitialConditions(); }

    // extract boundary condition information from u
    Vec uL;
    VecDuplicate(_bcL,&uL);
    PetscScalar minVal = 0;
    VecMin(_u,NULL,&minVal);
    PetscScalar v = 0.0;
    PetscInt Istart,Iend;
    ierr = VecGetOwnershipRange(_u,&Istart,&Iend);CHKERRQ(ierr);
    for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
      // put left boundary info into fault slip vector
      if ( Ii < _Nz ) {
        ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
        v = v + abs(minVal);
        ierr = VecSetValues(uL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }

      // put right boundary data into bcR
      if ( Ii > (_Ny*_Nz - _Nz - 1) ) {
        PetscInt zI =  Ii - (_Ny*_Nz - _Nz);
        //~ PetscPrintf(PETSC_COMM_WORLD,"Ny*Nz = %i, Ii = %i, zI = %i\n",_Ny*_Nz,Ii,zI);
        ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
        v = v + abs(minVal);
        ierr = VecSetValues(_bcRShift,1,&zI,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(_bcRShift);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(uL);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(_bcRShift);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(uL);CHKERRQ(ierr);
    VecCopy(_bcRShift,_bcR);

    if (!_bcLTauQS) {
      VecCopy(uL,_bcL);
    }

    VecDestroy(&uL);
  }
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute bcL from u so that fault's initial conditions can be set from it
PetscErrorCode LinearElastic::setInitialSlip(Vec& out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::setInitialSlip";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  std::string fileName = _inputDir + "slip";
  bool fileExists = doesFileExist(fileName);
  if (fileExists) {
    // load slip
    ierr = loadVecFromInputFile(out,_inputDir,"slip"); CHKERRQ(ierr);
  }
  else {
    PetscScalar minVal = 0;
    VecMin(_u,NULL,&minVal);
    ierr = VecGetOwnershipRange(_u,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart;Ii<Iend;Ii++) {
      if (Ii<_Nz) {
        ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
        v = 2. * (v + abs(minVal));
        ierr = VecSetValues(out,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(out);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(out);CHKERRQ(ierr);
  }


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// set up SBP operators
PetscErrorCode LinearElastic::setUpSBPContext(Domain& D, std::string _timeIntegrator)
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
    _sbp->setGrid(_y,_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setBCTypes(_bcRType,_bcTType,_bcLType,_bcBType);
  if (_timeIntegrator.compare("WaveEq")!=0){ _sbp->setMultiplyByH(1); }
  _sbp->computeMatrices(); // actually create the matrices


  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP(_sbp,_ksp,_pc);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


PetscErrorCode LinearElastic::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::setSurfDisp in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode LinearElastic::view(const double totRunTime)
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

  //~ierr = KSPView(_ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode LinearElastic::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = _sbp->writeOps(_outputDir + "ops_u_"); CHKERRQ(ierr);

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

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


  ierr = writeVec(_muVec,_outputDir + "mu"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::writeStep1D(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::writeStep1D";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();
  _stepCount = stepCount;

  if (_timeV1D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);

    _viewers["surfDisp"] = initiateViewer(_outputDir + "surfDisp");
    _viewers["bcL"] = initiateViewer(_outputDir + "bcL");
    _viewers["bcR"] = initiateViewer(_outputDir + "bcR");

    ierr = VecView(_surfDisp,_viewers["surfDisp"]); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["bcL"]); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["bcR"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["surfDisp"],_outputDir + "surfDisp");
    ierr = appendViewer(_viewers["bcL"],_outputDir + "bcL");
    ierr = appendViewer(_viewers["bcR"],_outputDir + "bcR");
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_surfDisp,_viewers["surfDisp"]); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["bcL"]); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["bcR"]); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



PetscErrorCode LinearElastic::writeStep2D(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::writeStep2D";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();


  if (_timeV2D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);

    _viewers["u"] = initiateViewer(_outputDir + "u");
    _viewers["sxy"] = initiateViewer(_outputDir + "sxy");

    ierr = VecView(_u,_viewers["u"]); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["sxy"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["u"],_outputDir + "u");
    ierr = appendViewer(_viewers["sxy"],_outputDir + "sxy");
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = VecView(_u,_viewers["u"]); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["sxy"]); CHKERRQ(ierr);
  }


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set slip based on uP
  Vec slip;
  VecDuplicate(_bcL,&slip);
  setInitialSlip(slip);
  varEx["slip"] = slip;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _currTime = time;
  if (_bcLTauQS==0) { // var holds slip, bcL is displacement at y=0+
    ierr = VecCopy(varEx.find("slip")->second,_bcL);CHKERRQ(ierr);
    ierr = VecScale(_bcL,0.5);CHKERRQ(ierr);
  } // else do nothing
  ierr = VecSet(_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::updateTemperature()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // do nothing

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::getSigmaDev(Vec& sdev)
{
  return 0;
}

// explicit time stepping
PetscErrorCode LinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
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

PetscErrorCode LinearElastic::initiateIntegrandWave(std::string _initialU, map<string,Vec>& _varEx){
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::d_dt_WaveEq in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  if(_initialU.compare("gaussian")==0){
    // PetscScalar yy[1], zz[1];
    // PetscScalar uu;

    PetscScalar *u, *uPrev, *y, *z;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_varEx["u"],&Istart,&Iend);
    VecGetArray(_varEx["u"],&u);
    VecGetArray(_varEx["uPrev"],&uPrev);
    VecGetArray(*_y, &y);
    VecGetArray(*_z, &z);

    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      u[Jj] = exp(-pow( y[Jj]-0.5*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.5*(_Lz), 2) /5);
      uPrev[Jj] = exp(-pow( y[Jj]-0.5*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.5*(_Lz), 2) /5);
      Jj++;
    }
    VecRestoreArray(*_y,&y);
    VecRestoreArray(*_z,&z);
    VecRestoreArray(_varEx["u"],&u);
    VecRestoreArray(_varEx["uPrev"],&uPrev);

    // Create matrix _ay
    VecDuplicate(*_y, &_ay);
    VecSet(_ay, 0.0);

    PetscScalar *yy, *zz, *ay;
    VecGetOwnershipRange(*_y,&Istart,&Iend);
    VecGetArray(_ay,&ay);
    VecGetArray(*_y, &yy);
    VecGetArray(*_z, &zz);
    Jj = 0;
    PetscScalar alphay, alphaz;
    PetscScalar dy,dz;
    if (_sbpType.compare("mfc_coordTrans")==0) { dy = 1./(_Ny-1); dz = 1./(_Nz-1); }
    else { dy = _Ly/(_Ny-1); dz = _Lz/(_Nz-1); }
    if (_order == 2 ) { alphay = 0.5 * dy; alphaz = 0.5 * dz; }
    if (_order == 4 ) { alphay = 0.4567e4/0.14400e5 * dy; alphaz = 0.4567e4/0.14400e5 * dz; }

    for (Ii=Istart;Ii<Iend;Ii++) {
      ay[Jj] = 0;
      if (zz[Jj] == 0){ay[Jj] = -0.5 * alphaz;}
      if (zz[Jj] == _Lz){ay[Jj] = -0.5 * alphaz;}
      if (yy[Jj] == 0){ay[Jj] = -0.5 * alphay;}
      if (yy[Jj] == _Ly){ay[Jj] = -0.5 * alphay;}
      Jj++;
    }
    VecRestoreArray(*_y,&y);
    VecRestoreArray(*_z,&z);
    VecRestoreArray(_ay,&ay);

     //~ for (Ii=Istart;Ii<Iend;Ii++) {
       //~ PetscInt II[0];
       //~ II[0] = Ii;
       //~ VecGetValues(*_y, 1, II, yy);
       //~ VecGetValues(*_z, 1, II, zz);
       //~ ay=0;
       //~ if (zz[0] == 0){ay[Jj] = -1.0 * alphaz;}
       //~ if (zz[0] == _Lz){ay[Jj] = -1.0 * alphaz;}
       //~ if (yy[0] == 0){ay[Jj] = -1.0 * alphay;}
       //~ if (yy[0] == _Ly){ay[Jj] = -1.0 * alphay;}
       //~ VecSetValues(_ay,1,&Ii,&ay,INSERT_VALUES);
       //~ }
     //~ VecAssemblyBegin(_ay);
     //~ VecAssemblyEnd(_ay);
    ierr = VecPointwiseMult(_ay, _ay, _cs);
    }
    _u = _varEx["u"];
    return ierr;
}

PetscErrorCode LinearElastic::d_dt_WaveEq(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscReal _deltaT)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::d_dt_WaveEq in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  Mat A;
  ierr = _sbp->getA(A);

  // Update the laplacian
  Vec Laplacian;
  VecDuplicate(*_y, &Laplacian);
  ierr = MatMult(A, varEx["u"], Laplacian);
  ierr = VecCopy(Laplacian, dvarEx["u"]);

  // Apply the time step
  Vec uNext, correction, previous, ones;

  VecDuplicate(varEx["u"], &ones);
  VecDuplicate(varEx["u"], &correction);
  VecSet(ones, 1.0);
  VecSet(correction, 0.0);
  ierr = VecAXPY(correction, _deltaT, _ay);
  ierr = VecAXPY(correction, -1.0, ones);

  VecDuplicate(varEx["u"], &previous);
  VecSet(previous, 0.0);
  ierr = VecPointwiseMult(previous, correction, varEx["uPrev"]);

  VecDuplicate(varEx["u"], &uNext);
  VecSet(uNext, 0.0);
  ierr = VecAXPY(uNext, pow(_deltaT, 2), dvarEx["u"]);
  ierr = VecPointwiseDivide(uNext, uNext, _rhoVec);

  ierr = VecAXPY(uNext, 2, varEx["u"]);
  ierr = VecAXPY(uNext, 1, previous);
  ierr = VecAXPY(correction, 2, ones);
  ierr = VecPointwiseDivide(uNext, uNext, correction);
  ierr = VecCopy(varEx["u"], varEx["uPrev"]);
  ierr = VecCopy(uNext, varEx["u"]);
  VecDestroy(&uNext);
  VecDestroy(&ones);
  VecDestroy(&correction);
  VecDestroy(&previous);

  _u = varEx["u"];
  // _u = _ay;

  PetscPrintf(PETSC_COMM_WORLD,"");
  #if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::d_dt_WaveEq in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  return ierr;
}


// explicit time stepping
PetscErrorCode LinearElastic::d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif

  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = _sbp->muxDy(_u,_sxy); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode LinearElastic::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  ierr = d_dt_eqCycle(time,varEx,dvarEx);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::d_dt IMEX in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode LinearElastic::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::d_dt_mms in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
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


  // set rhs, including body source term
  setMMSBoundaryConditions(time); // modifies _bcL,_bcR,_bcT, and _bcB
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

  // force u to be correct
  //~ ierr = mapToVec(_u,zzmms_uA,*_y,*_z,time); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::d_dt_mms in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode LinearElastic::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::setMMSBoundaryConditions";
  string fileName = "linearElastic.cpp";
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}

PetscErrorCode LinearElastic::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  string funcName = "LinearElastic::setMMSInitialConditions";
  string fileName = "linearElastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif


  PetscScalar time = _currTime;

  Vec source,Hxsource;
  VecDuplicate(_u,&source);
  VecDuplicate(_u,&Hxsource);

  if (_Nz == 1) { mapToVec(source,zzmms_uSource1D,*_y,_currTime); }
  else { mapToVec(source,zzmms_uSource,*_y,*_z,_currTime); }
  //~ ierr = mapToVec(source,zzmms_uSource,_Nz,_dy,_dz,time); CHKERRQ(ierr);
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
  writeVec(_u,"data/mms_uuu");
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



// Outputs data at each time step.
PetscErrorCode LinearElastic::debug(const PetscReal time,const PetscInt stepCount,
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

  ierr= VecGetOwnershipRange(_bcR,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcR,1,&Istart,&bcRval);CHKERRQ(ierr);

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

PetscErrorCode LinearElastic::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  _currTime = time;

  // measure error between analytical and numerical solution
  Vec uA;
  VecDuplicate(_u,&uA);
  if (_Nz == 1) { mapToVec(uA,zzmms_uA1D,*_y,_currTime); }
  else { mapToVec(uA,zzmms_uA,*_y,*_z,_currTime); }

  Vec sigmaxyA;
  VecDuplicate(_u,&sigmaxyA);
  //~ mapToVec(sigmaxyA,zzmms_sigmaxy,_Nz,_dy,_dz,_currTime);
    if (_Nz == 1) { mapToVec(sigmaxyA,zzmms_sigmaxy1D,*_y,_currTime); }
  else { mapToVec(sigmaxyA,zzmms_sigmaxy,*_y,*_z,_currTime); }


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


