
#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dy),_dz(D._dz),_y(&D._y),_z(&D._z),
  _kspTol(1e-10),
  _file(D._file),_outputDir(D._outputDir),_delim(D._delim),_inputDir(D._inputDir),
  _heatFieldsDistribution("unspecified"),_kFile("unspecified"),
  _rhoFile("unspecified"),_hFile("unspecified"),_cFile("unspecified"),
  _k(NULL),_rho(NULL),_c(NULL),_h(NULL),
  _TV(NULL),_vw(NULL),_timeV(NULL),
  _sbpT(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL),
  //~ _linSolver(D._linSolver),
  _linSolver("AMG"),
  _ksp(NULL),_pc(NULL),_I(NULL),_rhoC(NULL),_A(NULL),_pcMat(NULL),_sbpType(D._sbpType),_computePC(0),
  _linSolveTime(0),_linSolveCount(0),_pcRecomputeCount(0),
  _T(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcT);     PetscObjectSetName((PetscObject) _bcT, "_bcT");
  //~ VecSet(_bcT,273.0);
  PetscScalar bcTval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (0-_TDepths[0]) + _TVals[0];
  VecSet(_bcT,bcTval);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "bcB");
  PetscScalar bcBval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (_Lz-_TDepths[0]) + _TVals[0];
  VecSet(_bcB,bcBval);

  VecCreate(PETSC_COMM_WORLD,&_bcR);
  VecSetSizes(_bcR,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcR);     PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.0);

  VecDuplicate(_bcR,&_bcL); PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);

  // set fields
  setFields();

  // BC order: top, right, bottom, left; last argument makes A = Dzzmu + AT + AB
  // solve for steady state temperature profile
  {
    if (D._sbpType.compare("mfc")==0 || D._sbpType.compare("mc")==0) {
      _sbpT = new SbpOps_fc(D,_k,"Dirichlet","Dirichlet","Dirichlet","Dirichlet","z");
    }
    else if (D._sbpType.compare("mfc_coordTrans")==0) {
      _sbpT = new SbpOps_fc_coordTrans(D,_k,"Dirichlet","Dirichlet","Dirichlet","Dirichlet","z");
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
      assert(0); // automatically fail
    }
    computeSteadyStateTemp();
    setBCsforBE(); // update bcR with geotherm, correct sign for bcT, bcR, bcB
    //~ (*_sbpT).~SbpOps_fc();
    delete _sbpT;
  }
  if (D._sbpType.compare("mfc")==0 || D._sbpType.compare("mc")==0) {
    _sbpT = new SbpOps_fc(D,_k,"Dirichlet","Dirichlet","Dirichlet","Neumann","yz");
  }
  else if (D._sbpType.compare("mfc_coordTrans")==0) {
    _sbpT = new SbpOps_fc_coordTrans(D,_k,"Dirichlet","Dirichlet","Dirichlet","Neumann","yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }  // create identity matrix I


  Mat H;
  _sbpT->getH(H);
  if (D._sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    _sbpT->getCoordTrans(qy,rz,yq,zr);
    MatMatMatMult(yq,zr,H,MAT_INITIAL_MATRIX,1.0,&_I);
  }
  else {
    MatDuplicate(H,MAT_COPY_VALUES,&_I);
  }

  // set material properties to be 1
  VecSet(_rho,1.0);
  VecSet(_c,1.0);
  VecSet(_k,1.0);
  VecSet(_h,0.0);

  // create dt/rho*c matrix
  Vec rhoCV;
  VecDuplicate(_rho,&rhoCV);
  VecSet(rhoCV,1.0);
  VecPointwiseDivide(rhoCV,rhoCV,_rho);
  VecPointwiseDivide(rhoCV,rhoCV,_c);
  MatDuplicate(_I,MAT_DO_NOT_COPY_VALUES,&_rhoC);
  MatDiagonalSet(_rhoC,rhoCV,INSERT_VALUES);
  VecDestroy(&rhoCV);


  //~ setVecFromVectors(_T,_TVals,_TDepths); // FIX THIS LATER!!!

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

HeatEquation::~HeatEquation()
{

  KSPDestroy(&_ksp);
  MatDestroy(&_rhoC);
  MatDestroy(&_I);

  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);
  VecDestroy(&_h);

  VecDestroy(&_T);

  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);

  PetscViewerDestroy(&_TV);
  PetscViewerDestroy(&_vw);

  delete _sbpT;
}



// return temperature
PetscErrorCode HeatEquation::getTemp(Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::getTemp()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // return shallow copy of T:
  T = _T;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set temperature
PetscErrorCode HeatEquation::setTemp(Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::setTemp()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_T);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



// loads settings from the input text file
PetscErrorCode HeatEquation::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "HeatEquation::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
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

    if (var.compare("heatFieldsDistribution")==0) {
      _heatFieldsDistribution = line.substr(pos+_delim.length(),line.npos).c_str();
    }


    // names of each field's source file
    else if (var.compare("rhoFile")==0) {
      _rhoFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("kFile")==0) {
      _kFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hFile")==0) {
      _hFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("cFile")==0) {
      _cFile = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // if values are set by vector
    else if (var.compare("rhoVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoVals);
    }
    else if (var.compare("rhoDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoDepths);
    }

    else if (var.compare("kVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_kVals);
    }
    else if (var.compare("kDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_kDepths);
    }

    else if (var.compare("hVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_hVals);
    }
    else if (var.compare("hDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_hDepths);
    }

    else if (var.compare("cVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cVals);
    }
    else if (var.compare("cDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cDepths);
    }

    else if (var.compare("TVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_TVals);
    }
    else if (var.compare("TDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_TDepths);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

//parse input file and load values into data members
PetscErrorCode HeatEquation::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // load k
  ierr = VecCreate(PETSC_COMM_WORLD,&_k);CHKERRQ(ierr);
  ierr = VecSetSizes(_k,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_k);
  PetscObjectSetName((PetscObject) _k, "_k");
  ierr = loadVecFromInputFile(_k,_inputDir,_kFile);CHKERRQ(ierr);


  // load rho
  ierr = VecCreate(PETSC_COMM_WORLD,&_rho);CHKERRQ(ierr);
  ierr = VecSetSizes(_rho,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_rho);
  PetscObjectSetName((PetscObject) _rho, "_rho");
  ierr = loadVecFromInputFile(_rho,_inputDir,_rhoFile);CHKERRQ(ierr);

  // load h
  ierr = VecCreate(PETSC_COMM_WORLD,&_h);CHKERRQ(ierr);
  ierr = VecSetSizes(_h,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_h);
  PetscObjectSetName((PetscObject) _h, "_h");
  ierr = loadVecFromInputFile(_h,_inputDir,_hFile);CHKERRQ(ierr);

  // load c
  ierr = VecCreate(PETSC_COMM_WORLD,&_c);CHKERRQ(ierr);
  ierr = VecSetSizes(_c,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_c);
  PetscObjectSetName((PetscObject) _c, "_c");
  ierr = loadVecFromInputFile(_c,_inputDir,_cFile);CHKERRQ(ierr);



  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// initialize all fields
PetscErrorCode HeatEquation::setFields()
{
PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(*_y,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  VecDuplicate(_k,&_T);
  VecDuplicate(_bcL,&_kL);

  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_k,_kVals[0]);
    VecSet(_rho,_rhoVals[0]);
    VecSet(_h,_hVals[0]);
    VecSet(_c,_cVals[0]);
  }
  else {
    if (_heatFieldsDistribution.compare("mms")==0) {
      mapToVec(_k,MMS_he1_k,*_y,*_z);
      mapToVec(_rho,MMS_he1_rho,*_y,*_z);
      mapToVec(_c,MMS_he1_c,*_y,*_z);
      mapToVec(_h,MMS_he1_h,*_y,*_z);
      mapToVec(_T,MMS_he1_T,*_y,*_z,0.0);
    }
    else if (_heatFieldsDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
    else {
      ierr = setVecFromVectors(_k,_kVals,_kDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_rho,_rhoVals,_rhoDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_h,_hVals,_hDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_c,_cVals,_cDepths);CHKERRQ(ierr);
    }
  }

  setKL();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// create vector containing the left boundary values of k
PetscErrorCode HeatEquation::setKL()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting HeatEquation::setSurfDisp in heatEquation.cpp\n");CHKERRQ(ierr);
#endif

  PetscInt    Ii,Istart,Iend;
  PetscScalar k;
  ierr = VecGetOwnershipRange(_k,&Istart,&Iend);
  for (Ii=Istart;Ii<_Nz;Ii++) {
    ierr = VecGetValues(_k,1,&Ii,&k);CHKERRQ(ierr);
    ierr = VecSetValue(_kL,Ii,k,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_kL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_kL);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending HeatEquation::setSurfDisp in heatEquation.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode HeatEquation::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_heatFieldsDistribution.compare("mms")==0 ||
      _heatFieldsDistribution.compare("layered")==0 ||
      _heatFieldsDistribution.compare("loadFromFile")==0 );

  assert(_kVals.size() == _kDepths.size() );
  assert(_rhoVals.size() == _rhoDepths.size() );
  assert(_hVals.size() == _hDepths.size() );
  assert(_cVals.size() == _cDepths.size() );
  assert(_TVals.size() == _TDepths.size() );
  assert(_TVals.size() == 2 );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute T assuming that dT/dt and viscous strain rates = 0
PetscErrorCode HeatEquation::computeSteadyStateTemp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_Nz > 1) {
    // set up linear solver context
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD,&ksp);

    Mat A;
    _sbpT->getA(A);

    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
    //~ierr = KSPSetOperators(ksp,A,A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

    // perform computation of preconditioners now, rather than on first use
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    Vec rhs;
    VecDuplicate(_k,&rhs);
    _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);

    ierr = KSPSolve(ksp,rhs,_T);CHKERRQ(ierr);

    VecDestroy(&rhs);
    KSPDestroy(&ksp);
  }
  else {
    // set each field using it's vals and depths std::vectors
    if (_Nz == 1) { VecSet(_T,_TVals[0]); }
    else {
      if (_heatFieldsDistribution.compare("mms")==0) {
        if (_Nz == 1) { mapToVec(_T,MMS_T1D,*_y); }
        else { mapToVec(_T,MMS_T,*_y,*_z); }
        //~ mapToVec(_T,MMS_T,_Nz,_dy,_dz);
      }
      else if (_heatFieldsDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
      else { ierr = setVecFromVectors(_T,_TVals,_TDepths);CHKERRQ(ierr); }
    }
  }
  //~ierr = setVecFromVectors(_T,_TVals,_TDepths);CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::setupKSP(SbpOps* sbp, const PetscScalar dt)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting HeatEquation::setupKSP in heatequation.cpp\n");CHKERRQ(ierr);
#endif



  // A = I - dt/rho*c D2
  Mat D2;
  sbp->getA(D2);

  // create A
  MatMatMult(_rhoC,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_A);

  MatScale(_A,-dt);
  MatAXPY(_A,1.0,_I,DIFFERENT_NONZERO_PATTERN);

  // reuse old PC
  //~ if (_computePC>0) {
    //~ KSPSetOperators(_ksp,_A,_pcMat);
  //~ }
  //~ else {
    //~ if (_computePC==0) { ierr = MatConvert(_A,MATSAME,MAT_INITIAL_MATRIX,&_pcMat); CHKERRQ(ierr); }

    ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(_ksp,KSPCG); CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_A,_pcMat); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCICC); CHKERRQ(ierr);


    ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
    //~ // ierr = KSPSetOperators(_ksp,_A,_pcMat);CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_A,_A);CHKERRQ(ierr);
    //~ // ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE);CHKERRQ(ierr);

    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(_ksp,1e-9,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(_pc,4);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE);CHKERRQ(ierr);




    // use MUMPSCHOLESKY
    //~ ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    //~ ierr = KSPSetType(_ksp,KSPPREONLY);CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(_ksp,_A,_A);CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE);CHKERRQ(ierr);
    //~ PC pc;
    //~ ierr = KSPGetPC(_ksp,&pc);CHKERRQ(ierr);
    //~ PCSetType(pc,PCCHOLESKY);
    //~ PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    //~ PCFactorSetUpMatSolverPackage(pc);
  //~ }
  _computePC++;


  // perform computation of preconditioners now, rather than on first use
  double startTime = MPI_Wtime();
  ierr = KSPSetUp(_ksp);CHKERRQ(ierr);
  _factorTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending HeatEquation::setupKSP in heatequation.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode HeatEquation::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting HeatEquation::integrate in HeatEquation.cpp\n");CHKERRQ(ierr);
#endif


  OdeSolver *_quadEx = new RK32(5000,1.0,1e-3,"P"); //(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType
  ierr = _quadEx->setTolerance(1e-9); CHKERRQ(ierr);
  ierr = _quadEx->setTimeStepBounds(1e-3,1);CHKERRQ(ierr); // _minDeltaT,_maxDeltaT
  ierr = _quadEx->setTimeRange(0,1); CHKERRQ(ierr); // _initTime,_maxTime



  // control which fields are used to select step size
  int arrInds[] = {0}; // temp
  std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
  ierr = _quadEx->setErrInds(errInds);


  // put variables to be integrated into var
  std::vector<Vec>    _var; // holds variables for explicit integration in time
  Vec varT; VecDuplicate(_T,&varT);
  mapToVec(_T,MMS_he1_T,*_y,*_z,0.0);
  VecCopy(_T,varT);
  _var.push_back(varT);
  ierr = _quadEx->setInitialConds(_var);CHKERRQ(ierr);

  ierr = _quadEx->integrate(this);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::integrate in linearElastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// explicit time stepping for MMS
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting HeatEquation::d_dt in HeatEquation.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

    //~ d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sigmaxy,
      //~ const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt)

    //~ Vec slipVel; VecDuplicate(_bcL,&slipVel); VecSet(slipVel,0.0);
    //~ Vec tau; VecDuplicate(_bcL,&tau); VecSet(tau,0.0);
    //~ ierr = d_dt(time,slipVel,tau,NULL,NULL,NULL,NULL,*varBegin,*dvarBegin);

    ierr = d_dt_mms(time,*varBegin,*dvarBegin); CHKERRQ(ierr);

    //~ mapToVec(_T,MMS_he1_T,*_y,*_z,time);
    //~ VecCopy(*varBegin,_T);
    //~ mapToVec(*dvarBegin,MMS_he1_T_t,*_y,*_z,time);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLinearElastic::d_dt in linearElastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}
// Outputs data at each time step.
PetscErrorCode HeatEquation::debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin,const char *stage)
{
  PetscErrorCode ierr = 0;
  return ierr;
}
PetscErrorCode HeatEquation::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;

  if (stepCount == 0) {
    writeContext();
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV);CHKERRQ(ierr);
  }
  if ( stepCount % 1 == 0) {
    ierr = writeStep2D(stepCount);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV, "%.15e\n",time);CHKERRQ(ierr);
  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,time);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode HeatEquation::setMMSBoundaryConditions(const double time,
  std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
{
  PetscErrorCode ierr = 0;
  string funcName = "HeatEquation::setMMSBoundaryConditions";
  string fileName = "heatequation.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    z = _dz * Ii;
    y = 0;
    if (!bcLType.compare("Dirichlet")) { v = MMS_he1_T(y,z,time); }
    else if (!bcLType.compare("Neumann")) { v = MMS_he1_k(y,z)*MMS_he1_T_y(y,z,time); }
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!bcRType.compare("Dirichlet")) { v = MMS_he1_T(y,z,time); }
    else if (!bcRType.compare("Neumann")) { v = MMS_he1_k(y,z)*MMS_he1_T_y(y,z,time); }
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR);CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(_bcT,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = _dy * Ii;

    z = 0;
    if (!bcTType.compare("Dirichlet")) { v = MMS_he1_T(y,z,time); } // uAnal(y,z=0)
    else if (!bcTType.compare("Neumann")) { v = MMS_he1_k(y,z)*MMS_he1_T_z(y,z,time); }
    ierr = VecSetValues(_bcT,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    z = _Lz;
    if (!bcBType.compare("Dirichlet")) { v = MMS_he1_T(y,z,time); } // uAnal(y,z=Lz)
    else if (!bcBType.compare("Neumann")) { v = MMS_he1_k(y,z)*MMS_he1_T_z(y,z,time); }
    ierr = VecSetValues(_bcB,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcB);CHKERRQ(ierr);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}




// for thermomechanical coupling with explicity time stepping
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::d_dt";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_T); // so that the correct temperature is written out

  // left boundary: heat generated by fault motion
  Vec vel;
  VecDuplicate(_bcL,&vel);
  VecCopy(slipVel,vel);
  VecPointwiseMult(_bcL,tau,vel);
  VecDestroy(&vel);


  Mat A;
  _sbpT->getA(A);
  ierr = MatMult(A,T,dTdt); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(T,&rhs);
  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(dTdt,-1.0,rhs);CHKERRQ(ierr);
  VecDestroy(&rhs);

  Vec temp;
  VecDuplicate(dTdt,&temp);
  _sbpT->Hinv(dTdt,temp);
  VecCopy(temp,dTdt);
  VecDestroy(&temp);

  if (dgxy!=NULL && dgxz!=NULL) {
  // shear heating terms: simgaxy*dgxy + sigmaxz*dgxz (stresses times viscous strain rates)
  Vec shearHeat;
  VecDuplicate(sigmaxy,&shearHeat);
  VecSet(shearHeat,0.0);
  VecPointwiseMult(shearHeat,sigmaxy,dgxy);
  VecAXPY(dTdt,1.0,shearHeat);
  if (_Nz > 1) {
    VecSet(shearHeat,0.0);
    VecPointwiseMult(shearHeat,sigmaxz,dgxz);
    VecAXPY(dTdt,1.0,shearHeat);
  }
  VecDestroy(&shearHeat);
  }

  //~//!!! missing h*c term (heat production)

  VecPointwiseDivide(dTdt,dTdt,_rho);
  VecPointwiseDivide(dTdt,dTdt,_c);

  //~ VecSet(dTdt,0.0);
  mapToVec(dTdt,MMS_he1_T_t,*_y,*_z,time);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



// MMS test for thermomechanical coupling with explicity time stepping
PetscErrorCode HeatEquation::d_dt_mms(const PetscScalar time,const Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::d_dt";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_T); // so that the correct temperature is written out

  // update boundary conditions
  ierr = setMMSBoundaryConditions(time,"Dirichlet","Dirichlet","Neumann","Dirichlet"); CHKERRQ(ierr);


  Mat A;
  _sbpT->getA(A);
  ierr = MatMult(A,T,dTdt); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(T,&rhs);
  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(dTdt,-1.0,rhs);CHKERRQ(ierr);
  VecDestroy(&rhs);

  Vec temp;
  VecDuplicate(dTdt,&temp);
  _sbpT->Hinv(dTdt,temp);
  VecCopy(temp,dTdt);
  VecDestroy(&temp);

  VecPointwiseDivide(dTdt,dTdt,_rho);
  VecPointwiseDivide(dTdt,dTdt,_c);


  //~ VecSet(dTdt,0.0);
  //~ mapToVec(dTdt,MMS_he1_T_t,*_y,*_z,time);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical coupling using backward Euler (implicit time stepping)
PetscErrorCode HeatEquation::be(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // left boundary: heat generated by fault motion
  Vec vel;
  VecDuplicate(_bcL,&vel);
  VecCopy(slipVel,vel);
  VecPointwiseMult(_bcL,tau,vel);

  VecPointwiseDivide(_bcL,_bcL,_kL); // new

  VecScale(_bcL,-0.5); // negative sign is new
  VecDestroy(&vel);

  //~ VecSet(_bcL,0.0);
  //~ VecSet(_bcL,1);

  setupKSP(_sbpT,dt);

  Vec rhs,temp;
  VecDuplicate(sigmaxy,&rhs);
  VecDuplicate(sigmaxy,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);
  ierr = _sbpT->setRhs(temp,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  //~ assert(0);
  MatMult(_rhoC,temp,rhs);
  VecScale(rhs,dt);


  // add H * Tn to rhs
  VecSet(temp,0.0);
  _sbpT->H(To,temp);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(To,&temp1);
    VecSet(temp1,0.0);
    ierr = _sbpT->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(yq,temp,temp1);
    MatMult(zr,temp1,temp);
    //~ VecCopy(temp1,temp);
    VecDestroy(&temp1);
  }
  VecAXPY(rhs,1.0,temp);
  VecDestroy(&temp);


  //~ if (dgxy!=NULL && dgxz!=NULL) {
  //~ // shear heating terms: simgaxy*dgxy + sigmaxz*dgxz (stresses times viscous strain rates)
  //~ Vec shearHeat;
  //~ VecDuplicate(sigmaxy,&shearHeat);
  //~ VecSet(shearHeat,0.0);
  //~ VecPointwiseMult(shearHeat,sigmaxy,dgxy);
  //~ VecAXPY(dTdt,1.0,shearHeat);
  //~ if (_Nz > 1) {
    //~ VecSet(shearHeat,0.0);
    //~ VecPointwiseMult(shearHeat,sigmaxz,dgxz);
    //~ VecAXPY(dTdt,1.0,shearHeat);
  //~ }
  //~ VecDestroy(&shearHeat);
  //~ }
  //~ VecPointwiseDivide(shearHeat,shearHeat,_rho);
  //~ VecPointwiseDivide(shearHeat,shearHeat,_c);
  // if coordinate transform, weight this with yq*rz*H

  // solve for temperature
  double startTime = MPI_Wtime();
  KSPSolve(_ksp,rhs,_T);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  if (_linSolveCount==1) { _linSolveTime1 = _linSolveTime; }

  VecCopy(_T,T);

  VecDestroy(&rhs);
  MatDestroy(&_A);
  KSPDestroy(&_ksp);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set right boundary condition from computed geotherm
PetscErrorCode HeatEquation::setBCsforBE()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setBCs";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscInt    Istart,Iend,y;
  PetscScalar t = 0;
  ierr = VecGetOwnershipRange(_T,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    y = Ii/_Nz;
    if (y == _Ny-1) {
      PetscInt z = Ii-_Nz*(Ii/_Nz);
      //~PetscPrintf(PETSC_COMM_WORLD,"y=%i, z=%i Ii=%i\n",y,z,Ii);
      ierr = VecGetValues(_T,1,&Ii,&t);CHKERRQ(ierr);
      ierr = VecSetValue(_bcR,z,t,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR);CHKERRQ(ierr);

  // correct sign for Backward Euler
  VecScale(_bcT,-1.0);
  VecScale(_bcR,-1.0);
  VecScale(_bcB,-1.0);

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



PetscErrorCode HeatEquation::writeStep2D(const PetscInt stepCount)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif

  if (stepCount==0) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"T").c_str(),
                                 FILE_MODE_WRITE,&_TV);CHKERRQ(ierr);
    ierr = VecView(_T,_TV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_TV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"T").c_str(),
                                   FILE_MODE_APPEND,&_TV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcL").c_str(),
                                 FILE_MODE_WRITE,&_vw);CHKERRQ(ierr);
    ierr = VecView(_bcL,_vw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_vw);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcL").c_str(),
                                   FILE_MODE_APPEND,&_vw);CHKERRQ(ierr);

  }
  else {
    ierr = VecView(_T,_TV);CHKERRQ(ierr);
    ierr = VecView(_bcL,_vw);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::view()
{
  PetscErrorCode ierr = 0;
  //~ ierr = _quadEx->view();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Heat Equation Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent setting up linear solve context (e.g. factoring) (s): %g\n",_factorTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system 1st time (s): %g\n",_linSolveTime1);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return ierr;
}

// write out material properties
PetscErrorCode HeatEquation::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer    vw;

  std::string str = _outputDir + "k";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_k,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "rho";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_rho,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "c";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_c,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "h";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_h,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);


#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Fills vec with the linear interpolation between the pairs of points (vals,depths)
PetscErrorCode HeatEquation::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::setVecFromVectors";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
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
