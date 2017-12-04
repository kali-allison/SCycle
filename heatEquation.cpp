
#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _heatEquationType("transient"),_isMMS(D._isMMS),
  _file(D._file),_outputDir(D._outputDir),_delim(D._delim),_inputDir(D._inputDir),
  _heatFieldsDistribution("unspecified"),_kFile("unspecified"),
  _rhoFile("unspecified"),_hFile("unspecified"),_cFile("unspecified"),
  _surfaceHeatFlux(NULL),_heatFlux(NULL),
  _wViscShearHeating("yes"),_wFrictionalHeating("yes"),
  _sbpType(D._sbpType),_sbpT(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL),
  _linSolver("AMG"),_kspTol(1e-10),
  _ksp(NULL),_pc(NULL),_I(NULL),_rcInv(NULL),_B(NULL),_pcMat(NULL),_D2ath(NULL),
  _MapV(NULL),_Gw(NULL),_omega(NULL),_w(0),
  _linSolveTime(0),_factorTime(0),_beTime(0),_writeTime(0),_miscTime(0),
  _linSolveCount(0),_stride1D(D._stride1D),_stride2D(D._stride2D),
  _dT(NULL),_Tamb(NULL),_k(NULL),_rho(NULL),_c(NULL),_h(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  setFields(D);
  constructMapV();
  if (D._loadICs==1) { loadFieldsFromFiles(); }
  if (!_isMMS && D._loadICs!=1) { computeInitialSteadyStateTemp(D); }

  if (_heatEquationType.compare("transient")==0 ) { setUpTransientProblem(D); }
  else if (_heatEquationType.compare("steadyState")==0 ) { setUpSteadyStateProblem(D); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

HeatEquation::~HeatEquation()
{
  KSPDestroy(&_ksp);
  MatDestroy(&_B);
  MatDestroy(&_rcInv);
  MatDestroy(&_I);
  MatDestroy(&_D2ath);
  MatDestroy(&_pcMat);

  MatDestroy(&_MapV);
  VecDestroy(&_Gw);
  VecDestroy(&_omega);

  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);
  VecDestroy(&_h);

  VecDestroy(&_dT);
  VecDestroy(&_Tamb);
  VecDestroy(&_heatFlux);
  VecDestroy(&_surfaceHeatFlux);
  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);

  for (map<string,PetscViewer>::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first]);
  }

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

  VecWAXPY(T,1.0,_Tamb,_dT);

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

  VecCopy(T,_dT);

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

    if (var.compare("heatEquationType")==0) {
      _heatEquationType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    if (var.compare("heatFieldsDistribution")==0) {
      _heatFieldsDistribution = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("withViscShearHeating")==0) {
      _wViscShearHeating = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("withFrictionalHeating")==0) {
      _wFrictionalHeating = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // linear solver settings
    else if (var.compare("linSolver_heateq")==0) {
      _linSolver = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("kspTol_heateq")==0) {
      _kspTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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

    else if (var.compare("shearZoneWidth")==0) {
      _w = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
  //~ string vecSourceFile = _inputDir + "_k";
  //~ ierr = loadVecFromInputFile(_k,_inputDir,_kFile);CHKERRQ(ierr);


  //~ // load rho
  //~ ierr = VecCreate(PETSC_COMM_WORLD,&_rho);CHKERRQ(ierr);
  //~ ierr = VecSetSizes(_rho,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  //~ ierr = VecSetFromOptions(_rho);
  //~ PetscObjectSetName((PetscObject) _rho, "_rho");
  //~ ierr = loadVecFromInputFile(_rho,_inputDir,_rhoFile);CHKERRQ(ierr);

  //~ // load h
  //~ ierr = VecCreate(PETSC_COMM_WORLD,&_h);CHKERRQ(ierr);
  //~ ierr = VecSetSizes(_h,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  //~ ierr = VecSetFromOptions(_h);
  //~ PetscObjectSetName((PetscObject) _h, "_h");
  //~ ierr = loadVecFromInputFile(_h,_inputDir,_hFile);CHKERRQ(ierr);

  //~ // load c
  //~ ierr = VecCreate(PETSC_COMM_WORLD,&_c);CHKERRQ(ierr);
  //~ ierr = VecSetSizes(_c,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  //~ ierr = VecSetFromOptions(_c);
  //~ PetscObjectSetName((PetscObject) _c, "_c");
  //~ ierr = loadVecFromInputFile(_c,_inputDir,_cFile);CHKERRQ(ierr);


  // load T0 (background geotherm)
  ierr = loadVecFromInputFile(_Tamb,_inputDir,"T0"); CHKERRQ(ierr);

  // load dT (perturbation)
  ierr = loadVecFromInputFile(_dT,_inputDir,"dT"); CHKERRQ(ierr);
  writeVec(_dT,_outputDir+"init1_dT");

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// initialize all fields
PetscErrorCode HeatEquation::setFields(Domain& D)
{
PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcT);
  PetscObjectSetName((PetscObject) _bcT, "_bcT");
  PetscScalar bcTval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (0-_TDepths[0]) + _TVals[0];
  VecSet(_bcT,bcTval);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "bcB");
  PetscScalar bcBval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (_Lz-_TDepths[0]) + _TVals[0];
  VecSet(_bcB,bcBval);

  VecCreate(PETSC_COMM_WORLD,&_bcR);
  VecSetSizes(_bcR,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcR);
  PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.0);

  VecDuplicate(_bcR,&_bcL); PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);


  // set material properties
  VecDuplicate(*_y,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  VecDuplicate(_k,&_dT);
  VecDuplicate(_k,&_Q); VecSet(_Q,0.);
  VecDuplicate(_k,&_omega); VecSet(_omega,0.);

  VecDuplicate(_dT,&_Tamb);
  VecCopy(_dT,_Tamb);
  VecSet(_dT,0.0);

  // heat flux variables
  VecDuplicate(_bcT,&_surfaceHeatFlux); VecSet(_surfaceHeatFlux,0.0);
  VecDuplicate(_k,&_heatFlux); VecSet(_heatFlux,0.0);

  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_k,_kVals[0]);
    VecSet(_rho,_rhoVals[0]);
    VecSet(_h,_hVals[0]);
    VecSet(_c,_cVals[0]);
    VecSet(_dT,_TVals[0]);
  }
  else {
    if (_isMMS) {
      mapToVec(_k,zzmms_k,*_y,*_z);
      mapToVec(_rho,zzmms_rho,*_y,*_z);
      mapToVec(_c,zzmms_c,*_y,*_z);
      mapToVec(_h,zzmms_h,*_y,*_z);
      mapToVec(_Tamb,zzmms_T,*_y,*_z,D._initTime);
      mapToVec(_dT,zzmms_dT,*_y,*_z,D._initTime);
      setMMSBoundaryConditions(D._initTime,"Dirichlet","Dirichlet","Dirichlet","Dirichlet");
    }
    else {
      ierr = setVecFromVectors(_k,_kVals,_kDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_rho,_rhoVals,_rhoDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_h,_hVals,_hDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_c,_cVals,_cDepths);CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
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

  assert(_heatEquationType.compare("transient")==0 ||
      _heatEquationType.compare("steadyState")==0 );

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


PetscErrorCode HeatEquation::constructMapV()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::constructMapV";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  MatCreate(PETSC_COMM_WORLD,&_MapV);
  MatSetSizes(_MapV,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);
  MatSetFromOptions(_MapV);
  MatMPIAIJSetPreallocation(_MapV,_Ny*_Nz,NULL,_Ny*_Nz,NULL);
  MatSeqAIJSetPreallocation(_MapV,_Ny*_Nz,NULL);
  MatSetUp(_MapV);

  PetscScalar v=1.0;
  PetscInt Ii=0,Istart=0,Iend=0,Jj=0;
  MatGetOwnershipRange(_MapV,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    Jj = Ii % _Nz;
    MatSetValues(_MapV,1,&Ii,1,&Jj,&v,INSERT_VALUES);
  }
  MatAssemblyBegin(_MapV,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_MapV,MAT_FINAL_ASSEMBLY);

  //~ writeMat(_MapV,"MapV");

  // construct Gw = exp(-y^2/(2*w)) / sqrt(2*pi)/w
  VecDuplicate(_Tamb,&_Gw);
  VecSet(_Gw,0.);

  PetscScalar *y,*g;
  VecGetOwnershipRange(_Gw,&Istart,&Iend);
  VecGetArray(*_y,&y);
  VecGetArray(_Gw,&g);
  Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    g[Jj] = exp(-y[Jj]*y[Jj] / (2.*_w*_w)) / sqrt(2. * M_PI) / _w;
    Jj++;
  }
  VecRestoreArray(*_y,&y);
  VecRestoreArray(_Gw,&g);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute T assuming that dT/dt and viscous strain rates = 0
PetscErrorCode HeatEquation::computeInitialSteadyStateTemp(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_sbpType.compare("mc")==0) {
    _sbpT = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpT = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpT = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_k);
    if (_Ny > 1 && _Nz > 1) { _sbpT->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbpT->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbpT->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbpT->setBCTypes("Dirichlet","Dirichlet","Dirichlet","Dirichlet");
  _sbpT->setMultiplyByH(1);
  _sbpT->setLaplaceType("z");
  _sbpT->computeMatrices(); // actually create the matrices


  if (_Nz > 1) {
    // set up linear solver context
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD,&ksp);

    Mat A;
    _sbpT->getA(A);

    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
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

    // solve for temperature
    double startTime = MPI_Wtime();
    ierr = KSPSolve(ksp,rhs,_Tamb);CHKERRQ(ierr);
    _linSolveTime += MPI_Wtime() - startTime;
    _linSolveCount++;


    VecDestroy(&rhs);
    KSPDestroy(&ksp);
  }
  else{
    VecSet(_Tamb,_TVals[0]);
  }
  VecSet(_dT,0.0);
  computeHeatFlux();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Solve steady-state heat equation
PetscErrorCode HeatEquation::setupKSP_SS(SbpOps* sbp)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    std::string funcName = "HeatEquation::setupKSP_SS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Mat A;
  sbp->getA(A);

  // reuse preconditioner at each time step
  ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
  ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetOperators(_ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(_ksp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
  ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
  ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
  ierr = KSPSetTolerances(_ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(_pc,4);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE);CHKERRQ(ierr);

  // use MUMPSCHOLESKY
  //~ ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
  //~ ierr = KSPSetType(_ksp,KSPPREONLY);CHKERRQ(ierr);
  //~ ierr = KSPSetOperators(_ksp,A,A);CHKERRQ(ierr);
  //~ ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE);CHKERRQ(ierr);
  //~ PC pc;
  //~ ierr = KSPGetPC(_ksp,&pc);CHKERRQ(ierr);
  //~ PCSetType(pc,PCCHOLESKY);
  //~ PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
  //~ PCFactorSetUpMatSolverPackage(pc);

  double startTime = MPI_Wtime();
  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(_ksp);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(_ksp);CHKERRQ(ierr);
  _factorTime += MPI_Wtime() - startTime;

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
    std::string funcName = "HeatEquation::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif



  // create: A = I - dt/rho*c D2
  //~ Mat D2;
  //~ sbp->getA(D2);
  //~ MatMatMult(_rcInv,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_B);
  //~ MatScale(_B,-dt);
  //~ MatAXPY(_B,1.0,_I,DIFFERENT_NONZERO_PATTERN);

  // new version
  MatDuplicate(_D2ath,MAT_COPY_VALUES,&_B);
  MatScale(_B,-dt);
  MatAXPY(_B,1.0,_I,SUBSET_NONZERO_PATTERN);


  // set up KSP

    // don't reuse preconditioner
    ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_B,_B);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_ksp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(_ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(_pc,4);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE);CHKERRQ(ierr);



    // reuse preconditioner at each time step
    //~ ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    //~ ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(_ksp,_B,_B);CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE);CHKERRQ(ierr);
    //~ ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    //~ ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
    //~ ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
    //~ ierr = KSPSetTolerances(_ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    //~ ierr = PCFactorSetLevels(_pc,4);CHKERRQ(ierr);
    //~ ierr = KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE);CHKERRQ(ierr);


    // use MUMPSCHOLESKY
    //~ KSPDestroy(&_ksp);
    //~ ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    //~ ierr = KSPSetType(_ksp,KSPPREONLY);CHKERRQ(ierr);
    //~ ierr = KSPSetOperators(_ksp,_B,_B);CHKERRQ(ierr);
    //~ ierr = KSPSetReusePreconditioner(_ksp,PETSC_FALSE);CHKERRQ(ierr);
    //~ PC pc;
    //~ ierr = KSPGetPC(_ksp,&pc);CHKERRQ(ierr);
    //~ PCSetType(pc,PCCHOLESKY);
    //~ PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    //~ PCFactorSetUpMatSolverPackage(pc);

    // accept command line options
    ierr = KSPSetFromOptions(_ksp);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  double startTime = MPI_Wtime();
  ierr = KSPSetUp(_ksp);CHKERRQ(ierr);
  _factorTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// explicit time stepping for MMS
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    //~ d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sigmaxy,
      //~ const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt)

    //~ Vec slipVel; VecDuplicate(_bcL,&slipVel); VecSet(slipVel,0.0);
    //~ Vec tau; VecDuplicate(_bcL,&tau); VecSet(tau,0.0);
    //~ ierr = d_dt(time,slipVel,tau,NULL,NULL,NULL,NULL,*varBegin,*dvarBegin);

    //~ ierr = d_dt_mms(time,*varBegin,*dvarBegin); CHKERRQ(ierr);

    //~ mapToVec(_dT,zzmms_he1_T,*_y,*_z,time);
    //~ VecCopy(*varBegin,_dT);
    //~ mapToVec(*dvarBegin,zzmms_he1_T_t,*_y,*_z,time);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated implicity into varIm
  Vec T;
  VecDuplicate(_Tamb,&T);
  VecWAXPY(T,1.0,_Tamb,_dT);
  varIm["Temp"] = T;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ Vec T;
  //~ VecCopy(varIm.find("Temp")->second,_dT);
  // not needed for implicit solve

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::setMMSBoundaryConditions(const double time,
  std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setMMSBoundaryConditions";
    string fileName = "heatequation.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
    y = 0;
    if (!bcLType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
    else if (!bcLType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_y(y,z,time); }
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!bcRType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
    else if (!bcRType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_y(y,z,time); }
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
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
      if (!bcTType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
      else if (!bcTType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_z(y,z,time); }
      ierr = VecSetValues(_bcT,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);

      z = _Lz;
      if (!bcBType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
      else if (!bcBType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_z(y,z,time); }
      ierr = VecSetValues(_bcB,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
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

PetscErrorCode HeatEquation::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setMMSBoundaryConditions";
    string fileName = "heatequation.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // measure error between analytical and numerical solution
  Vec dTA;
  VecDuplicate(_dT,&dTA);

  //~ if (_Nz == 1) { mapToVec(dTA,zzmms_uA1D,*_y,time); }
  //~ else { mapToVec(dTA,zzmms_dT,*_y,*_z,time); }
  //~ mapToVec(dTA,zzmms_dT,*_y,*_z,time);
  mapToVec(dTA,zzmms_T,*_y,*_z,time);

  writeVec(dTA,_outputDir+"mms_dTA");
  writeVec(_dT,_outputDir+"mms_dT");

  writeVec(_bcL,_outputDir+"mms_he_bcL");
  writeVec(_bcR,_outputDir+"mms_he_bcR");
  writeVec(_bcT,_outputDir+"mms_he_bcT");
  writeVec(_bcB,_outputDir+"mms_he_bcB");

  double err2u = computeNormDiff_2(_dT,dTA);

  PetscPrintf(PETSC_COMM_WORLD,"%i %3i %.4e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u));

  VecDestroy(&dTA);
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}


// for thermomechanical coupling with explicit time stepping
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz,const Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::d_dt";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_dT); // so that the correct temperature is written out

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
  mapToVec(dTdt,zzmms_T_t,*_y,*_z,time);


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
    string funcName = "HeatEquation::d_dt_mms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_dT); // so that the correct temperature is written out

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
  //~ mapToVec(dTdt,zzmms_he1_T_t,*_y,*_z,time);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical coupling using backward Euler (implicit time stepping)
PetscErrorCode HeatEquation::be(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double beStartTime = MPI_Wtime();

  if (_isMMS && _heatEquationType.compare("transient")==0) {
    assert(0);
    //~ be_transient(time,slipVel,tau,sigmadev,dgxy,dgxz,T,To,dt);
  }
  else if (_isMMS && _heatEquationType.compare("steadyState")==0) {
    be_steadyStateMMS(time,slipVel,tau,sigmadev,dgxy,dgxz,T,To,dt);
  }
  else if (!_isMMS && _heatEquationType.compare("transient")==0) {
    be_transient(time,slipVel,tau,sigmadev,dgxy,dgxz,T,To,dt);
  }
  else if (!_isMMS && _heatEquationType.compare("steadyState")==0) {
    be_steadyState(time,slipVel,tau,sigmadev,dgxy,dgxz,T,To,dt);
  }

  computeHeatFlux();

  _beTime += MPI_Wtime() - beStartTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// for thermomechanical problem using implicit time stepping (backward Euler)
PetscErrorCode HeatEquation::be_transient(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_transient";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
//~ double startMiscTime = MPI_Wtime();
//~ _miscTime += MPI_Wtime() - startMiscTime;

  // set up matrix
  MatCopy(_D2ath,_B,SAME_NONZERO_PATTERN);
  MatScale(_B,-dt);
  MatAXPY(_B,1.0,_I,SUBSET_NONZERO_PATTERN);
  ierr = KSPSetOperators(_ksp,_B,_B);CHKERRQ(ierr);

  //~ setupKSP(_sbpT,dt);

  // set up boundary conditions and source terms
  Vec rhs,temp;
  VecDuplicate(_dT,&rhs);
  VecDuplicate(_dT,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);
  VecSet(_Q,0.0);

  // left boundary: heat generated by fault motion
  if (_wFrictionalHeating.compare("yes")==0) {
    // set bcL and/or omega depending on shear zone width
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,1.0,_omega);
  }

  // compute shear heating component
  if (_wViscShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL && sigmadev!=NULL) {
    Vec Qvisc;
    computeViscousShearHeating(Qvisc,sigmadev, dgxy, dgxz);
    VecAXPY(_Q,1.0,Qvisc);
    VecDestroy(&Qvisc);
  }

  ierr = _sbpT->setRhs(temp,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Vec temp1; VecDuplicate(_Q,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_Q,temp1);
    VecCopy(temp1,_Q);
    VecDestroy(&temp1);
  }
  Mat H; _sbpT->getH(H);
  ierr = MatMultAdd(H,_Q,temp,temp); CHKERRQ(ierr);
  MatMult(_rcInv,temp,rhs);
  VecScale(rhs,dt);


  // add H * dT = H * (Told - Tamb) to rhs
  VecSet(temp,0.0);
  VecWAXPY(_dT,-1.0,_Tamb,To); // _dT =  -_Tamb + To
  _sbpT->H(_dT,temp);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    Vec temp1; VecDuplicate(temp,&temp1);
    MatMult(J,temp,temp1);
    VecCopy(temp1,temp);
    VecDestroy(&temp1);
  }
  VecAXPY(rhs,1.0,temp);
  VecDestroy(&temp);

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  KSPSolve(_ksp,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  VecDestroy(&rhs);

  VecWAXPY(T,1.0,_dT,_Tamb); // T = dT + Tamb

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// for thermomechanical problem only the steady-state heat equation
PetscErrorCode HeatEquation::be_steadyState(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_steadyState";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
//~ double startMiscTime = MPI_Wtime();
//~ _miscTime += MPI_Wtime() - startMiscTime;

  // set up boundary conditions and source terms
  Vec rhs; VecDuplicate(_dT,&rhs); VecSet(rhs,0.0);
  VecSet(_Q,0.0);

  // left boundary: heat generated by fault motion
  if (_wFrictionalHeating.compare("yes")==0) {
    // set bcL and/or omega depending on shear zone width
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,-1.0,_omega);
    VecScale(_bcL,-1.);
  }

  // compute shear heating component
  if (_wViscShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL && sigmadev!=NULL) {
    Vec Qvisc;
    computeViscousShearHeating(Qvisc,sigmadev, dgxy, dgxz);
    VecAXPY(_Q,-1.0,Qvisc);
    VecDestroy(&Qvisc);
  }

  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  Mat H; _sbpT->getH(H);
  ierr = MatMultAdd(H,_Q,rhs,rhs); CHKERRQ(ierr);


  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  KSPSolve(_ksp,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  VecWAXPY(T,1.0,_dT,_Tamb); // T = dT + T0
  computeHeatFlux();

  VecDestroy(&rhs);
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// for thermomechanical coupling solving only the steady-state heat equation with MMS test
PetscErrorCode HeatEquation::be_steadyStateMMS(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_steadyStateMMS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // set up boundary conditions and source terms
  Vec rhs,temp;
  VecDuplicate(_dT,&rhs);
  VecDuplicate(_dT,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);


  setMMSBoundaryConditions(time,"Dirichlet","Dirichlet","Neumann","Dirichlet");
  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  Vec source,Hxsource;
  VecDuplicate(_dT,&source);
  VecDuplicate(_dT,&Hxsource);
  //~ mapToVec(source,zzmms_SSdTsource,*_y,*_z,time);
  mapToVec(source,zzmms_SSTsource,*_y,*_z,time);
  ierr = _sbpT->H(source,Hxsource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  //~ writeVec(source,_outputDir+"mms_SSdTsource");
  VecDestroy(&source);
  ierr = VecAXPY(rhs,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);


  //~ // compute shear heating component
  //~ if (_wViscShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL) {
    //~ Vec shearHeat;
    //~ computeViscousShearHeating(shearHeat,sigmadev, dgxy, dgxz);
    //~ VecSet(shearHeat,0.);
    //~ VecAXPY(temp,1.0,shearHeat);
    //~ VecDestroy(&shearHeat);
  //~ }

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  //~ VecCopy(To,_dT); // plausible guess
  KSPSolve(_ksp,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecCopy(_dT,T);
  //~ VecWAXPY(T,1.0,_dT,_Tamb); // T = dT + T0

  //~ mapToVec(_dT,zzmms_T,*_y,*_z,time);

  VecDestroy(&rhs);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::initiateVarSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated implicity into varIm
  Vec T;
  VecDuplicate(_Tamb,&T);
  VecWAXPY(T,1.0,_Tamb,_dT);
  varSS["Temp"] = T;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::updateSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::updateSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Vec slipVel = varSS.find("slipVel")->second;
  Vec tau = varSS.find("tau")->second;
  Vec sDev = varSS.find("sDev")->second;
  Vec gVxy_t = varSS.find("gVxy_t")->second;
  Vec gVxz_t = varSS.find("gVxz_t")->second;

  // final argument is output
  ierr = computeSteadyStateTemp(0,slipVel,tau,sDev,gVxy_t,gVxz_t,varSS["Temp"]); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute steady-state temperature given boundary conditions and shear heating source terms (assuming these remain constant)
PetscErrorCode HeatEquation::computeSteadyStateTemp(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double beStartTime = MPI_Wtime();
  //~ double startMiscTime = MPI_Wtime();
  //~ _miscTime += MPI_Wtime() - startMiscTime;

  // set up boundary conditions and source terms
  Vec rhs;
  VecDuplicate(_dT,&rhs);
  VecSet(rhs,0.0);

    // left boundary: heat generated by fault motion
  if (_wFrictionalHeating.compare("yes")==0 && slipVel!=NULL && tau!=NULL) {
    VecPointwiseMult(_bcL,tau,slipVel);
    VecScale(_bcL,0.5);
  }
  else { VecSet(_bcL,0.0); }

  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);


  // compute shear heating component
  if (_wViscShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL) {
    Vec shearHeat;
    computeViscousShearHeating(shearHeat,sigmadev, dgxy, dgxz);
    VecAXPY(rhs,-1.0,shearHeat);
    VecDestroy(&shearHeat);
  }

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  KSPSolve(_ksp,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  VecDestroy(&rhs);

  VecWAXPY(T,1.0,_dT,_Tamb);
  computeHeatFlux();

  _beTime += MPI_Wtime() - beStartTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute viscous shear heating term (uses temperature from previous time step)
PetscErrorCode HeatEquation::computeViscousShearHeating(Vec& Qvisc,const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeViscousShearHeating";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // shear heating terms: sigmadev * dgv  (stresses times viscous strain rates)
  // sigmadev = sqrt(sxy^2 + sxz^2)
  // dgv = sqrt(dgVxy^2 + dgVxz^2)
  VecDuplicate(sigmadev,&Qvisc);
  VecSet(Qvisc,0.0);


  // compute dgv (use shearHeat to store values)
  VecPointwiseMult(Qvisc,dgxy,dgxy);
  Vec temp;
  VecDuplicate(sigmadev,&temp);
  VecPointwiseMult(temp,dgxz,dgxz);
  VecAXPY(Qvisc,1.0,temp);
  VecDestroy(&temp);
  VecSqrtAbs(Qvisc);

  // multiply by deviatoric stress
  VecPointwiseMult(Qvisc,sigmadev,Qvisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute frictional shear heating term (uses temperature from previous time step)
PetscErrorCode HeatEquation::computeFrictionalShearHeating(const Vec& tau, const Vec& slipVel)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeFrictionalShearHeating";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // if left boundary condition is heat flux
  if (_w == 0) {
    VecPointwiseMult(_bcL,tau,slipVel);
    VecScale(_bcL,0.5);
    VecSet(_omega,0.);
  }

  // if using finite width shear zone
  else {
    VecSet(_bcL,0.); // q = 0, no flux
    Vec V,Tau;
    VecDuplicate(_Gw,&V);
    VecDuplicate(_Gw,&Tau);
    ierr = MatMult(_MapV,slipVel,V); CHKERRQ(ierr);
    ierr = MatMult(_MapV,tau,Tau); CHKERRQ(ierr);
    VecPointwiseMult(_omega,V,_Gw);
    VecPointwiseMult(_omega,Tau,_omega);
    VecDestroy(&V);
    VecDestroy(&Tau);
  }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set up KSP, matrices, boundary conditions for the steady state heat equation problem
PetscErrorCode HeatEquation::setUpSteadyStateProblem(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setUpSteadyStateProblem";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif


  setBCsforBE();

  std::string bcTType = "Dirichlet";
  std::string bcBType = "Dirichlet";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";

  delete _sbpT;

  // construct matrices
  if (_sbpType.compare("mc")==0) {
    _sbpT = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpT = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpT = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_k);
    if (_Ny > 1 && _Nz > 1) { _sbpT->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbpT->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbpT->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbpT->setBCTypes("Dirichlet","Dirichlet","Neumann","Dirichlet");
  _sbpT->setMultiplyByH(1);
  _sbpT->setLaplaceType("yz");
  _sbpT->computeMatrices(); // actually create the matrices

  setupKSP_SS(_sbpT);

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set up KSP, matrices, boundary conditions for the transient heat equation problem
PetscErrorCode HeatEquation::setUpTransientProblem(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setUpTransientProblem";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // update boundaries (for solving for perturbation from steady-state)
  setBCsforBE();

  delete _sbpT;
  // construct matrices
  // BC order: right,top, left, bottom
  if (_sbpType.compare("mc")==0) {
    _sbpT = new SbpOps_c(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpT = new SbpOps_fc(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpT = new SbpOps_fc_coordTrans(_order,_Ny,_Nz,_Ly,_Lz,_k);
    if (_Ny > 1 && _Nz > 1) { _sbpT->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbpT->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbpT->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbpT->setBCTypes("Dirichlet","Dirichlet","Neumann","Dirichlet");
  _sbpT->setMultiplyByH(1);
  _sbpT->setLaplaceType("yz");
  _sbpT->computeMatrices(); // actually create the matrices

  // create identity matrix I (multiplied by H)
  Mat H;
  _sbpT->getH(H);
  MatDuplicate(H,MAT_COPY_VALUES,&_I);
  if (D._sbpType.compare("mfc_coordTrans")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatMatMult(J,H,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_I);
  }
  else {
    MatDuplicate(H,MAT_COPY_VALUES,&_I);
  }

  // create (rho*c)^-1 vector and matrix
  Vec rhocV;
  VecDuplicate(_rho,&rhocV);
  VecSet(rhocV,1.);
  VecPointwiseDivide(rhocV,rhocV,_rho);
  VecPointwiseDivide(rhocV,rhocV,_c);
  MatDuplicate(_I,MAT_DO_NOT_COPY_VALUES,&_rcInv);
  MatDiagonalSet(_rcInv,rhocV,INSERT_VALUES);

  // create _D2ath = J^-1 (rho*c)^-1 H D2
  Mat D2;
  _sbpT->getA(D2);
  MatMatMult(_rcInv,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_D2ath);

  // ensure diagonal of _D2ath has been allocated, even if 0
  PetscScalar v=0.0;
  PetscInt Ii,Istart,Iend=0;
  MatGetOwnershipRange(_D2ath,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    MatSetValues(_D2ath,1,&Ii,1,&Ii,&v,ADD_VALUES);
  }
  MatAssemblyBegin(_D2ath,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_D2ath,MAT_FINAL_ASSEMBLY);

  VecDestroy(&rhocV);

  setupKSP(_sbpT,D._initDeltaT);

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
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

  /*
  PetscInt    Istart,Iend,y;
  PetscScalar t = 0;
  ierr = VecGetOwnershipRange(_dT,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    y = Ii/_Nz;
    if (y == _Ny-1) {
      PetscInt z = Ii-_Nz*(Ii/_Nz);
      //~PetscPrintf(PETSC_COMM_WORLD,"y=%i, z=%i Ii=%i\n",y,z,Ii);
      ierr = VecGetValues(_dT,1,&Ii,&t);CHKERRQ(ierr);
      ierr = VecSetValue(_bcR,z,t,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR);CHKERRQ(ierr);

  // correct sign for Backward Euler
  VecScale(_bcT,-1.0);
  VecScale(_bcR,-1.0);
  VecScale(_bcB,-1.0);
  */

  // only solve for change in T with linear solve
  VecSet(_bcR,0.0);
  VecSet(_bcT,0.0);
  VecSet(_bcB,0.0);
  VecSet(_bcL,0.0);

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute heat flux (full body field and surface heat flux) for output
PetscErrorCode HeatEquation::computeHeatFlux()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeHeatFlux";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif


  // total heat flux
  Vec totalTemp; // total temperature
  VecDuplicate(_dT,&totalTemp);
  getTemp(totalTemp);
  ierr = _sbpT->muxDz(totalTemp,_heatFlux); CHKERRQ(ierr);
  VecDestroy(&totalTemp);
  VecScale(_heatFlux,1e9);

  // extract surface heat flux
  PetscInt    Ii,Istart,Iend,y;
  PetscScalar v;
  ierr = VecGetOwnershipRange(_heatFlux,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    y = Ii / _Nz;
    if (Ii % _Nz == 0) {
      ierr = VecGetValues(_heatFlux,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValue(_surfaceHeatFlux,y,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfaceHeatFlux);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfaceHeatFlux);CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (stepCount == 0) {
    _viewers["surfaceHeatFlux"] = initiateViewer(outputDir + "surfaceHeatFlux");
    _viewers["he_bcR"] = initiateViewer(outputDir + "he_bcR");
    _viewers["he_bcT"] = initiateViewer(outputDir + "he_bcT");
    _viewers["he_bcL"] = initiateViewer(outputDir + "he_bcL");
    _viewers["he_bcB"] = initiateViewer(outputDir + "he_bcB");

    ierr = VecView(_surfaceHeatFlux,_viewers["surfaceHeatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["he_bcR"]); CHKERRQ(ierr);
    ierr = VecView(_bcT,_viewers["he_bcT"]); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["he_bcL"]); CHKERRQ(ierr);
    ierr = VecView(_bcB,_viewers["he_bcB"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["surfaceHeatFlux"],outputDir + "surfaceHeatFlux");
    ierr = appendViewer(_viewers["he_bcR"],outputDir + "he_bcR");
    ierr = appendViewer(_viewers["he_bcT"],outputDir + "he_bcT");
    ierr = appendViewer(_viewers["he_bcL"],outputDir + "he_bcL");
    ierr = appendViewer(_viewers["he_bcB"],outputDir + "he_bcB");
  }
  else {
    ierr = VecView(_surfaceHeatFlux,_viewers["surfaceHeatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_bcR,_viewers["he_bcR"]); CHKERRQ(ierr);
    ierr = VecView(_bcT,_viewers["he_bcT"]); CHKERRQ(ierr);
    ierr = VecView(_bcL,_viewers["he_bcL"]); CHKERRQ(ierr);
    ierr = VecView(_bcB,_viewers["he_bcB"]); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (stepCount == 0) {
    _viewers["dT"] = initiateViewer(outputDir + "dT");
    _viewers["heatFlux"] = initiateViewer(outputDir + "heatFlux");
    _viewers["surfaceHeatFlux"] = initiateViewer(outputDir + "surfaceHeatFlux");
    _viewers["omega"] = initiateViewer(outputDir + "he_omega");

    ierr = VecView(_dT,_viewers["dT"]); CHKERRQ(ierr);
    ierr = VecView(_heatFlux,_viewers["heatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_surfaceHeatFlux,_viewers["surfaceHeatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_omega,_viewers["omega"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["dT"],outputDir + "dT");
    ierr = appendViewer(_viewers["heatFlux"],outputDir + "heatFlux");
    ierr = appendViewer(_viewers["surfaceHeatFlux"],outputDir + "surfaceHeatFlux");
    ierr = appendViewer(_viewers["omega"],outputDir + "he_omega");
  }
  else {
    ierr = VecView(_dT,_viewers["dT"]); CHKERRQ(ierr);
    ierr = VecView(_heatFlux,_viewers["heatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_surfaceHeatFlux,_viewers["surfaceHeatFlux"]); CHKERRQ(ierr);
    ierr = VecView(_omega,_viewers["omega"]); CHKERRQ(ierr);
  }


  _writeTime += MPI_Wtime() - startTime;
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Heat Equation Runtime Summary:\n");CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   solver algorithm = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent setting up linear solve context (e.g. factoring) (s): %g\n",_factorTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in be (s): %g\n",_beTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% be time spent solving linear system: %g\n",_linSolveTime/_beTime*100.);CHKERRQ(ierr);

  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   misc time (s): %g\n",_miscTime);CHKERRQ(ierr);
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% misc time: %g\n",_miscTime/_beTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return ierr;
}

// Save all scalar fields to text file named he_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode HeatEquation::writeDomain(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = outputDir + "he_context.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"heatEquationType = %s\n",_heatEquationType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withShearHeating = %s\n",_wViscShearHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withFrictionalHeating = %s\n",_wFrictionalHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver_heateq = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sbpType_heateq = %s\n",_sbpType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol_heateq = %.15e\n",_kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"TVals = %s\n",vector2str(_TVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"TDepths = %s\n",vector2str(_TDepths).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"w = %.5e\n",_w);CHKERRQ(ierr);


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

// write out material properties
PetscErrorCode HeatEquation::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  writeDomain(outputDir);

  PetscViewer    vw;

  std::string str = outputDir + "he_k";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_k,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "he_rho";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_rho,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "he_c";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_c,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "he_h";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_h,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "he_T0";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_Tamb,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "he_Gw";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_Gw,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  // contextual fields of members
  ierr = _sbpT->writeOps(_outputDir + "ops_he_"); CHKERRQ(ierr);


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


//======================================================================
// MMS  tests


double HeatEquation::zzmms_rho(const double y,const double z) { return 1.0; }
double HeatEquation::zzmms_c(const double y,const double z) { return 1.0; }
double HeatEquation::zzmms_h(const double y,const double z) { return 0.0; }

//~ double HeatEquation::zzmms_k(const double y,const double z) { return sin(y)*sin(z) + 2.0; }
double HeatEquation::zzmms_k(const double y,const double z) { return sin(y)*sin(z) + 30.; }
double HeatEquation::zzmms_k_y(const double y,const double z) { return cos(y)*sin(z); }
double HeatEquation::zzmms_k_z(const double y,const double z) { return sin(y)*cos(z); }


//~ double HeatEquation::zzmms_f(const double y,const double z) { return cos(y)*sin(z) + 800.; }
double HeatEquation::zzmms_f(const double y,const double z) { return cos(y)*sin(z); }
double HeatEquation::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double HeatEquation::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double HeatEquation::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double HeatEquation::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }
double HeatEquation::zzmms_g(const double t) { return exp(-2.*t); }
double HeatEquation::zzmms_g_t(const double t) { return -2.*exp(-2.*t); }

double HeatEquation::zzmms_T(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_t(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g_t(t); }

double HeatEquation::zzmms_dT(const double y,const double z,const double t) { return zzmms_T(y,z,t) - zzmms_T(y,z,0.); }
double HeatEquation::zzmms_dT_y(const double y,const double z,const double t) { return zzmms_T_y(y,z,t) - zzmms_T_y(y,z,0.); }
double HeatEquation::zzmms_dT_yy(const double y,const double z,const double t) { return zzmms_T_yy(y,z,t) - zzmms_T_yy(y,z,0.); }
double HeatEquation::zzmms_dT_z(const double y,const double z,const double t) { return zzmms_T_z(y,z,t) - zzmms_T_z(y,z,0.); }
double HeatEquation::zzmms_dT_zz(const double y,const double z,const double t) { return zzmms_T_zz(y,z,t) - zzmms_T_zz(y,z,0.); }
double HeatEquation::zzmms_dT_t(const double y,const double z,const double t) { return zzmms_T_t(y,z,t) - zzmms_T_t(y,z,0.); }

double HeatEquation::zzmms_SSTsource(const double y,const double z,const double t)
{
  PetscScalar k = zzmms_k(y,z);
  PetscScalar k_y = zzmms_k_y(y,z);
  PetscScalar k_z = zzmms_k_z(y,z);
  PetscScalar T_y = zzmms_T_y(y,z,t);
  PetscScalar T_yy = zzmms_T_yy(y,z,t);
  PetscScalar T_z = zzmms_T_z(y,z,t);
  PetscScalar T_zz = zzmms_T_zz(y,z,t);
  return k*(T_yy + T_zz) + k_y*T_y + k_z*T_z;
}
double HeatEquation::zzmms_SSdTsource(const double y,const double z,const double t)
{
  PetscScalar k = zzmms_k(y,z);
  PetscScalar k_y = zzmms_k_y(y,z);
  PetscScalar k_z = zzmms_k_z(y,z);
  PetscScalar dT_y = zzmms_dT_y(y,z,t);
  PetscScalar dT_yy = zzmms_dT_yy(y,z,t);
  PetscScalar dT_z = zzmms_dT_z(y,z,t);
  PetscScalar dT_zz = zzmms_dT_zz(y,z,t);
  return k*(dT_yy + dT_zz) + k_y*dT_y + k_z*dT_z;
}







