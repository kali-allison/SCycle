
#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dy),_dz(D._dz),_y(&D._y),_z(&D._z),
  _file(D._file),_outputDir(D._outputDir),_delim(D._delim),_inputDir(D._inputDir),
  _heatFieldsDistribution("unspecified"),_kFile("unspecified"),
  _rhoFile("unspecified"),_hFile("unspecified"),_cFile("unspecified"),
  _surfaceHeatFlux(NULL),_heatFlux(NULL),
  _TV(NULL),_bcRVw(NULL),_bcTVw(NULL),_bcLVw(NULL),_bcBVw(NULL),_timeV(NULL),
  _heatFluxV(NULL),_surfaceHeatFluxV(NULL),
  _wShearHeating("yes"),_wFrictionalHeating("yes"),
  _sbpType(D._sbpType),_sbpT(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL),
  _linSolver("AMG"),_kspTol(1e-10),
  _ksp(NULL),_pc(NULL),_I(NULL),_rhoC(NULL),_A(NULL),_pcMat(NULL),_computePC(0),_D2divRhoC(NULL),
  _linSolveTime(0),_linSolveCount(0),_pcRecomputeCount(0),_stride1D(D._stride1D),_stride2D(D._stride2D),
  _T(NULL),_T0(NULL),_k(NULL),_rho(NULL),_c(NULL),_h(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  loadSettings(_file);
  checkInput();
  {
    setFields(D);
    delete _sbpT;
    _sbpT = NULL;
  }
  if (D._loadICs==1) { loadFieldsFromFiles(); }

  // set up linear system for time integration
  setBCsforBE(); // update bcR with geotherm, correct sign for bcT, bcR, bcB
  // BC order: top, right, bottom, left; last argument makes A = Dzzmu + AT + AB
  if (D._sbpType.compare("mfc")==0 || D._sbpType.compare("mc")==0) {
    _sbpT = new SbpOps_fc(D,_k,"Dirichlet","Dirichlet","Dirichlet","Neumann","yz");
  }
  else if (D._sbpType.compare("mfc_coordTrans")==0) {
    _sbpT = new SbpOps_fc_coordTrans(D,_k,"Dirichlet","Dirichlet","Dirichlet","Neumann","yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0);
  }

  // create identity matrix I
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

  // create dt/rho*c matrix
  Vec rhoCV;
  VecDuplicate(_rho,&rhoCV);
  VecSet(rhoCV,1.0);
  VecPointwiseDivide(rhoCV,rhoCV,_rho);
  VecPointwiseDivide(rhoCV,rhoCV,_c);
  MatDuplicate(_I,MAT_DO_NOT_COPY_VALUES,&_rhoC);
  MatDiagonalSet(_rhoC,rhoCV,INSERT_VALUES);
  VecDestroy(&rhoCV);

  // create D2 / rho / c _D2divrhoC
  Mat D2;
  _sbpT->getA(D2);
  MatMatMult(_rhoC,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_D2divRhoC);
  // ensure diagonal has been allocated, even if 0
  PetscScalar v=0.0;
  PetscInt Ii,Istart,Iend=0;
  MatGetOwnershipRange(_D2divRhoC,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    MatSetValues(_D2divRhoC,1,&Ii,1,&Ii,&v,ADD_VALUES);
  }
  MatAssemblyBegin(_D2divRhoC,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_D2divRhoC,MAT_FINAL_ASSEMBLY);
  MatConvert(_D2divRhoC,MATSAME,MAT_INITIAL_MATRIX,&_A);

   setupKSP(_sbpT,D._initDeltaT);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

}

HeatEquation::~HeatEquation()
{

  KSPDestroy(&_ksp);
  MatDestroy(&_A);
  MatDestroy(&_rhoC);
  MatDestroy(&_I);
  MatDestroy(&_D2divRhoC);

  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);
  VecDestroy(&_h);

  VecDestroy(&_T);
  VecDestroy(&_T0);

  VecDestroy(&_heatFlux);
  VecDestroy(&_surfaceHeatFlux);

  VecDestroy(&_bcL);
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);

  PetscViewerDestroy(&_TV);
  PetscViewerDestroy(&_bcRVw);
  PetscViewerDestroy(&_bcTVw);
  PetscViewerDestroy(&_bcLVw);
  PetscViewerDestroy(&_bcBVw);
  PetscViewerDestroy(&_timeV);
  PetscViewerDestroy(&_heatFluxV);
  PetscViewerDestroy(&_surfaceHeatFluxV);

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

  VecWAXPY(T,1.0,_T0,_T);

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
    else if (var.compare("withShearHeating")==0) {
      _wShearHeating = line.substr(pos+_delim.length(),line.npos).c_str();
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
  ierr = loadVecFromInputFile(_T0,_inputDir,"T0");CHKERRQ(ierr);

  // load dT (perturbation)
  ierr = loadVecFromInputFile(_T,_inputDir,"dT");CHKERRQ(ierr);

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
  VecSetFromOptions(_bcT);     PetscObjectSetName((PetscObject) _bcT, "_bcT");
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


  // set material properties
  VecDuplicate(*_y,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  VecDuplicate(_k,&_T);

  VecDuplicate(_T,&_T0);
  VecCopy(_T,_T0);
  VecSet(_T,0.0);

  // heat flux variables
  VecDuplicate(_bcT,&_surfaceHeatFlux); VecSet(_surfaceHeatFlux,0.0);
  VecDuplicate(_k,&_heatFlux); VecSet(_heatFlux,0.0);

  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_k,_kVals[0]);
    VecSet(_rho,_rhoVals[0]);
    VecSet(_h,_hVals[0]);
    VecSet(_c,_cVals[0]);
    VecSet(_T,_TVals[0]);
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
      //~ ierr = setVecFromVectors(_T,_TVals,_TDepths);CHKERRQ(ierr);
      computeSteadyStateTemp(D);
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
PetscErrorCode HeatEquation::computeSteadyStateTemp(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // Set up linear system
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
    ierr = KSPSolve(ksp,rhs,_T0);CHKERRQ(ierr);
    _linSolveTime += MPI_Wtime() - startTime;
    _linSolveCount++;


    VecDestroy(&rhs);
    KSPDestroy(&ksp);
  }
  else{
    VecSet(_T0,_TVals[0]);
  }
  /*
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
  }*/
  VecSet(_T,0.0);
  computeHeatFlux();

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




  // create: A = I - dt/rho*c D2
  //~ Mat D2;
  //~ sbp->getA(D2);
  //~ MatMatMult(_rhoC,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_A);
  //~ MatScale(_A,-dt);
  //~ MatAXPY(_A,1.0,_I,DIFFERENT_NONZERO_PATTERN);

  // new version
  MatCopy(_D2divRhoC,_A,SAME_NONZERO_PATTERN);
  MatScale(_A,-dt);
  MatShift(_A,1.0);


  // set up KSP

  // reuse old PC
  //~ if (_computePC>0) {
    //~ KSPSetOperators(_ksp,_A,_pcMat);
  //~ }
  //~ else {
    //~ if (_computePC==0) { ierr = MatConvert(_A,MATSAME,MAT_INITIAL_MATRIX,&_pcMat); CHKERRQ(ierr); }


    ierr = KSPCreate(PETSC_COMM_WORLD,&_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
    //~ // ierr = KSPSetOperators(_ksp,_A,_pcMat);CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_A,_A);CHKERRQ(ierr);
    //~ // ierr = KSPSetReusePreconditioner(_ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(_ksp,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
  }

  #if CALCULATE_ENERGY == 1
    VecCopy(_uP,_uPPrev);
  #endif
  if ( stepCount % _stride1D == 0) {
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep1D(stepCount);CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep2D(stepCount);CHKERRQ(ierr);
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
PetscErrorCode HeatEquation::be(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy,const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // set up matrix
  //~ setupKSP(_sbpT,dt);
  MatCopy(_D2divRhoC,_A,SAME_NONZERO_PATTERN);
  MatScale(_A,-dt);
  MatShift(_A,1.0);
  ierr = KSPSetOperators(_ksp,_A,_A);CHKERRQ(ierr);

  // set up boundary conditions and source terms
  Vec rhs,temp;
  VecDuplicate(_T,&rhs);
  VecDuplicate(_T,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);

    // left boundary: heat generated by fault motion
  if (_wFrictionalHeating.compare("yes")==0) {
    VecPointwiseMult(_bcL,tau,slipVel);
    VecScale(_bcL,0.5);
  }
  else { VecSet(_bcL,0.0); }

  ierr = _sbpT->setRhs(temp,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);

  // compute shear heating component
  if (_wShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL) {
    Vec shearHeat;
    computeShearHeating(shearHeat,sigmadev, dgxy, dgxz);
    //~ VecScale(shearHeat,0.0);
    VecAXPY(temp,1.0,shearHeat);
    VecDestroy(&shearHeat);
  }

  MatMult(_rhoC,temp,rhs);
  VecScale(rhs,dt);


  // add H * Tn to rhs
  VecSet(temp,0.0);
  _sbpT->H(To,temp);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,temp); CHKERRQ(ierr);
  }
  VecAXPY(rhs,1.0,temp);
  VecDestroy(&temp);


  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  VecCopy(To,_T); // plausible guess
  KSPSolve(_ksp,rhs,_T);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  VecDestroy(&rhs);
  //~ MatDestroy(&_A);
  //~ KSPDestroy(&_ksp);

  VecCopy(_T,T);
  computeHeatFlux();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute shear heating term (uses temperature from previous time step)
PetscErrorCode HeatEquation::computeShearHeating(Vec& shearHeat,const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeShearHeating";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif


  // shear heating terms: sigmadev * dgv  (stresses times viscous strain rates)
  // sigmadev = sqrt(sxy^2 + sxz^2)
  // dgv = sqrt(dgVxy^2 + dgVxz^2)
  VecDuplicate(sigmadev,&shearHeat);
  VecSet(shearHeat,0.0);


  //~ PetscInt Istart,Iend;
  //~ PetscScalar h,sdev,dgxyV,dgxzV = 0;
  //~ VecGetOwnershipRange(shearHeat,&Istart,&Iend);
  //~ for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    //~ ierr = VecGetValues(dgxy,1,&Ii,&dgxyV);CHKERRQ(ierr);
    //~ ierr = VecGetValues(dgxz,1,&Ii,&dgxzV);CHKERRQ(ierr);
    //~ ierr = VecGetValues(sigmadev,1,&Ii,&sdev);CHKERRQ(ierr);
    //~ h = sdev * sqrt( dgxyV*dgxyV + dgxzV*dgxzV);
    //~ VecSetValues(shearHeat,1,&Ii,&h,INSERT_VALUES);
    //~ assert(~isnan(h));
    //~ assert(~isinf(h));
  //~ }
  //~ VecAssemblyBegin(shearHeat);
  //~ VecAssemblyEnd(shearHeat);


  // compute dgv (use shearHeat to store values)
  VecPointwiseMult(shearHeat,dgxy,dgxy);
  Vec temp;
  VecDuplicate(sigmadev,&temp);
  VecPointwiseMult(temp,dgxz,dgxz);
  VecAXPY(shearHeat,1.0,temp);
  VecDestroy(&temp);
  VecSqrtAbs(shearHeat);

  // multiply by deviatoric stress
  VecPointwiseMult(shearHeat,sigmadev,shearHeat);

  // if coordinate transform, weight this with yq*rz*H
  Vec temp1;
  VecDuplicate(shearHeat,&temp1);
  VecSet(temp1,0.0);
  _sbpT->H(shearHeat,temp1);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbpT->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(shearHeat,yq,zr,temp1); CHKERRQ(ierr);
  }
  else{ VecCopy(temp1,shearHeat); }
  VecDestroy(&temp1);

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

  /*
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
  */

  // only solve for change in T with linear solve
  VecSet(_bcR,0.0);
  VecSet(_bcT,0.0);
  VecSet(_bcB,0.0);

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
  VecDuplicate(_T,&totalTemp);
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



PetscErrorCode HeatEquation::writeStep1D(const PetscInt stepCount)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
    CHKERRQ(ierr);
  #endif

  if (stepCount==0) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfaceHeatFlux").c_str(),
                                 FILE_MODE_WRITE,&_surfaceHeatFluxV);CHKERRQ(ierr);
    ierr = VecView(_surfaceHeatFlux,_surfaceHeatFluxV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfaceHeatFluxV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfaceHeatFlux").c_str(),
                                   FILE_MODE_APPEND,&_surfaceHeatFluxV);CHKERRQ(ierr);


    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcR").c_str(),
                                 FILE_MODE_WRITE,&_bcRVw);CHKERRQ(ierr);
    ierr = VecView(_bcR,_bcRVw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRVw);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcR").c_str(),
                                   FILE_MODE_APPEND,&_bcRVw);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcT").c_str(),
                                 FILE_MODE_WRITE,&_bcTVw);CHKERRQ(ierr);
    ierr = VecView(_bcT,_bcTVw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcTVw);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcT").c_str(),
                                   FILE_MODE_APPEND,&_bcTVw);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcL").c_str(),
                                 FILE_MODE_WRITE,&_bcLVw);CHKERRQ(ierr);
    ierr = VecView(_bcL,_bcLVw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcLVw);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcL").c_str(),
                                   FILE_MODE_APPEND,&_bcLVw);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcB").c_str(),
                                 FILE_MODE_WRITE,&_bcBVw);CHKERRQ(ierr);
    ierr = VecView(_bcB,_bcBVw);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcBVw);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"he_bcB").c_str(),
                                   FILE_MODE_APPEND,&_bcBVw);CHKERRQ(ierr);

  }
  else {
    ierr = VecView(_surfaceHeatFlux,_surfaceHeatFluxV);CHKERRQ(ierr);
    ierr = VecView(_bcR,_bcRVw);CHKERRQ(ierr);
    ierr = VecView(_bcT,_bcTVw);CHKERRQ(ierr);
    ierr = VecView(_bcL,_bcLVw);CHKERRQ(ierr);
    ierr = VecView(_bcB,_bcBVw);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),FILENAME,stepCount);
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
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"T0").c_str(),
                                 FILE_MODE_WRITE,&_TV);CHKERRQ(ierr);
    ierr = VecView(_T0,_TV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_TV);CHKERRQ(ierr);


    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dT").c_str(),
                                 FILE_MODE_WRITE,&_TV);CHKERRQ(ierr);
    ierr = VecView(_T,_TV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_TV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dT").c_str(),
                                   FILE_MODE_APPEND,&_TV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"heatFlux").c_str(),
                                 FILE_MODE_WRITE,&_heatFluxV);CHKERRQ(ierr);
    ierr = VecView(_heatFlux,_heatFluxV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_heatFluxV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"heatFlux").c_str(),
                                   FILE_MODE_APPEND,&_heatFluxV);CHKERRQ(ierr);

  }
  else {
    ierr = VecView(_T,_TV);CHKERRQ(ierr);
    ierr = VecView(_heatFlux,_heatFluxV);CHKERRQ(ierr);
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

// Save all scalar fields to text file named he_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode HeatEquation::writeDomain()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = _outputDir + "he_context.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"withShearHeating = %s\n",_wShearHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withFrictionalHeating = %s\n",_wFrictionalHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver_heateq = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sbpType_heateq = %s\n",_sbpType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol_heateq = %.15e\n",_kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"TVals = %s\n",vector2str(_TVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"TDepths = %s\n",vector2str(_TDepths).c_str());CHKERRQ(ierr);


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
PetscErrorCode HeatEquation::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  writeDomain();

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
