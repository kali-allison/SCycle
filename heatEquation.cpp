
#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dy),_dz(D._dz),_kspTol(D._kspTol),
  _file(D._file),_outputDir(D._outputDir),_delim(D._delim),_inputDir(D._inputDir),
  _heatFieldsDistribution("unspecified"),_kFile("unspecified"),
  _rhoFile("unspecified"),_hFile("unspecified"),_cFile("unspecified"),
  _k(NULL),_rho(NULL),_c(NULL),_h(NULL),
  _TV(NULL),_vw(NULL),
  _sbpT(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
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
  //~VecSet(_bcT,273.0);
  VecSet(_bcT,_TVals[0]);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "bcB");
  //~VecSet(_bcB,1643.0);
  VecSet(_bcB,_TVals.back());


  VecCreate(PETSC_COMM_WORLD,&_bcR);
  VecSetSizes(_bcR,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcR);     PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.0);

  VecDuplicate(_bcR,&_bcL); PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);

  // set fields
  VecDuplicate(D._muVecP,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  setFields();

  VecDuplicate(_k,&_T);
  VecSet(_T,_TVals[0]);

  // BC order: top, right, bottom, left; last argument makes A = Dzzmu + AT + AB
  {
    _sbpT = new SbpOps_fc(D,_k,"Dirichlet","Dirichlet","Dirichlet","Dirichlet","z");
    computeSteadyStateTemp();
    setBCs(); // update bcR with geotherm
  }
  _sbpT = new SbpOps_fc(D,_k,"Dirichlet","Dirichlet","Dirichlet","Neumann","yz");

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

HeatEquation::~HeatEquation()
{
  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);
  VecDestroy(&_h);

  PetscViewerDestroy(&_TV);
  PetscViewerDestroy(&_vw);

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


  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_k,_kVals[0]);
    VecSet(_rho,_rhoVals[0]);
    VecSet(_h,_hVals[0]);
    VecSet(_c,_cVals[0]);
  }
  else {
    if (_heatFieldsDistribution.compare("mms")==0) {
      //~mapToVec(_k,MMS_A,_Nz,_dy,_dz);
      //~mapToVec(_rho,MMS_B,_Nz,_dy,_dz);
      //~mapToVec(_h,MMS_n,_Nz,_dy,_dz);
      //~mapToVec(_c,MMS_c,_Nz,_dy,_dz);
      //~mapToVec(_T,MMS_T,_Nz,_dy,_dz);
    }
    else if (_heatFieldsDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
    else {
      ierr = setVecFromVectors(_k,_kVals,_kDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_rho,_rhoVals,_rhoDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_h,_hVals,_hDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_c,_cVals,_cDepths);CHKERRQ(ierr);
    }
  }

  //~ MatCreate(PETSC_COMM_WORLD,&_kMat);
  //~ ierr = MatSetSizes(_kMat,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  //~ ierr = MatSetFromOptions(_kMat);CHKERRQ(ierr);
  //~ ierr = MatMPIAIJSetPreallocation(_kMat,1,NULL,1,NULL);CHKERRQ(ierr);
  //~ ierr = MatSeqAIJSetPreallocation(_kMat,1,NULL);CHKERRQ(ierr);
  //~ ierr = MatSetUp(_kMat);CHKERRQ(ierr);

  //~ ierr = MatDiagonalSet(_kMat,_k,INSERT_VALUES);CHKERRQ(ierr);

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
    MatCreate(PETSC_COMM_WORLD,&A);
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
  }
  else {
    // set each field using it's vals and depths std::vectors
    if (_Nz == 1) { VecSet(_T,_TVals[0]); }
    else {
      if (_heatFieldsDistribution.compare("mms")==0) { mapToVec(_T,MMS_T,_Nz,_dy,_dz); }
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


// for thermomechanical coupling
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const Vec slipVel,const Vec& sigmaxy,
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
  PetscInt Istart,Iend;
  PetscScalar k,v,vel,s = 0;
  VecGetOwnershipRange(_k,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    //~PetscScalar z = _dz*(Ii-_Nz*(Ii/_Nz));
    PetscInt y = Ii/_Nz;
    if (y == 0) {
      //~PetscInt z = Ii-_Nz*(Ii/_Nz);
      //~PetscPrintf(PETSC_COMM_WORLD,"Ii = %i, y=%i\n",Ii,y);
      VecGetValues(_k,1,&Ii,&k);
      VecGetValues(sigmaxy,1,&Ii,&s);
      v = -s/k; // s in MPa, k in km^2 kPa/K/s
      VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_bcL);
  VecAssemblyEnd(_bcL);

  Vec absVel;
  VecDuplicate(_bcL,&absVel);
  VecCopy(slipVel,absVel);
  VecAbs(absVel);
  VecPointwiseMult(_bcL,_bcL,absVel);

  Mat A;
  MatCreate(PETSC_COMM_WORLD,&A);
  _sbpT->getA(A);
  ierr = MatMult(A,T,dTdt); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(T,&rhs);
  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(dTdt,-1.0,rhs);CHKERRQ(ierr);

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

  //~//!!! missing h*c term (heat production), also check if need to multiply by 0.5

  VecPointwiseDivide(dTdt,dTdt,_rho);
  VecPointwiseDivide(dTdt,dTdt,_c);

  //~VecSet(dTdt,0.0);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set right boundary condition from computed geotherm
PetscErrorCode HeatEquation::setBCs()
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

// write out material properties
PetscErrorCode HeatEquation::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer    vw;

  std::string str = outputDir + "k";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_k,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "rho";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_rho,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "c";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_c,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "h";
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
    z = _dz*(Ii-_Nz*(Ii/_Nz));
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
