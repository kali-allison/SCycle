#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dy),_dz(D._dz),
  _file(D._file),_delim(D._delim),_inputDir(D._inputDir),
  _heatFieldsDistribution("unspecified"),_kFile("unspecified"),
  _rhoFile("unspecified"),_hFile("unspecified"),_cFile("unspecified"),
  _k(NULL),_rho(NULL),_c(NULL),_h(NULL),
  _kArr(NULL),_kMat(NULL),
  _sbpT(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();

  // set fields
  VecDuplicate(D._muVP,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  setFields();
  //~VecSet(_k,3.0);
  //~VecSet(_rho,3000.0);
  //~VecSet(_c,3.0);
  //~VecSet(_h,0.0);


  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcT);     PetscObjectSetName((PetscObject) _bcT, "_bcT");
  VecSet(_bcT,273.0);

  VecDuplicate(_bcT,&_bcB); PetscObjectSetName((PetscObject) _bcB, "bcB");
  VecSet(_bcB,1643.0);


  VecCreate(PETSC_COMM_WORLD,&_bcR);
  VecSetSizes(_bcR,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcR);     PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,0.0);

  VecDuplicate(_bcR,&_bcL); PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);


  // BC order: top, right, bottom, left; last argument makes A = Dzzmu + AT + AB
  _sbpT = new SbpOps_c(D,*_kArr,_kMat,"Dirichlet","Dirichlet","Dirichlet","Neumann","z");

  computeSteadyStateTemp();
  assert(0);

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

  PetscFree(_kArr);
  MatDestroy(&_kMat);
}

// loads settings from the input text file
PetscErrorCode HeatEquation::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "HeatEquation::loadSettings()";
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
      mapToVec(_k,MMS_A,_Nz,_dy,_dz);
      mapToVec(_rho,MMS_B,_Nz,_dy,_dz);
      mapToVec(_h,MMS_n,_Nz,_dy,_dz);
      mapToVec(_c,MMS_T,_Nz,_dy,_dz);
    }
    else if (_heatFieldsDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
    else {
      ierr = setVecFromVectors(_k,_kVals,_kDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_rho,_rhoVals,_rhoDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_h,_hVals,_hDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_c,_cVals,_cDepths);CHKERRQ(ierr);
    }
  }

  // set conductivity matrix
  PetscInt *kInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&kInds);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_kArr);CHKERRQ(ierr);

  for (PetscInt Ii=0;Ii<_Ny*_Nz;Ii++) {
    //~z = _dz*(Ii-_Nz*(Ii/_Nz));
    //~y = _dy*(Ii/_Nz);
    _kArr[Ii] = 3.0;
    kInds[Ii] = Ii;
  }

  MatCreate(PETSC_COMM_WORLD,&_kMat);
  ierr = MatSetSizes(_kMat,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_kMat);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_kMat,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_kMat,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_kMat);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_kMat,_k,INSERT_VALUES);CHKERRQ(ierr);

  PetscFree(kInds);


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
    string funcName = "HeatEquation::setTempRate";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up linear solver context
  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  Mat A;
  _sbpT->getA(A);

  //~VecSet(_bcR

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical coupling
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const Vec slipVel,const Vec& sigmaxy,
      const Vec& sigmaxz, const Vec& dgxy, const Vec& dgxz, Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setTempRate";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  PetscInt Ii,Istart,Iend;
  PetscScalar k,v,vel,s = 0;
  PetscScalar dT,rho,c,h,dg=0;


  // left boundary: heat generated by fault motion
  VecGetOwnershipRange(_bcL,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_k,1,&Ii,&k);
    VecGetValues(sigmaxy,1,&Ii,&s);
    VecGetValues(slipVel,1,&Ii,&vel);
    v = -k*s*abs(vel)*0;
    VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);
  }
  VecAssemblyBegin(_bcL);
  VecAssemblyEnd(_bcL);


  Mat A;
  _sbpT->getA(A);
  ierr = MatMult(A,T,dTdt); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(T,&rhs);
  ierr = _sbpT->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(dTdt,1.0,rhs);CHKERRQ(ierr);


  VecGetOwnershipRange(T,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_rho,1,&Ii,&rho);
    VecGetValues(_c,1,&Ii,&c);
    VecGetValues(_h,1,&Ii,&h);
    VecGetValues(sigmaxy,1,&Ii,&s);
    VecGetValues(dgxy,1,&Ii,&dg);

    dT = 0.5*s*dg  + h*c;

    if (_Nz > 1) {
      VecGetValues(sigmaxz,1,&Ii,&s);
      VecGetValues(dgxz,1,&Ii,&dg);

      dT += 0.5*s*dg;
    }
    dT = dT / rho / c;
    VecSetValues(dTdt,1,&Ii,&dT,ADD_VALUES);
  }
  VecAssemblyBegin(dTdt);
  VecAssemblyEnd(dTdt);

  VecSet(dTdt,0.0);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// set left and right boundary conditions from computed geotherm
PetscErrorCode HeatEquation::setBCs()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLinearElastic::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  //~PetscInt    Ii,Istart,Iend;
  //~PetscScalar u,y,z;
  //~ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);
  //~for (Ii=Istart;Ii<Iend;Ii++) {
    //~z = Ii-_Nz*(Ii/_Nz);
    //~y = Ii/_Nz;
    //~if (z == 0) {
      //~ierr = VecGetValues(_uP,1,&Ii,&u);CHKERRQ(ierr);
      //~ierr = VecSetValue(_surfDispPlus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    //~}
  //~}
  //~ierr = VecAssemblyBegin(_surfDispPlus);CHKERRQ(ierr);
  //~ierr = VecAssemblyEnd(_surfDispPlus);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLinearElastic::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// Fills vec with the linear interpolation between the pairs of points (vals,depths)
// this probably won't work if the vector is 2D instead of 1D
PetscErrorCode HeatEquation::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::setVecFromVectors";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // Find the appropriate starting pair of points to interpolate between: (z0,v0) and (z1,v1)
  z1 = depths.back();
  depths.pop_back();
  z0 = depths.back();
  v1 = vals.back();
  vals.pop_back();
  v0 = vals.back();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  z = _dz*(Iend-1);
  while (z<z0) {
    z1 = depths.back();
    depths.pop_back();
    z0 = depths.back();
    v1 = vals.back();
    vals.pop_back();
    v0 = vals.back();
    //~PetscPrintf(PETSC_COMM_WORLD,"2: z = %g: z0 = %g   z1 = %g   v0 = %g  v1 = %g\n",z,z0,z1,v0,v1);
  }


  for (Ii=Iend-1; Ii>=Istart; Ii--) {
    z = _dz*Ii;
    if (z==z1) { v = v1; }
    else if (z==z0) { v = v0; }
    else if (z>z0 && z<z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }

    // if z is no longer bracketed by (z0,z1), move on to the next pair of points
    if (z<=z0) {
      z1 = depths.back();
      depths.pop_back();
      z0 = depths.back();
      v1 = vals.back();
      vals.pop_back();
      v0 = vals.back();
    }
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
