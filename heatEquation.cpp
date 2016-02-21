#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"


HeatEquation::HeatEquation(Domain& D)
:  _heatFieldsDistribution("unspecified"),_sbpT(NULL),
  _k(NULL),_rho(NULL),_c(NULL),_h(NULL),
  _kArr(NULL),_kMat(NULL),
  _bcT(NULL),_bcR(NULL),_bcB(NULL),_bcL(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~loadSettings(_file);
  //~checkInput();

  // set fields
  VecDuplicate(D._muVP,&_k);
  VecDuplicate(_k,&_rho);
  VecDuplicate(_k,&_c);
  VecDuplicate(_k,&_h);
  VecDuplicate(_k,&_T);
  setFields();

  VecSet(_k,3.0);
  VecSet(_rho,3000.0);
  VecSet(_c,3.0);
  VecSet(_h,0.0);

  _sbpT = new SbpOps_c(D,*D._muArrPlus,D._muP,"Dirichlet","Dirichlet","Dirichlet","Neumann"); // T, R, B, L

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
      loadVectorFromInputFile(str,_AVals);
    }
    else if (var.compare("rhoDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_ADepths);
    }

    else if (var.compare("kVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BVals);
    }
    else if (var.compare("kDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BDepths);
    }

    else if (var.compare("hVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nVals);
    }
    else if (var.compare("hDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nDepths);
    }

    else if (var.compare("cVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_TVals);
    }
    else if (var.compare("cDepths")==0) {
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

// initialize all fields
PetscErrorCode HeatEquation::setFields()
{
PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
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
    else if (_viscDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
    else {
      ierr = setVecFromVectors(_k,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_rho,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_h,_nVals,_nDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_c,_TVals,_TDepths);CHKERRQ(ierr);
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

  ierr = MatSetSizes(_kMat,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_kMat);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_kMat,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_kMat,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_kMat);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_kMat,_k,INSERT_VALUES);CHKERRQ(ierr);

  PetscFree(kInds);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
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

  assert(_viscDistribution.compare("mms")==0 ||
      _viscDistribution.compare("layered")==0 ||
      _viscDistribution.compare("loadFromFile")==0 );

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


// set off-fault material properties
PetscErrorCode PowerLaw::setFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = VecDuplicate(_uP,&_A);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_B);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_n);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_T);CHKERRQ(ierr);


  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_A,_AVals[0]);
    VecSet(_B,_BVals[0]);
    VecSet(_n,_nVals[0]);
    VecSet(_T,_TVals[0]);
  }
  else {
    if (_viscDistribution.compare("mms")==0) {
      mapToVec(_A,MMS_A,_Nz,_dy,_dz);
      mapToVec(_B,MMS_B,_Nz,_dy,_dz);
      mapToVec(_n,MMS_n,_Nz,_dy,_dz);
      mapToVec(_T,MMS_T,_Nz,_dy,_dz);
    }
    else if (_viscDistribution.compare("loadFromFile")==0) { loadFieldsFromFiles(); }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_B,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_n,_nVals,_nDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_T,_TVals,_TDepths);CHKERRQ(ierr);
    }
  }


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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical coupling
PetscErrorCode HeatEquation::setTempRate(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                          it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setTempRate";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  PetscInt Ii,Istart,Iend;
  PetscScalar v,k,slipVel,sigmaxy = 0;
  PetscScalar dT,rho,c,h,sigmaxz,dgxy,dgxz=0;

  // set up boundary conditions
  Vec bcT,bcB,bcR,bcL;
  VecDuplicate(_bcTP,&bcT);
  VecDuplicate(_bcBP,&bcB);
  VecDuplicate(_bcRP,&bcR);
  VecDuplicate(_bcLP,&bcL);

  VecSet(bcT,0.0); // surface temp
  VecSet(bcB,950.0); // temp at depth
  ierr = setVecFromVectors(bcR,_TVals,_TDepths);CHKERRQ(ierr); // remote boundary matches geotherm

  VecCopy(bcR,bcL);
  // left boundary: heat generated by fault motion
  //~VecGetOwnershipRange(bcL,&Istart,&Iend);
  //~for (Ii=Istart;Ii<Iend;Ii++) {
    //~VecGetValues(_k,1,&Ii,&k);
    //~VecGetValues(_stressxyP,1,&Ii,&sigmaxy);
    //~VecGetValues(*(dvarBegin+1),1,&Ii,&slipVel);
    //~v = -k*sigmaxy*abs(slipVel);
    //~VecSetValues(bcL,1,&Ii,&v,INSERT_VALUES);
  //~}
  //~VecAssemblyBegin(bcL);
  //~VecAssemblyEnd(bcL);


  Mat A;
  _sbpT->getA(A);
  ierr = MatMult(A,*(varBegin+4),*(dvarBegin+4)); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(_uP,&rhs);
  ierr = _sbpP->setRhs(rhs,bcL,bcR,bcT,bcB);CHKERRQ(ierr);
  ierr = VecAXPY(*(dvarBegin+4),1.0,rhs);CHKERRQ(ierr);


  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_rho,1,&Ii,&rho);
    VecGetValues(_c,1,&Ii,&c);
    VecGetValues(_h,1,&Ii,&h);
    VecGetValues(_stressxyP,1,&Ii,&sigmaxy);
    VecGetValues(*(dvarBegin+2),1,&Ii,&dgxy);
    VecGetValues(*(dvarBegin+4),1,&Ii,&dT);

    dT = dT/(rho*c) + 0.5*sigmaxy*dgxy  + h*c;

    if (_Nz > 1) {
      VecGetValues(_stressxzP,1,&Ii,&sigmaxz);
      VecGetValues(*(dvarBegin+3),1,&Ii,&dgxz);

      dT += 0.5*sigmaxz*dgxz;
    }
    VecSetValues(*(dvarBegin+4),1,&Ii,&dT,ADD_VALUES);
  }
  VecAssemblyBegin(*(dvarBegin+4));
  VecAssemblyEnd(*(dvarBegin+4));

  VecSet(*(dvarBegin+4),0.0);


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
