#include "powerLaw.hpp"

#define FILENAME "powerLaw.cpp"


PowerLaw::PowerLaw(Domain& D,HeatEquation& he,Vec& tau)
: LinearElastic(D,tau), _file(D._file),_delim(D._delim),
  _viscDistribution("unspecified"),_AFile("unspecified"),_BFile("unspecified"),_nFile("unspecified"),
  _A(NULL),_n(NULL),_B(NULL),_T(NULL),_effVisc(NULL),
  _sxyPV(NULL),_sxzPV(NULL),_sdevV(NULL),
  _gTxyV(NULL),_gTxzV(NULL),
  _gxyV(NULL),_dgxyV(NULL),
  _gxzV(NULL),_dgxzV(NULL),
  _TV(NULL),_effViscV(NULL),
  _sxz(NULL),_sdev(NULL),
  _gxy(NULL),_dgxy(NULL),
  _gxz(NULL),_dgxz(NULL),
  _gTxy(NULL),_gTxz(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  allocateFields(); // initialize fields
  he.setTemp(_T);
  setMaterialParameters();

  if (D._loadICs==1) {
    loadFieldsFromFiles();
    setUpSBPContext(D); // set up matrix operators
    setStresses(_currTime);
    computeViscosity();
  }
  else {
    guessSteadyStateEffVisc();
    setSSInitialConds(D,tau);
    setUpSBPContext(D); // set up matrix operators
    setStresses(_currTime);
  }

  if (_isMMS) { setMMSInitialConditions(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PowerLaw::~PowerLaw()
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::~PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_T);
  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_B);
  VecDestroy(&_effVisc);

  VecDestroy(&_sxz);
  VecDestroy(&_sdev);

  VecDestroy(&_gTxy);
  VecDestroy(&_gTxz);
  VecDestroy(&_gxy);
  VecDestroy(&_gxz);
  VecDestroy(&_dgxy);
  VecDestroy(&_dgxz);

  PetscViewerDestroy(&_sxyPV);
  PetscViewerDestroy(&_sxzPV);
  PetscViewerDestroy(&_sdevV);
  PetscViewerDestroy(&_gTxyV);
  PetscViewerDestroy(&_gTxzV);
  PetscViewerDestroy(&_gxyV);
  PetscViewerDestroy(&_gxzV);
  PetscViewerDestroy(&_dgxyV);
  PetscViewerDestroy(&_dgxzV);
  PetscViewerDestroy(&_effViscV);

  PetscViewerDestroy(&_timeV2D);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// loads settings from the input text file
PetscErrorCode PowerLaw::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "PowerLaw::loadSettings()";
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

    // viscosity for asthenosphere
    if (var.compare("viscDistribution")==0) {
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
    //~else if (var.compare("TFile")==0) {
      //~_TFile = line.substr(pos+_delim.length(),line.npos).c_str();
    //~}

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
    else if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("heatEquationType")==0) {
      _heatEquationType = line.substr(pos+_delim.length(),line.npos).c_str();
    }

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
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_viscDistribution.compare("layered")==0 ||
      _viscDistribution.compare("mms")==0 ||
      _viscDistribution.compare("loadFromFile")==0 ||
      _viscDistribution.compare("effectiveVisc")==0 );

  //~ if (_viscDistribution.compare("loadFromFile")==0) { assert(!_inputDir.compare("unspecified")); }
  if (_viscDistribution.compare("loadFromFile")==0) { assert(_inputDir.compare("unspecified")); }

  assert(_heatEquationType.compare("transient")==0 ||
      _heatEquationType.compare("steadyState")==0 );

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
PetscErrorCode PowerLaw::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = VecDuplicate(_u,&_A);CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&_B);CHKERRQ(ierr);
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
    std::string funcName = "PowerLaw::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif


  // set each field using it's vals and depths std::vectors
  if (_Nz == 1) {
    VecSet(_A,_AVals[0]);
    VecSet(_B,_BVals[0]);
    VecSet(_n,_nVals[0]);
  }
  else {
    if (_viscDistribution.compare("mms")==0) {
      if (_Nz == 1) { mapToVec(_A,zzmms_A1D,*_y); }
      else { mapToVec(_A,zzmms_A,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_B,zzmms_B1D,*_y); }
      else { mapToVec(_B,zzmms_B,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_n,zzmms_n1D,*_y); }
      else { mapToVec(_n,zzmms_n,*_y,*_z); }
    }
    else if (_viscDistribution.compare("loadFromFile")==0) { loadEffViscFromFiles(); }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_B,_BVals,_BDepths);CHKERRQ(ierr);
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
PetscErrorCode PowerLaw::loadEffViscFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::loadEffViscFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer inv; // in viewer

  // load effective viscosity
  string vecSourceFile = _inputDir + "EffVisc";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,inv);CHKERRQ(ierr);

  // load A
  vecSourceFile = _inputDir + "A";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_A,inv);CHKERRQ(ierr);

  // load B
  vecSourceFile = _inputDir + "B";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_B,inv);CHKERRQ(ierr);

  // load B
  vecSourceFile = _inputDir + "n";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_n,inv);CHKERRQ(ierr);

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
    std::string funcName = "PowerLaw::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer inv; // in viewer

  // load bcL
  string vecSourceFile = _inputDir + "bcL";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcL,inv);CHKERRQ(ierr);

  //~ // load bcR
  vecSourceFile = _inputDir + "bcR";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcRShift,inv);CHKERRQ(ierr);
  VecSet(_bcR,0.0);

  // load gxy
  vecSourceFile = _inputDir + "Gxy";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxy,inv);CHKERRQ(ierr);

  // load gxz
  vecSourceFile = _inputDir + "Gxz";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxz,inv);CHKERRQ(ierr);


   // load sxy
  vecSourceFile = _inputDir + "Sxy";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_sxy,inv);CHKERRQ(ierr);

  // load sxz
  vecSourceFile = _inputDir + "Sxz";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_sxz,inv);CHKERRQ(ierr);


  // load effective viscosity
  vecSourceFile = _inputDir + "EffVisc";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,inv);CHKERRQ(ierr);

  // load temperature
  vecSourceFile = _inputDir + "T";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_T,inv);CHKERRQ(ierr);

  // load power law parameters
  vecSourceFile = _inputDir + "A";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_A,inv);CHKERRQ(ierr);

  vecSourceFile = _inputDir + "B";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_B,inv);CHKERRQ(ierr);

  vecSourceFile = _inputDir + "n";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_n,inv);CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// try to speed up spin up by starting closer to steady state
PetscErrorCode PowerLaw::setSSInitialConds(Domain& D,Vec& tauRS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setSSInitialConds";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _sbp;

  // set up SBP operators
  //~ string bcT,string bcR,string bcB, string bcL
  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";

  if (_sbpType.compare("mc")==0) {
    _sbp = new SbpOps_c(D,_Ny,_Nz,_muVec,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp = new SbpOps_fc(D,_Ny,_Nz,_muVec,bcTType,bcRType,bcBType,bcLType,"yz"); // to spin up viscoelastic
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp = new SbpOps_fc_coordTrans(D,_Ny,_Nz,_muVec,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  KSPDestroy(&_ksp);
  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP(_sbp,_ksp,_pc);

  // set up boundary conditions
  VecSet(_bcR,0.0);
  PetscInt    Istart,Iend;
  PetscScalar v = 0;
  Vec faultVisc; VecDuplicate(_bcL,&faultVisc);
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < _Nz) {
      VecGetValues(_effVisc,1,&Ii,&v);
      VecSetValue(faultVisc,Ii,v,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(faultVisc); VecAssemblyEnd(faultVisc);

  VecGetOwnershipRange(_bcL,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar tauRSV = 0;
    ierr = VecGetValues(tauRS,1,&Ii,&tauRSV);CHKERRQ(ierr);

    // viscous strength
    VecGetValues(faultVisc,1,&Ii,&v);
    PetscScalar tauVisc = v*_vL/2.0/10.0; // 10 = seismogenic depth

    PetscScalar tau = min(tauRSV,tauVisc);
    //~ PetscScalar tau = tauRSV;
    //~ PetscScalar tau = tauVisc;
    VecSetValue(_bcL,Ii,tau,INSERT_VALUES);
  }
  VecAssemblyBegin(_bcL); VecAssemblyEnd(_bcL);

  _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  KSPDestroy(&_ksp);
  VecDestroy(&faultVisc);
  delete _sbp;
  _sbp = NULL;

  // extract boundary condition information from u
  Vec uL;
  VecDuplicate(_bcL,&uL);
  PetscScalar minVal = 0;
  VecMin(_u,NULL,&minVal);
  ierr = VecGetOwnershipRange(_u,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    // put left boundary info into fault slip vector
    if ( Ii < _Nz ) {
      ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
      v += abs(minVal);
      ierr = VecSetValues(uL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }

    // put right boundary data into bcR
    if ( Ii > (_Ny*_Nz - _Nz - 1) ) {
      PetscInt zI =  Ii - (_Ny*_Nz - _Nz);
      ierr = VecGetValues(_u,1,&Ii,&v);CHKERRQ(ierr);
      v += abs(minVal);
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

  // reset all BCs
  //~ VecSet(_bcRShift,0.0);
  //~ VecSet(_bcRShift,13.0);
  //~ VecSet(_bcR,_vL*_initTime/2.0);
  //~ VecSet(_bcL,0.0);
  //~ VecSet(_fault->_slip,0.0);
  //~ VecCopy(_fault->_slip,*(_var.begin()+2));
  //~ VecSet(_u,0.0);


  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// inititialize effective viscosity
PetscErrorCode PowerLaw::guessSteadyStateEffVisc()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::guessSteadyStateEffVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar strainRate = 1e-12,s=0.; // guess
  PetscScalar *A,*B,*n,*T,*effVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArray(_A,&A);
  VecGetArray(_B,&B);
  VecGetArray(_n,&n);
  VecGetArray(_T,&T);
  VecGetArray(_effVisc,&effVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    s = pow(strainRate/(A[Jj]*exp(-B[Jj]/T[Jj])),1.0/n[Jj]);
    effVisc[Jj] =  s/strainRate* 1e-3; // (GPa s)  in terms of strain rate
    Jj++;
  }
  VecRestoreArray(_A,&A);
  VecRestoreArray(_B,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


PetscErrorCode PowerLaw::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setMMSInitialConditions()";
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
  if (_Nz == 1) { mapToVec(_B,zzmms_B1D,*_y); }
  else { mapToVec(_B,zzmms_B,*_y,*_z); }
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
    Mat qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
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
  ierr = setStresses(time); CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

/*
PetscErrorCode PowerLaw::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();

  // ensure max time step is limited by Maxwell time
  PetscScalar maxTimeStep_tot, maxDeltaT_Tmax = 0.0;
  computeMaxTimeStep(maxDeltaT_Tmax);
  maxTimeStep_tot = min(_maxDeltaT,maxDeltaT_Tmax);

  _stepCount++;
  if (_timeIntegrator.compare("IMEX")==0) {
    _quadImex->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);

    // control which fields are used to select step size
    ierr = _quadImex->setErrInds(_timeIntInds);
    if (_bcLTauQS==1) {
      //~ int arrInds[] = {3,4}; // state: 0, slip: 1
      const char* tempList[] = {"gVxy","gVxz"};
      std::vector<std::string> errInds(tempList,tempList+2);
      ierr = _quadImex->setErrInds(errInds);
    }

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else { // fully explicit time integration
    // call odeSolver routine integrate here
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);

    // control which fields are used to select step size
    if (_isMMS) {
      //~ int arrInds[] = {3,4}; // state: 0, slip: 1
      //~ std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
      const char* tempList[] = {"gVxy","gVxz"};
      std::vector<string> errInds(tempList,tempList+2);
      ierr = _quadEx->setErrInds(errInds);
    }
    else if (_bcLTauQS==1) {
      //~ int arrInds[] = {3,4}; // state: 0, slip: 1
      //~ std::vector<int> errInds(arrInds,arrInds+2); // !! UPDATE THIS LINE TOO
      const char* tempList[] = {"gVxy","gVxz"};
      std::vector<string> errInds(tempList,tempList+2);
      ierr = _quadEx->setErrInds(errInds);
    }
    else  {
      ierr = _quadEx->setErrInds(_timeIntInds);
    }
    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
*/

// limited by Maxwell time
PetscErrorCode PowerLaw::computeMaxTimeStep(PetscScalar& maxTimeStep)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeMaxTimeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec Tmax;
  VecDuplicate(_u,&Tmax);
  VecSet(Tmax,0.0);
  VecPointwiseDivide(Tmax,_effVisc,_muVec);
  PetscScalar min_Tmax;
  VecMin(Tmax,NULL,&min_Tmax);

  maxTimeStep = 0.3 * min_Tmax;

  VecDestroy(&Tmax);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  LinearElastic::initiateIntegrand(time,varEx,varIm);

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

PetscErrorCode PowerLaw::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "LinearElastic::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  LinearElastic::updateFields(time,varEx,varIm);
  VecCopy(varEx.find("gVxy")->second,_gxy);
  VecCopy(varEx.find("gVxz")->second,_gxz);

  //~ if (_stepCount % 20 == 0) {
    if (varIm.find("Temp") != varIm.end() && _thermalCoupling.compare("coupled")==0) {
      VecCopy(varIm.find("Temp")->second,_T);
    }
  //~ }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
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

// implicit/explicit time stepping
PetscErrorCode PowerLaw::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::d_dt IMEX";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  assert(0);

  //~ if (_thermalCoupling.compare("coupled")==0 ) {
    //~ VecCopy(varImo.find("Temp")->second,_T);
    //~ _he.setTemp(_T);
    //~ _he.getTemp(_T);
  //~ }

  //~ ierr = d_dt_eqCycle(time,varEx,dvarEx);CHKERRQ(ierr);

  //~ if (_heatEquationType.compare("transient")==0 ) {
  //~ ierr = _he.be(time,*(dvarBegin+2),_fault->_tauQSP,_sdev,*(dvarBegin+3),
    //~ *(dvarBegin+4),*varBeginIm,*varBeginImo,dt);CHKERRQ(ierr);
  // arguments:
  // time, slipVel, sigmadev, dgxy, dgxz, T, dTdt
  //~ ierr = _he.be(time,dvarEx.find("slip")->second,_fault->_tauQSP,_sdev,dvarEx.find("gVxy")->second,
    //~ dvarEx.find("gVxz")->second,varIm.find("Temp")->second,varImo.find("Temp")->second,dt);CHKERRQ(ierr);



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode PowerLaw::d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::d_dt_eqCycle";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

//~ double startMiscTime = MPI_Wtime();
//~ _miscTime += MPI_Wtime() - startMiscTime;


  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);

  // set up rhs vector
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;


  // update stresses, viscosity, and set shear traction on fault
  ierr = setStresses(time);CHKERRQ(ierr);
  //~ computeViscosity();

  ierr = setViscStrainRates(time,_gxy,_gxz,dvarEx["gVxy"],dvarEx["gVxz"]); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::computeTotalStrainRates(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::computeTotalStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,dvarEx["gVxy"],dvarEx["gVxz"]);CHKERRQ(ierr);

  // set up rhs vector
  Vec bcL_t, bcR_t;
  VecDuplicate(_bcL,&bcL_t);
  VecDuplicate(_bcL,&bcR_t);
  VecSet(bcR_t,_vL/2.);
  VecCopy(dvarEx["slip"],bcL_t);
  ierr = _sbp->setRhs(_rhs,bcL_t,bcR_t,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);
  VecDestroy(&bcL_t);
  VecDestroy(&bcR_t);

  // solve for u_t
  Vec u_t;
  VecDuplicate(_u,&u_t);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,u_t);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // solve for total strain rate
  ierr = _sbp->Dy(u_t,dvarEx["gTxy"]); CHKERRQ(ierr);
  ierr = _sbp->Dz(u_t,dvarEx["gTxz"]); CHKERRQ(ierr);
  PetscScalar *gTxy,*gVxy,*gTxz,*gVxz=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_gxy,&Istart,&Iend);
  VecGetArray(_gxy,&gVxy);
  VecGetArray(_gxz,&gVxz);
  VecGetArray(dvarEx["gTxy"],&gTxy);
  VecGetArray(dvarEx["gTxz"],&gTxz);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    gTxy[Jj] = gTxy[Jj] - gVxy[Jj];
    gTxz[Jj] = gTxz[Jj] - gVxz[Jj];
    Jj++;
  }
  VecRestoreArray(dvarEx["gTxy"],&gTxy);
  VecRestoreArray(dvarEx["gTxz"],&gTxz);
  VecRestoreArray(_gxy,&gVxy);
  VecRestoreArray(_gxz,&gVxz);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::d_dt_mms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  _currTime = time;

  // force viscous strains to be correct
  //~ if (_Nz == 1) { mapToVec(_gxy,zzmms_gxy1D,*_y,time); }
  //~ else { mapToVec(_gxy,zzmms_gxy,*_y,*_z,time); }
  //~ if (_Nz == 1) { mapToVec(_gxz,zzmms_gxy1D,*_y,time); }
  //~ else { mapToVec(_gxz,zzmms_gxz,*_y,*_z,time); }

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time); CHKERRQ(ierr); // modifies _bcL,_bcR,_bcT, and _bcB
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB); CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_u,&viscSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&viscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxviscSourceMMS); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&uSource); CHKERRQ(ierr);
  ierr = VecDuplicate(_u,&HxuSource); CHKERRQ(ierr);

  //~ ierr = setViscStrainSourceTerms(viscSource,_var.begin());CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz); CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,zzmms_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,zzmms_gSource,*_y,*_z,time); }
  ierr = _sbp->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,zzmms_uSource1D,*_y,time); }
  else { mapToVec(uSource,zzmms_uSource,*_y,*_z,time); }
  ierr = _sbp->H(uSource,HxuSource);
  VecDestroy(&uSource);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxviscSourceMMS); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,HxuSource); CHKERRQ(ierr);
  }

  ierr = VecAXPY(_rhs,1.0,viscSource); CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhs,1.0,HxviscSourceMMS); CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhs,1.0,HxuSource); CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u); CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  //~ mapToVec(_u,zzmms_uA,*_y,*_z,time);

  // update stresses
  ierr = setStresses(time); CHKERRQ(ierr);
  //~ mapToVec(_sxy,zzmms_pl_sigmaxy,*_y,*_z,_currTime);
  //~ mapToVec(_sxz,zzmms_pl_sigmaxz,*_y,*_z,_currTime);
  //~ mapToVec(_sdev,zzmms_sdev,*_y,*_z,_currTime);
  computeViscosity();

  // update rates
  //~ ierr = setViscStrainRates(time,_gxy,_gxz,*(dvarBegin+3),*(dvarBegin+4)); CHKERRQ(ierr);
  ierr = setViscStrainRates(time,_gxy,_gxz,dvarEx["gVxy"],dvarEx["gVxz"]); CHKERRQ(ierr);
  Vec source;
  VecDuplicate(_u,&source);
  if (_Nz == 1) { mapToVec(source,zzmms_pl_gxy_t_source1D,*_y,_currTime); }
  else { mapToVec(source,zzmms_pl_gxy_t_source,*_y,*_z,_currTime); }
  VecAXPY(dvarEx["gVxy"],1.0,source);
  if (_Nz == 1) { mapToVec(source,zzmms_pl_gxz_t_source1D,*_y,_currTime); }
  else { mapToVec(source,zzmms_pl_gxz_t_source,*_y,*_z,_currTime); }
  VecAXPY(dvarEx["gVxz"],1.0,source);
  VecDestroy(&source);


  // force rates to be correct
  //~ mapToVec(dvarEx["gVxy"],zzmms_gxy_t,*_y,*_z,time);
  //~ mapToVec(dvarEx["gVxz"],zzmms_gxz_t,*_y,*_z,time);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::setViscStrainSourceTerms(Vec& out,Vec& gxy, Vec& gxz)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setViscStrainSourceTerms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(gxy,&source);
  VecSet(source,0.0);

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz)
  // + Hz^-1 E0z mu gxz - Hz^-1 ENz mu gxz
  Vec sourcexy_y;
  VecDuplicate(_u,&sourcexy_y);
  VecSet(sourcexy_y,0.0);
  ierr = _sbp->Dyxmu(gxy,sourcexy_y);CHKERRQ(ierr);

  // if bcL is shear stress, then also add Hy^-1 E0y mu gxy
  if (_bcLTauQS==1) {
    Vec temp1,bcL;
    VecDuplicate(gxy,&temp1); VecSet(temp1,0.0);
    VecDuplicate(gxy,&bcL); VecSet(bcL,0.0);
    _sbp->HyinvxE0y(gxy,temp1);
    ierr = VecPointwiseMult(bcL,_muVec,temp1); CHKERRQ(ierr);
    VecDestroy(&temp1);
    ierr = VecAXPY(sourcexy_y,1.0,bcL);CHKERRQ(ierr);
    VecDestroy(&bcL);
  }

  ierr = VecCopy(sourcexy_y,source);CHKERRQ(ierr); // sourcexy_y -> source
  VecDestroy(&sourcexy_y);

  if (_Nz > 1)
  {
    Vec sourcexz_z;
    VecDuplicate(gxz,&sourcexz_z);
    ierr = _sbp->Dzxmu(gxz,sourcexz_z);CHKERRQ(ierr);
    ierr = VecAXPY(source,1.0,sourcexz_z);CHKERRQ(ierr); // source += Hxsourcexz_z
    VecDestroy(&sourcexz_z);

    // enforce traction boundary condition
    Vec temp1,bcT,bcB;
    VecDuplicate(gxz,&temp1); VecSet(temp1,0.0);
    VecDuplicate(gxz,&bcT);
    VecDuplicate(gxz,&bcB);

    _sbp->HzinvxE0z(gxz,temp1);
    ierr = VecPointwiseMult(bcT,_muVec,temp1); CHKERRQ(ierr);

    _sbp->HzinvxENz(gxz,temp1);
    ierr = VecPointwiseMult(bcB,_muVec,temp1); CHKERRQ(ierr);

    ierr = VecAXPY(source,1.0,bcT);CHKERRQ(ierr);
    ierr = VecAXPY(source,-1.0,bcB);CHKERRQ(ierr);

    VecDestroy(&temp1);
    VecDestroy(&bcT);
    VecDestroy(&bcB);
  }

  // apply effects of coordinate transform
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = multMatsVec(yq,zr,source); CHKERRQ(ierr);
  }
  ierr = _sbp->H(source,out); CHKERRQ(ierr);
  VecDestroy(&source);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode PowerLaw::computeViscosity()
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeViscosity";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // compute effective viscosity
  PetscScalar *sigmadev,*A,*B,*n,*T,*effVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArray(_sdev,&sigmadev);
  VecGetArray(_A,&A);
  VecGetArray(_B,&B);
  VecGetArray(_n,&n);
  VecGetArray(_T,&T);
  VecGetArray(_effVisc,&effVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    effVisc[Jj] = 1e-3 / ( A[Jj]*pow(sigmadev[Jj],n[Jj]-1.0)*exp(-B[Jj]/T[Jj]) ) ;
    effVisc[Jj] = min(effVisc[Jj],1e30);

    assert(~isnan(effVisc[Jj]));
    assert(~isinf(effVisc[Jj]));
    Jj++;
  }
  VecRestoreArray(_sdev,&sigmadev);
  VecRestoreArray(_A,&A);
  VecRestoreArray(_B,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


//~ PetscErrorCode PowerLaw::setViscStrainRates(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
PetscErrorCode PowerLaw::setViscStrainRates(const PetscScalar time,const Vec& gVxy, const Vec& gVxz,
  Vec& gVxy_t, Vec& gVxz_t)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setViscStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

// add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_gTxy,&SAT);
  ierr = setViscousStrainRateSAT(_u,_bcL,_bcR,SAT);CHKERRQ(ierr);
  VecSet(SAT,0.0); // !!!

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  VecPointwiseMult(gVxy_t,_muVec,SAT);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gxy,&temp1);
    ierr = _sbp->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,gVxy_t,temp1);
    VecCopy(temp1,gVxy_t);
    VecDestroy(&temp1);
  }
  VecAXPY(gVxy_t,1.0,_sxy);
  VecPointwiseDivide(gVxy_t,gVxy_t,_effVisc);


  if (_Nz > 1) {
    VecCopy(_sxz,gVxz_t);
    VecPointwiseDivide(gVxz_t,gVxz_t,_effVisc);
  }

  VecDestroy(&SAT);

  // add

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode PowerLaw::setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::viscousStrainRateSAT";
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

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes sigmaxy, sigmaxz, and sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::setStresses(const PetscScalar time)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setStresses";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  _sbp->Dy(_u,_gTxy);
  VecCopy(_gTxy,_sxy);
  VecAXPY(_sxy,-1.0,_gxy);
  VecPointwiseMult(_sxy,_sxy,_muVec);

  // deviatoric stress: part 1/3
  VecPointwiseMult(_sdev,_sxy,_sxy);

  if (_Nz > 1) {
    _sbp->Dz(_u,_gTxz);
    VecCopy(_gTxz,_sxz);
    VecAXPY(_sxz,-1.0,_gxz);
    VecPointwiseMult(_sxz,_sxz,_muVec);

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

PetscErrorCode PowerLaw::getSigmaDev(Vec& sdev)
{
  sdev = _sdev;
  return 0;
}



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

PetscErrorCode PowerLaw::measureMMSError(const PetscScalar time)
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



//======================================================================
// IO functions
//======================================================================

// Save all scalar fields to text file named pl_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode PowerLaw::writeDomain()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = _outputDir + "pl_context.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"viscDistribution = %s\n",_viscDistribution.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);

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

PetscErrorCode PowerLaw::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  LinearElastic::writeContext();

  writeDomain();

  PetscViewer    vw;

  std::string str = _outputDir + "powerLawA";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_A,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "powerLawB";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_B,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "n";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_n,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  // contextual fields of members
  //~ ierr = _sbp->writeOps(_outputDir + "ops_u_"); CHKERRQ(ierr);
  //~ ierr = _fault->writeContext(_outputDir); CHKERRQ(ierr);
  //~ ierr = _he.writeContext(); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep1D(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  LinearElastic::writeStep1D(time);

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep2D(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();
  LinearElastic::writeStep2D(time);

  if (_gTxyV==NULL) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
              FILE_MODE_WRITE,&_gTxyV);CHKERRQ(ierr);
    ierr = VecView(_gTxy,_gTxyV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gTxyV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
                                   FILE_MODE_APPEND,&_gTxyV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
              FILE_MODE_WRITE,&_sxyPV);CHKERRQ(ierr);
    ierr = VecView(_sxy,_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
                                   FILE_MODE_APPEND,&_sxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
              FILE_MODE_WRITE,&_gxyV);CHKERRQ(ierr);
    ierr = VecView(_gxy,_gxyV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gxyV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
                                   FILE_MODE_APPEND,&_gxyV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"effVisc").c_str(),
              FILE_MODE_WRITE,&_effViscV);CHKERRQ(ierr);
    ierr = VecView(_effVisc,_effViscV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_effViscV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"effVisc").c_str(),
                                   FILE_MODE_APPEND,&_effViscV);CHKERRQ(ierr);

    //~ // write out boundary conditions for testing purposes
    //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
              //~ FILE_MODE_WRITE,&_bcRlusV);CHKERRQ(ierr);
    //~ ierr = VecView(_bcR,_bcRlusV);CHKERRQ(ierr);
    //~ ierr = PetscViewerDestroy(&_bcRlusV);CHKERRQ(ierr);
    //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
                                   //~ FILE_MODE_APPEND,&_bcRlusV);CHKERRQ(ierr);

    //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
              //~ FILE_MODE_WRITE,&_bcLlusV);CHKERRQ(ierr);
    //~ ierr = VecView(_bcL,_bcLlusV);CHKERRQ(ierr);
    //~ ierr = PetscViewerDestroy(&_bcLlusV);CHKERRQ(ierr);
    //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
                                   //~ FILE_MODE_APPEND,&_bcLlusV);CHKERRQ(ierr);

    //~if (_isMMS) {
      //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                //~FILE_MODE_WRITE,&_uAnalV);CHKERRQ(ierr);
      //~ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);
      //~ierr = PetscViewerDestroy(&_uAnalV);CHKERRQ(ierr);
      //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                                     //~FILE_MODE_APPEND,&_uAnalV);CHKERRQ(ierr);
    //~}
    if (_Nz>1)
    {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxzP").c_str(),
              FILE_MODE_WRITE,&_gTxzV);CHKERRQ(ierr);
      ierr = VecView(_gTxz,_gTxzV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gTxzV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxzP").c_str(),
                                     FILE_MODE_APPEND,&_gTxzV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
               FILE_MODE_WRITE,&_sxzPV);CHKERRQ(ierr);
      ierr = VecView(_sxz,_sxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_sxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
                                     FILE_MODE_APPEND,&_sxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
               FILE_MODE_WRITE,&_gxzV);CHKERRQ(ierr);
      ierr = VecView(_gxz,_gxzV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gxzV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
                                   FILE_MODE_APPEND,&_gxzV);CHKERRQ(ierr);
    }
  }
  else {
    //~ ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);
    //~ _he.writeStep2D(_stepCount);

    //~ ierr = VecView(_bcR,_bcRlusV);CHKERRQ(ierr);
    //~ ierr = VecView(_bcL,_bcLlusV);CHKERRQ(ierr);

    //~ ierr = VecView(_u,_uV);CHKERRQ(ierr);
    ierr = VecView(_gTxy,_gTxyV);CHKERRQ(ierr);
    ierr = VecView(_sxy,_sxyPV);CHKERRQ(ierr);
    ierr = VecView(_gxy,_gxyV);CHKERRQ(ierr);
    ierr = VecView(_effVisc,_effViscV);CHKERRQ(ierr);
    //~if (_isMMS) {ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);}
    if (_Nz>1)
    {
      ierr = VecView(_gTxz,_gTxzV);CHKERRQ(ierr);
      ierr = VecView(_sxz,_sxzPV);CHKERRQ(ierr);
      ierr = VecView(_gxz,_gxzV);CHKERRQ(ierr);
    }
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::view(const double totRunTime)
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







// why not use the genFuncs implementation??
// Fills vec with the linear interpolation between the pairs of points (vals,depths)
PetscErrorCode PowerLaw::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setVecFromVectors";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}



// Play around with psuedo-timestepping
PetscErrorCode PowerLaw::psuedoTS_main()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::psuedoTS";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  VecSet(_effVisc,1e11);

  // compute mu*effVisc^(-1) for Jacobian
  Vec muDivVisc;
  ierr = VecCreate(PETSC_COMM_WORLD,&muDivVisc);CHKERRQ(ierr);
  ierr = VecSetSizes(muDivVisc,PETSC_DECIDE,2*_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muDivVisc);CHKERRQ(ierr);
  VecSet(muDivVisc,0.0);
  Vec temp; VecDuplicate(_muVec,&temp);
  VecPointwiseDivide(temp, _muVec, _effVisc);
  repVec(muDivVisc,temp, 2);
  VecDestroy(&temp);


  // create Jacobian
  Mat J;
  MatCreate(PETSC_COMM_WORLD,&J);
  MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2*_Ny*_Nz,2*_Ny*_Nz);
  MatSetFromOptions(J);
  MatMPIAIJSetPreallocation(J,1,NULL,0,NULL); // nnz per row
  MatSeqAIJSetPreallocation(J,1,NULL); // nnz per row
  MatSetUp(J);
  MatDiagonalSet(J,muDivVisc,INSERT_VALUES);

  // create Vec to contain output
  Vec g;
  VecDuplicate(muDivVisc,&g);
  VecSet(g,0.);
  VecDestroy(&muDivVisc);

  // create time stepper context
  TS ts;
  TSCreate(PETSC_COMM_WORLD,&ts);
  TSSetProblemType(ts,TS_NONLINEAR);
  TSSetSolution(ts,g); // where to compute solution
  TSSetInitialTimeStep(ts,0.0,1e-3); // set initial time (meaningless), and time step
  TSPseudoSetTimeStep(ts,TSPseudoTimeStepDefault,0); // strategy for increasing time step
  TSSetDuration(ts,1e5,1e10); // # of timesteps and final time
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);

  // provide call-back functions
  void* ctx = this;
  TSSetIJacobian(ts,J,J,computeIJacobian,ctx);
  TSSetIFunction(ts,NULL,evaluateIRHS,ctx);
  //~ TSSetRHSJacobian(ts,J,J,computeJacobian,ctx);
  //~ TSSetRHSFunction(ts,NULL,evaluateRHS,ctx);
  TSMonitorSet(ts,monitor,ctx,NULL);


  TSSetFromOptions(ts);
  TSSetUp(ts);

  //~ TSGetTolerances(TS ts,PetscReal *atol,Vec *vatol,PetscReal *rtol,Vec *vrtol)
  //~ PetscReal atol, rtol;
  //~ TSGetTolerances(ts,&atol,NULL,&rtol,NULL);
  //~ PetscPrintf(PETSC_COMM_WORLD,"atol = %g, %rtol = %g\n",atol,rtol);


  TSSolve(ts,g);


  /*
  Vec gxy, gxz, g;
  VecDuplicate(mu,&g);
  PetscInt Istart,Iend;
  VecGetOwnershipRange(g,&Istart,&Iend);
  for( PetscInt Ii=Istart; Ii<Iend; Ii++) {
    PetscScalar v = Ii;
    VecSetValue(g,Ii,v,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(g);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(g);CHKERRQ(ierr);

  VecDuplicate(_muVec,&gxy); VecSet(gxy,0.0);
  VecDuplicate(_muVec,&gxz); VecSet(gxz,0.0);
  sepVec(gxy,g,0,_Ny*_Nz);
  sepVec(gxz,g,_Ny*_Nz,2*_Ny*_Nz);
  VecSet(g,0.0);
  distributeVec(g,gxy,0,_Ny*_Nz);
  distributeVec(g,gxz,_Ny*_Nz,2*_Ny*_Nz);
  VecView(g,PETSC_VIEWER_STDOUT_WORLD);
  */




  //~ VecDestroy(&mu);
  //~ VecDestroy(&effVisc);
  //~ VecDestroy(&muDivVisc);
  MatDestroy(&J);

  PetscPrintf(PETSC_COMM_WORLD,"hello world!\n");
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// returns F(X,Xdot)
PetscErrorCode PowerLaw::psuedoTS_evaluateIRHS(Vec&F,PetscReal time,Vec& g,Vec& g_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::psuedoTS_evaluateIRHS";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %f\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif


  // extract gxy and gxz from g
  sepVec(_gxy,g,0,_Ny*_Nz);
  sepVec(_gxz,g,_Ny*_Nz,2*_Ny*_Nz);

  // extract _gxy_t and _gxz_t from g_t
  Vec _gxy_t, _gxz_t;
  VecDuplicate(_gxy,&_gxy_t); VecSet(_gxy_t,0.0);
  VecDuplicate(_gxz,&_gxz_t); VecSet(_gxz_t,0.0);
  sepVec(_gxy_t,g_t,0,_Ny*_Nz);
  sepVec(_gxz_t,g_t,_Ny*_Nz,2*_Ny*_Nz);


  // solve for u
  // add source terms to rhs: d/dy( 2*mu*gVxy ) + d/dz( 2*mu*gVxz )
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);

  // set up rhs vector
  //~ ierr = VecSet(_bcR,_vL*time/2.0);CHKERRQ(ierr);
  //~ ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);


  // solve for displacement u
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;


  // solve for u_t
  Vec bcL;
  VecDuplicate(_bcR,&bcL);
  VecSet(bcL,0.0);
  //~ VecSet(_bcR,_vL/2.0);
  VecSet(_bcR,0);
  ierr = _sbp->setRhs(_rhs,bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = setViscStrainSourceTerms(viscSource,_gxy_t,_gxz_t);CHKERRQ(ierr);
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);

  // solve for u_t
  Vec u_t;
  VecDuplicate(_u,&u_t);
  startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,u_t);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecDestroy(&viscSource);
  VecDestroy(&bcL);


  // evaluate RHS
  ierr = setStresses(time);CHKERRQ(ierr); // also computes gTxy, gTxz
  //~ computeViscosity();
  VecSet(_effVisc,1e11);
  Vec _gTxy_t, _gTxz_t;
  VecDuplicate(_gxy,&_gTxy_t); VecSet(_gTxy_t,0.0);
  VecDuplicate(_gxy,&_gTxz_t); VecSet(_gTxz_t,0.0);
  _sbp->Dy(_u,_gTxy);
  _sbp->Dz(_u,_gTxz);

  Vec _gExy_t, _gExz_t;
  VecDuplicate(_gxy,&_gExy_t); VecSet(_gExy_t,0.0);
  VecDuplicate(_gxz,&_gExz_t); VecSet(_gExz_t,0.0);
  PetscInt Istart,Iend;
  PetscScalar mu,effVisc,gTxy_t,gTxy,gVxy,gExy_t = 0.;
  PetscScalar gTxz_t,gTxz,gVxz,gExz_t = 0.;
  VecGetOwnershipRange(_gExy_t,&Istart,&Iend);
  for( PetscInt Ii=Istart; Ii<Iend; Ii++) {
    ierr = VecGetValues(_muVec,1,&Ii,&mu);CHKERRQ(ierr);
    ierr = VecGetValues(_effVisc,1,&Ii,&effVisc);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxy,1,&Ii,&gTxy);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxz,1,&Ii,&gTxz);CHKERRQ(ierr);
    ierr = VecGetValues(_gxy,1,&Ii,&gVxy);CHKERRQ(ierr);
    ierr = VecGetValues(_gxz,1,&Ii,&gVxz);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxy_t,1,&Ii,&gTxy_t);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxz_t,1,&Ii,&gTxz_t);CHKERRQ(ierr);

    gExy_t = gTxy_t - mu/effVisc*(gTxy - gVxy);
    gExz_t = gTxz_t - mu/effVisc*(gTxz - gVxz);

    VecSetValue(_gExy_t,Ii,gExy_t,INSERT_VALUES);
    VecSetValue(_gExz_t,Ii,gExz_t,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(_gExy_t);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_gExz_t);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_gExy_t);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_gExz_t);CHKERRQ(ierr);

  // place elastic strain rates into output vector
  distributeVec(F,_gExy_t,0,_Ny*_Nz);
  distributeVec(F,_gExz_t,_Ny*_Nz,2*_Ny*_Nz);


  VecDestroy(&_gxy_t);
  VecDestroy(&_gxz_t);
  VecDestroy(&u_t);
  VecDestroy(&_gTxy_t);
  VecDestroy(&_gTxz_t);
  VecDestroy(&_gExy_t);
  VecDestroy(&_gExz_t);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// returns F(X,Xdot)
PetscErrorCode PowerLaw::psuedoTS_evaluateRHS(Vec& F,PetscReal time,Vec& g)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::psuedoTS_evaluateRHS";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %f\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif


  // extract gxy and gxz from g
  sepVec(_gxy,g,0,_Ny*_Nz);
  sepVec(_gxz,g,_Ny*_Nz,2*_Ny*_Nz);

  // extract _gxy_t and _gxz_t from g_t
  Vec _gxy_t, _gxz_t;
  VecDuplicate(_gxy,&_gxy_t); VecSet(_gxy_t,0.0);
  VecDuplicate(_gxz,&_gxz_t); VecSet(_gxz_t,0.0);
  //~ sepVec(_gxy_t,g_t,0,_Ny*_Nz);
  //~ sepVec(_gxz_t,g_t,_Ny*_Nz,2*_Ny*_Nz);


  // solve for u
  // add source terms to rhs: d/dy( 2*mu*gVxy ) + d/dz( 2*mu*gVxz )
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);

  // set up rhs vector
  ierr = VecSet(_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);


  // solve for displacement u
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;


  // compute intermediate fields
  ierr = setStresses(time);CHKERRQ(ierr); // also computes gTxy, gTxz
  //~ computeViscosity();
  Vec _gTxy_t, _gTxz_t;
  VecDuplicate(_gxy,&_gTxy_t); VecSet(_gTxy_t,0.0);
  VecDuplicate(_gxy,&_gTxz_t); VecSet(_gTxz_t,0.0);
  _sbp->Dy(_u,_gTxy);
  _sbp->Dz(_u,_gTxz);

  // compute viscous strains


  // solve for u_t
  Vec bcL;
  VecDuplicate(_bcR,&bcL);
  VecSet(bcL,0.0);
  VecSet(_bcR,_vL/2.0);
  ierr = _sbp->setRhs(_rhs,bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = setViscStrainSourceTerms(viscSource,_gxy_t,_gxz_t);CHKERRQ(ierr);
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);

  // solve for u_t
  Vec u_t;
  VecDuplicate(_u,&u_t);
  startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,u_t);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecDestroy(&viscSource);
  VecDestroy(&bcL);




  // evaluate RHS
  Vec _gExy_t, _gExz_t;
  VecDuplicate(_gxy,&_gExy_t); VecSet(_gExy_t,0.0);
  VecDuplicate(_gxz,&_gExz_t); VecSet(_gExz_t,0.0);
  PetscInt Istart,Iend;
  PetscScalar mu,effVisc,gTxy_t,gTxy,gVxy,gExy_t = 0.;
  PetscScalar gTxz_t,gTxz,gVxz,gExz_t = 0.;
  VecGetOwnershipRange(_gExy_t,&Istart,&Iend);
  for( PetscInt Ii=Istart; Ii<Iend; Ii++) {
    ierr = VecGetValues(_muVec,1,&Ii,&mu);CHKERRQ(ierr);
    ierr = VecGetValues(_effVisc,1,&Ii,&effVisc);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxy,1,&Ii,&gTxy);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxz,1,&Ii,&gTxz);CHKERRQ(ierr);
    ierr = VecGetValues(_gxy,1,&Ii,&gVxy);CHKERRQ(ierr);
    ierr = VecGetValues(_gxz,1,&Ii,&gVxz);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxy_t,1,&Ii,&gTxy_t);CHKERRQ(ierr);
    ierr = VecGetValues(_gTxz_t,1,&Ii,&gTxz_t);CHKERRQ(ierr);

    gExy_t = gTxy_t - mu/effVisc*(gTxy - gVxy);
    gExz_t = gTxz_t - mu/effVisc*(gTxz - gVxz);

    VecSetValue(_gExy_t,Ii,gExy_t,INSERT_VALUES);
    VecSetValue(_gExz_t,Ii,gExz_t,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(_gExy_t);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_gExz_t);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_gExy_t);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_gExz_t);CHKERRQ(ierr);

  // place elastic strain rates into output vector
  distributeVec(F,_gExy_t,0,_Ny*_Nz);
  distributeVec(F,_gExz_t,_Ny*_Nz,2*_Ny*_Nz);


  VecDestroy(&_gxy_t);
  VecDestroy(&_gxz_t);
  VecDestroy(&u_t);
  VecDestroy(&_gTxy_t);
  VecDestroy(&_gTxz_t);
  VecDestroy(&_gExy_t);
  VecDestroy(&_gExz_t);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// returns Jacobian for explicit solve
PetscErrorCode PowerLaw::psuedoTS_computeJacobian(Mat& J,PetscReal time,Vec& g)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::psuedoTS_computeJacobian(Mat& J,PetscReal time,Vec g)";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %f\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // for now assume effective viscosity is constant
/*
  // extract gxy and gxz from g
  sepVec(_gxy,g,0,_Ny*_Nz);
  sepVec(_gxz,g,_Ny*_Nz,2*_Ny*_Nz);

  // solve for u
  // add source terms to rhs: d/dy( 2*mu*gVxy ) + d/dz( 2*mu*gVxz )
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);

  // set up rhs vector
  ierr = VecSet(_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);


  // solve for displacement u
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // evaluate RHS
  ierr = setStresses(time);CHKERRQ(ierr);
  computeViscosity();
  * */


  Vec muDivVisc;
  VecDuplicate(g,&muDivVisc);
  Vec temp; VecDuplicate(_muVec,&temp);
  VecPointwiseDivide(temp, _muVec, _effVisc);
  repVec(muDivVisc,temp, 2);
  MatDiagonalSet(J,muDivVisc,INSERT_VALUES);


  VecDestroy(&temp);
  VecDestroy(&muDivVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// returns Jacobian for implicit
PetscErrorCode PowerLaw::psuedoTS_computeIJacobian(Mat& J,PetscReal time,Vec& g,Vec& g_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::psuedoTS_computeIJacobian";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %f\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

/*
  // extract gxy and gxz from g
  sepVec(_gxy,g,0,_Ny*_Nz);
  sepVec(_gxz,g,_Ny*_Nz,2*_Ny*_Nz);


  // solve for u
  // add source terms to rhs: d/dy( 2*mu*gVxy ) + d/dz( 2*mu*gVxz )
  Vec viscSource;
  ierr = VecDuplicate(_gxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,_gxy,_gxz);CHKERRQ(ierr);

  // set up rhs vector
  ierr = VecSet(_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);
  ierr = _sbp->setRhs(_rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhs,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);


  // solve for displacement u
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_u);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // evaluate RHS
  ierr = setStresses(time);CHKERRQ(ierr);
  computeViscosity();
  */


  Vec muDivVisc;
  VecDuplicate(g,&muDivVisc);
  VecSet(muDivVisc,-30./1e11);
  //~ Vec temp; VecDuplicate(_muVec,&temp);
  //~ VecPointwiseDivide(temp, _muVec, _effVisc);
  //~ repVec(muDivVisc,temp, 2);
  MatDiagonalSet(J,muDivVisc,INSERT_VALUES);


  //~ VecDestroy(&temp);
  VecDestroy(&muDivVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// MMS functions
double PowerLaw::zzmms_sigmaxz(const double y,const double z, const double t)
{ return zzmms_mu(y,z)*zzmms_uA_z(y,z,t); }


// specific MMS functions
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

//======================================================================
//======================================================================

// call-back function that returns Jacobian for PETSc'c TSSetIJacobian
PetscErrorCode computeIJacobian(TS ts,PetscReal time,Vec g,Vec g_t,PetscReal a,Mat Amat,Mat Pmat,void *ptr)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "computeIJacobian";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  PowerLaw *pl = (PowerLaw*) ptr; // from PETSc tutorial
  //~ PowerLaw *pl = static_cast<PowerLaw*> (ptr); // from stack overflow

  pl->psuedoTS_computeIJacobian(Amat,time,g,g_t);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// call-back function that returns Jacobian for PETSc'c TSSetRHSJacobian
PetscErrorCode computeJacobian(TS ts,PetscReal time,Vec g,Mat Amat,Mat Pmat,void *ptr)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "computeJacobian";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  //~ PowerLaw *pl = (PowerLaw*) ptrTSPSU; // from PETSc tutorial
  PowerLaw *pl = static_cast<PowerLaw*> (ptr); // from stack overflow

  pl->psuedoTS_computeJacobian(Amat,time,g);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// call-back function that returns F(X,Xdot) for PETSc'c TSSetIFunction
PetscErrorCode evaluateIRHS(TS ts,PetscReal time,Vec g,Vec g_t,Vec F,void *ptr)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "evaluateIRHS";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  //~ PowerLaw *pl = (PowerLaw*) ptr; // from PETSc tutorial
  PowerLaw *pl = static_cast<PowerLaw*> (ptr); // from stack overflow

  pl->psuedoTS_evaluateIRHS(F,time,g,g_t);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// call-back function that returns F(X,Xdot) for PETSc'c TSSetRHSFunction
PetscErrorCode evaluateRHS(TS ts,PetscReal time,Vec g,Vec F,void *ptr)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "evaluateRHS";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  //~ PowerLaw *pl = (PowerLaw*) ptr; // from PETSc tutorial
  PowerLaw *pl = static_cast<PowerLaw*> (ptr); // from stack overflow

  pl->psuedoTS_evaluateRHS(F,time,g);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// call-back function that writes relevent data to memory
PetscErrorCode monitor(TS ts,PetscInt stepCount,PetscReal time,Vec g,void *ptr)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "monitor";
    std::string fileName = "PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  //~ PowerLaw *pl = (PowerLaw*) ptr; // from PETSc tutorial
  PowerLaw *pl = static_cast<PowerLaw*> (ptr); // from stack overflow

  sepVec(pl->_gxy,g,0,pl->_Ny*pl->_Nz);
  sepVec(pl->_gxz,g,pl->_Ny*pl->_Nz,2*pl->_Ny*pl->_Nz);
  //~ pl->timeMonitor(time,stepCount);



  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
