#include "powerLaw.hpp"

#define FILENAME "powerLaw.cpp"


PowerLaw::PowerLaw(Domain& D,HeatEquation& he,Vec& tau)
: LinearElastic(D,tau), _file(D._file),_delim(D._delim),
  _momBalType("transient"),
  _viscDistribution("unspecified"),_AFile("unspecified"),_BFile("unspecified"),_nFile("unspecified"),
  _A(NULL),_n(NULL),_QR(NULL),_T(NULL),_effVisc(NULL),SATL(NULL),
  _B(NULL),_C(NULL),
  _sbp_eta(NULL),_ksp_eta(NULL),_pc_eta(NULL),_v(NULL),
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
  he.getTemp(_T);
  setMaterialParameters();

  // guess steady state conditions
  //~ guessSteadyStateEffVisc(1e-12);
  //~ setSSInitialConds(D,tau);
  setUpSBPContext(D); // set up matrix operators
  initializeMomBalMats();
  //~ computeTotalStrains(_currTime);
  //~ computeStresses(_currTime);
  if (D._loadICs==1) {
    loadFieldsFromFiles();
    setUpSBPContext(D); // set up matrix operators
    computeTotalStrains(_currTime);
    computeStresses(_currTime);
    computeViscosity();
  }

  //~ if (_momBalType.compare("steadyState")==0) {
    //~ initializeSSMatrices(D); // initialize Bss and Css
    //~ solveSSProblem(10);
  //~ }

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

  VecDestroy(&SATL);

  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_T);
  VecDestroy(&_QR);
  VecDestroy(&_effVisc);
  VecDestroy(&_v);
  VecDestroy(&_sxz);
  VecDestroy(&_sdev);
  VecDestroy(&_gTxy);
  VecDestroy(&_gTxz);
  VecDestroy(&_gxy);
  VecDestroy(&_gxz);
  VecDestroy(&_dgxy);
  VecDestroy(&_dgxz);

  PetscViewerDestroy(&_timeV2D);

  MatDestroy(&_B);
  MatDestroy(&_C);

  KSPDestroy(&_ksp_eta);
  delete _sbp_eta; _sbp = NULL;

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
    if (var.compare("powerLawMomBalType")==0) {
      _momBalType = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    else if (var.compare("viscDistribution")==0) {
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

  if (_viscDistribution.compare("loadFromFile")==0) { assert(_inputDir.compare("unspecified")); }

  assert(_momBalType.compare("transient")==0 || _momBalType.compare("steadyState")==0 );

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
  ierr = VecDuplicate(_u,&_QR);CHKERRQ(ierr);
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

  VecDuplicate(_bcL,&SATL); VecSet(SATL,0.0);

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
    VecSet(_QR,_BVals[0]);
    VecSet(_n,_nVals[0]);
  }
  else {
    if (_viscDistribution.compare("mms")==0) {
      if (_Nz == 1) { mapToVec(_A,zzmms_A1D,*_y); }
      else { mapToVec(_A,zzmms_A,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_QR,zzmms_B1D,*_y); }
      else { mapToVec(_QR,zzmms_B,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_n,zzmms_n1D,*_y); }
      else { mapToVec(_n,zzmms_n,*_y,*_z); }
    }
    else if (_viscDistribution.compare("loadFromFile")==0) { loadEffViscFromFiles(); }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_QR,_BVals,_BDepths);CHKERRQ(ierr);
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
  ierr = VecLoad(_QR,inv);CHKERRQ(ierr);

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

  // load bcL and bcR
  ierr = loadVecFromInputFile(_bcL,_inputDir,"bcL"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_bcRShift,_inputDir,"bcR"); CHKERRQ(ierr);
  VecSet(_bcR,0.0);

  // load viscous strains
  ierr = loadVecFromInputFile(_gxy,_inputDir,"Gxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_gxz,_inputDir,"Gxz"); CHKERRQ(ierr);

  // load stresses
  ierr = loadVecFromInputFile(_sxy,_inputDir,"Sxy"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_sxz,_inputDir,"Sxz"); CHKERRQ(ierr);

  // load effective viscosity
  ierr = loadVecFromInputFile(_effVisc,_inputDir,"EffVisc"); CHKERRQ(ierr);

  // load temperature
  ierr = loadVecFromInputFile(_T,_inputDir,"T"); CHKERRQ(ierr);

  // load power law parameters
  ierr = loadVecFromInputFile(_A,_inputDir,"A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"B"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"n"); CHKERRQ(ierr);


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

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// compute B and C
// B = H*Dy*mu + SAT terms
// C = H*Dz*mu + SAT terms
PetscErrorCode PowerLaw::initializeMomBalMats()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initializeMomBalMats";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // get necessary matrix factors
  Mat Dy,Dz;
  Mat Hyinv,Hzinv,H;
  Mat muqy,murz,mu;
  Mat E0y,ENy,E0z,ENz;
  Mat qy,rz,yq,zr;
  _sbp->getDs(Dy,Dz);
  _sbp->getHinvs(Hyinv,Hzinv);
  _sbp->getH(H);
  _sbp->getMus(mu,muqy,murz);
  _sbp->getEs(E0y,ENy,E0z,ENz);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp->getCoordTrans(qy,rz,yq,zr);
  }

  // helpful factor qyxrzxH = qy * rz * H, and yqxzrxH = yq * zr * H
  Mat yqxHy,zrxHz,yqxzrxH;
  if (_sbpType.compare("mfc_coordTrans")==0) {
    ierr = MatMatMatMult(yq,zr,H,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxzrxH); CHKERRQ(ierr);
    ierr = MatMatMult(yqxzrxH,Hzinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy); CHKERRQ(ierr); // correct
    ierr = MatMatMult(yqxzrxH,Hyinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&zrxHz); CHKERRQ(ierr);
  }
  else {
    ierr = MatMatMult(H,Hzinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&yqxHy);
    ierr = MatMatMult(H,Hyinv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&zrxHz);
    ierr = MatConvert(H,MATSAME,MAT_INITIAL_MATRIX,&yqxzrxH); CHKERRQ(ierr);
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

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// compute Bss and Css
PetscErrorCode PowerLaw::initializeSSMatrices(Domain &D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::initializeSSMatrices";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";

  if (_sbpType.compare("mc")==0) {
    _sbp_eta = new SbpOps_c(D,_Ny,_Nz,_effVisc,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp_eta = new SbpOps_fc(D,_Ny,_Nz,_effVisc,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp_eta = new SbpOps_fc_coordTrans(D,_Ny,_Nz,_effVisc,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  KSPCreate(PETSC_COMM_WORLD,&_ksp_eta);
  setupKSP(_sbp_eta,_ksp_eta,_pc_eta);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// inititialize effective viscosity
PetscErrorCode PowerLaw::guessSteadyStateEffVisc(const PetscScalar strainRate)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::guessSteadyStateEffVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ PetscScalar strainRate = 1e-12; // guess
  PetscScalar s=0.;
  PetscScalar *A,*B,*n,*T,*effVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  VecGetArray(_A,&A);
  VecGetArray(_QR,&B);
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
  VecRestoreArray(_QR,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  return ierr;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// for steady state computations
// compute initial tauVisc (from guess at effective viscosity)
PetscErrorCode PowerLaw::getTauVisc(Vec& tauVisc, const PetscScalar ess_t)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::getTauVisc";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // get steady state effective viscosity
  guessSteadyStateEffVisc(ess_t);

  // use effective viscosity to compute strength of off-fault material
  if (tauVisc == NULL) { VecDuplicate(_bcL,&tauVisc); }

  // first get viscosity just on fault
  PetscInt Istart,Iend;
  PetscScalar v = 0;
  ierr = VecGetOwnershipRange(_effVisc,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_Nz) {
      ierr = VecGetValues(_effVisc,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(tauVisc,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(tauVisc);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(tauVisc);CHKERRQ(ierr);

  VecScale(tauVisc,ess_t);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
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
  if (_Nz == 1) { mapToVec(_QR,zzmms_B1D,*_y); }
  else { mapToVec(_QR,zzmms_B,*_y,*_z); }
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
  ierr = computeTotalStrains(time); CHKERRQ(ierr);
  ierr = computeStresses(time); CHKERRQ(ierr);


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
  if (_momBalType.compare("transient")==0) {
    Vec vargxyP; VecDuplicate(_u,&vargxyP); VecCopy(_gxy,vargxyP);
    Vec vargxzP; VecDuplicate(_u,&vargxzP); VecCopy(_gxz,vargxzP);
    varEx["gVxy"] = vargxyP;
    varEx["gVxz"] = vargxzP;
  }

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

  // if integrating viscous strains in time
    VecCopy(varEx.find("gVxy")->second,_gxy);
    VecCopy(varEx.find("gVxz")->second,_gxz);

  // if also solving coupled heat equation
  if (varIm.find("Temp") != varIm.end() && _thermalCoupling.compare("coupled")==0) {
      VecCopy(varIm.find("Temp")->second,_T);
    }

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


  // add source terms to rhs: d/dy(mu*gVxy) + d/dz(mu*gVxz)
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
  ierr = computeTotalStrains(time);CHKERRQ(ierr);
  ierr = computeStresses(time);CHKERRQ(ierr);
  //~ computeViscosity();

  ierr = setViscStrainRates(time,_gxy,_gxz,dvarEx["gVxy"],dvarEx["gVxz"]); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

// solve for steady-state v, viscous strain rates
PetscErrorCode PowerLaw::updateSS(Domain& D,const Vec& tau)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::updateSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  if (_sbp_eta == NULL) { initializeSSMatrices(D); }

  // set up rhs vector
  Vec bcR_v,bcT,bcB;
  VecDuplicate(_bcR,&bcR_v); VecSet(bcR_v,_vL/2.);
  VecDuplicate(_bcB,&bcB); VecSet(bcB,0.);
  VecDuplicate(_bcT,&bcT); VecSet(bcT,0.);
  VecCopy(tau,_bcL);
  ierr = _sbp_eta->setRhs(_rhs,_bcL,bcR_v,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs

  Vec effVisc_old,gVxy_t,gVxz_t;
  VecDuplicate(_effVisc,&effVisc_old);
  VecDuplicate(_effVisc,&gVxy_t); VecSet(gVxy_t,0.0);
  VecDuplicate(_effVisc,&gVxz_t); VecSet(gVxz_t,0.0);
  VecDuplicate(_effVisc,&_v); VecSet(_v,0.0);

  // set up IO to evaluate loop
  _viewers["SS_effVisc"] = initiateViewer(_outputDir + "SS_effVisc");
  _viewers["SS_gVxy_t"] = initiateViewer(_outputDir + "SS_gVxy_t");
  _viewers["SS_gVxz_t"] = initiateViewer(_outputDir + "SS_gVxz_t");
  _viewers["SS_sxy"] = initiateViewer(_outputDir + "SS_sxy");
  _viewers["SS_sxz"] = initiateViewer(_outputDir + "SS_sxz");
  _viewers["SS_v"] = initiateViewer(_outputDir + "SS_v");

  ierr = VecView(_effVisc,_viewers["SS_effVisc"]); CHKERRQ(ierr);
  ierr = VecView(gVxy_t,_viewers["SS_gVxy_t"]); CHKERRQ(ierr);
  ierr = VecView(gVxz_t,_viewers["SS_gVxz_t"]); CHKERRQ(ierr);
  ierr = VecView(_sxy,_viewers["SS_sxy"]); CHKERRQ(ierr);
  ierr = VecView(_sxz,_viewers["SS_sxz"]); CHKERRQ(ierr);
  ierr = VecView(_v,_viewers["SS_v"]); CHKERRQ(ierr);

  ierr = appendViewer(_viewers["SS_effVisc"],_outputDir + "SS_effVisc");
  ierr = appendViewer(_viewers["SS_gVxy_t"],_outputDir + "SS_gVxy_t");
  ierr = appendViewer(_viewers["SS_gVxz_t"],_outputDir + "SS_gVxz_t");
  ierr = appendViewer(_viewers["SS_sxy"],_outputDir + "SS_sxy");
  ierr = appendViewer(_viewers["SS_sxz"],_outputDir + "SS_sxz");
  ierr = appendViewer(_viewers["SS_v"],_outputDir + "SS_v");

  double err = 1e10;
  int Ii = 0;
  while ( Ii < 50 && err > 1e-3) {
    VecCopy(_effVisc,effVisc_old);

    _sbp_eta->updateVarCoeff(_effVisc);
    Mat A;
    _sbp_eta->getA(A);
    ierr = KSPSetOperators(_ksp_eta,A,A);CHKERRQ(ierr); // update operator

    // solve for steady-state velocity
    ierr = KSPSolve(_ksp_eta,_rhs,_v);CHKERRQ(ierr);

    // update viscous strain rates
    _sbp_eta->Dy(_v,gVxy_t);
    _sbp_eta->Dz(_v,gVxz_t);

    // update stresses
    ierr = VecPointwiseMult(_sxy,_effVisc,gVxy_t); CHKERRQ(ierr);
    ierr = VecPointwiseMult(_sxz,_effVisc,gVxz_t); CHKERRQ(ierr);

    // update effective viscosity
    //~ ierr = computeViscosity(); CHKERRQ(ierr);

    err = computeNormDiff_2(effVisc_old,_effVisc);
    PetscPrintf(PETSC_COMM_WORLD,"    %i %e\n",Ii,err);
    Ii++;

    ierr = VecView(_effVisc,_viewers["SS_effVisc"]); CHKERRQ(ierr);
    ierr = VecView(gVxy_t,_viewers["SS_gVxy_t"]); CHKERRQ(ierr);
    ierr = VecView(gVxz_t,_viewers["SS_gVxz_t"]); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["SS_sxy"]); CHKERRQ(ierr);
    ierr = VecView(_sxz,_viewers["SS_sxz"]); CHKERRQ(ierr);
    ierr = VecView(_v,_viewers["SS_v"]); CHKERRQ(ierr);
  }

  // update u and viscous strains
  //~ PetscScalar time = 1e12;
  PetscScalar time = 100;
  VecCopy(_v,_u);
  VecScale(_u,time);

  _viewers["SS_u"] = initiateViewer(_outputDir + "SS_u");
  ierr = VecView(_u,_viewers["SS_u"]); CHKERRQ(ierr);

  PetscScalar *mu,*gxy_t,*gxz_t,*gxy,*gxz,*sxy,*sxz=0;
  PetscInt Istart,Iend;
  VecGetOwnershipRange(_sxy,&Istart,&Iend);
  VecGetArray(_muVec,&mu);
  VecGetArray(_sxy,&sxy);
  VecGetArray(_sxz,&sxz);
  VecGetArray(_gxy,&gxy);
  VecGetArray(_gxz,&gxz);
  VecGetArray(gVxy_t,&gxy_t);
  VecGetArray(gVxz_t,&gxz_t);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar gVxy0 = -sxy[Jj]/mu[Jj];
    PetscScalar gVxz0 = -sxz[Jj]/mu[Jj];
    gxy[Jj] = gxy_t[Jj] * time + gVxy0;
    gxz[Jj] = gxz_t[Jj] * time + gVxz0;

    Jj++;
  }
  VecRestoreArray(_muVec,&mu);
  VecRestoreArray(_sxy,&sxy);
  VecRestoreArray(_sxz,&sxz);
  VecRestoreArray(_gxy,&gxy);
  VecRestoreArray(_gxz,&gxz);
  VecRestoreArray(gVxy_t,&gxy_t);
  VecRestoreArray(gVxz_t,&gxz_t);

  _viewers["SS_gxy"] = initiateViewer(_outputDir + "SS_gxy");
  ierr = VecView(_gxy,_viewers["SS_gxy"]); CHKERRQ(ierr);
  _viewers["SS_gxz"] = initiateViewer(_outputDir + "SS_gxz");
  ierr = VecView(_gxz,_viewers["SS_gxz"]); CHKERRQ(ierr);

  VecDestroy(&effVisc_old);
  VecDestroy(&gVxy_t);
  VecDestroy(&gVxz_t);
  VecDestroy(&bcR_v);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}




// solve for steady-state u, gVxy, gVxz and iterate on effective viscosity
PetscErrorCode PowerLaw::solveSSProblem(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::solveSSProblem";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Dirichlet";
  if (_bcLTauQS==1) { _bcLType = "Neumann"; }

  // set up rhs vector
  Vec bcR_v,bcT,bcB;
  VecDuplicate(_bcR,&bcR_v); VecSet(bcR_v,_vL/2.);
  VecDuplicate(_bcB,&bcB); VecSet(bcB,0.);
  VecDuplicate(_bcT,&bcT); VecSet(bcT,0.);
  ierr = _sbp_eta->setRhs(_rhs,_bcL,bcR_v,_bcT,_bcB);CHKERRQ(ierr); // update rhs from BCs

  Vec effVisc_old,gVxy_t,gVxz_t;
  VecDuplicate(_effVisc,&effVisc_old);
  VecDuplicate(_effVisc,&gVxy_t); VecSet(gVxy_t,0.0);
  VecDuplicate(_effVisc,&gVxz_t); VecSet(gVxz_t,0.0);
  VecDuplicate(_effVisc,&_v); VecSet(_v,0.0);

    // set up IO to evaluate loop
  _viewers["SS_effVisc"] = initiateViewer(_outputDir + "SS_effVisc");
  _viewers["SS_gVxy_t"] = initiateViewer(_outputDir + "SS_gVxy_t");
  _viewers["SS_gVxz_t"] = initiateViewer(_outputDir + "SS_gVxz_t");
  _viewers["SS_sxy"] = initiateViewer(_outputDir + "SS_sxy");
  _viewers["SS_sxz"] = initiateViewer(_outputDir + "SS_sxz");

  ierr = VecView(_effVisc,_viewers["SS_effVisc"]); CHKERRQ(ierr);
  ierr = VecView(gVxy_t,_viewers["SS_gVxy_t"]); CHKERRQ(ierr);
  ierr = VecView(gVxz_t,_viewers["SS_gVxz_t"]); CHKERRQ(ierr);
  ierr = VecView(_sxy,_viewers["SS_sxy"]); CHKERRQ(ierr);
  ierr = VecView(_sxz,_viewers["SS_sxz"]); CHKERRQ(ierr);

  ierr = appendViewer(_viewers["SS_effVisc"],_outputDir + "SS_effVisc");
  ierr = appendViewer(_viewers["SS_gVxy_t"],_outputDir + "SS_gVxy_t");
  ierr = appendViewer(_viewers["SS_gVxz_t"],_outputDir + "SS_gVxz_t");
  ierr = appendViewer(_viewers["SS_sxy"],_outputDir + "SS_sxy");
  ierr = appendViewer(_viewers["SS_sxz"],_outputDir + "SS_sxz");

  double err = 1e10;
  int Ii = 0;
  while ( Ii < 10 && err > 1e-3) {
    VecCopy(_effVisc,effVisc_old);

    _sbp_eta->updateVarCoeff(_effVisc);
    Mat A;
    _sbp_eta->getA(A);
    ierr = KSPSetOperators(_ksp_eta,A,A);CHKERRQ(ierr); // update operator

    // solve for steady-state velocity
    double startTime = MPI_Wtime();
    ierr = KSPSolve(_ksp_eta,_rhs,_v);CHKERRQ(ierr); // why does this hang??????????
    _linSolveTime += MPI_Wtime() - startTime;
    _linSolveCount++;

    // update viscous strain rates
    _sbp_eta->Dy(_v,gVxy_t);
    _sbp_eta->Dz(_v,gVxz_t);

    // update stresses
    ierr = VecPointwiseMult(_effVisc,gVxy_t,_sxy); CHKERRQ(ierr);
    ierr = VecPointwiseMult(_effVisc,gVxz_t,_sxz); CHKERRQ(ierr);

    // update effective viscosity
    ierr = computeViscosity(); CHKERRQ(ierr);


    err = computeNormDiff_2(effVisc_old,_effVisc);
    PetscPrintf(PETSC_COMM_WORLD,"    %i %e\n",Ii,err);
    Ii++;

    ierr = VecView(_effVisc,_viewers["SS_effVisc"]); CHKERRQ(ierr);
    ierr = VecView(gVxy_t,_viewers["SS_gVxy_t"]); CHKERRQ(ierr);
    ierr = VecView(gVxz_t,_viewers["SS_gVxz_t"]); CHKERRQ(ierr);
    ierr = VecView(_sxy,_viewers["SS_sxy"]); CHKERRQ(ierr);
    ierr = VecView(_sxz,_viewers["SS_sxz"]); CHKERRQ(ierr);
  }


  // update u and viscous strains
  VecCopy(_v,_u);
  VecScale(_u,time);

  PetscScalar *mu,*gxy_t,*gxz_t,*gxy,*gxz,*sxy,*sxz=0;
  PetscInt Istart,Iend;
  VecGetOwnershipRange(_sxy,&Istart,&Iend);
  VecGetArray(_muVec,&mu);
  VecGetArray(_sxy,&sxy);
  VecGetArray(_sxz,&sxz);
  VecGetArray(_gxy,&gxy);
  VecGetArray(_gxz,&gxz);
  VecGetArray(gVxy_t,&gxy_t);
  VecGetArray(gVxz_t,&gxz_t);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar gVxy0 = sxy[Jj]/mu[Jj];
    PetscScalar gVxz0 = sxz[Jj]/mu[Jj];
    gxy[Jj] = gxy_t[Jj] * time + gVxy0;
    gxz[Jj] = gxz_t[Jj] * time + gVxz0;

    Jj++;
  }
  VecRestoreArray(_muVec,&mu);
  VecRestoreArray(_sxy,&sxy);
  VecRestoreArray(_sxz,&sxz);
  VecRestoreArray(_gxy,&gxy);
  VecRestoreArray(_gxz,&gxz);
  VecRestoreArray(gVxy_t,&gxy_t);
  VecRestoreArray(gVxz_t,&gxz_t);

  VecDestroy(&effVisc_old);
  VecDestroy(&gVxy_t);
  VecDestroy(&gVxz_t);
  VecDestroy(&bcR_v);

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
  ierr = computeTotalStrains(time); CHKERRQ(ierr);
  ierr = computeStresses(time); CHKERRQ(ierr);
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

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz) + SAT
  ierr = MatMult(_B,_gxy,out); CHKERRQ(ierr);
  if (_Nz > 1) {
    ierr = MatMultAdd(_C,_gxz,out,out); CHKERRQ(ierr);
  }

/*
  Vec source;
  VecDuplicate(_u,&source);
  VecSet(source,0.0);

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


  // Hz^-1 E0z mu gxz - Hz^-1 ENz mu gxz
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
*/

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
  VecGetArray(_QR,&B);
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
  VecRestoreArray(_QR,&B);
  VecRestoreArray(_n,&n);
  VecRestoreArray(_T,&T);
  VecRestoreArray(_effVisc,&effVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


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
  ierr = setViscousStrainRateSAT(_u,_bcL,_bcR,SAT); CHKERRQ(ierr);
  VecSet(SAT,0.0); // !!!

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  VecPointwiseMult(gVxy_t,_muVec,SAT);
  VecAXPY(gVxy_t,1.0,_sxy);
  VecPointwiseDivide(gVxy_t,gVxy_t,_effVisc);

  if (_Nz > 1) {
    VecCopy(_sxz,gVxz_t);
    VecPointwiseDivide(gVxz_t,gVxz_t,_effVisc);
  }

  VecDestroy(&SAT);


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

  // include effects of coordinate transform
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gxy,&temp1);
    ierr = _sbp->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,out,temp1);
    VecCopy(temp1,out);
    VecDestroy(&temp1);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes gTxy and gTxz
PetscErrorCode PowerLaw::computeTotalStrains(const PetscScalar time)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeTotalStrains";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  _sbp->Dy(_u,_gTxy);
  if (_Nz > 1) {
    _sbp->Dz(_u,_gTxz);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

// computes sigmaxy, sigmaxz, and sigmadev = sqrt(sigmaxy^2 + sigmaxz^2)
PetscErrorCode PowerLaw::computeStresses(const PetscScalar time)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::computeStresses";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(_gTxy,_sxy);
  VecAXPY(_sxy,-1.0,_gxy);
  VecPointwiseMult(_sxy,_sxy,_muVec);

  // deviatoric stress: part 1/3
  VecPointwiseMult(_sdev,_sxy,_sxy);

  if (_Nz > 1) {
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


  //~ // create SBP operators with viscosity as coefficient

  writeDomain();

  PetscViewer    vw;

  std::string str = _outputDir + "powerLawA";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_A,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = _outputDir + "powerLawB";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_QR,vw);CHKERRQ(ierr);
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


PetscErrorCode PowerLaw::writeStep1D(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  LinearElastic::writeStep1D(stepCount,time);

  if (stepCount == 0) {
    //~ _viewers["pl_SATL"] = initiateViewer(_outputDir + "pl_SATL");
    //~ _viewers["pl_SATR"] = initiateViewer(_outputDir + "pl_SATR");

    //~ ierr = VecView(SATL,_viewers["pl_SATL"]); CHKERRQ(ierr);
    //~ ierr = VecView(SATR,_viewers["pl_SATR"]); CHKERRQ(ierr);

    //~ ierr = appendViewer(_viewers["pl_SATL"],_outputDir + "pl_SATL");
    //~ ierr = appendViewer(_viewers["pl_SATR"],_outputDir + "pl_SATR");
  }
  else {
    //~ ierr = VecView(SATL,_viewers["pl_SATL"]);CHKERRQ(ierr);
    //~ ierr = VecView(SATR,_viewers["pl_SATR"]);CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep2D(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();
  LinearElastic::writeStep2D(stepCount,time);

  if (stepCount == 0) {
    _viewers["gTxy"] = initiateViewer(_outputDir + "gTxy");
    _viewers["gxy"] = initiateViewer(_outputDir + "gxy");
    _viewers["effVisc"] = initiateViewer(_outputDir + "effVisc");

    ierr = VecView(_gTxy,_viewers["gTxy"]); CHKERRQ(ierr);
    ierr = VecView(_gxy,_viewers["gxy"]); CHKERRQ(ierr);
    ierr = VecView(_effVisc,_viewers["effVisc"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["gTxy"],_outputDir + "gTxy");
    ierr = appendViewer(_viewers["gxy"],_outputDir + "gxy");
    ierr = appendViewer(_viewers["effVisc"],_outputDir + "effVisc");

    if (_Nz>1) {
      _viewers["gTxz"] = initiateViewer(_outputDir + "gTxz");
      _viewers["gxz"] = initiateViewer(_outputDir + "gxz");
      _viewers["sxz"] = initiateViewer(_outputDir + "sxz");

      ierr = VecView(_gTxz,_viewers["gTxz"]); CHKERRQ(ierr);
      ierr = VecView(_gxz,_viewers["gxz"]); CHKERRQ(ierr);
      ierr = VecView(_sxz,_viewers["sxz"]); CHKERRQ(ierr);

      ierr = appendViewer(_viewers["gTxz"],_outputDir + "gTxz");
      ierr = appendViewer(_viewers["gxz"],_outputDir + "gxz");
      ierr = appendViewer(_viewers["sxz"],_outputDir + "sxz");
    }
  }
  else {
    ierr = VecView(_gTxy,_viewers["gTxy"]); CHKERRQ(ierr);
    ierr = VecView(_gxy,_viewers["gxy"]); CHKERRQ(ierr);
    ierr = VecView(_effVisc,_viewers["effVisc"]); CHKERRQ(ierr);
    if (_Nz>1) {
      ierr = VecView(_gTxz,_viewers["gTxz"]); CHKERRQ(ierr);
      ierr = VecView(_gxz,_viewers["gxz"]); CHKERRQ(ierr);
      ierr = VecView(_sxz,_viewers["sxz"]); CHKERRQ(ierr);
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
  ierr = computeTotalStrains(time); CHKERRQ(ierr);
  ierr = computeStresses(time);CHKERRQ(ierr);
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
  ierr = computeTotalStrains(time); CHKERRQ(ierr);
  ierr = computeStresses(time);CHKERRQ(ierr); // also computes gTxy, gTxz
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
  ierr = computeStresses(time);CHKERRQ(ierr);
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
  ierr = computeStresses(time);CHKERRQ(ierr);
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
