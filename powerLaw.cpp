#include "powerLaw.hpp"

#define FILENAME "powerLaw.cpp"


PowerLaw::PowerLaw(Domain& D)
: SymmLinearElastic(D), _file(D._file),_delim(D._delim),_inputDir(D._inputDir),
  _viscDistribution("unspecified"),_AFile("unspecified"),_BFile("unspecified"),_nFile("unspecified"),
  _A(NULL),_n(NULL),_B(NULL),_effVisc(NULL),
  _sxzP(NULL),_sigmadev(NULL),
  _gxyP(NULL),_dgxyP(NULL),
  _gxzP(NULL),_dgxzP(NULL),
  _gTxyP(NULL),_gTxzP(NULL),
  //~ _T(NULL),
  _sxyPV(NULL),_sxzPV(NULL),_sigmadevV(NULL),
  _gTxyPV(NULL),_gTxzPV(NULL),
  _gxyPV(NULL),_dgxyPV(NULL),
  _gxzPV(NULL),_dgxzPV(NULL),
  _TV(NULL),_effViscV(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  allocateFields(); // initialize fields
  setMaterialParameters();

  if (D._loadICs==1) {
    loadFieldsFromFiles();
    setUpSBPContext(D); // set up matrix operators
    setStresses(_initTime);
    }
  else {
    setSSInitialConds(D);
    setUpSBPContext(D); // set up matrix operators
    setStresses(_initTime);
    guessSteadyStateEffVisc();
  }

  if (_isMMS) { setMMSInitialConditions(); }


  // add viscous strain to integrated variables, stored in _var
  Vec vargxyP; VecDuplicate(_uP,&vargxyP); VecCopy(_gxyP,vargxyP);
  Vec vargxzP; VecDuplicate(_uP,&vargxzP); VecCopy(_gxzP,vargxzP);
  _var.push_back(vargxyP);
  _var.push_back(vargxzP);


  _he.getTemp(_T); // should already be done from linear elastic
  // if also solving heat equation
  if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    Vec T;
    VecDuplicate(_uP,&T);
    _he.getTemp(T);
    _varIm.push_back(T);
    _he.getTemp(_T);
  }

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

  VecDestroy(&_bcRPShift);
  VecDestroy(&_effVisc);
  VecDestroy(&_T);

  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_B);

  VecDestroy(&_sxzP);
  VecDestroy(&_sigmadev);

  VecDestroy(&_gTxyP);
  VecDestroy(&_gTxzP);
  VecDestroy(&_gxyP);
  VecDestroy(&_gxzP);
  VecDestroy(&_dgxyP);
  VecDestroy(&_dgxzP);

  PetscViewerDestroy(&_sxyPV);
  PetscViewerDestroy(&_sxzPV);
  PetscViewerDestroy(&_sigmadevV);
  PetscViewerDestroy(&_gTxyPV);
  PetscViewerDestroy(&_gTxzPV);
  PetscViewerDestroy(&_gxyPV);
  PetscViewerDestroy(&_gxzPV);
  PetscViewerDestroy(&_dgxyPV);
  PetscViewerDestroy(&_dgxzPV);
  PetscViewerDestroy(&_effViscV);

  //~ PetscViewerDestroy(&_timeV2D);

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

  ierr = VecDuplicate(_uP,&_A);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_B);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_n);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&_effVisc);CHKERRQ(ierr);


  // allocate space for stress and strain vectors
  VecDuplicate(_uP,&_sxzP); VecSet(_sxzP,0.0);
  VecDuplicate(_uP,&_sigmadev); VecSet(_sigmadev,0.0);

  VecDuplicate(_uP,&_gxyP);
  PetscObjectSetName((PetscObject) _gxyP, "_gxyP");
  VecSet(_gxyP,0.0);
  VecDuplicate(_uP,&_dgxyP);
  PetscObjectSetName((PetscObject) _dgxyP, "_dgxyP");
  VecSet(_dgxyP,0.0);

  VecDuplicate(_uP,&_gxzP);
  PetscObjectSetName((PetscObject) _gxzP, "_gxzP");
  VecSet(_gxzP,0.0);
  VecDuplicate(_uP,&_dgxzP);
  PetscObjectSetName((PetscObject) _dgxzP, "_dgxzP");
  VecSet(_dgxzP,0.0);

  VecDuplicate(_uP,&_gTxyP); VecSet(_gTxyP,0.0);
  VecDuplicate(_uP,&_gTxzP); VecSet(_gTxzP,0.0);

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
      //~ mapToVec(_A,MMS_A,_Nz,_dy,_dz);
      //~ mapToVec(_B,MMS_B,_Nz,_dy,_dz);
      //~ mapToVec(_n,MMS_n,_Nz,_dy,_dz);

      if (_Nz == 1) { mapToVec(_A,MMS_A1D,*_y); }
      else { mapToVec(_A,MMS_A,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_B,MMS_B1D,*_y); }
      else { mapToVec(_B,MMS_B,*_y,*_z); }
      if (_Nz == 1) { mapToVec(_n,MMS_n1D,*_y); }
      else { mapToVec(_n,MMS_n,*_y,*_z); }
    }
    else if (_viscDistribution.compare("loadFromFile")==0) { loadEffViscFromFiles(); }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_B,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_n,_nVals,_nDepths);CHKERRQ(ierr);
    }
  }
  _he.getTemp(_T);
  //~ guessSteadyStateEffVisc();


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
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,inv);CHKERRQ(ierr);

  // load A
  vecSourceFile = _inputDir + "A";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_A,inv);CHKERRQ(ierr);

  // load B
  vecSourceFile = _inputDir + "B";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_B,inv);CHKERRQ(ierr);

  // load B
  vecSourceFile = _inputDir + "n";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
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


  /*  // load T (initial condition)
  ierr = VecCreate(PETSC_COMM_WORLD,&_T);CHKERRQ(ierr);
  ierr = VecSetSizes(_T,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_T);
  PetscObjectSetName((PetscObject) _T, "_T");
  ierr = loadVecFromInputFile(_T,_inputDir,_TFile);CHKERRQ(ierr);
  */


  // load bcL
  string vecSourceFile = _inputDir + "bcL";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcLP,inv);CHKERRQ(ierr);

  //~ // load bcR
  vecSourceFile = _inputDir + "bcR";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_bcRPShift,inv);CHKERRQ(ierr);

  // load gxy
  vecSourceFile = _inputDir + "Gxy";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxyP,inv);CHKERRQ(ierr);

  // load gxz
  vecSourceFile = _inputDir + "Gxz";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxzP,inv);CHKERRQ(ierr);


   // load sxy
  vecSourceFile = _inputDir + "Sxy";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_sxyP,inv);CHKERRQ(ierr);

  // load sxz
  vecSourceFile = _inputDir + "Sxz";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_sxzP,inv);CHKERRQ(ierr);


  // load effective viscosity
  vecSourceFile = _inputDir + "EffVisc";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_effVisc,inv);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// try to speed up spin up by starting closer to steady state
PetscErrorCode PowerLaw::setSSInitialConds(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::setSSInitialConds";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  delete _sbpP;

  // set up SBP operators
  //~ string bcT,string bcR,string bcB, string bcL
  std::string bcTType = "Neumann";
  std::string bcBType = "Neumann";
  std::string bcRType = "Dirichlet";
  std::string bcLType = "Neumann";

  if (_sbpType.compare("mc")==0) {
    _sbpP = new SbpOps_c(D,D._muVecP,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbpP = new SbpOps_fc(D,D._muVecP,bcTType,bcRType,bcBType,bcLType,"yz"); // to spin up viscoelastic
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbpP = new SbpOps_fc_coordTrans(D,D._muVecP,bcTType,bcRType,bcBType,bcLType,"yz");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  KSPDestroy(&_kspP);
  KSPCreate(PETSC_COMM_WORLD,&_kspP);
  setupKSP(_sbpP,_kspP,_pcP);

  // set up boundary conditions
  VecSet(_bcRP,0.0);
  PetscInt    Istart,Iend;
  PetscScalar v = 0;
  Vec faultVisc; VecDuplicate(_bcLP,&faultVisc);
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < _Nz) {
      VecGetValues(_effVisc,1,&Ii,&v);
      VecSetValue(faultVisc,Ii,v,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(faultVisc); VecAssemblyEnd(faultVisc);

  VecGetOwnershipRange(_bcLP,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar tauRS = _fault.getTauSS(Ii); // rate-and-state strength

    // viscous strength
    VecGetValues(faultVisc,1,&Ii,&v);
    PetscScalar tauVisc = v*_vL/2.0/10.0; // 10 = seismogenic depth

    //~ PetscScalar tau = min(tauRS,tauVisc);
    PetscScalar tau = tauRS;
    VecSetValue(_bcLP,Ii,tau,INSERT_VALUES);
  }
  VecAssemblyBegin(_bcLP); VecAssemblyEnd(_bcLP);

  writeVec(_bcLP,(_outputDir+"init1_bcL").c_str());
  writeVec(_bcRP,(_outputDir+"init1_bcR").c_str());

  _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  KSPDestroy(&_kspP);
  delete _sbpP;
  _sbpP = NULL;

  // extract boundary condition information from u
  PetscScalar minVal = 0;
  VecMin(_uP,NULL,&minVal);
  ierr = VecGetOwnershipRange(_uP,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    // put left boundary info into fault slip vector
    if ( Ii < _Nz ) {
      ierr = VecGetValues(_uP,1,&Ii,&v);CHKERRQ(ierr);
      v = 2.0*(v + abs(minVal));
      ierr = VecSetValues(_fault._slip,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }

    // put right boundary data into bcR
    if ( Ii > (_Ny*_Nz - _Nz - 1) ) {
      PetscInt zI =  Ii - (_Ny*_Nz - _Nz);
      ierr = VecGetValues(_uP,1,&Ii,&v);CHKERRQ(ierr);
      v = v + abs(minVal);
      ierr = VecSetValues(_bcRPShift,1,&zI,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcRPShift);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_fault._slip);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRPShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_fault._slip);CHKERRQ(ierr);
  VecCopy(_bcRPShift,_bcRP);
  VecCopy(_fault._slip,_bcLP);
  VecScale(_bcLP,0.5);
  VecCopy(_fault._slip,*(_var.begin()+2));

  writeVec(_bcLP,(_outputDir+"init2_bcL").c_str());
  writeVec(_bcRP,(_outputDir+"init2_bcR").c_str());
  writeVec(_bcRPShift,(_outputDir+"init2_bcRPShift").c_str());
  writeVec(_uP,(_outputDir+"init2_u").c_str());


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

  PetscScalar s,A,B,n,T,effVisc;
  PetscInt Ii,Istart,Iend;
  PetscScalar strainRate = 1e-12; // guess
  VecGetOwnershipRange(_effVisc,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_A,1,&Ii,&A);
    VecGetValues(_B,1,&Ii,&B);
    VecGetValues(_n,1,&Ii,&n);
    VecGetValues(_T,1,&Ii,&T);
    s = pow(strainRate/(A*exp(-B/T)),1.0/n);
    effVisc =  s/strainRate* 1e-3; // (GPa s)  in terms of strain rate

    VecSetValues(_effVisc,1,&Ii,&effVisc,INSERT_VALUES);
    assert(!isnan(effVisc));
  }
  VecAssemblyBegin(_effVisc);
  VecAssemblyEnd(_effVisc);


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

  PetscScalar time = _initTime;
  if (_Nz == 1) { mapToVec(_gxyP,MMS_gxy1D,*_y,time); }
  else { mapToVec(_gxyP,MMS_gxy,*_y,*_z,time); }
  if (_Nz == 1) { VecSet(_gxzP,0.0); }
  else { mapToVec(_gxzP,MMS_gxz,*_y,*_z,time); }

  // set material properties
  if (_Nz == 1) { mapToVec(_muVecP,MMS_mu1D,*_y); }
  else { mapToVec(_muVecP,MMS_mu,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_A,MMS_A1D,*_y); }
  else { mapToVec(_A,MMS_A,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_B,MMS_B1D,*_y); }
  else { mapToVec(_B,MMS_B,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_n,MMS_n1D,*_y); }
  else { mapToVec(_n,MMS_n,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_T,MMS_T1D,*_y); }
  else { mapToVec(_T,MMS_T,*_y,*_z); }

  VecCopy(_gxyP,*(_var.begin()+2));
  VecCopy(_gxzP,*(_var.begin()+3));

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcLP,_bcRP,_bcTP, and _bcBP
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_uP,&viscSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&viscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxviscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&uSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxuSource);CHKERRQ(ierr);

  ierr = setViscStrainSourceTerms(viscSource,_var.begin());CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,MMS_gSource1D,*_y,_currTime); }
  else { mapToVec(viscSourceMMS,MMS_gSource,*_y,*_z,_currTime); }
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,MMS_uSource1D,*_y,_currTime); }
  else { mapToVec(uSource,MMS_uSource,*_y,*_z,_currTime); }
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // set stresses
  ierr = setStresses(time);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();

  _stepCount++;
  if (_timeIntegrator.compare("IMEX")==0) {
    _quadImex->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_var,_varIm);CHKERRQ(ierr);

    // control which fields are used to select step size
    ierr = _quadImex->setErrInds(_timeIntInds);

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else { // fully explicit time integration
    // call odeSolver routine integrate here
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setInitialConds(_var);CHKERRQ(ierr);

    // control which fields are used to select step size
    if (_isMMS) {
      int arrInds[] = {3}; // state: 0, slip: 1
      std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
      ierr = _quadEx->setErrInds(errInds);
    }
    else if (_bcLTauQS==1) {
      int arrInds[] = {3,4}; // state: 0, slip: 1
      std::vector<int> errInds(arrInds,arrInds+2); // !! UPDATE THIS LINE TOO
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
  VecDuplicate(_uP,&Tmax);
  VecSet(Tmax,0.0);
  VecPointwiseDivide(Tmax,_effVisc,_muVecP);
  PetscScalar min_Tmax;
  VecMin(Tmax,NULL,&min_Tmax);

  maxTimeStep = 0.3 * min_Tmax;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  if (_isMMS) {
    ierr = d_dt_mms(time,varBegin,dvarBegin);CHKERRQ(ierr);
  }
  else {
    ierr = d_dt_eqCycle(time,varBegin,dvarBegin);CHKERRQ(ierr);
  }
  return ierr;
}

// implicit/explicit time stepping
PetscErrorCode PowerLaw::d_dt(const PetscScalar time,
  const_it_vec varBegin,it_vec dvarBegin,it_vec varBeginIm,const_it_vec varBeginImo,
  const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting PowerLaw::d_dt IMEX in powerLaw.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  if (_thermalCoupling.compare("coupled")==0) {
    //~ VecCopy(*varBeginImo,_T);
    //~ _he.getTemp(_T);
    VecWAXPY(_T,1.0,_he._T0,*varBeginImo);
  }

  ierr = d_dt_eqCycle(time,varBegin,dvarBegin);CHKERRQ(ierr);

  ierr = _he.be(time,*(dvarBegin+2),_fault._tauQSP,_sigmadev,*(dvarBegin+3),
    *(dvarBegin+4),*varBeginIm,*varBeginImo,dt);CHKERRQ(ierr);
  // arguments:
  // time, slipVel, sigmadev, dgxy, dgxz, T, dTdt


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending PowerLaw::d_dt IMEX in powerLaw.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode PowerLaw::d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::PowerLaw";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(*(varBegin+3),_gxyP);
  VecCopy(*(varBegin+4),_gxzP);

  // update boundaries
  if (_bcLTauQS==0) {
    ierr = VecCopy(*(varBegin+2),_bcLP);CHKERRQ(ierr);
    ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr);
  } // else do nothing
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = VecDuplicate(_gxyP,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,varBegin);CHKERRQ(ierr);

  // set up rhs vector
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // set shear traction on fault
  ierr = setStresses(time);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_sxyP,NULL);CHKERRQ(ierr);

  // set rates
  if (_bcLTauQS==0) {
    ierr = _fault.d_dt(varBegin,dvarBegin); // sets rates for slip and state
  }
  else {
    VecSet(*dvarBegin,0.0); // dstate psi
    VecSet(*(dvarBegin+1),0.0); // // dstate theta
    VecSet(*(dvarBegin+2),0.0); // slip vel
  }
  ierr = setViscStrainRates(time,varBegin,dvarBegin); CHKERRQ(ierr); // sets viscous strain rates

  //~ if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    //~ ierr = _he.d_dt(time,*(dvarBegin+1),_fault._tauQSP,_sxyP,_sxzP,*(dvarBegin+2),
      //~ *(dvarBegin+3),*(varBegin+4),*(dvarBegin+4),dt);CHKERRQ(ierr);
      //~ // arguments:
      //~ // time, slipVel, sigmaxy, sigmaxz, dgxy, dgxz, T, dTdt
  //~ }

  //~VecSet(*dvarBegin,0.0);
  //~VecSet(*(dvarBegin+1),0.0);
  //~VecSet(*(dvarBegin+2),0.0);
  //~ VecSet(*(dvarBegin+3),0.0);
  //~ VecSet(*(dvarBegin+4),0.0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::d_dt_mms(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::d_dt_mms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  _currTime = time;

  VecCopy(*(varBegin+3),_gxyP);
  VecCopy(*(varBegin+4),_gxzP);

  // force viscous strains to be correct
  //~ if (_Nz == 1) { mapToVec(_gxyP,MMS_gxy1D,*_y,time); }
  //~ else { mapToVec(_gxyP,MMS_gxy,*_y,*_z,time); }
  //~ if (_Nz == 1) { mapToVec(_gxzP,MMS_gxy1D,*_y,time); }
  //~ else { mapToVec(_gxzP,MMS_gxz,*_y,*_z,time); }

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcLP,_bcRP,_bcTP, and _bcBP
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_uP,&viscSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&viscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxviscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&uSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxuSource);CHKERRQ(ierr);

  ierr = setViscStrainSourceTerms(viscSource,_var.begin());CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,MMS_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,MMS_gSource,*_y,*_z,time); }
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,MMS_uSource1D,*_y,time); }
  else { mapToVec(uSource,MMS_uSource,*_y,*_z,time); }
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  //~ mapToVec(_uP,MMS_uA,*_y,*_z,_currTime);

  // update stresses
  ierr = setStresses(time);CHKERRQ(ierr);
  //~ mapToVec(_sxyP,MMS_pl_sigmaxy,*_y,*_z,_currTime);
  //~ mapToVec(_sxzP,MMS_pl_sigmaxz,*_y,*_z,_currTime);
  //~ mapToVec(_sigmadev,MMS_sigmadev,*_y,*_z,_currTime);

  // update rates
  ierr = setViscStrainRates(time,varBegin,dvarBegin);CHKERRQ(ierr); // set viscous strain rates
  Vec source;
  VecDuplicate(_uP,&source);
  if (_Nz == 1) { mapToVec(source,MMS_pl_gxy_t_source1D,*_y,_currTime); }
  else { mapToVec(source,MMS_pl_gxy_t_source,*_y,*_z,_currTime); }
  VecAXPY(*(dvarBegin+3),1.0,source);
  if (_Nz == 1) { mapToVec(source,MMS_pl_gxz_t_source1D,*_y,_currTime); }
  else { mapToVec(source,MMS_pl_gxz_t_source,*_y,*_z,_currTime); }
  VecAXPY(*(dvarBegin+4),1.0,source);
  VecDestroy(&source);


  // update rates
  VecSet(*dvarBegin,0.0); // d/dt psi
  VecSet(*(dvarBegin+1),0.0); // d/dt theta
  VecSet(*(dvarBegin+2),0.0); // slip vel

  //~ mapToVec(*(dvarBegin+2),MMS_gxy_t,*_y,*_z,_currTime);
  //~ mapToVec(*(dvarBegin+3),MMS_gxz_t,*_y,*_z,_currTime);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::setViscStrainSourceTerms(Vec& out,const_it_vec varBegin)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setViscStrainSourceTerms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(_gxyP,&source);
  VecSet(source,0.0);

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz)
  // + Hz^-1 E0z mu gxz - Hz^-1 ENz mu gxz
  Vec sourcexy_y;
  VecDuplicate(_uP,&sourcexy_y);
  VecSet(sourcexy_y,0.0);
  ierr = _sbpP->Dyxmu(_gxyP,sourcexy_y);CHKERRQ(ierr);


  // apply effects of coordinate transform
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1,temp2;
    VecDuplicate(_gxyP,&temp1);
    VecDuplicate(_gxyP,&temp2);
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(yq,sourcexy_y,temp1);
    MatMult(zr,temp1,temp2);
    VecCopy(temp2,sourcexy_y);
    VecDestroy(&temp1);
    VecDestroy(&temp2);
  }

  // if bcL is shear stress, then also add Hy^-1 E0y mu gxy
  if (_bcLTauQS==1) {
    Vec temp1,bcL;
    VecDuplicate(_gxyP,&temp1); VecSet(temp1,0.0);
    VecDuplicate(_gxyP,&bcL);
    _sbpP->HyinvxE0y(_gxyP,temp1);
    ierr = VecPointwiseMult(bcL,_muVecP,temp1); CHKERRQ(ierr);
    VecDestroy(&temp1);

    // apply effects of coordinate transform
    if (_sbpType.compare("mfc_coordTrans")==0) {
      Mat qy,rz,yq,zr;
      Vec temp1;
      VecDuplicate(_gxyP,&temp1);
      ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
      MatMult(yq,bcL,temp1);
      MatMult(zr,temp1,bcL);
      VecDestroy(&temp1);
    }
    ierr = VecAXPY(sourcexy_y,1.0,bcL);CHKERRQ(ierr);
    VecDestroy(&bcL);
  }

  ierr = VecCopy(sourcexy_y,source);CHKERRQ(ierr); // sourcexy_y -> source
  VecDestroy(&sourcexy_y);



  if (_Nz > 1)
  {
    Vec sourcexz_z;
    VecDuplicate(_gxzP,&sourcexz_z);
    ierr = _sbpP->Dzxmu(_gxzP,sourcexz_z);CHKERRQ(ierr);
    if (_sbpType.compare("mfc_coordTrans")==0) {
      Mat qy,rz,yq,zr;
      Vec temp1,temp2;
      VecDuplicate(_gxzP,&temp1);
      VecDuplicate(_gxzP,&temp2);
      ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
      MatMult(zr,sourcexz_z,temp1);
      MatMult(yq,temp1,temp2);
      VecCopy(temp2,sourcexz_z);
      VecDestroy(&temp1);
      VecDestroy(&temp2);
    }
    ierr = VecAXPY(source,1.0,sourcexz_z);CHKERRQ(ierr); // source += Hxsourcexz_z
    VecDestroy(&sourcexz_z);

    // enforce traction boundary condition
    Vec temp1,bcT,bcB;
    VecDuplicate(_gxzP,&temp1); VecSet(temp1,0.0);
    VecDuplicate(_gxzP,&bcT);
    VecDuplicate(_gxzP,&bcB);

    _sbpP->HzinvxE0z(_gxzP,temp1);
    ierr = VecPointwiseMult(bcT,_muVecP,temp1); CHKERRQ(ierr);

    _sbpP->HzinvxENz(_gxzP,temp1);
    ierr = VecPointwiseMult(bcB,_muVecP,temp1); CHKERRQ(ierr);

    if (_sbpType.compare("mfc_coordTrans")==0) {
      Mat qy,rz,yq,zr;
      Vec temp2;
      VecDuplicate(_gxzP,&temp2);
      ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);

      MatMult(yq,bcB,temp2);
      MatMult(zr,temp2,bcB);

      MatMult(yq,bcT,temp2);
      MatMult(zr,temp2,bcT);
      VecDestroy(&temp2);
    }

    ierr = VecAXPY(source,1.0,bcT);CHKERRQ(ierr);
    ierr = VecAXPY(source,-1.0,bcB);CHKERRQ(ierr);

    VecDestroy(&temp1);
    VecDestroy(&bcT);
    VecDestroy(&bcB);
  }

  ierr = _sbpP->H(source,out); CHKERRQ(ierr);
  VecDestroy(&source);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
}







PetscErrorCode PowerLaw::setViscStrainRates(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
    PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::setViscStrainRates";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecSet(*(dvarBegin+3),0.0);
  VecSet(*(dvarBegin+4),0.0);

  // compute effective viscosity
  PetscScalar sigmadev,A,B,n,T,effVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(*(dvarBegin+3),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_sigmadev,1,&Ii,&sigmadev);
    VecGetValues(_A,1,&Ii,&A);
    VecGetValues(_B,1,&Ii,&B);
    VecGetValues(_n,1,&Ii,&n);
    VecGetValues(_T,1,&Ii,&T);
    effVisc = 1.0/( A*pow(sigmadev,n-1.0)*exp(-B/T) ) * 1e-3;
    VecSetValues(_effVisc,1,&Ii,&effVisc,INSERT_VALUES);
  }
  VecAssemblyBegin(_effVisc);
  VecAssemblyEnd(_effVisc);

// add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_gTxyP,&SAT);
  ierr = setViscousStrainRateSAT(_uP,_bcLP,_bcRP,SAT);CHKERRQ(ierr);

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  VecPointwiseMult(*(dvarBegin+3),_muVecP,SAT);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gxyP,&temp1);
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,*(dvarBegin+3),temp1);
    VecCopy(temp1,*(dvarBegin+3));
    VecDestroy(&temp1);
  }
  VecAXPY(*(dvarBegin+3),1.0,_sxyP);
  VecPointwiseDivide(*(dvarBegin+3),*(dvarBegin+3),_effVisc);

  if (_Nz > 1) {
    VecCopy(_sxzP,*(dvarBegin+4));
    VecPointwiseDivide(*(dvarBegin+4),*(dvarBegin+4),_effVisc);
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
    ierr = _sbpP->HyinvxE0y(u,temp1);CHKERRQ(ierr);
    ierr = _sbpP->Hyinvxe0y(gL,GL);CHKERRQ(ierr);
    VecAXPY(out,1.0,temp1);
    VecAXPY(out,-1.0,GL);
  }

  // right displacement boundary
  ierr = _sbpP->HyinvxENy(u,temp1);CHKERRQ(ierr);
  ierr = _sbpP->HyinvxeNy(gR,GR);CHKERRQ(ierr);
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

  _sbpP->Dy(_uP,_gTxyP);
  VecCopy(_gTxyP,_sxyP);
  VecAXPY(_sxyP,-1.0,_gxyP);
  VecPointwiseMult(_sxyP,_sxyP,_muVecP);

  // deviatoric stress: part 1/3
  VecPointwiseMult(_sigmadev,_sxyP,_sxyP);

  if (_Nz > 1) {
    _sbpP->Dz(_uP,_gTxzP);
    VecCopy(_gTxzP,_sxzP);
    VecAXPY(_sxzP,-1.0,_gxzP);
    VecPointwiseMult(_sxzP,_sxzP,_muVecP);

  // deviatoric stress: part 2/3
  Vec temp;
  VecDuplicate(_sxzP,&temp);
  VecPointwiseMult(temp,_sxzP,_sxzP);
  VecAXPY(_sigmadev,1.0,temp);
  VecDestroy(&temp);
  }

  // deviatoric stress: part 3/3
  VecSqrtAbs(_sigmadev);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr = 0;
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
  ierr = VecGetOwnershipRange(_bcLP,&Istart,&Iend);CHKERRQ(ierr);
  if (_Nz == 1) {
    Ii = Istart;
    y = 0;
    if (!_bcLType.compare("Dirichlet")) { v = MMS_uA1D(y,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("Neumann")) { v = MMS_mu1D(y) * (MMS_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("Dirichlet")) { v = MMS_uA1D(y,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("Neumann")) { v = MMS_mu1D(y) * (MMS_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  else {
    for(Ii=Istart;Ii<Iend;Ii++) {
      z = _dz * Ii;

      y = 0;
      if (!_bcLType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y=0,z)
      else if (!_bcLType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time)- MMS_gxy(y,z,time));} // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      y = _Ly;
      if (!_bcRType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y=Ly,z)
      else if (!_bcRType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time)- MMS_gxy(y,z,time)); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRP);CHKERRQ(ierr);


  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(_bcTP,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = _dy * Ii;

    z = 0;
    if (!_bcTType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y,z=0)
    else if (!_bcTType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time) - MMS_gxz(y,z,time)); }
    //~ else if (!_bcTType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcTP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    z = _Lz;
    if (!_bcBType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y,z=Lz)
    else if (!_bcBType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time) - MMS_gxz(y,z,time));}
    //~ else if (!_bcBType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcBP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcBP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcBP);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}





PetscErrorCode PowerLaw::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;

  _stepCount = stepCount;
  _currTime = time;
  ierr = setSurfDisp();
  if ( stepCount % _stride1D == 0) {
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep1D();CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep2D();CHKERRQ(ierr);
  }

  if (stepCount % 10 == 0) {
    PetscScalar maxTimeStep;
    computeMaxTimeStep(maxTimeStep);

  if (_timeIntegrator.compare("IMEX")==0) {
     _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep);CHKERRQ(ierr);
  }
  else {
    _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep);CHKERRQ(ierr);
  }

  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}

// Outputs data at each time step.
PetscErrorCode PowerLaw::debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;
  PetscScalar    epsVxy,depsVxy;

  ierr= VecGetOwnershipRange(*varBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*varBegin,1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+1),1,&Istart,&uVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(*dvarBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*dvarBegin,1,&Istart,&dQVal);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+1),1,&Istart,&velVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+2),1,&Istart,&epsVxy);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+2),1,&Istart,&depsVxy);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcRP,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRP,1,&Istart,&bcRval);CHKERRQ(ierr);

  ierr = VecGetValues(_fault._tauQSP,1,&Istart,&tauQS);CHKERRQ(ierr);

  if (stepCount == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s  | %-15s %-15s %-15s | %-15s %-15s %-16s | %-15s\n",
                       "Step","Stage", "bcR","D","eVxy", "tauQS","V","deVxy","time");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ",stepCount,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",bcRval,uVal,epsVxy);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",tauQS,velVal,depsVxy);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e\n",time);CHKERRQ(ierr);


  //~VecView(_fault._tauQSP,PETSC_VIEWER_STDOUT_WORLD);
#endif
  return ierr;
}

PetscErrorCode PowerLaw::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // measure error between analytical and numerical solution
  Vec uA,gxyA,gxzA;
  VecDuplicate(_uP,&uA);
  VecDuplicate(_uP,&gxyA);
  VecDuplicate(_uP,&gxzA);
  //~ mapToVec(uA,MMS_uA,*_y,*_z,_currTime);
  //~ mapToVec(gxyA,MMS_gxy,_Nz,_dy,_dz,_currTime);
  //~ mapToVec(gxzA,MMS_gxz,_Nz,_dy,_dz,_currTime);

  if (_Nz == 1) { mapToVec(uA,MMS_uA1D,*_y,_currTime); }
  else { mapToVec(uA,MMS_uA,*_y,*_z,_currTime); }
    if (_Nz == 1) { mapToVec(gxyA,MMS_gxy1D,*_y,_currTime); }
  else { mapToVec(gxyA,MMS_gxy,*_y,*_z,_currTime); }
  if (_Nz == 1) { mapToVec(gxzA,MMS_gxy1D,*_y,_currTime); }
  else { mapToVec(gxzA,MMS_gxz,*_y,*_z,_currTime); }

  double err2u = computeNormDiff_2(_uP,uA);
  double err2epsxy = computeNormDiff_2(*(_var.begin()+2),gxyA);
  double err2epsxz = computeNormDiff_2(*(_var.begin()+3),gxzA);

  PetscPrintf(PETSC_COMM_WORLD,"%i %3i %.4e %.4e % .15e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u),err2epsxy,log2(err2epsxy),err2epsxz,log2(err2epsxz));

  //~// measure error for stresses as well
  //~mapToVec(_gxyP,MMS_gxy,_Nz,_dy,_dz,_currTime);
  //~mapToVec(_gxzP,MMS_gxz,_Nz,_dy,_dz,_currTime);
  //~mapToVec(_uP,MMS_uA,_Nz,_dy,_dz,_currTime);
  //~ierr = setStresses(_currTime,_var.begin(),_var.end());CHKERRQ(ierr); // numerical solution
  //~Vec sigmaxyA, sigmaxzA, sigmadevA;
  //~VecDuplicate(_uP,&sigmaxyA);
  //~VecDuplicate(_uP,&sigmaxzA);
  //~VecDuplicate(_uP,&sigmadevA);
  //~mapToVec(sigmaxyA,MMS_pl_sigmaxy,_Nz,_dy,_dz,_currTime);
  //~mapToVec(sigmaxzA,MMS_pl_sigmaxz,_Nz,_dy,_dz,_currTime);
  //~mapToVec(sigmadevA,MMS_sigmadev,_Nz,_dy,_dz,_currTime);
  //~double err2sigmaxyA = computeNormDiff_2(_sxyP,sigmaxyA);
  //~double err2sigmaxzA = computeNormDiff_2(_sxzP,sigmaxzA);
  //~double err2sigmadevA = computeNormDiff_2(_sigmadev,sigmadevA);
  //~VecDestroy(&sigmaxyA);
  //~VecDestroy(&sigmaxzA);
  //~VecDestroy(&sigmadevA);

  //~PetscPrintf(PETSC_COMM_WORLD,"%3i %3i %.4e %.4e % .15e %.4e % .15e\n",
              //~_order,_Ny,_dy,err2sigmaxyA,log2(err2sigmaxyA),err2sigmaxzA,log2(err2sigmaxzA));


  VecDestroy(&uA);
  VecDestroy(&gxyA);
  VecDestroy(&gxzA);
  return ierr;
}



//======================================================================
// IO functions
//======================================================================

PetscErrorCode PowerLaw::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer    vw;

  std::string str = outputDir + "powerLawA";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_A,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "powerLawB";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_B,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);

  str = outputDir + "n";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);CHKERRQ(ierr);
  ierr = VecView(_n,vw);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vw);CHKERRQ(ierr);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep1D()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,_stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    // write contextual fields
    _he.writeContext();
    //~ierr = _sbpP->writeOps(_outputDir);CHKERRQ(ierr);
    ierr = writeContext(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                 FILE_MODE_WRITE,&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
              //~FILE_MODE_WRITE,&_bcRPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRPlusV);CHKERRQ(ierr);

    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);
    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);

    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    //~ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),FILENAME,_stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PowerLaw::writeStep2D()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PowerLaw::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),FILENAME,_stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    _he.writeStep2D(_stepCount);

    // write contextual fields
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
              FILE_MODE_WRITE,&_uPV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
                                   FILE_MODE_APPEND,&_uPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
              FILE_MODE_WRITE,&_gTxyPV);CHKERRQ(ierr);
    ierr = VecView(_gTxyP,_gTxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gTxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
                                   FILE_MODE_APPEND,&_gTxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
              FILE_MODE_WRITE,&_sxyPV);CHKERRQ(ierr);
    ierr = VecView(_sxyP,_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_sxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
                                   FILE_MODE_APPEND,&_sxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
              FILE_MODE_WRITE,&_gxyPV);CHKERRQ(ierr);
    ierr = VecView(_gxyP,_gxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
                                   FILE_MODE_APPEND,&_gxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"effVisc").c_str(),
              FILE_MODE_WRITE,&_effViscV);CHKERRQ(ierr);
    ierr = VecView(_effVisc,_effViscV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_effViscV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"effVisc").c_str(),
                                   FILE_MODE_APPEND,&_effViscV);CHKERRQ(ierr);

    // write out boundary conditions for testing purposes
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
              FILE_MODE_WRITE,&_bcRPlusV);CHKERRQ(ierr);
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
              FILE_MODE_WRITE,&_gTxzPV);CHKERRQ(ierr);
      ierr = VecView(_gTxzP,_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxzP").c_str(),
                                     FILE_MODE_APPEND,&_gTxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
               FILE_MODE_WRITE,&_sxzPV);CHKERRQ(ierr);
      ierr = VecView(_sxzP,_sxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_sxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
                                     FILE_MODE_APPEND,&_sxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
               FILE_MODE_WRITE,&_gxzPV);CHKERRQ(ierr);
      ierr = VecView(_gxzP,_gxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
                                   FILE_MODE_APPEND,&_gxzPV);CHKERRQ(ierr);
    }
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);
    _he.writeStep2D(_stepCount);

    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcLP,_bcLPlusV);CHKERRQ(ierr);

    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = VecView(_gTxyP,_gTxyPV);CHKERRQ(ierr);
    ierr = VecView(_sxyP,_sxyPV);CHKERRQ(ierr);
    ierr = VecView(_gxyP,_gxyPV);CHKERRQ(ierr);
    ierr = VecView(_effVisc,_effViscV);CHKERRQ(ierr);
    //~if (_isMMS) {ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);}
    if (_Nz>1)
    {
      ierr = VecView(_gTxzP,_gTxzPV);CHKERRQ(ierr);
      ierr = VecView(_sxzP,_sxzPV);CHKERRQ(ierr);
      ierr = VecView(_gxzP,_gxzPV);CHKERRQ(ierr);
    }
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),FILENAME,_stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PowerLaw::view()
{
  PetscErrorCode ierr = 0;
  if (_timeIntegrator.compare("IMEX")==0) { ierr = _quadImex->view(); }
  if (_timeIntegrator.compare("RK32")==0) { ierr = _quadEx->view(); }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
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
