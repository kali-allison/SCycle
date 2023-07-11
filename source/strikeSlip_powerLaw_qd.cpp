#include "strikeSlip_powerLaw_qd.hpp"

#define FILENAME "strikeSlip_powerLaw_qd.cpp"

using namespace std;


StrikeSlip_PowerLaw_qd::StrikeSlip_PowerLaw_qd(Domain&D)
  : _D(&D),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
    _guessSteadyStateICs(0),_isMMS(D._isMMS),
    _thermalCoupling("no"),_grainSizeEvCoupling("no"),
    _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
    _stateLaw("agingLaw"),_forcingType("no"),
    _evolveTemperature(0),_evolveGrainSize(0),_computeSSTemperature(0),_computeSSGrainSize(0),
    _vL(1e-9),_faultTypeScale(2.0),
    _timeIntegrator("RK43"),_timeControlType("PID"),
    _stride1D(1),_stride2D(1),_strideChkpt(1e4),
    _maxStepCount(1e8),_initTime(0),_currTime(0),_maxTime(1e15),_minDeltaT(-1),_maxDeltaT(1e10),
    _time1DVec(NULL), _dtime1DVec(NULL),_time2DVec(NULL), _dtime2DVec(NULL),
    _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
    _chkptTimeStep1D(0), _chkptTimeStep2D(0),
    _JjSSVec(NULL),
    _fss_T(0.15),_fss_EffVisc(0.2),_fss_grainSize(0.2),_gss_t(1e-10),
    _SS_index(0),_maxSSIts_effVisc(50),_maxSSIts_tot(100),
    _atolSS_effVisc(1e-4),_maxSSIts_time(5e10),
    _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),
    _startTime(MPI_Wtime()),_miscTime(0),
    _viewer_context(NULL),_viewer1D(NULL),_viewer2D(NULL),_viewerSS(NULL),_viewer_chkpt(NULL),
    _forcingTerm(NULL),_forcingTermPlain(NULL),_forcingVal(0),
    _bcT_L(0),
    _bcRType("remoteLoading"),_bcTType("freeSurface"),
    _bcLType("symmFault"),_bcBType("freeSurface"),
    _quadEx(NULL),_quadImex(NULL),
    _fault(NULL),_material(NULL),_he(NULL),_p(NULL),_grainDist(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::StrikeSlip_PowerLaw_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  parseBCs();
  allocateFields();

  if (_D->_restartFromChkpt) {
    loadCheckpoint();
    _guessSteadyStateICs = 0;
  }
  if (_D->_restartFromChkptSS) {
    loadCheckpointSS();
    _guessSteadyStateICs = 0;
  }

  // initiate momentum balance equation
  _material = new PowerLaw(D,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);

  _he = new HeatEquation(D); // heat equation
  if (_thermalCoupling == "coupled") { VecCopy(_he->_T,_material->_T); }

  _body2fault = &(D._scatters["body2L"]);
  _fault = new Fault_qd(D,D._scatters["body2L"],_faultTypeScale); // fault
  if (_thermalCoupling.compare("no")!=0 && _stateLaw.compare("flashHeating")==0) {
    Vec T; VecDuplicate(_D->_y,&T);
    _he->getTemp(T);
    _fault->setThermalFields(T,_he->_k,_he->_c);
  }

  //~ // pressure diffusion equation
  if (_hydraulicCoupling != "no") { _p = new PressureEq(D); }
  if (_hydraulicCoupling == "coupled") { _fault->setSNEff(_p->_p); }

  //~ // grain size distribution
  if (_evolveGrainSize == 1 || _computeSSGrainSize == 1) { _grainDist = new GrainSizeEvolution(D); }
  if (_grainSizeEvCoupling == "coupled") { VecCopy(_grainDist->_d, _material->_grainSize); }

  // body forcing term for ice stream
  _forcingTerm = NULL; _forcingTermPlain = NULL;
  if (_forcingType == "iceStream") { constructIceStreamForcingTerm(); }

  // compute min allowed time step for adaptive time stepping method
  computeMinTimeStep();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_PowerLaw_qd::~StrikeSlip_PowerLaw_qd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::~StrikeSlip_PowerLaw_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }

  PetscViewerDestroy(&_viewer1D);
  PetscViewerDestroy(&_viewer2D);
  PetscViewerDestroy(&_viewer_context);
  PetscViewerDestroy(&_viewerSS);


  delete _quadImex;    _quadImex = NULL;
  delete _quadEx;      _quadEx = NULL;
  delete _material;    _material = NULL;
  delete _fault;       _fault = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;
  delete _grainDist;   _grainDist = NULL;

  if (_varSS.find("v") != _varSS.end()) { VecDestroy(&_varSS["v"]); }
  if (_varSS.find("grainSize") != _varSS.end()) { VecDestroy(&_varSS["grainSize"]); }
  VecDestroy(&_forcingTerm);
  VecDestroy(&_forcingTermPlain);
  VecDestroy(&_time1DVec);
  VecDestroy(&_dtime1DVec);
  VecDestroy(&_time2DVec);
  VecDestroy(&_dtime2DVec);
  VecDestroy(&_JjSSVec);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_PowerLaw_qd::loadSettings(const char *file)
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
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);
    rhs = "";
    if (line.length() > (pos + _delim.length())) {
      rhs = line.substr(pos+_delim.length(),line.npos);
    }
    rhsFull = rhs; // everything after _delim

    // interpret everything after the appearance of a space on the line as a comment
    pos = rhs.find(" ");
    rhs = rhs.substr(0,pos);

    if (var.compare("thermalCoupling")==0) { _thermalCoupling = rhs.c_str(); }
    else if (var.compare("grainSizeEvCoupling")==0) { _grainSizeEvCoupling = rhs.c_str(); }
    else if (var.compare("hydraulicCoupling")==0) { _hydraulicCoupling = rhs.c_str(); }
    else if (var.compare("stateLaw")==0) { _stateLaw = rhs.c_str(); }
    else if (var.compare("guessSteadyStateICs")==0) { _guessSteadyStateICs = atoi( rhs.c_str() ); }
    else if (var.compare("forcingType")==0) { _forcingType = rhs.c_str(); }

    // for steady state iteration
    else if (var.compare("fss_T")==0) { _fss_T = atof( rhs.c_str() ); }
    else if (var.compare("fss_EffVisc")==0) { _fss_EffVisc = atof( rhs.c_str() ); }
    else if (var.compare("gss_t")==0) { _gss_t = atof( rhs.c_str() ); }
    else if (var.compare("maxSSIts_effVisc")==0) { _maxSSIts_effVisc = atoi( rhs.c_str() ); }
    else if (var.compare("maxSSIts_tot")==0) { _maxSSIts_tot = atoi( rhs.c_str() ); }
    else if (var.compare("atolSS_effVisc")==0) { _atolSS_effVisc = atof( rhs.c_str() ); }
    else if (var.compare("maxSSIts_time")==0) { _maxSSIts_time = atof( rhs.c_str() ); }
    else if (var.compare("evolveTemperature")==0) { _evolveTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("evolveGrainSize")==0) { _evolveGrainSize = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSHeatEq")==0) { _computeSSTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSGrainSize")==0) { _computeSSGrainSize = (int) atoi( rhs.c_str() ); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) { _timeIntegrator = rhs; }
    else if (var.compare("timeControlType")==0) { _timeControlType = rhs; }
    else if (var.compare("stride1D")==0){ _stride1D = (int)atof( rhs.c_str() ); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof( rhs.c_str() ); }
    else if (var.compare("strideChkpt")==0){ _strideChkpt = (int)atof(rhs.c_str()); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( rhs.c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( rhs.c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( rhs.c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("timeStepTol")==0) { _timeStepTol = atof( rhs.c_str() ); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }
    else if (var.compare("scale")==0) { loadVectorFromInputFile(rhsFull,_scale); }
    else if (var.compare("normType")==0) { _normType = rhs.c_str(); }

    else if (var.compare("vL")==0) { _vL = atof( rhs.c_str() ); }

    else if (var.compare("bodyForce")==0) { _forcingVal = atof( rhs.c_str() ); }

    else if (var.compare("bcT_L")==0) { _bcT_L = atof( rhs.c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_qd")==0) { _bcRType = rhs.c_str(); }
    else if (var.compare("momBal_bcT_qd")==0) { _bcTType = rhs.c_str(); }
    else if (var.compare("momBal_bcL_qd")==0) { _bcLType = rhs.c_str(); }
    else if (var.compare("momBal_bcB_qd")==0) { _bcBType = rhs.c_str(); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_PowerLaw_qd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);

  assert(_thermalCoupling == "coupled" || _thermalCoupling == "uncoupled" || _thermalCoupling == "no" );
  assert(_grainSizeEvCoupling == "coupled" || _grainSizeEvCoupling == "uncoupled" || _grainSizeEvCoupling == "no" );
  assert(_hydraulicCoupling == "coupled" || _hydraulicCoupling == "uncoupled" || _hydraulicCoupling == "no" );

  assert(_forcingType == "iceStream" || _forcingType == "no" );

  assert(_timeIntegrator == "FEuler" ||
      _timeIntegrator == "RK32" ||
      _timeIntegrator == "RK43" ||
      _timeIntegrator == "RK32_WBE" ||
      _timeIntegrator == "RK43_WBE" );

  assert(_timeControlType == "P" ||
         _timeControlType == "PI" ||
         _timeControlType == "PID" );

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {_initDeltaT = _minDeltaT; }
  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_timeStepTol >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);

  // check boundary condition types for momentum balance equation
  assert(_bcRType == "freeSurface" || _bcRType == "remoteLoading");
  assert(_bcTType == "freeSurface" || _bcTType == "remoteLoading" || _bcTType == "atan_u");
  assert(_bcLType == "symmFault"   || _bcLType == "rigidFault" );
  assert(_bcBType == "freeSurface" || _bcBType == "remoteLoading");

  if (_bcTType == "atan_u") { assert(_bcT_L > 0.); }

  if (_stateLaw=="flashHeating") {
    assert(_thermalCoupling!="no");
    assert(_evolveTemperature == 1);
  }

  if (_thermalCoupling=="coupled" || _thermalCoupling=="uncoupled") {
    assert(_evolveTemperature == 1 || _computeSSTemperature == 1);
  }

  if (_grainSizeEvCoupling=="coupled" || _grainSizeEvCoupling=="uncoupled") {
    assert(_evolveGrainSize == 1 || _computeSSGrainSize == 1);
  }

  if (_thermalCoupling != "no" && _evolveTemperature ==1 && (_timeIntegrator != "RK32_WBE" && _timeIntegrator != "RK43_WBE")) {
    assert(0);
  }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// allocate space for member fields
PetscErrorCode StrikeSlip_PowerLaw_qd::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_PowerLaw_qd::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initiate Vecs to hold current time and time step
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_time1DVec); CHKERRQ(ierr);
  ierr = VecSetBlockSize(_time1DVec, 1); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _time1DVec, "time1D"); CHKERRQ(ierr);
  ierr = VecSet(_time1DVec,_initTime); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_dtime1DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _dtime1DVec, "dtime1D"); CHKERRQ(ierr);
  VecSet(_dtime1DVec,_deltaT); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_time2DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _time2DVec, "time2D"); CHKERRQ(ierr);
  VecSet(_time2DVec,_initTime); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_dtime2DVec); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _dtime2DVec, "dtime2D"); CHKERRQ(ierr);
  ierr = VecSet(_dtime2DVec,_deltaT); CHKERRQ(ierr);

  // initiate Vecs to hold index Jj
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
  VecSetBlockSize(_JjSSVec, 1);
  PetscObjectSetName((PetscObject) _JjSSVec, "SS_index");
  VecSet(_JjSSVec,_SS_index);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

return ierr;
}

// compute recommended smallest time step based on grid spacing and shear wave speed
// Note: defaults to user specified value
// recommended minDeltaT <= min(dy/cs, dz/cs)
PetscErrorCode StrikeSlip_PowerLaw_qd::computeMinTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::computeTimeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // compute grid spacing in y and z
  Vec dy, dz;
  ierr = VecDuplicate(_D->_y,&dy);
  ierr = VecDuplicate(_D->_y,&dz);

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatGetDiagonal(yq, dy); CHKERRQ(ierr);
    ierr = VecScale(dy,1.0/(_D->_Ny-1)); CHKERRQ(ierr);
    ierr = MatGetDiagonal(zr, dz); CHKERRQ(ierr);
    ierr = VecScale(dz,1.0/(_D->_Nz-1)); CHKERRQ(ierr);
  }
  else {
    ierr = VecSet(dy,_D->_Ly/(_D->_Ny-1.0)); CHKERRQ(ierr);
    ierr = VecSet(dz,_D->_Lz/(_D->_Nz-1.0)); CHKERRQ(ierr);
  }

  // print smallest grid spacing
  //~ PetscScalar min_dy, min_dz;
  //~ ierr = VecMin(dy,NULL,&min_dy); CHKERRQ(ierr);
  //~ ierr = VecMin(dz,NULL,&min_dz); CHKERRQ(ierr);
  //~ PetscScalar min_cs,max_cs;
  //~ ierr = VecMin(_material->_cs,NULL,&min_cs); CHKERRQ(ierr);
  //~ ierr = VecMax(_material->_cs,NULL,&max_cs); CHKERRQ(ierr);
  //~ PetscPrintf(PETSC_COMM_WORLD," input: min_dy = %g, min_dz = %g, min_cs = %g, max_cs = %g\n",min_dy,min_dz,min_cs,max_cs);

  // compute time for shear wave to travel one dy or dz
  Vec ts_dy,ts_dz;
  ierr = VecDuplicate(_D->_y,&ts_dy); CHKERRQ(ierr);
  ierr = VecDuplicate(_D->_z,&ts_dz); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(ts_dy,dy,_material->_cs); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(ts_dz,dz,_material->_cs); CHKERRQ(ierr);

  PetscScalar min_ts_dy, min_ts_dz;
  ierr = VecMin(ts_dy,NULL,&min_ts_dy); CHKERRQ(ierr);
  ierr = VecMin(ts_dz,NULL,&min_ts_dz); CHKERRQ(ierr);

  // clean up memory usage
  VecDestroy(&dy);
  VecDestroy(&dz);
  VecDestroy(&ts_dy);
  VecDestroy(&ts_dz);

  // smallest reasonable time step
  PetscScalar min_deltaT = min(min_ts_dy,min_ts_dz);

  // provide if not user specified
  if (_minDeltaT == -1) {
    _minDeltaT = min_deltaT;
  }
  else if (_minDeltaT > min_deltaT) {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: minimum requested time step (minDeltaT) is larger than recommended.");
    PetscPrintf(PETSC_COMM_WORLD," Requested: %e s, Recommended (min(dy/cs,dz/cs)): %e s\n",_minDeltaT,min_deltaT);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// parse boundary conditions
PetscErrorCode StrikeSlip_PowerLaw_qd::parseBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::parseBCs()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_bcRType.compare("symmFault")==0 || _bcRType.compare("rigidFault")==0 || _bcRType.compare("remoteLoading")==0) {
    _mat_bcRType = "Dirichlet";
  }
  else if (_bcRType.compare("freeSurface")==0 || _bcRType.compare("outGoingCharacteristics")==0) {
    _mat_bcRType = "Neumann";
  }

  if (_bcTType.compare("atan_u")==0 || _bcTType.compare("symmFault")==0 || _bcTType.compare("rigidFault")==0 || _bcTType.compare("remoteLoading")==0) {
    _mat_bcTType = "Dirichlet";
  }
  else if (_bcTType.compare("freeSurface")==0 || _bcTType.compare("outGoingCharacteristics")==0) {
    _mat_bcTType = "Neumann";
  }

  if (_bcLType.compare("symmFault")==0 || _bcLType.compare("rigidFault")==0 || _bcLType.compare("remoteLoading")==0) {
    _mat_bcLType = "Dirichlet";
  }
  else if (_bcLType.compare("freeSurface")==0 || _bcLType.compare("outGoingCharacteristics")==0 ) {
    _mat_bcLType = "Neumann";
  }

  if (_bcBType.compare("symmFault")==0 || _bcBType.compare("rigidFault")==0 || _bcBType.compare("remoteLoading")==0) {
    _mat_bcBType = "Dirichlet";
  }
  else if (_bcBType.compare("freeSurface")==0 || _bcBType.compare("outGoingCharacteristics")==0) {
    _mat_bcBType = "Neumann";
  }

  // determine if material is symmetric about the fault, or if one side is rigid
  _faultTypeScale = 2.0;
  if (_bcLType.compare("rigidFault")==0 ) { _faultTypeScale = 1.0; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode StrikeSlip_PowerLaw_qd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_varEx.find("slip") == _varEx.end() ) { ierr = VecDuplicate(_material->_bcL, &_varEx["slip"]); CHKERRQ(ierr); }
  VecCopy(_material->_bcL, _varEx["slip"]);
  if (_bcLType == "symmFault" || _bcLType == "rigidFault") {
    VecScale(_varEx["slip"],_faultTypeScale);
  }
  if (!_D->_restartFromChkpt) {
    ierr = loadVecFromInputFile(_varEx["slip"],_inputDir,"slip"); CHKERRQ(ierr);
  }
  VecCopy(_varEx["slip"],_fault->_slip);

  if (_guessSteadyStateICs) {
    initiateIntegrandSS();
    solveSS(0);
    writeSS(0);
    VecDestroy(&_JjSSVec);
  }
  { // set up KSP context for time integration
    Mat A;
    _material->_sbp->getA(A);
    _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans);
  }

  _material->initiateIntegrand(_initTime,_varEx);
  _fault->initiateIntegrand(_initTime,_varEx);

  if (_evolveTemperature == 1) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
     _fault->updateTemperature(_he->_T);
  }

  if (_hydraulicCoupling!="no") {
     _p->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  if (_evolveGrainSize == 1) {
     _grainDist->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// monitoring function for ode solvers
PetscErrorCode StrikeSlip_PowerLaw_qd::timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::timeMonitor";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _deltaT = deltaT;
  _currTime = time;
  VecSet(_time1DVec,time);
  VecSet(_dtime1DVec,_deltaT);
  VecSet(_time2DVec,time);
  VecSet(_dtime2DVec,_deltaT);

  if ( (_stride1D>0 &&_currTime == _maxTime) || (_stride1D>0 && stepCount % _stride1D == 0)) {
    ierr = writeStep1D(stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_viewer1D); CHKERRQ(ierr);
    ierr = _fault->writeStep(_viewer1D); CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr =  _he->writeStep1D(_viewer1D); CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { _p->writeStep(_viewer1D); }
  }

  if ( (_stride2D>0 &&_currTime == _maxTime) || (_stride2D>0 && stepCount % _stride2D == 0)) {
    ierr = writeStep2D(stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_viewer2D);CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr =  _he->writeStep2D(_viewer2D);CHKERRQ(ierr); }
    if (_evolveGrainSize == 1) { ierr =  _grainDist->writeStep(_viewer2D);CHKERRQ(ierr); }
  }

  if ( _D->_saveChkpts == 1 && ((_strideChkpt > 0 && stepCount % _strideChkpt == 0) || (_currTime == _maxTime)) ) {
    ierr = writeCheckpoint();                                           CHKERRQ(ierr);
    ierr = _D->writeCheckpoint(_viewer_chkpt);                          CHKERRQ(ierr);
    ierr = _material->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault->writeCheckpoint(_viewer_chkpt);                      CHKERRQ(ierr);
    ierr = _he->writeCheckpoint(_viewer_chkpt);                         CHKERRQ(ierr);
    if (_quadEx != NULL) { ierr = _quadEx->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadImex != NULL) { ierr = _quadImex->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_grainDist != NULL) { ierr =  _grainDist->writeCheckpoint(_viewer_chkpt);CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeCheckpoint(_viewer_chkpt);  CHKERRQ(ierr); }
  }

  // ensure time step does not exceed limits: Maxwell time, and characteristic time step of grain size evolution
  PetscScalar maxTimeStep_tot, maxDeltaT_momBal = 0.0;
  ierr =  _material->computeMaxTimeStep(maxDeltaT_momBal);              CHKERRQ(ierr);
  maxTimeStep_tot = min(_maxDeltaT,0.9*maxDeltaT_momBal);

  if (_evolveGrainSize == 1 && _grainDist->_grainSizeEvType == "transient") {
    PetscScalar maxDeltaT_grainSizeEv = 0;
    ierr =  _grainDist->computeMaxTimeStep(maxDeltaT_grainSizeEv,_material->_sdev,_material->_dgVdev_disl,_material->_T); CHKERRQ(ierr);
    maxTimeStep_tot = min(_maxDeltaT,0.9*maxDeltaT_grainSizeEv);
  }

  // communicate maximum allowed time step to time integrator
  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    ierr = _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
  }
  else { ierr = _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr); }

  // stopping criteria for time integration
  if (_D->_systemEvolutionType == "steadyStateIts") {
    if (time >= _maxSSIts_time) { stopIntegration = 1; }
  }

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e, min Tmax = %.5e\n",stepCount,_currTime,_deltaT,maxDeltaT_momBal);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::writeStep1D(PetscInt stepCount, PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer1D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_1D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer1D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer1D, PETSC_TRUE);CHKERRQ(ierr);

    // write time
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer1D, _D->_prevChkptTimeStep1D +1); CHKERRQ(ierr);
    }

    ierr = VecView(_time1DVec, _viewer1D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                             CHKERRQ(ierr);
    if (_JjSSVec != NULL) { ierr = VecView(_JjSSVec, _viewer1D); CHKERRQ(ierr); }
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                       CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);               CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer1D);              CHKERRQ(ierr);
    ierr = VecView(_time1DVec, _viewer1D);                           CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                          CHKERRQ(ierr);
    if (_JjSSVec != NULL) { ierr = VecView(_JjSSVec, _viewer1D); CHKERRQ(ierr); }
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                       CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::writeStep2D(PetscInt stepCount, PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeStep2D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  if (_viewer2D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_2D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer2D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer2D, PETSC_TRUE);CHKERRQ(ierr);

    // write time
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer2D, _D->_prevChkptTimeStep2D +1); CHKERRQ(ierr);
    }

    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    if (_JjSSVec != NULL) { ierr = VecView(_JjSSVec, _viewer2D); CHKERRQ(ierr); }
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer2D);                 CHKERRQ(ierr);
    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    if (_JjSSVec != NULL) { ierr = VecView(_JjSSVec, _viewer2D); CHKERRQ(ierr); }
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode StrikeSlip_PowerLaw_qd::writeCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer_chkpt == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "checkpoint.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewer_chkpt);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer_chkpt, PETSC_TRUE);CHKERRQ(ierr);
  }

  if (_viewer1D != NULL) {
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5GetTimestep(_viewer1D,&_chkptTimeStep1D);     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);

  }
  if (_viewer2D != NULL) {
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5GetTimestep(_viewer2D,&_chkptTimeStep2D);     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
  }

  ierr = PetscViewerFileSetMode(_viewer_chkpt,FILE_MODE_WRITE);         CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(_viewer_chkpt, "/time1D");            CHKERRQ(ierr);
  ierr = VecView(_time1DVec, _viewer_chkpt);                            CHKERRQ(ierr);
  ierr = VecView(_dtime1DVec, _viewer_chkpt);                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep1D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "deltaT", PETSC_SCALAR, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                        CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewer_chkpt, "/time2D");          CHKERRQ(ierr);
  ierr = VecView(_time2DVec, _viewer_chkpt);                          CHKERRQ(ierr);
  ierr = VecView(_dtime2DVec, _viewer_chkpt);                         CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep2D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                      CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

    // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);         CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/time1D");            CHKERRQ(ierr);
  ierr = VecLoad(_time1DVec, viewer);                            CHKERRQ(ierr);
  ierr = VecLoad(_dtime1DVec, viewer);                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "currTime", PETSC_SCALAR, NULL, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                        CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/time2D");            CHKERRQ(ierr);
  ierr = VecLoad(_time2DVec, viewer);                            CHKERRQ(ierr);
  ierr = VecLoad(_dtime2DVec, viewer);                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                        CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _initTime = _currTime;
  _initDeltaT = _deltaT;
  _maxStepCount = _maxStepCount + _stepCount;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_qd::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::loadCheckpointSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer viewer;

  string fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/steadyState");              CHKERRQ(ierr);
  ierr = VecLoad(_JjSSVec, viewer);                                     CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "SS_index", "SS_index", PETSC_INT, NULL, &_SS_index); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _SS_index++;
  _maxSSIts_tot += _SS_index;


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode StrikeSlip_PowerLaw_qd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  if (_timeIntegrator.compare("IMEX")==0&& _quadImex!=NULL) { ierr = _quadImex->view(); }
  if (_timeIntegrator.compare("RK32")==0 && _quadEx!=NULL) { ierr = _quadEx->view(); }

  _material->view(_integrateTime);
  _fault->view(_integrateTime);
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_PowerLaw_qd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // output scalar fields
  std::string str = _outputDir + "mediator.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"grainSizeEvCoupling = %s\n",_grainSizeEvCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"forcingType = %s\n",_forcingType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"vL = %g\n",_vL);CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"minDeltaT = %.15e # (s)\n",_minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxDeltaT = %.15e # (s)\n",_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initDeltaT = %.15e # (s)\n",_initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeStepTol = %g\n",_timeStepTol);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"evolveTemperature = %i\n",_evolveTemperature);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"evolveGrainSize = %i\n",_evolveGrainSize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeSSHeatEq = %i\n",_computeSSTemperature);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeSSGrainSize = %i\n",_computeSSGrainSize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"strideChkpt = %i\n",_strideChkpt);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"_D->_saveChkpts = %i\n",_D->_saveChkpts);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  if (_scale.size() > 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"scale = %s\n",vector2str(_scale).c_str());CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"normType = %s\n",_normType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // boundary conditions for momentum balance equation
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR = %s\n",_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT = %s\n",_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL = %s\n",_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB = %s\n",_bcBType.c_str());CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  // write non-ascii context
  string outFileName = _outputDir + "data_context.h5";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewer_context); CHKERRQ(ierr);
  ierr = PetscViewerSetType(_viewer_context, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewer_context);CHKERRQ(ierr);

  _D->write(_viewer_context);
  _fault->writeContext(_outputDir, _viewer_context);
  _material->writeContext(_outputDir, _viewer_context);
  if (_he != NULL) { _he->writeContext(_outputDir, _viewer_context); }
  if (_hydraulicCoupling!="no") { _p->writeContext(_outputDir, _viewer_context); }
  if (_grainSizeEvCoupling != "no") { _grainDist->writeContext(_outputDir, _viewer_context); }

  if (_forcingType == "iceStream") {
    ierr = PetscViewerHDF5PushGroup(viewer, "/momBal");                 CHKERRQ(ierr);
    ierr = VecView(_forcingTermPlain, viewer);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================

PetscErrorCode StrikeSlip_PowerLaw_qd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();
  _startIntegrateTime = MPI_Wtime();

  // update momentum balance equation boundary conditions
  ierr = _material->changeBCTypes(_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType); CHKERRQ(ierr);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

  _stepCount = 0;

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    _quadEx = new FEuler(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadEx = new RK32(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    _quadEx = new RK43(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    _quadImex = new RK32_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex = new RK43_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);
    ierr = _quadImex->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadImex->setErrInds(_timeIntInds,_scale);

    if (_D->_restartFromChkpt) { ierr = _quadImex->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else {
    _quadEx->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);
    ierr = _quadEx->setErrInds(_timeIntInds,_scale);

    if (_D->_restartFromChkpt) { ierr = _quadEx->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }


  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_PowerLaw_qd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // 1. update fields based on varEx

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType=="symmFault" || _bcLType=="rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType == "remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  if (_bcTType == "atan_u") { updateBCT_atan_u(time); }

  _material->updateFields(time,varEx);
  _fault->updateFields(time,varEx);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->updateFields(time,varEx);
  }
  if ( varEx.find("grainSize") != varEx.end() ) { _grainDist->updateFields(time,varEx); }
  if ( _grainSizeEvCoupling == "coupled" ) { _material->updateGrainSize(_grainDist->_d); }


  // 2. compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->d_dt(time,varEx,dvarEx);
  }

  // compute grain size rate, or value from either piezometric relation or steady-state
  if ( _grainSizeEvCoupling!="no" && varEx.find("grainSize") != varEx.end() && _grainDist->_grainSizeEvType != "steadyState" && _grainDist->_grainSizeEvType != "piezometer") {
    _grainDist->d_dt(dvarEx["grainSize"],varEx.find("grainSize")->second,_material->_sdev,_material->_dgVdev_disl,_material->_T);
  }
  else if ( _grainSizeEvCoupling!="no" && _grainDist->_grainSizeEvType == "piezometer") {
    _grainDist->computeGrainSizeFromPiez(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }
  else if ( _grainSizeEvCoupling!="no" && _grainDist->_grainSizeEvType == "steadyState") {
    _grainDist->computeSteadyStateGrainSize(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }


  // update fields on fault from other classes
  ierr = VecScatterBegin(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  if (_hydraulicCoupling=="coupled") { _fault->setSNEff(_p->_p); }

  // rates for fault
  if (_bcLType=="symmFault" || _bcLType=="rigidFault") {
    ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state


    //~ // impose ceiling on fault velocity: slipVel <= vL
    //~ PetscScalar *V;
    //~ ierr = VecGetArray(_fault->_slipVel,&V);
    //~ PetscInt Kk = 0; // local array index
    //~ PetscInt Istart, Iend;
    //~ ierr = VecGetOwnershipRange(_fault->_slipVel,&Istart,&Iend); // local portion of global Vec index
    //~ for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
      //~ V[Kk] = min(V[Kk],_vL);
      //~ Kk++;
    //~ }
    //~ VecRestoreArray(_fault->_slipVel,&V);

    // compute frictional strength of fault based on updated slip velocity
    ierr = strength_psi_Vec(_fault->_strength, _fault->_psi, _fault->_slipVel, _fault->_a, _fault->_sNEff, _fault->_v0); CHKERRQ(ierr);
    ierr = VecCopy(_fault->_strength,_fault->_tauP); CHKERRQ(ierr);
    ierr = VecCopy(_fault->_slipVel,_fault->_tauQSP); CHKERRQ(ierr); // V -> tauQS
    ierr = VecPointwiseMult(_fault->_tauQSP,_fault->_eta_rad,_fault->_tauQSP); CHKERRQ(ierr); // tauQS = V * eta_rad
    ierr = VecAYPX(_fault->_tauQSP,1.0,_fault->_tauP); CHKERRQ(ierr); // tauQS = tau + V*eta_rad
  }
  else {
    ierr = VecSet(dvarEx["psi"],0.); CHKERRQ(ierr);
    ierr = VecSet(dvarEx["slip"],0.); CHKERRQ(ierr);
  }

  return ierr;
}



// implicit/explicit time stepping
PetscErrorCode StrikeSlip_PowerLaw_qd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::d_dt IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // 1. update state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType=="symmFault" || _bcLType=="rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType=="remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  if (_bcTType=="atan_u") { updateBCT_atan_u(time); }

  _material->updateFields(time,varEx);
  _fault->updateFields(time,varEx);

  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->updateFields(time,varEx,varImo);
  }

  // update temperature in momBal and fault
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling=="coupled") {
    _fault->updateTemperature(varImo.find("Temp")->second);
    _material->updateTemperature(varImo.find("Temp")->second);
  }

  // update grain size in material
  if ( _grainSizeEvCoupling!="no" && varEx.find("grainSize") != varEx.end() ) {
    _grainDist->updateFields(time,varEx);
  }
  if ( _grainSizeEvCoupling == "coupled" ) { _material->updateGrainSize(_grainDist->_d); }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling=="coupled") { _fault->setSNEff(_p->_p); }


  // 2. compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
  }

  // compute grain size rate, or value from either piezometric relation or steady-state
  if ( _grainSizeEvCoupling!="no" && varEx.find("grainSize") != varEx.end() && _grainDist->_grainSizeEvType != "steadyState" && _grainDist->_grainSizeEvType != "piezometer") {
    _grainDist->d_dt(dvarEx["grainSize"],varEx.find("grainSize")->second,_material->_sdev,_material->_dgVdev_disl,_material->_T);
  }
  else if ( _grainSizeEvCoupling!="no" && _grainDist->_grainSizeEvType == "piezometer") {
    _grainDist->computeGrainSizeFromPiez(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }
  else if ( _grainSizeEvCoupling!="no" && _grainDist->_grainSizeEvType == "steadyState") {
    _grainDist->computeSteadyStateGrainSize(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }


  // update fields on fault from other classes
  ierr = VecScatterBegin(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  if (_bcLType.compare("symmFault")==0 || _bcLType.compare("rigidFault")==0) {
    ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state
  }
  else {
    VecSet(dvarEx["psi"],0.);
    VecSet(dvarEx["slip"],0.);
  }

  // 3. implicitly integrated variables

  // heat equation
  if (_evolveTemperature == 1 && varIm.find("Temp") != varIm.end()) {

    // frictional shear heating source terms
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault->_tauP;

    // compute viscous strain rate that contributes to viscous shear heating:
    Vec dgV_sh;
    VecDuplicate(_material->_dgVdev,&dgV_sh);
    if ( _grainSizeEvCoupling!="no") {
      // relevant visc strain rate = (total) - (portion contributing to grain size reduction)
      ierr = VecPointwiseMult(dgV_sh,_grainDist->_f,_material->_dgVdev_disl);CHKERRQ(ierr);
      ierr = VecScale(dgV_sh,-1.0);CHKERRQ(ierr);
      ierr = VecAXPY(dgV_sh,1.0,_material->_dgVdev);CHKERRQ(ierr);
    }
    else {
      ierr = VecCopy(_material->_dgVdev,dgV_sh);CHKERRQ(ierr);
    }

    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,_material->_sdev,dgV_sh,varIm["Temp"],Told,dt); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgdev, T, old T, dt
    VecDestroy(&dgV_sh);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode StrikeSlip_PowerLaw_qd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // compute source terms to rhs: d/dy(mu*gVxy) + d/dz(mu*gVxz)
  Vec viscSource;
  ierr = VecDuplicate(_material->_gVxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = _material->computeViscStrainSourceTerms(viscSource); CHKERRQ(ierr);

  // set up rhs vector
  //~ if (_isMMS) { _material->setMMSBoundaryConditions(time); }
  _material->setRHS();
  ierr = VecAXPY(_material->_rhs,1.0,viscSource); CHKERRQ(ierr);
  VecDestroy(&viscSource);
  //~ if (_isMMS) { _material->addRHS_MMSSource(time,_material->_rhs); }

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) { VecAXPY(_material->_rhs,1.0,_forcingTerm); }


  // solve for displacement
  ierr = _material->computeU(); CHKERRQ(ierr);

  // update stresses, viscosity, and set shear traction on fault
  ierr = _material->computeTotalStrains(); CHKERRQ(ierr);
  ierr = _material->computeStresses(); CHKERRQ(ierr);
  ierr = _material->computeViscosity(_material->_effViscCap); CHKERRQ(ierr);

  // compute viscous strain rates
  //~ Vec gVxy = varEx.find("gVxy")->second;
  //~ Vec gVxz = varEx.find("gVxz")->second;
  //~ ierr = _material->computeViscStrainRates(time,gVxy,gVxz,dvarEx["gVxy"],dvarEx["gVxz"]); CHKERRQ(ierr);
  ierr = _material->computeViscStrainRates(time); CHKERRQ(ierr);
  VecCopy(_material->_dgVxy,dvarEx["gVxy"]);
  VecCopy(_material->_dgVxz,dvarEx["gVxz"]);
  //~ if (_isMMS) { _material->addViscStrainRates_MMSSource(time,dvarEx["gVxy"],dvarEx["gVxz"]); }

  return ierr;
}


// for solving fixed point iteration problem, with or without the heat equation
PetscErrorCode StrikeSlip_PowerLaw_qd::integrateSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::integrateSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  // initial guess for (thermo)mechanical problem
  initiateIntegrandSS();
  if (!_D->_restartFromChkptSS){
    solveSS(_SS_index);
    writeSS(_SS_index);
    _SS_index++;
  }

  // iterate to converge to steady-state solution
  while (_SS_index < _maxSSIts_tot) {
    PetscPrintf(PETSC_COMM_WORLD,"Jj = %i\n",_SS_index);

    // iterate to find effective viscosity etc
    solveSSViscoelasticProblem(_SS_index);

    // save current state
    writeSS(_SS_index);

    // brute force time integrate for steady-state shear stress the fault
    solveSStau(_SS_index);

    // find steady-state temperature
    if (_computeSSTemperature == 1) { solveSSHeatEquation(_SS_index); }
    if (_thermalCoupling == "coupled") {
      _material->updateTemperature(_varSS["Temp"]);
      _fault->updateTemperature(_varSS["Temp"]);
    }



    _SS_index++;
  }



  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute the forcing term for top bc in terms of particle velocity
PetscErrorCode StrikeSlip_PowerLaw_qd::updateBCT_atan_v()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 2
    string funcName = "StrikeSlip_PowerLaw_qd::updateBCT_atan_v";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    Vec yT;
  VecDuplicate(_D->_z0,&yT);
  VecScatterBegin(_D->_scatters["body2T"], _D->_y, yT, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2T"], _D->_y, yT, INSERT_VALUES, SCATTER_FORWARD);

  PetscInt                Istart,Iend;
  PetscScalar            *bcT;
  const PetscScalar      *y;
  VecGetOwnershipRange(_material->_bcT,&Istart,&Iend);
  VecGetArray(_material->_bcT,&bcT);
  VecGetArrayRead(yT,&y);

  PetscInt Jj = 0;
  PetscScalar amp = atan(_D->_Ly/(2.0*PETSC_PI*_bcT_L));
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    bcT[Jj] = (_vL/_faultTypeScale/amp) * atan(y[Jj]/(2.0*PETSC_PI*_bcT_L));
    Jj++;
  }
  VecRestoreArray(_material->_bcT,&bcT);
  VecRestoreArrayRead(_D->_y0,&y);

  VecDestroy(&yT);

  #if VERBOSE > 2
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// compute the forcing term for top bc in terms of displacement
PetscErrorCode StrikeSlip_PowerLaw_qd::updateBCT_atan_u(const PetscScalar time)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 2
    string funcName = "StrikeSlip_PowerLaw_qd::updateBCT_atan_u";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt           Istart,Iend;
  PetscScalar       *bcT;
  const PetscScalar *y,*bcTShift;
  VecGetOwnershipRange(_material->_bcT,&Istart,&Iend);
  VecGetArray(_material->_bcT,&bcT);
  VecGetArrayRead(_D->_y0,&y);
  VecGetArrayRead(_material->_bcTShift,&bcTShift);

  PetscInt Jj = 0;
  PetscScalar amp = atan(_D->_Ly/(2.0*PETSC_PI*_bcT_L));
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    bcT[Jj] = (_vL/_faultTypeScale/amp) * atan(y[Jj]/(2.0*PETSC_PI*_bcT_L)) * time + bcTShift[Jj];
    Jj++;
  }
  VecRestoreArray(_material->_bcT,&bcT);
  VecRestoreArrayRead(_D->_y0,&y);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// estimate steady state shear stress on fault, store in varSS
PetscErrorCode StrikeSlip_PowerLaw_qd::guessTauSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::guessTauSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // steady-state shear stress on fault
  bool loadTauSS = 0;
  loadVecFromInputFile(_fault->_tauP,_inputDir,"tauSS",loadTauSS);

  // if steady-state shear stress not provided: tauSS = min(tauRS,tauVisc)
  if (loadTauSS == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"\n\n Note: didn't find tauSS and am constructing it\n\n");
    // viscous strength of material, evaluated only at fault
    Vec tauVisc = NULL;
    VecDuplicate(_fault->_tauP,&tauVisc);
    VecScatterBegin(_D->_scatters["body2L"], _material->_effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2L"], _material->_effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
    VecScale(tauVisc,_gss_t);

    VecPointwiseMin(_fault->_tauP,_fault->_tauP,tauVisc);
    VecDestroy(&tauVisc);
  }
  VecCopy(_fault->_tauP,_fault->_tauQSP);
  _fault->computeVel();

  // impose ceiling on fault velocity: slipVel <= vL
  PetscScalar *V;
  VecGetArray(_fault->_slipVel,&V);
  PetscInt Kk = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_fault->_slipVel,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    V[Kk] = min(V[Kk],_vL);
    Kk++;
  }
  VecRestoreArray(_fault->_slipVel,&V);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_qd::initiateIntegrandSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::initiateIntegrandSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up varSS
  VecDuplicate(_material->_u,&_varSS["v"]); VecSet(_varSS["v"],0.);
  _varSS["effVisc"] = _material->_effVisc;
  _varSS["sxy"] = _material->_sxy;
  _varSS["sxz"] = _material->_sxz;
  _varSS["sDev"] = _material->_sdev;
  _varSS["u"] = _material->_u;
  _varSS["gVxy"] = _material->_gVxy;
  _varSS["gVxz"] = _material->_gVxz;
  _varSS["dgVxy"] = _material->_dgVxy;
  _varSS["dgVxz"] = _material->_dgVxz;
  _varSS["Temp"] = _he->_T;
  _varSS["tau"] = _fault->_tauP;
  _varSS["slipVel"] = _fault->_slipVel;
  _varSS["psi"] = _fault->_psi;


  if (_varSS.find("Temp") == _varSS.end() ) {
    Vec Temp; VecDuplicate(_he->_T,&Temp); VecCopy(_he->_T,Temp); _varSS["Temp"] = Temp;
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::solveSS(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // estimate steady-state conditions for fault, material based on strain rate
  _fault->guessSS(_vL); // sets: slipVel, psi, tau
  _material->guessSteadyStateEffVisc(_gss_t);

  // estimate steady-state shear stress at y = 0
  guessTauSS(_varSS);
  solveSSViscoelasticProblem(Jj); // converge to steady state eta etc

  if (_computeSSTemperature == 1) { solveSSHeatEquation(Jj); }
  if (_thermalCoupling == "coupled") {
      _material->updateTemperature(_varSS["Temp"]);
      _fault->updateTemperature(_varSS["Temp"]);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_qd::solveSStau(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::solveSStau";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // create output path with Jj appended on end
  //~ char buff[5]; sprintf(buff,"%04d",Jj); _outputDir = baseOutDir + string(buff) + "_";
  //~ PetscPrintf(PETSC_COMM_WORLD,"baseDir = %s\n\n",_outputDir.c_str());

  _stepCount = 0;
  _initTime = 0;
  _currTime = _initTime;

  // ensure varEx and varIm contain updated versions of all fields
  _material->initiateIntegrand(_initTime,_varEx);
  _fault->initiateIntegrand(_initTime,_varEx);
  if (_evolveTemperature == 1) { _he->initiateIntegrand(_initTime,_varEx,_varIm); }
  if (_evolveGrainSize == 1) { _grainDist->initiateIntegrand(_initTime,_varEx,_varIm); }

  if (_varEx.find("slip") != _varEx.end() ) { VecCopy(_material->_bcL,_varEx["slip"]); }
  else { Vec varSlip; VecDuplicate(_material->_bcL,&varSlip); VecCopy(_material->_bcL,varSlip); _varEx["slip"] = varSlip; }
  VecScale(_varEx["slip"],_faultTypeScale);


  delete _quadEx; _quadEx = NULL;
  delete _quadImex; _quadImex = NULL;

  integrate();

  delete _quadEx; _quadEx = NULL;
  delete _quadImex; _quadImex = NULL;
  PetscViewerDestroy(&_viewer1D); _viewer1D = NULL;
  PetscViewerDestroy(&_viewer2D); _viewer2D = NULL;

  // impose ceiling on fault velocity: slipVel <= vL
  PetscScalar *V;
  VecGetArray(_fault->_slipVel,&V);
  PetscInt Kk = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_fault->_slipVel,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    V[Kk] = min(V[Kk],_vL);
    Kk++;
  }
  VecRestoreArray(_fault->_slipVel,&V);

  // compute frictional strength of fault based on updated slip velocity
  strength_psi_Vec(_fault->_strength, _fault->_psi, _fault->_slipVel, _fault->_a, _fault->_sNEff, _fault->_v0);
  ierr = VecCopy(_fault->_strength,_fault->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault->_slipVel,_fault->_tauQSP); CHKERRQ(ierr); // V -> tauQS
  ierr = VecPointwiseMult(_fault->_tauQSP,_fault->_eta_rad,_fault->_tauQSP); CHKERRQ(ierr); // tauQS = V * eta_rad
  ierr = VecAYPX(_fault->_tauQSP,1.0,_fault->_tauP); // tauQS = tau + V*eta_rad

  // update momentum balance equation boundary conditions to be compatible with steady-state solve step
  //~ _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);
  //~ Mat A; _material->_sbp->getA(A);
  //~ ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// converge to steady state: effective viscosity, sxy, sxz, gVxy, gVxz, gVxy_t, gVxz_t, u
PetscErrorCode StrikeSlip_PowerLaw_qd::solveSSViscoelasticProblem(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::solveSSViscoelasticProblem";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up KSP for steady-state solution
  //~ _material->changeBCTypes(_mat_bcRType,_mat_bcTType,"Neumann",_mat_bcBType);
  //~ _material->changeBCTypes("Dirichlet",_mat_bcTType_SS,"Neumann","Neumann");
  //~ Mat A;
  //~ _material->_sbp->getA(A);
  //~ _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS);

  // set up rhs vector containing boundary condition data
  VecCopy(_varSS["tau"],_material->_bcL);
  VecSet(_material->_bcR,_vL/2.);
  string  _mat_bcTType_SS = "Neumann";
  VecSet(_material->_bcT,0.);
  VecSet(_material->_bcB,0.);
  if (_bcTType == "atan_u") {
    updateBCT_atan_v();
    _mat_bcTType_SS = "Dirichlet";
  }

  // loop over effective viscosity
  Vec effVisc_old; VecDuplicate(_varSS["effVisc"],&effVisc_old);

  Vec temp; VecDuplicate(_varSS["effVisc"],&temp); VecSet(temp,0.);
  double err = 1e10;
  int Ii = 0;
  while (Ii < _maxSSIts_effVisc && err >= _atolSS_effVisc) {
    VecCopy(_varSS["effVisc"],effVisc_old);

    _material->setSSRHS(_varSS,"Dirichlet",_mat_bcTType_SS,"Neumann","Neumann");
    _material->updateSSa(_varSS); // compute v, viscous strain rates

    // update grain size
    if (_computeSSGrainSize == 1) { solveSSGrainSize(Jj); }
    if (_grainSizeEvCoupling == "coupled" && _computeSSGrainSize == 1) { _material->updateGrainSize(_varSS["grainSize"]); }

    _material->computeViscosity(_material->_effViscCap); // new viscosity

    // update effective viscosity: log10(accepted viscosity) = (1-f)*log10(old viscosity) + f*log10(new viscosity):
    MyVecLog10AXPBY(temp,1.-_fss_EffVisc,effVisc_old,_fss_EffVisc,_varSS["effVisc"]);
    VecCopy(temp,_varSS["effVisc"]);

    // evaluate convergence of this iteration
    err = computeMaxDiff_scaleVec1(effVisc_old,_varSS["effVisc"]); // total eff visc

    PetscPrintf(PETSC_COMM_WORLD,"    effective viscosity loop: %i %e\n",Ii,err);
    //~ ierr = writeViscLoopSS(Ii); CHKERRQ(ierr);
    Ii++;
  }
  VecDestroy(&effVisc_old);
  VecDestroy(&temp);


  // update u, gVxy, gVxz, boundary conditions based on effective viscosity
  ierr = _material->updateSSb(_varSS,_initTime); CHKERRQ(ierr); // solve for gVxy, gVxz
  setSSBCs(); // update u, boundary conditions to be positive, consistent with varEx

  // update shear stress on fault
  ierr = VecScatterBegin(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _material->_sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  _fault->updateTauP();
  _fault->updateStrength();
  //VecCopy(_fault->_tauQSP,_fault->_tauP);
  //VecCopy(_fault->_tauQSP,_fault->_strength);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// solve steady-state heat equation for temperature
// update temperature using damping:
//   Tnew = (1-f)*Told + f*Tnew
PetscErrorCode StrikeSlip_PowerLaw_qd::solveSSHeatEquation(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::solveSSHeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // ensure temperature is stored in varSS
  if (_varSS.find("Temp") == _varSS.end() ) {
    Vec Temp; VecDuplicate(_he->_T,&Temp); VecCopy(_he->_T,Temp); _varSS["Temp"] = Temp;
  }

  // save previous temperature for damping
  Vec T_old;
  VecDuplicate(_varSS["Temp"],&T_old);
  VecCopy(_varSS["Temp"],T_old);

  // frictional shear heating source terms
  Vec V; VecDuplicate(_varSS["slipVel"],&V); VecSet(V,_D->_vL);
  VecPointwiseMin(_varSS["slipVel"],V,_varSS["slipVel"]);

  // viscous shear heating source terms
  Vec dgV_sh;
  VecDuplicate(_material->_dgVdev,&dgV_sh);
  if ( _grainSizeEvCoupling != "no") {
    // relevant visc strain rate = (total) - (portion contributing to grain size reduction)
    ierr = VecPointwiseMult(dgV_sh,_grainDist->_f,_material->_dgVdev_disl); CHKERRQ(ierr);
    ierr = VecScale(dgV_sh,-1.0); CHKERRQ(ierr);
    ierr = VecAXPY(dgV_sh,1.0,_material->_dgVdev); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy(_material->_dgVdev,dgV_sh); CHKERRQ(ierr);
  }

  // compute new steady-state temperature
  _he->computeSteadyStateTemp(_currTime,_varSS["slipVel"],_fault->_tauP,_material->_sdev,dgV_sh,_varSS["Temp"]);
  VecDestroy(&dgV_sh);

  // If this is first iteration, keep Temp.
  // If not, apply damping parameter for update
  if (Jj > 0) {
    ierr = VecScale(_varSS["Temp"],_fss_T); CHKERRQ(ierr);
    ierr = VecAXPY(_varSS["Temp"],1.-_fss_T,T_old); CHKERRQ(ierr);
    ierr = VecWAXPY(_he->_dT,-1.0,_he->_Tamb,_varSS["Temp"]); CHKERRQ(ierr);
  }

  // clean up memory usage
  VecDestroy(&T_old);
  VecDestroy(&V);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// solve for steady-state grain size distribution
// update grain size using damping:
//   gnew = (1-f)*gold + f*gnew
PetscErrorCode StrikeSlip_PowerLaw_qd::solveSSGrainSize(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::solveSSGrainSize";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // ensure grain size is stored in varSS
  if (_varSS.find("grainSize") == _varSS.end() ) {
    VecDuplicate(_grainDist->_d,&_varSS["grainSize"]);
    VecCopy(_grainDist->_d,_varSS["grainSize"]);
  }

  // get source terms for grain size distribution equation
  Vec sdev = _material->_sdev;
  Vec dgVdev = _material->_dgVdev_disl;

  // compute new steady-state grain size distribution
  if (_grainDist->_grainSizeEvTypeSS == "piezometer") {
    _grainDist->computeGrainSizeFromPiez(sdev, dgVdev, _varSS["Temp"]);
  }
  else {
    _grainDist->computeSteadyStateGrainSize(sdev, dgVdev, _varSS["Temp"]);
  }

  // update to new grain size with no damping
  VecCopy(_grainDist->_d,_varSS["grainSize"]);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::writeSS(const int Ii)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  bool needToDestroyJjSSVec = 0;
  if (_JjSSVec == NULL) {
    // initiate Vec to hold index Jj
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
    VecSetBlockSize(_JjSSVec, 1);
    PetscObjectSetName((PetscObject) _JjSSVec, "index");
    VecSet(_JjSSVec,Ii);
    needToDestroyJjSSVec = 1;
  }
  else {
    VecSet(_JjSSVec,Ii);
  }

  ierr = VecSet(_JjSSVec,Ii);                                           CHKERRQ(ierr);

  if (_viewerSS == NULL) {
    // set up viewer for output of steady-state data
    string outFileName = _outputDir + "data_steadyState.h5";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewerSS);             CHKERRQ(ierr);
    ierr = PetscViewerSetType(_viewerSS, PETSCVIEWERBINARY);            CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewerSS);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(_viewerSS, _SS_index);            CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewerSS);                 CHKERRQ(ierr);
  }

  ierr = VecView(_JjSSVec, _viewerSS);                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewerSS, "SS_index", "SS_index", PETSC_INT, &_SS_index); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(_viewerSS);                     CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewerSS);                            CHKERRQ(ierr);

  ierr = _material->writeStep1D(_viewerSS);                             CHKERRQ(ierr);
  ierr = _fault->writeStep(_viewerSS);                                  CHKERRQ(ierr);
  if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_viewerSS); CHKERRQ(ierr);}
  if (_he!=NULL) { ierr =  _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

  ierr = _material->writeStep2D(_viewerSS);                             CHKERRQ(ierr);
  if (_he!=NULL) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }
  if (_computeSSGrainSize == 1) { ierr =  _grainDist->writeStep(_viewerSS); CHKERRQ(ierr); }

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::writeViscLoopSS(const int Ii)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::writeViscLoopSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  bool needToDestroyJjSSVec = 0;
  if (_JjSSVec == NULL) {
    // initiate Vec to hold index Jj
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
    VecSetBlockSize(_JjSSVec, 1);
    PetscObjectSetName((PetscObject) _JjSSVec, "index");
    VecSet(_JjSSVec,Ii);
    needToDestroyJjSSVec = 1;
  }
  else {
    VecSet(_JjSSVec,Ii);
  }

  ierr = VecSet(_JjSSVec,Ii);                                           CHKERRQ(ierr);

  if (_viewerSS == NULL) {
    // set up viewer for output of steady-state data
    string outFileName = _outputDir + "data_viscLoop.h5";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewerSS);             CHKERRQ(ierr);
    ierr = PetscViewerSetType(_viewerSS, PETSCVIEWERBINARY);            CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewerSS);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(_viewerSS, _SS_index);            CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewerSS);                 CHKERRQ(ierr);
  }

  ierr = VecView(_JjSSVec, _viewerSS);                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewerSS, "SS_index", "SS_index", PETSC_INT, &_SS_index); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewerSS, "/momBal");                   CHKERRQ(ierr);
  ierr = VecView(_material->_surfDisp,_viewerSS);                                     CHKERRQ(ierr);
  ierr = VecView(_material->_bcL,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_bcR,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_bcB,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_bcT,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_bcRShift, _viewerSS);                                    CHKERRQ(ierr);
  ierr = VecView(_material->_u,_viewerSS);                                            CHKERRQ(ierr);
  ierr = VecView(_material->_sxy,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_sxz,_viewerSS);                                          CHKERRQ(ierr);
  ierr = VecView(_material->_sdev,_viewerSS);                                         CHKERRQ(ierr);
  ierr = VecView(_material->_gTxy,_viewerSS);                                         CHKERRQ(ierr);
  ierr = VecView(_material->_gTxz,_viewerSS);                                         CHKERRQ(ierr);
  ierr = VecView(_material->_gVxy,_viewerSS);                                         CHKERRQ(ierr);
  ierr = VecView(_material->_gVxz,_viewerSS);                                         CHKERRQ(ierr);
  ierr = VecView(_material->_dgVxy,_viewerSS);                                        CHKERRQ(ierr);
  ierr = VecView(_material->_dgVxz,_viewerSS);                                        CHKERRQ(ierr);
  ierr = VecView(_material->_effVisc,_viewerSS);                                      CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewerSS);                             CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewerSS, "/momBal/dislocationCreep");CHKERRQ(ierr);
  ierr = VecView(_material->_dgVdev_disl,_viewerSS);                                CHKERRQ(ierr);
  ierr = VecView(_material->_disl->_invEffVisc,_viewerSS);                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewerSS);                             CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewerSS, "/fault");                  CHKERRQ(ierr);
  ierr = VecView(_fault->_slip, _viewerSS);                                      CHKERRQ(ierr);
  ierr = VecView(_fault->_slipVel, _viewerSS);                                   CHKERRQ(ierr);
  ierr = VecView(_fault->_tauP, _viewerSS);                                      CHKERRQ(ierr);
  ierr = VecView(_fault->_tauQSP, _viewerSS);                                    CHKERRQ(ierr);
  ierr = VecView(_fault->_strength, _viewerSS);                                  CHKERRQ(ierr);
  ierr = VecView(_fault->_psi, _viewerSS);                                       CHKERRQ(ierr);
  ierr = VecView(_fault->_sNEff, _viewerSS);                                     CHKERRQ(ierr);
  ierr = VecView(_fault->_sN, _viewerSS);                                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewerSS);                             CHKERRQ(ierr);

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// update the boundary conditions based on new steady state u
PetscErrorCode StrikeSlip_PowerLaw_qd::setSSBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd::setSSBCs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // adjust u so it has no negative values
  PetscScalar minVal = 0;
  VecMin(_material->_u,NULL,&minVal);
  if (minVal < 0) {
    minVal = abs(minVal);
    Vec temp;
    VecDuplicate(_material->_u,&temp);
    VecSet(temp,minVal);
    VecAXPY(_material->_u,1.,temp);
    VecDestroy(&temp);
  }

  // extract R boundary from u, to set _material->bcR
  VecScatterBegin(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
  VecCopy(_material->_bcRShift,_material->_bcR);

  // extract L boundary from u to set slip, possibly _material->_bcL
  Vec uL; VecDuplicate(_material->_bcL,&uL);
  VecScatterBegin(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);
  VecCopy(uL,_material->_bcL);

  // extract top boundary from u to set bcT, if using Dirichlet loading for top BC
  //~ VecScatterBegin(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecScatterEnd(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecCopy(_material->_bcTShift,_material->_bcT);

  if (_varEx.find("slip") != _varEx.end() ) { VecCopy(uL,_varEx["slip"]); }
  else {
    Vec slip;
    VecDuplicate(_material->_bcL,&slip);
    VecCopy(uL,slip);
    _varEx["slip"] = slip;
  }
  if (_bcLType == "symmFault") {
    VecScale(_varEx["slip"],2.);
  }

  VecDestroy(&uL);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// constructs the body forcing term for an ice stream
// includes allocation of memory for this forcing term
PetscErrorCode StrikeSlip_PowerLaw_qd::constructIceStreamForcingTerm()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_PowerLaw_qd::constructIceStreamForcingTerm";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif



  //~ // matrix to map the value for the forcing term, which lives on the fault, to all other processors
  //~ Mat MapV;
  //~ MatCreate(PETSC_COMM_WORLD,&MapV);
  //~ MatSetSizes(MapV,PETSC_DECIDE,PETSC_DECIDE,_D->_Ny*_D->_Nz,_D->_Nz);
  //~ MatSetFromOptions(MapV);
  //~ MatMPIAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL,_D->_Ny*_D->_Nz,NULL);
  //~ MatSeqAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL);
  //~ MatSetUp(MapV);

  //~ PetscScalar v=1.0;
  //~ PetscInt Ii=0,Istart=0,Iend=0,Jj=0;
  //~ MatGetOwnershipRange(MapV,&Istart,&Iend);
  //~ for (Ii = Istart; Ii < Iend; Ii++) {
    //~ Jj = Ii % _D->_Nz;
    //~ MatSetValues(MapV,1,&Ii,1,&Jj,&v,INSERT_VALUES);
  //~ }
  //~ MatAssemblyBegin(MapV,MAT_FINAL_ASSEMBLY);
  //~ MatAssemblyEnd(MapV,MAT_FINAL_ASSEMBLY);


  // compute forcing term for momentum balance equation
  // forcing = - tau_ss / Ly
  //~ Vec tauSS = NULL;
  //~ _fault->guessSS(tauSS,_vL);
  //~ VecScale(tauSS,-1./_D->_Ly);

  //~ VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  //~ MatMult(MapV,tauSS,_forcingTerm);

  //~ MatDestroy(&MapV);
  //~ VecDestroy(&tauSS);

  // compute forcing term for momentum balance equation
  // forcing = (1/Ly) * (tau_ss + eta_rad*V_ss)
  //~ Vec tauSS = NULL,radDamp=NULL,V=NULL;
  //~ VecDuplicate(_fault->_eta_rad,&V); VecSet(V,_vL);
  //~ VecDuplicate(_fault->_eta_rad,&radDamp); VecPointwiseMult(radDamp,_fault->_eta_rad,V);
  //~ _fault->guessSS(tauSS,_vL);
  //~ VecAXPY(tauSS,1.0,radDamp);
  //~ VecScale(tauSS,-1./_D->_Ly);

  //~ VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  //~ MatMult(MapV,tauSS,_forcingTerm);

  //~ MatDestroy(&MapV);
  //~ VecDestroy(&tauSS);
  //~ VecDestroy(&radDamp);

  // compute forcing term using scalar input
  VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,_forcingVal);
  PetscObjectSetName((PetscObject) _forcingTerm, "forcingTerm");
  VecDuplicate(_material->_u,&_forcingTermPlain); VecCopy(_forcingTerm,_forcingTermPlain);
  PetscObjectSetName((PetscObject) _forcingTermPlain, "forcingTermPlain");

  // alternatively, load forcing term from user input
  ierr = loadVecFromInputFile(_forcingTerm,_inputDir,"iceForcingTerm"); CHKERRQ(ierr);

  // multiply forcing term by H, or by J*H if using a curvilinear grid
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    // multiply this term by H*J (the H matrix and the Jacobian)
    Vec temp1; VecDuplicate(_forcingTerm,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_forcingTerm,temp1); CHKERRQ(ierr);
    Mat H; _material->_sbp->getH(H);
    ierr = MatMult(H,temp1,_forcingTerm); CHKERRQ(ierr);
    VecDestroy(&temp1);
  }
  else{ // multiply forcing term by H
    Vec temp1; VecDuplicate(_forcingTerm,&temp1);
    Mat H; _material->_sbp->getH(H);
    ierr = MatMult(H,_forcingTerm,temp1); CHKERRQ(ierr);
    VecCopy(temp1,_forcingTerm);
    VecDestroy(&temp1);
  }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // _material->measureMMSError(_currTime);

  //~ _he->measureMMSError(_currTime);
  _p->measureMMSError(_currTime);

  return ierr;
}



