#include "strikeSlip_powerLaw_qd_fd.hpp"

#define FILENAME "strikeSlip_powerLaw_qd_fd.cpp"

using namespace std;


StrikeSlip_PowerLaw_qd_fd::StrikeSlip_PowerLaw_qd_fd(Domain&D)
: _D(&D),_y(&D._y),_z(&D._z),_delim(D._delim),
  _inputDir(D._inputDir),_outputDir(D._outputDir),_vL(1e-9),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _guessSteadyStateICs(0),_forcingType("no"),_faultTypeScale(2.0),
  _evolveTemperature(0),_evolveGrainSize(0),_computeSSTemperature(0),_computeSSGrainSize(0),
  _cycleCount(0),_maxNumCycles(1e3),_phaseCount(0),
  _deltaT(1e-3),_deltaT_fd(-1),_CFL(0.5),
  _ay(NULL),_Fhat(NULL),_alphay(NULL),
  _inDynamic(false),_allowed(false), _trigger_qd2fd(1e-3), _trigger_fd2qd(1e-3),
  _limit_qd(10*_vL), _limit_fd(1e-1),_limit_stride_fd(1e-2),
  _JjSSVec(NULL),
  _fss_T(0.15),_fss_EffVisc(0.2),_fss_grainSize(0.2),_gss_t(1e-10),
  _SS_index(0),_maxSSIts_effVisc(50),_maxSSIts_tot(100),
  _atolSS_effVisc(1e-4),
  _u0(NULL),
  _timeIntegrator("RK43"),_timeControlType("PID"),
  _stride1D(1),_stride2D(1),_strideChkpt_qd(1e4),_strideChkpt_fd(1e4),_maxStepCount(1e8),
  _initTime(0),_currTime(0),_maxTime(1e15),
  _minDeltaT(1e-3),_maxDeltaT(1e10),
  _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
  _chkptTimeStep1D(0), _chkptTimeStep2D(0),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),
  _startTime(MPI_Wtime()),_miscTime(0),
  _time1DVec(NULL), _dtime1DVec(NULL),_time2DVec(NULL), _dtime2DVec(NULL),_regime1DVec(NULL),_regime2DVec(NULL),
  _viewer_context(NULL),_viewer1D(NULL),_viewer2D(NULL),_viewerSS(NULL),_viewer_chkpt(NULL),
  _forcingVal(0),
  _qd_bcRType("remoteLoading"),_qd_bcTType("freeSurface"),_qd_bcLType("symmFault"),_qd_bcBType("freeSurface"),
  _fd_bcRType("outGoingCharacteristics"),_fd_bcTType("freeSurface"),_fd_bcLType("symmFault"),_fd_bcBType("outGoingCharacteristics"),
  _mat_fd_bcRType("Neumann"),_mat_fd_bcTType("Neumann"),_mat_fd_bcLType("Neumann"),_mat_fd_bcBType("Neumann"),
  _quadEx_qd(NULL),_quadImex_qd(NULL),_quadWaveEx(NULL),
  _fault_qd(NULL),_material(NULL),_he(NULL),_p(NULL),_grainDist(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::StrikeSlip_PowerLaw_qd_fd()";
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
  if (_D->_restartFromChkpt && _inDynamic) {
    _material = new PowerLaw(D,_mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType);
  }
  else {
    _material = new PowerLaw(D,_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);
  }

  _he = new HeatEquation(D); // heat equation
  if (_thermalCoupling == "coupled") { VecCopy(_he->_T,_material->_T); }

  _body2fault = &(D._scatters["body2L"]);
  _fault_qd = new Fault_qd(D,*_body2fault,_faultTypeScale); // quasidynamic fault
  _fault_fd = new Fault_fd(D,*_body2fault,_faultTypeScale); // fully dynamic fault
  if (_thermalCoupling!="no" && _stateLaw == "flashHeating") {
    Vec T; VecDuplicate(_D->_y,&T);
    _he->getTemp(T);
    _fault_qd->setThermalFields(T,_he->_k,_he->_c);
    _fault_fd->setThermalFields(T,_he->_k,_he->_c);
  }

  // pressure diffusion equation
  if (_hydraulicCoupling!="no") { _p = new PressureEq(D); }
  if (_hydraulicCoupling == "coupled") {
    _fault_qd->setSNEff(_p->_p);
    _fault_fd->setSNEff(_p->_p);
  }

  // grain size distribution
  if (_evolveGrainSize == 1 || _computeSSGrainSize == 1) { _grainDist = new GrainSizeEvolution(D); }
  if (_grainSizeEvCoupling == "coupled") { VecCopy(_grainDist->_d, _material->_grainSize); }

  computePenaltyVectors();
  computeTimeStep(); // compute fully dynamic time step

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_PowerLaw_qd_fd::~StrikeSlip_PowerLaw_qd_fd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::~StrikeSlip_PowerLaw_qd_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // adaptive time stepping containers
  map<string,Vec>::iterator it;
  for (it = _varQSEx.begin(); it!=_varQSEx.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }

  // wave equation time stepping containers
  for (it = _varFD.begin(); it!=_varFD.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varFDPrev.begin(); it!=_varFDPrev.end(); it++ ) {
    VecDestroy(&it->second);
  }

  PetscViewerDestroy(&_viewer1D);
  PetscViewerDestroy(&_viewer2D);
  PetscViewerDestroy(&_viewer_context);

  VecDestroy(&_time1DVec);
  VecDestroy(&_dtime1DVec);
  VecDestroy(&_time2DVec);
  VecDestroy(&_dtime2DVec);
  VecDestroy(&_u0);
  VecDestroy(&_ay);


  delete _quadImex_qd;    _quadImex_qd = NULL;
  delete _quadEx_qd;      _quadEx_qd = NULL;
  delete _material;    _material = NULL;
  delete _fault_qd;    _fault_qd = NULL;
  delete _fault_fd;    _fault_fd = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;
  delete _grainDist;   _grainDist = NULL;

  VecDestroy(&_forcingTerm);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::loadSettings(const char *file)
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
    else if (var.compare("grainSizeEvCouplingSS")==0) { _grainSizeEvCouplingSS = rhs.c_str(); }
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
    else if (var.compare("evolveTemperature")==0) { _evolveTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("evolveGrainSize")==0) { _evolveGrainSize = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSHeatEq")==0) { _computeSSTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSGrainSize")==0) { _computeSSGrainSize = (int) atoi( rhs.c_str() ); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) { _timeIntegrator = rhs; }
    else if (var.compare("timeControlType")==0) { _timeControlType = rhs; }
    else if (var.compare("stride1D_qd")==0){
      _stride1D_qd = (int)atof( rhs.c_str() );
      _stride1D = _stride1D_qd;
    }
    else if (var.compare("stride2D_qd")==0){
      _stride2D_qd = (int)atof( rhs.c_str() );
      _stride2D = _stride2D_qd;
    }
    else if (var.compare("strideChkpt")==0){ _strideChkpt_qd = (int)atof(rhs.c_str()); }
    else if (var.compare("strideChkpt_fd")==0){ _strideChkpt_fd = (int)atof(rhs.c_str()); }
    else if (var.compare("stride1D_fd")==0){ _stride1D_fd = (int)atof( rhs.c_str() ); }
    else if (var.compare("stride2D_fd")==0){ _stride2D_fd = (int)atof( rhs.c_str() ); }
    else if (var.compare("stride1D_fd_end")==0){ _stride1D_fd_end = (int)atof( rhs.c_str() ); }
    else if (var.compare("stride2D_fd_end")==0){ _stride2D_fd_end = (int)atof( rhs.c_str() ); }

    else if (var.compare("initTime")==0) { _initTime = atof( rhs.c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( rhs.c_str() ); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( rhs.c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("timeStepTol")==0) { _timeStepTol = atof( rhs.c_str() ); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }
    else if (var.compare("scale")==0) { loadVectorFromInputFile(rhsFull,_scale); }
    else if (var.compare("normType")==0) { _normType = rhs.c_str(); }
    else if (var.compare("vL")==0) { _vL = atof( rhs.c_str() ); }

    else if (var.compare("bodyForce")==0) { _forcingVal = atof( rhs.c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_fd")==0) { _fd_bcRType = rhs.c_str(); }
    else if (var.compare("momBal_bcT_fd")==0) { _fd_bcTType = rhs.c_str(); }
    else if (var.compare("momBal_bcL_fd")==0) { _fd_bcLType = rhs.c_str(); }
    else if (var.compare("momBal_bcB_fd")==0) { _fd_bcBType = rhs.c_str(); }

    else if (var.compare("momBal_bcR_qd")==0) { _qd_bcRType = rhs.c_str(); }
    else if (var.compare("momBal_bcT_qd")==0) { _qd_bcTType = rhs.c_str(); }
    else if (var.compare("momBal_bcL_qd")==0) { _qd_bcLType = rhs.c_str(); }
    else if (var.compare("momBal_bcB_qd")==0) { _qd_bcBType = rhs.c_str(); }

    else if (var.compare("trigger_qd2fd")==0) { _trigger_qd2fd = atof( rhs.c_str() ); }
    else if (var.compare("trigger_fd2qd")==0) { _trigger_fd2qd = atof( rhs.c_str() ); }
    else if (var.compare("limit_qd")==0) { _limit_qd = atof( rhs.c_str() ); }
    else if (var.compare("limit_fd")==0) { _limit_fd = atof( rhs.c_str() ); }
    else if (var.compare("limit_stride_fd")==0) { _limit_stride_fd = atof( rhs.c_str() ); }

    else if (var.compare("deltaT_fd")==0) { _deltaT_fd = atof( rhs.c_str() ); }
    else if (var.compare("CFL")==0) { _CFL = atof( rhs.c_str() ); }
    else if (var.compare("maxNumCycles")==0) { _maxNumCycles = atoi( rhs.c_str() ); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);

  assert(_thermalCoupling=="coupled" || _thermalCoupling=="uncoupled" || _thermalCoupling == "no" );
  assert(_hydraulicCoupling=="coupled" || _hydraulicCoupling=="uncoupled" || _hydraulicCoupling == "no" );
  assert(_grainSizeEvCoupling=="coupled" || _grainSizeEvCoupling=="uncoupled" || _grainSizeEvCoupling == "no" );

  assert(_forcingType == "iceStream" || _forcingType == "no" );

  assert(_timeIntegrator == "FEuler" ||
      _timeIntegrator == "RK32" ||
      _timeIntegrator == "RK43" ||
      _timeIntegrator == "RK32_WBE" ||
      _timeIntegrator == "RK43_WBE" );

  assert(_timeControlType == "P" || _timeControlType == "PI" || _timeControlType == "PID");

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {_initDeltaT = _minDeltaT; }
  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_timeStepTol >= 1e-14);
  assert(_minDeltaT >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);

  // check boundary condition types for momentum balance equation
  assert(_qd_bcRType.compare("freeSurface")==0 || _qd_bcRType.compare("remoteLoading")==0 );
  assert(_qd_bcTType.compare("freeSurface")==0 );
  assert(_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0 );
  assert(_qd_bcBType.compare("freeSurface")==0 );

  assert(_fd_bcRType.compare("freeSurface")==0 || _fd_bcRType.compare("outGoingCharacteristics")==0 );
  assert(_fd_bcTType.compare("freeSurface")==0 || _fd_bcTType.compare("outGoingCharacteristics")==0 );
  assert(_fd_bcLType.compare("symmFault")==0 || _fd_bcLType.compare("rigidFault")==0 );
  assert(_fd_bcBType.compare("freeSurface")==0 || _fd_bcBType.compare("outGoingCharacteristics")==0 );

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
  if (_thermalCoupling != "no" && (_timeIntegrator != "RK32_WBE" && _timeIntegrator != "RK43_WBE")) {
    assert(0);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// parse boundary conditions
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::parseBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::parseBCs()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_qd_bcRType.compare("symmFault")==0 || _qd_bcRType.compare("rigidFault")==0 || _qd_bcRType.compare("remoteLoading")==0) {
    _mat_qd_bcRType = "Dirichlet";
  }
  else if (_qd_bcRType.compare("freeSurface")==0 || _qd_bcRType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcRType = "Neumann";
  }

  if (_qd_bcTType.compare("symmFault")==0 || _qd_bcTType.compare("rigidFault")==0 || _qd_bcTType.compare("remoteLoading")==0) {
    _mat_qd_bcTType = "Dirichlet";
  }
  else if (_qd_bcTType.compare("freeSurface")==0 || _qd_bcTType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcTType = "Neumann";
  }

  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0 || _qd_bcLType.compare("remoteLoading")==0) {
    _mat_qd_bcLType = "Dirichlet";
  }
  else if (_qd_bcLType.compare("freeSurface")==0 || _qd_bcLType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcLType = "Neumann";
  }

  if (_qd_bcBType.compare("symmFault")==0 || _qd_bcBType.compare("rigidFault")==0 || _qd_bcBType.compare("remoteLoading")==0) {
    _mat_qd_bcBType = "Dirichlet";
  }
  else if (_qd_bcBType.compare("freeSurface")==0 || _qd_bcBType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcBType = "Neumann";
  }

  // determine if material is symmetric about the fault, or if one side is rigid
  _faultTypeScale = 2.0;
  if (_qd_bcLType.compare("rigidFault")==0 ) { _faultTypeScale = 1.0; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// allocate space for member fields
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_LinearElastic_qd_fd::allocateFields";
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

  ierr = VecDuplicate(_time1DVec,&_regime1DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _regime1DVec, "regime1D"); CHKERRQ(ierr);
  VecSet(_regime1DVec,0); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_time2DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _time2DVec, "time2D"); CHKERRQ(ierr);
  VecSet(_time2DVec,_initTime); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_regime2DVec); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _regime2DVec, "regime2D"); CHKERRQ(ierr);
  ierr = VecSet(_regime2DVec,_deltaT); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_dtime2DVec); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _dtime2DVec, "dtime2D"); CHKERRQ(ierr);
  ierr = VecSet(_dtime2DVec,_deltaT); CHKERRQ(ierr);

  // initial displacement at start of fully dynamic phase
  ierr = VecDuplicate(_D->_y, &_u0); VecSet(_u0,0.0);
  ierr = PetscObjectSetName((PetscObject) _u0, "u0"); CHKERRQ(ierr);

  // body forcing term for ice stream
  _forcingTerm = NULL; _forcingTermPlain = NULL;
  if (_forcingType.compare("iceStream")==0) { constructIceStreamForcingTerm(); }


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

// returns true if it's time to switch from qd to fd, or fd to qd, or if
// the maximum time or step count has been reached
bool StrikeSlip_PowerLaw_qd_fd::checkSwitchRegime(const Fault* _fault)
{
  bool mustSwitch = false;

  // if using max slip velocity as switching criteria
  Vec absSlipVel;
  VecDuplicate(_fault->_slipVel, &absSlipVel);
  VecCopy(_fault->_slipVel, absSlipVel);
  PetscScalar maxV;
  VecAbs(absSlipVel);
  VecMax(absSlipVel, NULL, &maxV);
  VecDestroy(&absSlipVel);

  // if using R = eta*V / tauQS
  //~ Vec R; VecDuplicate(_fault->_slipVel,&R);
  //~ VecPointwiseMult(R,_fault_qd->_eta_rad,_fault->_slipVel);
  //~ VecPointwiseDivide(R,R,_fault->_tauQSP);
  //~ PetscScalar maxV;
  //~ VecMax(R,NULL,&maxV);
  //~ VecDestroy(&R);


  // if integrating past allowed time or step count, force switching now
  if(_currTime > _maxTime || _stepCount > _maxStepCount){
    mustSwitch = true;
    return mustSwitch;
  }

  // Otherwise, first check if switching from qd to fd, or from fd to qd, is allowed:
  // switching from fd to qd is allowed if maxV has ever been > limit_fd
  if( _inDynamic && !_allowed && maxV > _limit_fd) { _allowed = true; }

  // switching from qd to fd is allowed if maxV has ever been < limit_qd
  if( !_inDynamic && !_allowed && maxV < _limit_qd) { _allowed = true; }


  // If switching is allowed, assess if the switching criteria has been reached:
  // switching from fd to qd happens if maxV < _trigger_fd2qd
  if (_inDynamic && _allowed && maxV < _trigger_fd2qd) { mustSwitch = true; }

  // switching from qd to fd happens if maxV > _trigger_qd2fd
  if (!_inDynamic && _allowed && maxV > _trigger_qd2fd) { mustSwitch = true; }


  // also change stride for IO to avoid writing out too many time steps
  // at the end of an earthquake
  if (_inDynamic && _allowed && maxV < _limit_stride_fd) {
    _stride1D = _stride1D_fd_end;
    _stride2D = _stride2D_fd_end;
  }

  return mustSwitch;
}


// compute allowed time step based on CFL condition and user input
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::computeTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::computeTimeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // coefficient for CFL condition
  PetscScalar gcfl = 0.7071; // if order = 2
  if (_D->_order == 4) { gcfl = 0.7071/sqrt(1.4498); }
  if (_D->_order == 6) { gcfl = 0.7071/sqrt(2.1579); }


  // compute grid spacing in y and z
  Vec dy, dz;
  VecDuplicate(*_y,&dy);
  VecDuplicate(*_y,&dz);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatGetDiagonal(yq, dy); VecScale(dy,1.0/(_D->_Ny-1));
    MatGetDiagonal(zr, dz); VecScale(dz,1.0/(_D->_Nz-1));
  }
  else {
    VecSet(dy,_D->_Ly/(_D->_Ny-1.0));
    VecSet(dz,_D->_Lz/(_D->_Nz-1.0));
  }

  // compute time for shear wave to travel 1 dy or dz
  Vec ts_dy,ts_dz;
  VecDuplicate(*_y,&ts_dy);
  VecDuplicate(*_z,&ts_dz);
  VecPointwiseDivide(ts_dy,dy,_material->_cs);
  VecPointwiseDivide(ts_dz,dz,_material->_cs);
  PetscScalar min_ts_dy, min_ts_dz;
  VecMin(ts_dy,NULL,&min_ts_dy);
  VecMin(ts_dz,NULL,&min_ts_dz);

  // clean up memory usage
  VecDestroy(&dy);
  VecDestroy(&dz);
  VecDestroy(&ts_dy);
  VecDestroy(&ts_dz);

  // largest possible time step permitted by CFL condition
  PetscScalar max_deltaT = gcfl * min(abs(min_ts_dy),abs(min_ts_dz));


  // compute time step requested by user
  PetscScalar cfl_deltaT = _CFL * gcfl *  max_deltaT;
  PetscScalar request_deltaT = _deltaT_fd;

  _deltaT = max_deltaT; // ensure deltaT is assigned something sensible even if the conditionals have an error
  if (request_deltaT <= 0. && cfl_deltaT <= 0.) {
    // if user did not specify deltaT or CFL
    _deltaT = max_deltaT;
  }
  else if (request_deltaT > 0. && cfl_deltaT <= 0.) {
    // if user specified deltaT but not CFL
    _deltaT = request_deltaT;
    assert(request_deltaT > 0.);
    if (request_deltaT > max_deltaT) {
      PetscPrintf(PETSC_COMM_WORLD,"Warning: requested deltaT of %g is larger than maximum recommended deltaT of %g\n",request_deltaT,max_deltaT);
    }
  }
  else if (request_deltaT <= 0. && cfl_deltaT > 0.) {
    // if user specified CLF but not deltaT
    _deltaT = cfl_deltaT;
    assert(_CFL <= 1. && _CFL >= 0.);
  }
  else if (request_deltaT > 0. && cfl_deltaT > 0.) {
    // if user specified both CLF and deltaT
    _deltaT = request_deltaT;
    if (request_deltaT > max_deltaT) {
      PetscPrintf(PETSC_COMM_WORLD,"Warning: requested deltaT of %g is larger than maximum recommended deltaT of %g\n",request_deltaT,max_deltaT);
    }
  }

  _deltaT_fd = _deltaT;

  PetscPrintf(PETSC_COMM_WORLD,"Note: maximum recommended deltaT of %g\n",max_deltaT);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// initiate varQSEx, varIm, and varFD
// includes computation of steady-state initial conditions if necessary
// should only be called once before the 1st earthquake cycle
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = initiateIntegrand_qd(); CHKERRQ(ierr);
  ierr = initiateIntegrand_fd(); CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// initiate integrand for quasidynamic period
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::initiateIntegrand_qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::initiateIntegrand_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  if (_varQSEx.find("slip") == _varQSEx.end() ) { ierr = VecDuplicate(_material->_bcL, &_varQSEx["slip"]); CHKERRQ(ierr); }
  if (!_D->_restartFromChkpt) {
    ierr = VecCopy(_material->_bcL, _varQSEx["slip"]); CHKERRQ(ierr);
    if (_qd_bcLType == "symmFault" || _qd_bcLType == "rigidFault") {
      ierr = VecScale(_varQSEx["slip"],_faultTypeScale); CHKERRQ(ierr);
    }
    ierr = loadVecFromInputFile(_varQSEx["slip"],_inputDir,"slip"); CHKERRQ(ierr);
    ierr = VecCopy(_varQSEx["slip"],_fault_qd->_slip); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy(_fault_qd->_slip,_varQSEx["slip"]); CHKERRQ(ierr);
  }


  if (_guessSteadyStateICs) {
    ierr = initiateIntegrandSS(); CHKERRQ(ierr);
    ierr = solveSS(0); CHKERRQ(ierr);
    ierr = writeSS(0); CHKERRQ(ierr);
    VecDestroy(&_JjSSVec); _JjSSVec = NULL;

    // ensure fault_fd == fault_qd
    ierr = VecCopy(_fault_qd->_psi,      _fault_fd->_psi); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slipVel,  _fault_fd->_slipVel); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slip,     _fault_fd->_slip); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slip,     _fault_fd->_slip0); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_tauP,     _fault_fd->_tau0); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_tauQSP,   _fault_fd->_tauQSP); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_tauP,     _fault_fd->_tauP); CHKERRQ(ierr);
  }

  ierr = _material->initiateIntegrand(_initTime,_varQSEx); CHKERRQ(ierr);
  ierr = _fault_qd->initiateIntegrand(_initTime,_varQSEx); CHKERRQ(ierr);


  if (_evolveTemperature == 1) {
    ierr = _he->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr);
    ierr = _fault_qd->updateTemperature(_he->_T); CHKERRQ(ierr);
    ierr = _fault_fd->updateTemperature(_he->_T); CHKERRQ(ierr);
  }
  if (_evolveGrainSize == 1) {
    ierr = _grainDist->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr);
  }
  if (_hydraulicCoupling!="no" ) {
    ierr = _p->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::initiateIntegrand_fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::initiateIntegrand_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // add psi and slip to varFD
  ierr = _fault_fd->initiateIntegrand(_initTime,_varFD); CHKERRQ(ierr); // adds psi and slip

  // add u
  if (_varFD.find("u") != _varFD.end() ) { ierr = VecCopy(_material->_u,_varFD["u"]); CHKERRQ(ierr); }
  else {
    Vec var;
    ierr = VecDuplicate(_material->_u,&var);
    ierr = VecCopy(_material->_u,var);
    _varFD["u"] = var;
  }
  //~ if (!_D->_restartFromChkpt) {
    VecSet(_u0,0.0);
  //~ }


  // if solving the heat equation, add temperature to varFD
  if (_evolveTemperature == 1) {
    if (_varFD.find("Temp") != _varFD.end() ) { ierr = VecCopy(_he->_T,_varFD["Temp"]);CHKERRQ(ierr);  }
    else {
      Vec var;
      ierr = VecDuplicate(_he->_T,&var); CHKERRQ(ierr);
      ierr = VecCopy(_he->_T,var); CHKERRQ(ierr);
      _varFD["Temp"] = var;
    }
  }

  // if solving the grain size evolution equation, add to varFD
  if (_evolveGrainSize == 1) {
    ierr = _grainDist->initiateIntegrand(_initTime,_varFD,_varIm); CHKERRQ(ierr);
  }
   // copy varFD into varFDPrev
  for (map<string,Vec>::iterator it = _varFD.begin(); it != _varFD.end(); it++ ) {
    if (_varFDPrev.find(it->first) == _varFDPrev.end() ) {
      Vec var;
      ierr = VecDuplicate(_varFD[it->first],&var); CHKERRQ(ierr);
      _varFDPrev[it->first] = var;
    }
    ierr = VecCopy(_varFD[it->first],_varFDPrev[it->first]); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// monitoring function for ode solvers
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::timeMonitor";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  // update to reflect current status
  _deltaT = deltaT;
  _currTime = time;
  VecSet(_time1DVec,time);
  VecSet(_dtime1DVec,_deltaT);
  VecSet(_regime1DVec,(int) _inDynamic);
  VecSet(_time2DVec,time);
  VecSet(_dtime2DVec,_deltaT);
  VecSet(_regime2DVec,(int) _inDynamic);

  if (_stepCount == stepCount && _stepCount != 0) { return ierr; } // don't write out the same step twice
  _stepCount = stepCount;

  if ( (_stride1D>0 && _currTime == _maxTime) || (_stride1D>0 && stepCount % _stride1D == 0) ) {
    ierr = writeStep1D(stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_viewer1D); CHKERRQ(ierr);
    if(_inDynamic){ ierr = _fault_fd->writeStep(_viewer1D); CHKERRQ(ierr); }
    else { ierr = _fault_qd->writeStep(_viewer1D); CHKERRQ(ierr); }
    if (_evolveTemperature == 1) { ierr = _he->writeStep1D(_viewer1D); CHKERRQ(ierr); }
    if (_hydraulicCoupling != "no") { ierr = _p->writeStep(_viewer1D); CHKERRQ(ierr); }

  }

  if ( (_stride2D>0 &&_currTime == _maxTime) || (_stride2D>0 && stepCount % _stride2D == 0) ) {
    ierr = writeStep2D(stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_viewer2D);CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr = _he->writeStep2D(_viewer2D);CHKERRQ(ierr); }
    if (_evolveGrainSize == 1) { ierr = _grainDist->writeStep(_viewer2D);CHKERRQ(ierr); }
  }

    // checkpointing
  PetscInt strideChkpt = _strideChkpt_qd;
  if (_inDynamic) {strideChkpt = _strideChkpt_fd;}
  if ( _D->_saveChkpts == 1 && ((strideChkpt > 0 && stepCount % strideChkpt == 0) || (_currTime == _maxTime)) ) {
    ierr = writeCheckpoint();                                           CHKERRQ(ierr);
    ierr = _D->writeCheckpoint(_viewer_chkpt);                          CHKERRQ(ierr);
    ierr = _material->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault_qd->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault_fd->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _he->writeCheckpoint(_viewer_chkpt);                         CHKERRQ(ierr);
    if (_quadEx_qd != NULL && !_inDynamic) { ierr = _quadEx_qd->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadImex_qd != NULL && !_inDynamic) { ierr = _quadImex_qd->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadWaveEx != NULL && _inDynamic) { ierr = _quadWaveEx->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_evolveGrainSize == 1) { ierr = _grainDist->writeCheckpoint(_viewer_chkpt);CHKERRQ(ierr); }
    if (_hydraulicCoupling != "no") { ierr = _p->writeCheckpoint(_viewer_chkpt);  CHKERRQ(ierr); }
  }

  // prevent adaptive time stepper from taking time steps > Maxwell time
  PetscScalar maxTimeStep_tot, maxDeltaT_momBal = 0.0;
  ierr = _material->computeMaxTimeStep(maxDeltaT_momBal);CHKERRQ(ierr);
  maxTimeStep_tot = min(_maxDeltaT,0.9*maxDeltaT_momBal);

  if (_evolveGrainSize == 1 && _grainDist->_grainSizeEvType == "transient") {
    PetscScalar maxDeltaT_grainSizeEv = 0;
    ierr =  _grainDist->computeMaxTimeStep(maxDeltaT_grainSizeEv,_material->_sdev,_material->_dgVdev_disl,_material->_T); CHKERRQ(ierr);
    maxTimeStep_tot = min(_maxDeltaT,0.9*maxDeltaT_grainSizeEv);
  }


  if (_quadImex_qd!=NULL) {
    _quadImex_qd->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
  }
  else if (_quadEx_qd!=NULL) {
    _quadEx_qd->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
  }

  // check if time to switch from quasidynamic to fully dynamic
  stopIntegration = 0;
  if(_inDynamic){
    if(checkSwitchRegime(_fault_fd)){ stopIntegration = 1;}
  }
  else {
    if(checkSwitchRegime(_fault_qd)){ stopIntegration = 1; }
  }

  // additional stopping criteria for steadyStateIterations
  //~ stopIntegration = 0;
  //~ if (_D->_systemEvolutionType == "steadyStateIts") {
    //~ PetscScalar maxVel = 0;
    //~ if(_inDynamic){ VecMax(_fault_fd->_slipVel,NULL,&maxVel); }
    //~ else { VecMax(_fault_qd->_slipVel,NULL,&maxVel); }
    //~ if (maxVel < 1.2e-9 && time > 1e11) { stopIntegration = 1; }
  //~ }

  #if VERBOSE > 0
    //~ double _currIntegrateTime = MPI_Wtime() - _startIntegrateTime;
    std::string regime = "quasidynamic";
    if(_inDynamic){ regime = "fully dynamic"; }
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s\n",stepCount,_currTime,_deltaT,regime.c_str());CHKERRQ(ierr);

    PetscScalar maxVel = 0;
    if(_inDynamic){ VecMax(_fault_fd->_slipVel,NULL,&maxVel); }
    else { VecMax(_fault_qd->_slipVel,NULL,&maxVel); }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s, allowed = %i  maxVel = %.15e\n",stepCount,_currTime,_deltaT,regime.c_str(), _allowed,maxVel);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::writeStep1D(PetscInt stepCount, PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer1D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_1D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer1D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer1D, PETSC_TRUE);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer1D, _D->_prevChkptTimeStep1D +1); CHKERRQ(ierr);
    }

    ierr = VecView(_time1DVec, _viewer1D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                             CHKERRQ(ierr);
    ierr = VecView(_regime1DVec, _viewer1D);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer1D);                 CHKERRQ(ierr);
    ierr = VecView(_time1DVec, _viewer1D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                             CHKERRQ(ierr);
    ierr = VecView(_regime1DVec, _viewer1D);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                          CHKERRQ(ierr);
  }


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::writeStep2D(PetscInt stepCount, PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer2D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_2D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer2D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer2D, PETSC_TRUE);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer2D, _D->_prevChkptTimeStep2D +1); CHKERRQ(ierr);
    }

    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    ierr = VecView(_regime2DVec, _viewer2D);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer2D);                 CHKERRQ(ierr);
    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    ierr = VecView(_regime2DVec, _viewer2D);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::writeCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::writeCheckpoint";
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
  ierr = VecView(_regime1DVec, _viewer_chkpt);                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep1D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "deltaT", PETSC_SCALAR, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "cycleCount", PETSC_INT, &_cycleCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "phaseCount", PETSC_INT, &_phaseCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "inDynamic", PETSC_INT, &_inDynamic); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "allowed", PETSC_INT, &_allowed); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                        CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewer_chkpt, "/time2D");            CHKERRQ(ierr);
  ierr = VecView(_time2DVec, _viewer_chkpt);                            CHKERRQ(ierr);
  ierr = VecView(_dtime2DVec, _viewer_chkpt);                           CHKERRQ(ierr);
  ierr = VecView(_regime2DVec, _viewer_chkpt);                          CHKERRQ(ierr);
  ierr = VecView(_u0, _viewer_chkpt);                                   CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep2D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                      CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/time1D");                   CHKERRQ(ierr);
  ierr = VecLoad(_time1DVec, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_dtime1DVec, viewer);                                  CHKERRQ(ierr);
  ierr = VecLoad(_regime1DVec, viewer);                                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "currTime", PETSC_SCALAR, NULL, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "cycleCount", PETSC_INT, NULL, &_cycleCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "phaseCount", PETSC_INT, NULL, &_phaseCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "inDynamic", PETSC_INT, NULL, &_inDynamic); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "allowed", PETSC_INT, NULL, &_allowed); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/time2D");                   CHKERRQ(ierr);
  ierr = VecLoad(_time2DVec, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_dtime2DVec, viewer);                                  CHKERRQ(ierr);
  ierr = VecLoad(_regime2DVec, viewer);                                 CHKERRQ(ierr);
  ierr = VecLoad(_u0, viewer);                                          CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _initTime = _currTime;
  _initDeltaT = _deltaT;
  _maxStepCount = _maxStepCount + _stepCount;
  _maxNumCycles = _maxNumCycles + _cycleCount;

  if(_inDynamic) {
    _stride1D = _stride1D_fd;
    _stride2D = _stride2D_fd;
  }
  else {
    _stride1D = _stride1D_qd;
    _stride2D = _stride2D_qd;
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::loadCheckpointSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "data_steadyState.h5";

  // load saved checkpoint data
  PetscViewer viewer;

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

  _inDynamic = false;
  _allowed = false;
  _stride1D = _stride1D_qd;
  _stride2D = _stride2D_qd;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_qd_fd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  if (_timeIntegrator.compare("IMEX")==0&& _quadImex_qd!=NULL) { ierr = _quadImex_qd->view(); }
  if (_timeIntegrator.compare("RK32")==0 && _quadEx_qd!=NULL) { ierr = _quadEx_qd->view(); }

  _material->view(_integrateTime);
  _fault_qd->view(_integrateTime);
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_PowerLaw_qd_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::writeContext";
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

  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  if (_scale.size() > 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"scale = %s\n",vector2str(_scale).c_str());CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"normType = %s\n",_normType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_qd = %i\n",_stride1D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_qd = %i\n",_stride2D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_fd = %i\n",_stride1D_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_fd = %i\n",_stride2D_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_fd_end = %i\n",_stride1D_fd_end);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_fd_end = %i\n",_stride2D_fd_end);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"trigger_qd2fd = %.15e\n",_trigger_qd2fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"trigger_fd2qd = %.15e\n",_trigger_fd2qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_qd = %.15e\n",_limit_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_fd = %.15e\n",_limit_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_stride_fd = %.15e\n",_limit_stride_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"CFL = %.15e\n",_CFL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"deltaT_fd = %.15e\n",_deltaT_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // boundary conditions for momentum balance equation
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR_qd = %s\n",_qd_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT_qd = %s\n",_qd_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL_qd = %s\n",_qd_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB_qd = %s\n",_qd_bcBType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR_fd = %s\n",_fd_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT_fd = %s\n",_fd_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL_fd = %s\n",_fd_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB_fd = %s\n",_fd_bcBType.c_str());CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  // write non-ascii context
  string outFileName = _outputDir + "data_context.h5";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewer_context); CHKERRQ(ierr);
  ierr = PetscViewerSetType(_viewer_context, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewer_context);CHKERRQ(ierr);

  _D->write(_viewer_context);
  _fault_qd->writeContext(_outputDir, _viewer_context);
  _material->writeContext(_outputDir, _viewer_context);
  if (_he != NULL) { _he->writeContext(_outputDir, _viewer_context); }
  if (_hydraulicCoupling != "no" ) { _p->writeContext(_outputDir, _viewer_context); }
  if (_grainDist != NULL) { _grainDist->writeContext(_outputDir, _viewer_context); }
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


// integrate over multiple earthquake cycles, beginning with a quasidynamic period
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::integrate(){
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime_integrateTime = MPI_Wtime();

  int isFirstPhase = 1;

  //~ // first phase
  if (_inDynamic) {
    double startTime_fd = MPI_Wtime();
    _inDynamic = true;
    integrate_fd(isFirstPhase);
    _phaseCount++;
    _dynTime += MPI_Wtime() - startTime_fd;
  }
  if (!_inDynamic) {
    double startTime_qd = MPI_Wtime();
    _inDynamic = false;
    integrate_qd(isFirstPhase);
    _phaseCount++;
    _qdTime += MPI_Wtime() - startTime_qd;
  }

  if(_currTime >= _maxTime || _stepCount >= _maxStepCount){ return 0; }


  // for all cycles after 1st cycle
  int maxPhaseCount = _maxNumCycles * 2;
  while (_phaseCount < maxPhaseCount && _stepCount <= _maxStepCount && _currTime <= _maxTime) {
    if(_inDynamic) {
      double startTime_qd = MPI_Wtime();
      _allowed = false;
      _inDynamic = false;
      prepare_fd2qd();
      integrate_qd(0);
      _qdTime += MPI_Wtime() - startTime_qd;
    }
    else {
      double startTime_fd = MPI_Wtime();
      _allowed = false;
      _inDynamic = true;
      prepare_qd2fd();
      integrate_fd(0);
      _dynTime += MPI_Wtime() - startTime_fd;
    }

    _phaseCount++;
    _cycleCount = floor(_phaseCount/2);
  }

  _integrateTime += MPI_Wtime() - startTime_integrateTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// integrate through quasidynamic period
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::integrate_qd(int isFirstPhase)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update momentum balance equation boundary conditions
  ierr = _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType); CHKERRQ(ierr);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

  delete _quadEx_qd; _quadEx_qd = NULL;
  delete _quadImex_qd; _quadImex_qd = NULL;

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    _quadEx_qd = new FEuler(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadEx_qd = new RK32(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    _quadEx_qd = new RK43(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    _quadImex_qd = new RK32_WBE(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex_qd = new RK43_WBE(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  // integrate
  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex_qd->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadImex_qd->setTimeStepBounds(_minDeltaT,_maxDeltaT);
    _quadImex_qd->setTimeRange(_currTime,_maxTime);
    _quadImex_qd->setInitialStepCount(_stepCount);
    _quadImex_qd->setInitialConds(_varQSEx,_varIm);
    _quadImex_qd->setToleranceType(_normType);
    _quadImex_qd->setErrInds(_timeIntInds,_scale);

    if (isFirstPhase == 1 && _D->_restartFromChkpt) { ierr = _quadImex_qd->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadImex_qd->integrate(this); CHKERRQ(ierr);

    std::map<string,Vec> varOut = _quadImex_qd->_varEx;
    for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
      VecCopy(varOut[it->first],_varQSEx[it->first]);
    }
  }
  else {
    _quadEx_qd->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadEx_qd->setTimeStepBounds(_minDeltaT,_maxDeltaT);
    _quadEx_qd->setTimeRange(_currTime,_maxTime);
    _quadEx_qd->setInitialStepCount(_stepCount);
    _quadEx_qd->setToleranceType(_normType);
    _quadEx_qd->setInitialConds(_varQSEx);
    _quadEx_qd->setErrInds(_timeIntInds,_scale);

    if (isFirstPhase == 1 && _D->_restartFromChkpt) { ierr = _quadEx_qd->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadEx_qd->integrate(this); CHKERRQ(ierr);

    std::map<string,Vec> varOut = _quadEx_qd->_var;
    for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
      VecCopy(varOut[it->first],_varQSEx[it->first]);
    }
  }

  delete _quadEx_qd; _quadEx_qd = NULL;
  delete _quadImex_qd; _quadImex_qd = NULL;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// integrate through fully dynamic period
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::integrate_fd(int isFirstPhase)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::integrate_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update momentum balance equation boundary conditions
  ierr = _material->changeBCTypes(_mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType); CHKERRQ(ierr);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

  delete _quadWaveEx; _quadWaveEx = NULL;

  // initialize time integrator
  _quadWaveEx = new OdeSolver_WaveEq(_maxStepCount,_currTime,_maxTime,_deltaT_fd);
  _quadWaveEx->setInitialConds(_varFD);
  _quadWaveEx->setInitialStepCount(_stepCount);

  if (isFirstPhase == 1 && _D->_restartFromChkpt) { ierr = _quadWaveEx->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

  ierr = _quadWaveEx->integrate(this);CHKERRQ(ierr);

  std::map<string,Vec> varOut = _quadWaveEx->_var;
  for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
    VecCopy(varOut[it->first],_varFD[it->first]);
  }

  delete _quadWaveEx; _quadWaveEx = NULL;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// move from a fully dynamic phase to a quasidynamic phase
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::prepare_fd2qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::prepare_fd2qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ // Force writing output
  //~ PetscInt stopIntegration = 0;
  //~ if(_stride1D > 0){ _stride1D = 1; }
  //~ if(_stride2D > 0){ _stride2D = 1; }
  //~ timeMonitor(_currTime, _deltaT,_stepCount, stopIntegration);

  // switch strides to qd values
  _stride1D = _stride1D_qd;
  _stride2D = _stride2D_qd;


  // update explicitly integrated variables
  VecCopy(_fault_fd->_psi, _varQSEx["psi"]);
  VecCopy(_fault_fd->_slip, _varQSEx["slip"]);

  // update implicitly integrated T
  if (_evolveTemperature == 1) { VecCopy(_varFD["Temp"],_varIm["Temp"]); } // if solving the heat equation

  // update fault internal variables
  ierr = VecCopy(_fault_fd->_psi,       _fault_qd->_psi); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slipVel,   _fault_qd->_slipVel); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slip,      _fault_qd->_slip); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slip0,     _fault_qd->_slip0); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_strength,  _fault_qd->_strength); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_tauP,      _fault_qd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_tauQSP,    _fault_qd->_tauQSP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_prestress, _fault_qd->_prestress); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_sN,        _fault_qd->_sN); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_sNEff,     _fault_qd->_sNEff); CHKERRQ(ierr);
  if (_fault_qd->_stateLaw.compare("flashHeating") == 0) {
    ierr = VecCopy(_fault_fd->_T,       _fault_qd->_T); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_Tw,      _fault_qd->_Tw); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_Vw,      _fault_qd->_Vw); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// move from a fully dynamic phase to a quasidynamic phase
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::prepare_qd2fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::prepare_qd2fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // switch strides to qd values
  _stride1D = _stride1D_fd;
  _stride2D = _stride1D_fd;

  // save current variables as n-1 time step
  VecCopy(_fault_qd->_slip,_varFDPrev["slip"]);
  VecCopy(_fault_qd->_psi,_varFDPrev["psi"]);
  VecCopy(_material->_u,_varFDPrev["u"]);
  if (_thermalCoupling.compare("no")!=0 ) { VecCopy(_varIm["Temp"], _varFDPrev["Temp"]); } // if solving the heat equation

  // take 1 quasidynamic time step to compute variables at time n
  _inDynamic = 0;
  integrate_singleQDTimeStep();
  _inDynamic = 1;

  // update varFD to reflect latest values
  VecCopy(_fault_qd->_slip,_varFD["slip"]);
  VecCopy(_fault_qd->_psi,_varFD["psi"]);
  VecCopy(_material->_u,_varFD["u"]);
  if (_thermalCoupling.compare("no")!=0 ) { VecCopy(_varIm["Temp"], _varFD["Temp"]); } // if solving the heat equation

  // now change u to du
  VecAXPY(_varFD["u"],-1.0,_varFDPrev["u"]);
  VecCopy(_varFDPrev["u"],_u0);
  VecSet(_varFDPrev["u"],0.0);


  // update fault internal variables
  ierr = VecCopy(_fault_qd->_psi,       _fault_fd->_psi); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_slipVel,   _fault_fd->_slipVel); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_slip,      _fault_fd->_slip); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_slip,      _fault_fd->_slip0); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_strength); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_tau0); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauP,      _fault_fd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauQSP,    _fault_fd->_tauQSP); CHKERRQ(ierr);
  if (_fault_qd->_stateLaw.compare("flashHeating") == 0) {
    ierr = VecCopy(_fault_qd->_T,         _fault_fd->_T); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_Tw,         _fault_fd->_Tw); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_Vw,         _fault_fd->_Vw); CHKERRQ(ierr);
  }


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// take 1 quasidynamic time step to set up varFDPrev and varFD
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::integrate_singleQDTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::integrate_singleQDTimeStep()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  OdeSolver      *quadEx = NULL; // explicit time stepping
  OdeSolverImex  *quadImex = NULL; // implicit time stepping

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    quadEx = new FEuler(1,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    quadEx = new RK32(1,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    quadEx = new RK43(1,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    quadImex = new RK32_WBE(1,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    quadImex = new RK43_WBE(1,_maxTime,_deltaT_fd,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  // integrate
  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    quadImex->setTolerance(_timeStepTol);CHKERRQ(ierr);
    quadImex->setTimeStepBounds(_deltaT_fd,_deltaT_fd);
    quadImex->setTimeRange(_currTime,_currTime+_deltaT_fd);
    quadImex->setInitialStepCount(_stepCount);
    quadImex->setInitialConds(_varQSEx,_varIm);
    quadImex->setToleranceType(_normType);
    quadImex->setErrInds(_timeIntInds,_scale);

    ierr = quadImex->integrate(this); CHKERRQ(ierr);
  }
  else {
    quadEx->setTolerance(_timeStepTol);CHKERRQ(ierr);
    quadEx->setTimeStepBounds(_deltaT_fd,_deltaT_fd);
    quadEx->setTimeRange(_currTime,_currTime+_deltaT_fd);
    quadEx->setInitialStepCount(_stepCount);
    quadEx->setToleranceType(_normType);
    quadEx->setInitialConds(_varQSEx);
    quadEx->setErrInds(_timeIntInds,_scale);

    ierr = quadEx->integrate(this); CHKERRQ(ierr);
  }

  delete quadEx;
  delete quadImex;


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// purely explicit adaptive time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update fields based on varEx

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType == "symmFault" || _qd_bcLType == "rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType == "remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  _material->updateFields(time,varEx);
  _fault_qd->updateFields(time,varEx);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling!="no") {
    _p->updateFields(time,varEx);
  }
  if ( _evolveGrainSize==1 && varEx.find("grainSize") != varEx.end() ) {
    _grainDist->updateFields(time,varEx);
  }
  if ( _grainSizeEvCoupling == "coupled" ) { _material->updateGrainSize(_grainDist->_d); }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->d_dt(time,varEx,dvarEx);
  }

  // compute grain size rate, or value from either piezometric relation or steady-state
  if ( _evolveGrainSize==1 && varEx.find("grainSize") != varEx.end() ) {
    _grainDist->d_dt(dvarEx["grainSize"],varEx.find("grainSize")->second,_material->_sdev,_material->_dgVdev_disl,_material->_T);
  }
  else if ( _evolveGrainSize==1 && _grainDist->_grainSizeEvType == "piezometer") {
    _grainDist->computeGrainSizeFromPiez(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }
  else if ( _evolveGrainSize==1 && _grainDist->_grainSizeEvType == "steadyState") {
    _grainDist->computeSteadyStateGrainSize(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }

  // update fields on fault from other classes
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  if (_hydraulicCoupling == "coupled") { _fault_qd->setSNEff(_p->_p); }

  // rates for fault
  if (_qd_bcLType == "symmFault" || _qd_bcLType == "rigidFault") {
    ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state
  }
  else {
    VecSet(dvarEx["psi"],0.);
    VecSet(dvarEx["slip"],0.);
  }

  return ierr;
}



// implicit/explicit adaptive time stepping
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::d_dt IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType=="symmFault" || _qd_bcLType=="rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType=="remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  _material->updateFields(time,varEx);
  _fault_qd->updateFields(time,varEx);

  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->updateFields(time,varEx,varImo);
  }
  if ( _evolveGrainSize==1 && varEx.find("grainSize") != varEx.end() ) {
    _grainDist->updateFields(time,varEx);
  }
  if ( _grainSizeEvCoupling == "coupled" ) { _material->updateGrainSize(_grainDist->_d); }


  // update temperature in momBal and fault
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling == "coupled") {
    _fault_qd->updateTemperature(varImo.find("Temp")->second);
    _material->updateTemperature(varImo.find("Temp")->second);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault_qd->setSNEff(_p->_p);
  }


  // compute grain size rate, or value from either piezometric relation or steady-state
  if ( _evolveGrainSize==1 && varEx.find("grainSize") != varEx.end() ) {
    _grainDist->d_dt(dvarEx["grainSize"],varEx.find("grainSize")->second,_material->_sdev,_material->_dgVdev_disl,_material->_T);
  }
  else if ( _evolveGrainSize==1 && _grainDist->_grainSizeEvType == "piezometer") {
    _grainDist->computeGrainSizeFromPiez(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }
  else if ( _evolveGrainSize==1 && _grainDist->_grainSizeEvType == "steadyState") {
    _grainDist->computeSteadyStateGrainSize(_material->_sdev, _material->_dgVdev_disl, _material->_T);
  }


  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);
  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
  }

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  //~ ierr = _fault_qd->setTauQS(sxy); CHKERRQ(ierr); // new
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  if (_qd_bcLType == "symmFault" || _qd_bcLType=="rigidFault") {
    ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state
  }
  else {
    VecSet(dvarEx["psi"],0.);
    VecSet(dvarEx["slip"],0.);
  }


  // heat equation
  if (varIm.find("Temp") != varIm.end()) {

    // frictional shear heating source terms
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault_qd->_tauP;

    // compute viscous strain rate that contributes to viscous shear heating:
    Vec dgV_sh;
    VecDuplicate(_material->_dgVdev,&dgV_sh);
    if ( _grainSizeEvCoupling!="no") {
      // relevant visc strain rate = (total) - (portion contributing to grain size reduction)
      VecPointwiseMult(dgV_sh,_grainDist->_f,_material->_dgVdev_disl);
      VecScale(dgV_sh,-1.0);
      VecAXPY(dgV_sh,1.0,_material->_dgVdev);
    }
    else {
      VecCopy(_material->_dgVdev,dgV_sh);
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


// fully dynamic: purely explicit time stepping
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::d_dt fd explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // momentum balance equation except for fault boundary
  propagateWaves(time, deltaT, varNext, var, varPrev);

  // effect of fault: update body u from fault u
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);
  ierr = VecScatterBegin(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  // compute stresses and effective viscosity
  VecCopy(varNext.find("u")->second, _material->_u);
  VecAXPY(_material->_u,1.0,_u0);
  ierr = _material->computeTotalStrains(); CHKERRQ(ierr);
  ierr = _material->computeStresses(); CHKERRQ(ierr);
  ierr = _material->computeViscosity(_material->_effViscCap); CHKERRQ(ierr);

  // update fault shear stress and quasi-static shear stress
  Vec sxy,sxz,sdev; _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  // update shear stress: tau = tauQS - eta_rad * slipVel
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = -eta_rad*slipVel + tauQSP

  // update surface displacement
  ierr = _material->setSurfDisp(); CHKERRQ(ierr);


  // update boundary conditions so they are consistent during output
  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  // explicitly integrate heat equation using forward Euler
  if (_evolveTemperature == 1) {
    // frictional shear heating source terms
    Vec V = _fault_fd->_slipVel;
    Vec tau = _fault_fd->_tauP;

    // compute viscous strain rate that contributes to viscous shear heating:
    Vec dgV_sh;
    VecDuplicate(_material->_dgVdev,&dgV_sh);
    if ( _evolveGrainSize == 1) {
      // relevant visc strain rate = (total) - (portion contributing to grain size reduction)
      VecPointwiseMult(dgV_sh,_grainDist->_f,_material->_dgVdev_disl);
      VecScale(dgV_sh,-1.0);
      VecAXPY(dgV_sh,1.0,_material->_dgVdev);
    }
    else {
      VecCopy(_material->_dgVdev,dgV_sh);
    }

    Vec Tn = var.find("Temp")->second;
    Vec dTdt; VecDuplicate(Tn,&dTdt);
    ierr = _he->d_dt(time,V,tau,sdev,dgV_sh,Tn,dTdt); CHKERRQ(ierr);
    VecWAXPY(varNext["Temp"], deltaT, dTdt, Tn); // Tn+1 = deltaT * dTdt + Tn
    _he->setTemp(varNext["Temp"]); // keep heat equation T up to date
    VecDestroy(&dTdt);
    VecDestroy(&dgV_sh);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: IMEX time stepping
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev, map<string,Vec>& varIm, const map<string,Vec>& varImPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::d_dt fd imex";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // momentum balance equation except for fault boundary
  propagateWaves(time, deltaT, varNext, var, varPrev);

  // effect of fault: update body u from fault u
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);
  ierr = VecScatterBegin(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  // compute stresses and effective viscosity
  VecCopy(varNext.find("u")->second, _material->_u);
  VecAXPY(_material->_u,1.0,_u0);
  ierr = _material->computeTotalStrains(); CHKERRQ(ierr);
  ierr = _material->computeStresses(); CHKERRQ(ierr);
  ierr = _material->computeViscosity(_material->_effViscCap); CHKERRQ(ierr);

  // update fault shear stress and quasi-static shear stress
  Vec sxy,sxz,sdev; _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  // update shear stress: tau = tauQS - eta_rad * slipVel
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = -eta_rad*slipVel + tauQSP

  // update surface displacement
  ierr = _material->setSurfDisp(); CHKERRQ(ierr);


  // update boundary conditions so they are consistent during output
  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  // put implicitly integrated variables here

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // compute source terms to rhs: d/dy(mu*gVxy) + d/dz(mu*gVxz)
  Vec viscSource;
  ierr = VecDuplicate(_material->_gVxy,&viscSource);CHKERRQ(ierr);
  ierr = VecSet(viscSource,0.0);CHKERRQ(ierr);
  ierr = _material->computeViscStrainSourceTerms(viscSource); CHKERRQ(ierr);

  _material->setRHS();
  ierr = VecAXPY(_material->_rhs,1.0,viscSource); CHKERRQ(ierr);
  VecDestroy(&viscSource);

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) { VecAXPY(_material->_rhs,1.0,_forcingTerm); }


  // solve for displacement
  ierr = _material->computeU(); CHKERRQ(ierr);

  // update stresses, viscosity, and set shear traction on fault
  ierr = _material->computeTotalStrains(); CHKERRQ(ierr);
  ierr = _material->computeStresses(); CHKERRQ(ierr);
  ierr = _material->computeViscosity(_material->_effViscCap); CHKERRQ(ierr);

  // compute viscous strain rates
  ierr = _material->computeViscStrainRates(time); CHKERRQ(ierr);
  VecCopy(_material->_dgVxy,dvarEx["gVxy"]);
  VecCopy(_material->_dgVxz,dvarEx["gVxz"]);

  return ierr;
}


// fully dynamic: off-fault portion of the momentum balance equation
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::propagateWaves(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::propagateWaves";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

double startPropagation = MPI_Wtime();

  // compute D2u = (Dyy+Dzz)*u
  Vec D2u, temp;
  VecDuplicate(*_y, &D2u);
  VecDuplicate(*_y, &temp);
  Mat A; _material->_sbp->getA(A);
  ierr = MatMult(A, var.find("u")->second, temp);
  ierr = _material->_sbp->Hinv(temp, D2u);
  VecDestroy(&temp);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
      Mat J,Jinv,qy,rz,yq,zr;
      ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
      Vec temp;
      VecDuplicate(D2u, &temp);
      MatMult(Jinv, D2u, temp);
      VecCopy(temp, D2u);
      VecDestroy(&temp);
  }
  ierr = VecScatterBegin(*_body2fault, D2u, _fault_fd->_d2u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, D2u, _fault_fd->_d2u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);


  // Propagate waves and compute displacement at the next time step
  // includes boundary conditions except for fault

  PetscInt       Ii,Istart,Iend;
  PetscScalar   *uNextA; // changed in this loop
  const PetscScalar   *u, *uPrev, *d2u, *ay, *rho; // unchchanged in this loop
  ierr = VecGetArray(varNext["u"], &uNextA);
  ierr = VecGetArrayRead(var.find("u")->second, &u);
  ierr = VecGetArrayRead(varPrev.find("u")->second, &uPrev);
  ierr = VecGetArrayRead(_ay, &ay);
  ierr = VecGetArrayRead(D2u, &d2u);
  ierr = VecGetArrayRead(_material->_rho, &rho);

  ierr = VecGetOwnershipRange(varNext["u"],&Istart,&Iend);CHKERRQ(ierr);
  PetscInt       Jj = 0;
  for (Ii = Istart; Ii < Iend; Ii++){
    PetscScalar c1 = deltaT*deltaT / rho[Jj];
    PetscScalar c2 = deltaT*ay[Jj] - 1.0;
    PetscScalar c3 = deltaT*ay[Jj] + 1.0;

    uNextA[Jj] = (c1*d2u[Jj] + 2.*u[Jj] + c2*uPrev[Jj]) / c3;
    Jj++;
  }
  ierr = VecRestoreArray(varNext["u"], &uNextA);
  ierr = VecRestoreArrayRead(var.find("u")->second, &u);
  ierr = VecRestoreArrayRead(varPrev.find("u")->second, &uPrev);
  ierr = VecRestoreArrayRead(_ay, &ay);
  ierr = VecRestoreArrayRead(D2u, &d2u);
  ierr = VecRestoreArrayRead(_material->_rho, &rho);

  VecDestroy(&D2u);

_propagateTime += MPI_Wtime() - startPropagation;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute alphay and alphaz for use in time stepping routines
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::computePenaltyVectors()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::computePenaltyVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar h11y, h11z;
  _material->_sbp->geth11(h11y, h11z);

  Vec alphay,alphaz;
  VecDuplicate(*_y, &alphay); VecSet(alphay,h11y);
  VecDuplicate(*_y, &alphaz); VecSet(alphaz,h11z);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr);
    Vec temp1, temp2;
    VecDuplicate(alphay, &temp1);
    VecDuplicate(alphay, &temp2);
    MatMult(yq, alphay, temp1);
    MatMult(zr, alphaz, temp2);
    VecCopy(temp1, alphay);
    VecCopy(temp2, alphaz);
    VecDestroy(&temp1);
    VecDestroy(&temp2);
  }
  VecScatterBegin(_D->_scatters["body2L"], alphay, _fault_fd->_alphay, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], alphay, _fault_fd->_alphay, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(&alphay);
  VecDestroy(&alphaz);

  // compute vectors
  VecDuplicate(*_y, &_ay);
  VecSet(_ay, 0.0);

  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_ay,&Istart,&Iend);
  PetscScalar *ay;
  VecGetArray(_ay,&ay);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    ay[Jj] = 0;
    if ( (Ii/_D->_Nz == _D->_Ny-1) && (_fd_bcRType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11y; }
    if ( (Ii%_D->_Nz == 0) && (_fd_bcTType.compare("outGoingCharacteristics") == 0 )) { ay[Jj] += 0.5 / h11z; }
    if ( ((Ii+1)%_D->_Nz == 0) && (_fd_bcBType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11z; }

    if ( (Ii/_D->_Nz == 0) && ( _fd_bcLType.compare("outGoingCharacteristics") == 0 ||
      _fd_bcLType.compare("symmFault") == 0 || _fd_bcLType.compare("rigidFault") == 0 ) )
    Jj++;
  }
  VecRestoreArray(_ay,&ay);

  ierr = VecPointwiseMult(_ay, _ay, _material->_cs); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// estimate steady state shear stress on fault, store in varSS
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::guessTauSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::guessTauSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // steady-state shear stress on fault
  bool loadTauSS = 0;
  ierr = loadVecFromInputFile(_fault_qd->_tauP,_inputDir,"tauSS",loadTauSS); CHKERRQ(ierr);

  // if steady-state shear stress not provided: tauSS = min(tauRS,tauVisc)
  if (loadTauSS == 0) {
    // viscous strength of material, evaluated only at fault
    Vec tauVisc = NULL;
    VecDuplicate(_fault_qd->_tauP,&tauVisc);
    VecScatterBegin(_D->_scatters["body2L"], _material->_effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2L"], _material->_effVisc, tauVisc, INSERT_VALUES, SCATTER_FORWARD);
    VecScale(tauVisc,_gss_t);

    VecPointwiseMin(_fault_qd->_tauP,_fault_qd->_tauP,tauVisc);
    VecDestroy(&tauVisc);
  }
  ierr = VecCopy(_fault_qd->_tauP,_fault_qd->_tauQSP); CHKERRQ(ierr);
  ierr = _fault_qd->computeVel(); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// for solving fixed point iteration problem, with or without the heat equation
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::integrateSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::integrateSS";
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
    PetscPrintf(PETSC_COMM_WORLD,"SS_index = %i\n",_SS_index);

    // brute force time integrate for steady-state shear stress the fault
    solveSStau(_SS_index);

    // iterate to find effective viscosity etc
    solveSSViscoelasticProblem(_SS_index);

    // find steady-state temperature
    if (_computeSSTemperature == 1) { solveSSHeatEquation(_SS_index); }
    if (_thermalCoupling == "coupled") {
      _material->updateTemperature(_varSS["Temp"]);
      _fault_qd->updateTemperature(_varSS["Temp"]);
      _fault_fd->updateTemperature(_varSS["Temp"]);
    }

    writeSS(_SS_index);
    _SS_index++;
  }



  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::initiateIntegrandSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::initiateIntegrandSS";
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
  _varSS["tau"] = _fault_qd->_tauP;
  _varSS["slipVel"] = _fault_qd->_slipVel;
  _varSS["psi"] = _fault_qd->_psi;

  if (_varSS.find("Temp") == _varSS.end() ) {
    Vec Temp; VecDuplicate(_he->_T,&Temp); VecCopy(_he->_T,Temp); _varSS["Temp"] = Temp;
  }


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveSS(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::solveSS";
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
  _varSS["tau"] = _fault_qd->_tauP;
  _varSS["slipVel"] = _fault_qd->_slipVel;
  _varSS["psi"] = _fault_qd->_psi;

  if (_varSS.find("Temp") == _varSS.end() ) {
    Vec Temp; VecDuplicate(_he->_T,&Temp); VecCopy(_he->_T,Temp); _varSS["Temp"] = Temp;
  }

  if (!_D->_restartFromChkptSS) {
    // estimate steady-state conditions for fault, material based on strain rate
    _fault_qd->guessSS(_vL); // sets: slipVel, psi, tau
    _material->guessSteadyStateEffVisc(_gss_t);

    // estimate steady-state shear stress at y = 0
    guessTauSS(_varSS);
    solveSSViscoelasticProblem(Jj); // converge to steady state eta etc

    if (_computeSSTemperature == 1) { solveSSHeatEquation(Jj); }
    if (_thermalCoupling == "coupled") {
      _material->updateTemperature(_varSS["Temp"]);
      _fault_qd->updateTemperature(_varSS["Temp"]);
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveSStau(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::solveSStau";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

// create output path with Jj appended on end
  //~ char buff[5]; sprintf(buff,"%04d",Jj); _outputDir = baseOutDir + string(buff) + "_";
  //~ PetscPrintf(PETSC_COMM_WORLD,"baseDir = %s\n\n",_outputDir.c_str());

  _stepCount = 0;
  _initTime = 0;
  _currTime = _initTime;

  // ensure varQSSEx, varIm, and varFD contain updated versions of all fields
  ierr = _material->initiateIntegrand(_initTime,_varQSEx); CHKERRQ(ierr);
  ierr = _fault_qd->initiateIntegrand(_initTime,_varQSEx); CHKERRQ(ierr);
  if (_evolveTemperature == 1) { ierr = _he->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr); }
  if (_evolveGrainSize == 1) { ierr = _grainDist->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr); }

  if (_varQSEx.find("slip") != _varQSEx.end() ) { VecCopy(_material->_bcL,_varQSEx["slip"]); }
  else { Vec varSlip; VecDuplicate(_material->_bcL,&varSlip); VecCopy(_material->_bcL,varSlip); _varQSEx["slip"] = varSlip; }
  ierr = VecScale(_varQSEx["slip"],_faultTypeScale); CHKERRQ(ierr);
  ierr = VecCopy(_varQSEx["slip"],_fault_qd->_slip); CHKERRQ(ierr);
  ierr = VecCopy(_varQSEx["slip"],_fault_fd->_slip); CHKERRQ(ierr);
  initiateIntegrand_fd();


  delete _quadEx_qd; _quadEx_qd = NULL;
  delete _quadImex_qd; _quadImex_qd = NULL;
  delete _quadWaveEx; _quadWaveEx = NULL;

  ierr = integrate(); CHKERRQ(ierr);

  delete _quadEx_qd; _quadEx_qd = NULL;
  delete _quadImex_qd; _quadImex_qd = NULL;
  delete _quadWaveEx; _quadWaveEx = NULL;
  PetscViewerDestroy(&_viewer1D); _viewer1D = NULL;
  PetscViewerDestroy(&_viewer2D); _viewer2D = NULL;

  // update momentum balance equation boundary conditions to be compatible with steady-state solve step
  _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS); CHKERRQ(ierr);

  // identify which phase the integrate ended in and sync up fault_qd and fault_fd
  if (_inDynamic == true) {
    ierr = VecCopy(_fault_fd->_psi,       _fault_qd->_psi); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_slipVel,   _fault_qd->_slipVel); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_slip,      _fault_qd->_slip); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_strength,  _fault_qd->_strength); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_tauP,      _fault_qd->_tauP); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_tauQSP,    _fault_qd->_tauQSP); CHKERRQ(ierr);
    if (_fault_qd->_stateLaw.compare("flashHeating") == 0) {
      ierr = VecCopy(_fault_fd->_T,       _fault_qd->_T); CHKERRQ(ierr);
      ierr = VecCopy(_fault_fd->_Tw,      _fault_qd->_Tw); CHKERRQ(ierr);
      ierr = VecCopy(_fault_fd->_Vw,      _fault_qd->_Vw); CHKERRQ(ierr);
    }
  }
  else {

    ierr = VecCopy(_fault_qd->_psi,       _fault_fd->_psi); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slipVel,   _fault_fd->_slipVel); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slip,      _fault_fd->_slip); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_slip,      _fault_fd->_slip0); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_strength); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_tau0); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_tauP,      _fault_fd->_tauP); CHKERRQ(ierr);
    ierr = VecCopy(_fault_qd->_tauQSP,    _fault_fd->_tauQSP); CHKERRQ(ierr);
    if (_fault_qd->_stateLaw.compare("flashHeating") == 0) {
      ierr = VecCopy(_fault_qd->_T,         _fault_fd->_T); CHKERRQ(ierr);
      ierr = VecCopy(_fault_qd->_Tw,         _fault_fd->_Tw); CHKERRQ(ierr);
      ierr = VecCopy(_fault_qd->_Vw,         _fault_fd->_Vw); CHKERRQ(ierr);
    }
  }

  /* This may have improved convergence rate for grain size evolution.
   * Not sure it makes sense for simulations with flash heating.
  // impose ceiling on fault velocity: slipVel <= vL
  PetscScalar *V;
  VecGetArray(_fault_qd->_slipVel,&V);
  PetscInt Kk = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_fault_qd->_slipVel,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    V[Kk] = min(V[Kk],_vL);
    Kk++;
  }
  VecRestoreArray(_fault_qd->_slipVel,&V);

  // compute frictional strength of fault_qd based on updated slip velocity
  strength_psi_Vec(_fault_qd->_strength, _fault_qd->_psi, _fault_qd->_slipVel, _fault_qd->_a, _fault_qd->_sNEff, _fault_qd->_v0);
  ierr = VecCopy(_fault_qd->_strength,_fault_qd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_slipVel,_fault_qd->_tauQSP); CHKERRQ(ierr); // V -> tauQS
  ierr = VecPointwiseMult(_fault_qd->_tauQSP,_fault_qd->_eta_rad,_fault_qd->_tauQSP); CHKERRQ(ierr); // tauQS = V * eta_rad
  ierr = VecAYPX(_fault_qd->_tauQSP,1.0,_fault_qd->_tauP); // tauQS = tau + V*eta_rad

  // update fault_fd as well
  ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_strength); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauP,      _fault_fd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauQSP,    _fault_fd->_tauQSP); CHKERRQ(ierr);
  */


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// converge to steady state: effective viscosity, sxy, sxz, gVxy, gVxz, gVxy_t, gVxz_t, u
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveSSViscoelasticProblem(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::solveSSViscoelasticProblem";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set up KSP for steady-state solution
  ierr = _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,"Neumann",_mat_qd_bcBType); CHKERRQ(ierr);
  Mat A;
  _material->_sbp->getA(A);
  _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS);

  // set up rhs vector containing boundary condition data
  VecCopy(_varSS["tau"],_material->_bcL);
  VecSet(_material->_bcR,_vL/2.);
  string  _mat_bcTType_SS = "Neumann";

  // loop over effective viscosity
  Vec effVisc_old; VecDuplicate(_varSS["effVisc"],&effVisc_old);

  Vec temp; VecDuplicate(_varSS["effVisc"],&temp); VecSet(temp,0.);
  double err = 1e10;
  int Ii = 0;
  while (Ii < _maxSSIts_effVisc && err >= _atolSS_effVisc) {
    VecCopy(_varSS["effVisc"],effVisc_old);

    ierr = _material->setSSRHS(_varSS,"Dirichlet",_mat_bcTType_SS,"Neumann","Neumann");
    ierr = _material->updateSSa(_varSS); CHKERRQ(ierr); // compute v, viscous strain rates

    // update grain size
    if (_computeSSGrainSize == 1) { solveSSGrainSize(Jj); }
    if (_grainSizeEvCoupling == "coupled") { _material->updateGrainSize(_varSS["grainSize"]); }

    _material->computeViscosity(_material->_effViscCap); // new viscosity

    // update effective viscosity: log10(accepted viscosity) = (1-f)*log10(old viscosity) + f*log10(new viscosity):
    MyVecLog10AXPBY(temp,1.-_fss_EffVisc,effVisc_old,_fss_EffVisc,_varSS["effVisc"]);
    ierr = VecCopy(temp,_varSS["effVisc"]); CHKERRQ(ierr);

    // evaluate convergence of this iteration
    err = computeMaxDiff_scaleVec1(effVisc_old,_varSS["effVisc"]); // total eff visc

    PetscPrintf(PETSC_COMM_WORLD,"    effective viscosity loop: %i %e\n",Ii,err);
    Ii++;
  }
  VecDestroy(&effVisc_old);
  VecDestroy(&temp);


  // update u, gVxy, gVxz, boundary conditions based on effective viscosity
  ierr = _material->updateSSb(_varSS,_initTime); CHKERRQ(ierr); // solve for gVxy, gVxz
  ierr = setSSBCs(); CHKERRQ(ierr); // update u, boundary conditions to be positive, consistent with varEx

  // update shear stress on fault
  ierr = VecScatterBegin(*_body2fault, _material->_sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _material->_sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauQSP,_fault_qd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauQSP,_fault_qd->_strength); CHKERRQ(ierr);

  // update fault_fd as well
  ierr = VecCopy(_fault_qd->_strength,  _fault_fd->_strength); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauP,      _fault_fd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_tauQSP,    _fault_fd->_tauQSP); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// solve steady-state heat equation for temperature
// update temperature using damping:
//   Tnew = (1-f)*Told + f*Tnew
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveSSHeatEquation(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::solveSSHeatEquation";
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
  _he->computeSteadyStateTemp(_currTime,_varSS["slipVel"],_fault_qd->_tauP,_material->_sdev,dgV_sh,_varSS["Temp"]);
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
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::solveSSGrainSize(const PetscInt Jj)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::solveSSGrainSize";
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

PetscErrorCode StrikeSlip_PowerLaw_qd_fd::writeSS(const int Ii)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::writeSS";
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
  ierr = _fault_qd->writeStep(_viewerSS);                               CHKERRQ(ierr);
  if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_viewerSS); CHKERRQ(ierr);}
  if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

  ierr = _material->writeStep2D(_viewerSS);                             CHKERRQ(ierr);
  if (_computeSSTemperature == 1) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }
  if (_computeSSGrainSize == 1) { ierr =  _grainDist->writeStep(_viewerSS); CHKERRQ(ierr); }

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// update the boundary conditions based on new steady state u
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::setSSBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_qd_fd::setSSBCs";
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

  if (_varQSEx.find("slip") != _varQSEx.end() ) { VecCopy(uL,_varQSEx["slip"]); }
  else {
    Vec slip;
    VecDuplicate(_material->_bcL,&slip);
    VecCopy(uL,slip);
    _varQSEx["slip"] = slip;
  }
  if (_qd_bcLType == "symmFault") {
    VecScale(_varQSEx["slip"],2.);
  }

  VecDestroy(&uL);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// constructs the body forcing term for an ice stream
// includes allocation of memory for this forcing term
PetscErrorCode StrikeSlip_PowerLaw_qd_fd::constructIceStreamForcingTerm()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_PowerLaw_qd_fd::constructIceStreamForcingTerm";
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


  //~ // compute forcing term for momentum balance equation
  //~ // forcing = - tau_ss / Ly
  //~ Vec tauSS = NULL;
  //~ _fault_qd->computeTauRS(tauSS,_vL);
  //~ VecScale(tauSS,-1./_D->_Ly);

  //~ VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  //~ MatMult(MapV,tauSS,_forcingTerm);

  //~ MatDestroy(&MapV);
  //~ VecDestroy(&tauSS);

  // compute forcing term for momentum balance equation
  // forcing = (1/Ly) * (tau_ss + eta_rad*V_ss)
  //~ Vec tauSS = NULL,radDamp=NULL,V=NULL;
  //~ VecDuplicate(_fault_qd->_eta_rad,&V); VecSet(V,_vL);
  //~ VecDuplicate(_fault_qd->_eta_rad,&radDamp); VecPointwiseMult(radDamp,_fault_qd->_eta_rad,V);
  //~ _fault_qd->computeTauRS(tauSS,_vL);
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
