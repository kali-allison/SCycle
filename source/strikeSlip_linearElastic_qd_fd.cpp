#include "strikeSlip_linearElastic_qd_fd.hpp"

#define FILENAME "strikeSlip_linearElastic_qd_fd.cpp"

using namespace std;


StrikeSlip_LinearElastic_qd_fd::StrikeSlip_LinearElastic_qd_fd(Domain&D)
  : _D(&D),_delim(D._delim),
    _inputDir(D._inputDir),_outputDir(D._outputDir),_vL(1e-9),
    _thermalCoupling("no"),_heatEquationType("transient"),
    _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
    _guessSteadyStateICs(0),_computeSSMomBal(0),_forcingType("no"),_faultTypeScale(2.0),
    _evolveTemperature(0),_computeSSHeatEq(0),
    _cycleCount(0),_maxNumCycles(1e3),_phaseCount(0),
    _deltaT(-1), _CFL(-1),_y(&D._y),_z(&D._z),_Req(NULL),
    _inDynamic(false),_allowed(false),
    _trigger_qd2fd(1e-3), _trigger_fd2qd(1e-3),
    _limit_qd(10*_vL), _limit_fd(1e-1),_limit_stride_fd(-1),_u0(NULL),
    _timeIntegrator("RK43"),_timeControlType("PID"),
    _stride1D(10),_stride2D(10),_strideChkpt_qd(1e4), _strideChkpt_fd(1e4),
    _stride1D_qd(10),_stride2D_qd(10),_stride1D_fd(10),
    _stride2D_fd(10),_stride1D_fd_end(10),_stride2D_fd_end(10),
    _maxStepCount(1e8),
    _initTime(0),_currTime(0),_minDeltaT(1e-3),_maxDeltaT(1e10),_maxTime(1e15),
    _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
    _chkptTimeStep1D(0), _chkptTimeStep2D(0),
    _JjSSVec(NULL),
    _time1DVec(NULL), _dtime1DVec(NULL),_time2DVec(NULL), _dtime2DVec(NULL),_regime1DVec(NULL),_regime2DVec(NULL),
    _viewer_context(NULL),_viewer1D(NULL),_viewer2D(NULL),_viewerSS(NULL),_viewer_chkpt(NULL),
    _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),
    _startTime(MPI_Wtime()),_miscTime(0),_dynTime(0), _qdTime(0),
    _forcingVal(0),
    _qd_bcRType("remoteLoading"),_qd_bcTType("freeSurface"),_qd_bcLType("symmFault"),_qd_bcBType("freeSurface"),
    _fd_bcRType("outGoingCharacteristics"),_fd_bcTType("freeSurface"),_fd_bcLType("symmFault"),_fd_bcBType("outGoingCharacteristics"),
    _mat_fd_bcRType("Neumann"),_mat_fd_bcTType("Neumann"),_mat_fd_bcLType("Neumann"),_mat_fd_bcBType("Neumann"),
    _quadEx_qd(NULL),_quadImex_qd(NULL), _quadWaveEx(NULL), _quadWaveImex(NULL),
    _fault_qd(NULL),_fault_fd(NULL), _material(NULL),_he(NULL),_p(NULL)
{
#if VERBOSE > 1
  std::string funcName = "StrikeSlip_LinearElastic_qd_fd::StrikeSlip_LinearElastic_qd_fd()";
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

  _body2fault = &(D._scatters["body2L"]);
  _fault_qd = new Fault_qd(D,D._scatters["body2L"],_faultTypeScale); // fault for quasidynamic problem
  _fault_fd = new Fault_fd(D, D._scatters["body2L"],_faultTypeScale); // fault for fully dynamic problem

  if (_evolveTemperature == 1 || _computeSSHeatEq == 1) { _he = new HeatEquation(D); }
  if (_thermalCoupling != "no" && _stateLaw == "flashHeating") {
    Vec T; VecDuplicate(_D->_y,&T);
    _he->getTemp(T);
    _fault_qd->setThermalFields(T,_he->_k,_he->_c);
    _fault_fd->setThermalFields(T,_he->_k,_he->_c);
  }

  if (_hydraulicCoupling != "no") { _p = new PressureEq(D); }
  if (_hydraulicCoupling == "coupled") {
    _fault_qd->setSNEff(_p->_p);
    _fault_fd->setSNEff(_p->_p);
  }

  // initiate momentum balance equation
  if (_guessSteadyStateICs == 1 && _computeSSMomBal == 1) { _material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,"Neumann",_mat_qd_bcBType); }
  else {_material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType); }
  computePenaltyVectors();

  // body forcing term for ice stream
  _forcingTerm = NULL; _forcingTermPlain = NULL;
  if (_forcingType.compare("iceStream")==0) { constructIceStreamForcingTerm(); }

  computeTimeStep(); // compute fully dynamic time step size
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_LinearElastic_qd_fd::~StrikeSlip_LinearElastic_qd_fd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::~StrikeSlip_LinearElastic_qd_fd()";
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
  VecDestroy(&_u0);
  VecDestroy(&_ay);
  VecDestroy(&_forcingTerm);
  VecDestroy(&_forcingTermPlain);
  VecDestroy(&_time1DVec);
  VecDestroy(&_dtime1DVec);
  VecDestroy(&_time2DVec);
  VecDestroy(&_dtime2DVec);
  VecDestroy(&_Req);

  delete _quadImex_qd;    _quadImex_qd = NULL;
  delete _quadEx_qd;      _quadEx_qd = NULL;
  delete _material;       _material = NULL;
  delete _fault_qd;       _fault_qd = NULL;
  delete _fault_fd;       _fault_fd = NULL;
  delete _he;             _he = NULL;
  delete _p;              _p = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::loadSettings(const char *file)
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
    else if (var.compare("hydraulicCoupling")==0) { _hydraulicCoupling = rhs.c_str(); }
    else if (var.compare("stateLaw")==0) { _stateLaw = rhs.c_str(); }
    else if (var.compare("guessSteadyStateICs")==0) { _guessSteadyStateICs = atoi(rhs.c_str() ); }
    else if (var.compare("computeSSMomBal")==0) { _computeSSMomBal = atoi( rhs.c_str() ); }
    else if (var.compare("forcingType")==0) { _forcingType = rhs.c_str(); }
    else if (var.compare("evolveTemperature")==0) { _evolveTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSHeatEq")==0) { _computeSSHeatEq = (int) atoi( rhs.c_str() ); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) { _timeIntegrator = rhs; }
    else if (var.compare("timeControlType")==0) { _timeControlType = rhs; }
    else if (var.compare("stride1D_qd")==0){
      _stride1D_qd = (int)atof(rhs.c_str() );
      _stride1D = _stride1D_qd;
    }
    else if (var.compare("stride2D_qd")==0){
      _stride2D_qd = (int)atof(rhs.c_str() );
      _stride2D = _stride2D_qd;
    }
    else if (var.compare("stride1D_fd")==0){ _stride1D_fd = (int)atof(rhs.c_str() ); }
    else if (var.compare("stride2D_fd")==0){ _stride2D_fd = (int)atof(rhs.c_str() ); }
    else if (var.compare("stride1D_fd_end")==0){ _stride1D_fd_end = (int)atof(rhs.c_str() ); }
    else if (var.compare("stride2D_fd_end")==0){ _stride2D_fd_end = (int)atof(rhs.c_str() ); }
    else if (var.compare("strideChkpt")==0){ _strideChkpt_qd = (int)atof(rhs.c_str()); }
    else if (var.compare("strideChkpt_fd")==0){ _strideChkpt_fd = (int)atof(rhs.c_str()); }
    else if (var.compare("initTime")==0) {
      _initTime = atof(rhs.c_str() );
      _currTime = _initTime;
    }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( rhs.c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof(rhs.c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("timeStepTol")==0) { _timeStepTol = atof(rhs.c_str() ); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }
    else if (var.compare("scale")==0) { loadVectorFromInputFile(rhsFull,_scale); }
    else if (var.compare("normType")==0) { _normType = rhs.c_str(); }

    else if (var.compare("vL")==0) { _vL = atof(rhs.c_str() ); }

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

    else if (var.compare("trigger_qd2fd")==0) { _trigger_qd2fd = atof(rhs.c_str() ); }
    else if (var.compare("trigger_fd2qd")==0) { _trigger_fd2qd = atof(rhs.c_str() ); }
    else if (var.compare("limit_qd")==0) { _limit_qd = atof(rhs.c_str() ); }
    else if (var.compare("limit_fd")==0) { _limit_fd = atof(rhs.c_str() ); }
    else if (var.compare("limit_stride_fd")==0) { _limit_stride_fd = atof(rhs.c_str() ); }
    else if (var.compare("deltaT_fd")==0) { _deltaT = atof(rhs.c_str() ); }
    else if (var.compare("CFL")==0) { _CFL = atof(rhs.c_str() ); }
    else if (var.compare("maxNumCycles")==0) { _maxNumCycles = atoi(rhs.c_str() ); }

  }
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);

  assert(_thermalCoupling.compare("coupled")==0 ||
      _thermalCoupling.compare("uncoupled")==0 ||
      _thermalCoupling.compare("no")==0 );

  assert(_hydraulicCoupling.compare("coupled")==0 ||
      _hydraulicCoupling.compare("uncoupled")==0 ||
      _hydraulicCoupling.compare("no")==0 );

  assert(_forcingType.compare("iceStream")==0 || _forcingType.compare("no")==0 );

  assert(_timeIntegrator.compare("FEuler")==0 ||
      _timeIntegrator.compare("RK32")==0 ||
      _timeIntegrator.compare("RK43")==0 ||
      _timeIntegrator.compare("RK32_WBE")==0 ||
      _timeIntegrator.compare("RK43_WBE")==0 );

  assert(_timeControlType.compare("P")==0 ||
         _timeControlType.compare("PI")==0 ||
         _timeControlType.compare("PID")==0 );

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {_initDeltaT = _minDeltaT; }

  assert(_initTime >= 0);

  assert(_timeStepTol >= 1e-14);
  assert(_minDeltaT >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);

  // check boundary condition types for momentum balance equation
  assert(_qd_bcRType.compare("freeSurface")==0 || _qd_bcRType.compare("remoteLoading")==0 );
  assert(_qd_bcTType.compare("freeSurface")==0 || _qd_bcTType.compare("remoteLoading")==0 );
  assert(_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0 );
  assert(_qd_bcBType.compare("freeSurface")==0 || _qd_bcBType.compare("remoteLoading")==0 );

  assert(_fd_bcRType.compare("freeSurface")==0 || _fd_bcRType.compare("outGoingCharacteristics")==0 );
  assert(_fd_bcTType.compare("freeSurface")==0 || _fd_bcTType.compare("outGoingCharacteristics")==0 );
  assert(_fd_bcLType.compare("symmFault")==0 || _fd_bcLType.compare("rigidFault")==0 );
  assert(_fd_bcBType.compare("freeSurface")==0 || _fd_bcBType.compare("outGoingCharacteristics")==0 );

  if (_stateLaw.compare("flashHeating")==0) {
    assert(_thermalCoupling != "no");
  }

  if (_limit_fd < _trigger_qd2fd){
    _limit_fd = 10 * _trigger_qd2fd;
  }

  if (_limit_qd > _trigger_fd2qd){
    _limit_qd = _trigger_qd2fd / 10.0;
  }

  if (_limit_stride_fd == -1){
    _limit_stride_fd = _limit_fd / 10.0;
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
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::parseBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::parseBCs()";
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
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::allocateFields()
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

  ierr = VecDuplicate(_D->_y, &_u0); VecSet(_u0,0.0);
  ierr = PetscObjectSetName((PetscObject) _u0, "u0"); CHKERRQ(ierr);

  ierr = VecDuplicate(_D->_y0, &_Req); VecSet(_Req,0.0);
  ierr = PetscObjectSetName((PetscObject) _Req, "Req"); CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

return ierr;
}


// compute allowed time step based on CFL condition and user input
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::computeTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::computeTimeStep";
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
  PetscScalar request_deltaT = _deltaT;

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

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute alphay and alphaz for use in time stepping routines
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::computePenaltyVectors()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::computePenaltyVectors";
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

  ierr = VecPointwiseMult(_ay, _ay, _material->_cs);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




/*
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime_integrateTime = MPI_Wtime();

  // first cycle
  initiateIntegrand();

  // if start with quasidynamic phase
  {
double startTime_qd = MPI_Wtime();
    _allowed = false;
    _inDynamic = false;
    integrate_qd();
_qdTime += MPI_Wtime() - startTime_qd;

  if(_currTime >= _maxTime || _stepCount >= _maxStepCount){ return 0; }

double startTime_fd = MPI_Wtime();
    _allowed = false;
    _inDynamic = true;
    prepare_qd2fd();
    integrate_fd();
_dynTime += MPI_Wtime() - startTime_fd;
  }

  if(_currTime >= _maxTime || _stepCount >= _maxStepCount || _maxNumCycles <= 1){ return 0; }

  // if start with fully dynamic phase
  //~ {
//~ double startTime_fd = MPI_Wtime();
    //~ _allowed = false;
    //~ _inDynamic = true;
    //~ integrate_fd();
//~ _dynTime += MPI_Wtime() - startTime_fd;
  //~ }

  // for all cycles after 1st cycle
  _cycleCount++;
  while (_cycleCount < _maxNumCycles && _stepCount <= _maxStepCount && _currTime <= _maxTime) {
    _allowed = false;
    _inDynamic = false;
    double startTime_qd = MPI_Wtime();
    prepare_fd2qd();
    integrate_qd();
    _qdTime += MPI_Wtime() - startTime_qd;

    double startTime_fd = MPI_Wtime();
    _allowed = false;
    _inDynamic = true;
    prepare_qd2fd();
    integrate_fd();
    _dynTime += MPI_Wtime() - startTime_fd;

    _cycleCount++;
  }


  _integrateTime += MPI_Wtime() - startTime_integrateTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}
*/

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::integrate_new";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime_integrateTime = MPI_Wtime();

  initiateIntegrand();

  int isFirstPhase = 1;

  // first phase
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




// returns true if it's time to switch from qd to fd, or fd to qd, or if
// the maximum time or step count has been reached
bool StrikeSlip_LinearElastic_qd_fd::checkSwitchRegime(const Fault* fault)
{
  bool mustSwitch = false;

  // if using max slip velocity as switching criteria
  //~ Vec absSlipVel;
  //~ VecDuplicate(fault->_slipVel, &absSlipVel);
  //~ VecCopy(fault->_slipVel, absSlipVel);
  //~ PetscScalar maxV;
  //~ VecAbs(absSlipVel);
  //~ VecMax(absSlipVel, NULL, &maxV);
  //~ VecDestroy(&absSlipVel);

  // if using R = eta*V / tauQS
  Vec R; VecDuplicate(fault->_slipVel,&R);
  VecPointwiseMult(R,_fault_qd->_eta_rad,fault->_slipVel);
  VecPointwiseDivide(R,R,fault->_tauQSP);
  PetscScalar maxV;
  VecMax(R,NULL,&maxV);
  VecCopy(R,_Req);
  VecDestroy(&R);



  //~ // if integrating past allowed time or step count, force switching now
  //~ if(_currTime > _maxTime || _stepCount > _maxStepCount){
    //~ mustSwitch = true;
    //~ return mustSwitch;
  //~ }

  // Otherwise, first check if switching from qd to fd, or from fd to qd, is allowed:
  // switching from fd to qd is allowed if maxV has ever been > limit_dyn
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
  //~ if (_inDynamic && _allowed && maxV < _limit_stride_fd) {
    //~ _stride1D = _stride1D_fd_end;
    //~ _stride2D = _stride2D_fd_end;
  //~ }

  return mustSwitch;
}

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  initiateIntegrand_qd(); // also calls solveSS if necessary
  initiateIntegrand_fd();


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// initiate integrand for quasidynamic phase
// includes using solveSS() to guess steady-state initial conditions
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::initiateIntegrand_qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::initiateIntegrand_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (!_D->_restartFromChkpt) {
    Vec slip;
    ierr = VecDuplicate(_material->_bcL,&slip); CHKERRQ(ierr);
    ierr = VecCopy(_material->_bcL,slip); CHKERRQ(ierr);
    if (_qd_bcLType.compare("symmFault")==0) {
      ierr = VecScale(slip,_faultTypeScale); CHKERRQ(ierr);
    }
    ierr = loadVecFromInputFile(slip,_inputDir,"slip"); CHKERRQ(ierr);
    _varQSEx["slip"] = slip;
    ierr = VecCopy(_varQSEx["slip"],_fault_qd->_slip); CHKERRQ(ierr);
  }
  else {
    if (_varQSEx.find("slip") != _varQSEx.end() ) { ierr = VecCopy(_fault_qd->_slip,_varQSEx["slip"]); CHKERRQ(ierr); }
    else {
      Vec var;
      ierr = VecDuplicate(_fault_qd->_slip,&var); CHKERRQ(ierr);
      ierr = VecCopy(_fault_qd->_slip,var); CHKERRQ(ierr);
      _varQSEx["slip"] = var;
    }
  }


  if (_guessSteadyStateICs) { solveSS(); }

  // LinearElastic does not set up its KSP, so must set it up here
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

  // initiate varQSEx and (if needed varIm)
  ierr = _fault_qd->initiateIntegrand(_initTime,_varQSEx); CHKERRQ(ierr);

  // initiate integrand for varIm
  if (_evolveTemperature == 1) {
    ierr = _he->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr);
  }
  if (_hydraulicCoupling!= "no" ) {
    ierr = _p->initiateIntegrand(_initTime,_varQSEx,_varIm); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::initiateIntegrand_fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::initiateIntegrand_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // add psi and slip to varFD
  ierr = _fault_fd->initiateIntegrand(_initTime,_varFD); CHKERRQ(ierr); // adds psi and slip

  // add u
  if (_varFD.find("u") != _varFD.end() ) { ierr = VecCopy(_material->_u,_varFD["u"]);CHKERRQ(ierr);  }
  else {
    Vec var;
    ierr = VecDuplicate(_material->_u,&var); CHKERRQ(ierr);
    ierr = VecCopy(_material->_u,var); CHKERRQ(ierr);
    _varFD["u"] = var;
  }
  ierr = VecSet(_u0,0.0); CHKERRQ(ierr);

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
  //~ if (_hydraulicCoupling != "no" ) {
    //~ VecDuplicate(_varIm["pressure"], &_varFD["pressure"]);
    //~ VecCopy(_varIm["pressure"], _varFD["pressure"]);
    //~ if ((_p->_permSlipDependent).compare("yes")==0) {
      //~ VecDuplicate(_varQSEx["permeability"], &_varFD["permeability"]);
      //~ VecCopy(_varQSEx["permeability"], _varFD["permeability"]);
    //~ }
  //~ }

   // copy varFD into varFDPrev
  for (map<string,Vec>::iterator it = _varFD.begin(); it != _varFD.end(); it++ ) {
    if (_varFDPrev.find(it->first) == _varFDPrev.end() ) {
      Vec var;
      VecDuplicate(_varFD[it->first],&var);
      _varFDPrev[it->first] = var;
    }
    VecCopy(_varFD[it->first],_varFDPrev[it->first]);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// move from a fully dynamic phase to a quasidynamic phase
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::prepare_fd2qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::prepare_fd2qd()";
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
  ierr = VecCopy(_fault_fd->_psi, _varQSEx["psi"]); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slip, _varQSEx["slip"]); CHKERRQ(ierr);

  // update implicitly integrated T
  if (_evolveTemperature == 1) { ierr = VecCopy(_varFD["Temp"],_varIm["Temp"]); CHKERRQ(ierr); } // if solving the heat equation

  if (_hydraulicCoupling != "no" ) {
    VecCopy(_varFD["pressure"], _varIm["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      VecCopy(_varFD["permeability"], _varQSEx["permeability"]);
    }
    if (_hydraulicCoupling.compare("coupled")==0){
      // _fault_qd->setSNEff(_varIm["pressure"]);
      _fault_qd->setSNEff(_p->_p);
    }
  }

  // update fault internal variables
  ierr = VecCopy(_fault_fd->_psi,       _fault_qd->_psi); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slipVel,   _fault_qd->_slipVel); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_slip,      _fault_qd->_slip); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_tauP,      _fault_qd->_tauP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_tauQSP,    _fault_qd->_tauQSP); CHKERRQ(ierr);
  ierr = VecCopy(_fault_fd->_strength,  _fault_qd->_strength); CHKERRQ(ierr);
  if (_fault_fd->_stateLaw.compare("flashHeating") == 0) {
    ierr = VecCopy(_fault_fd->_T,         _fault_qd->_T); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_Tw,         _fault_qd->_Tw); CHKERRQ(ierr);
    ierr = VecCopy(_fault_fd->_Vw,         _fault_qd->_Vw); CHKERRQ(ierr);
  }




  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// move from a quasidynamic phase to a fully dynamic phase
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::prepare_qd2fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::prepare_qd2fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ // Force writing output
  //~ PetscInt stopIntegration = 0;
  //~ if(_stride1D > 0){ _stride1D = 1; }
  //~ if(_stride2D > 0){ _stride2D = 1; }
  //~ timeMonitor(_currTime, _deltaT,_stepCount, stopIntegration);

  // switch strides to qd values
  _stride1D = _stride1D_fd;
  _stride2D = _stride1D_fd;

  // save current variables as n-1 time step
  ierr = VecCopy(_fault_qd->_slip,_varFDPrev["slip"]); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_psi,_varFDPrev["psi"]); CHKERRQ(ierr);
  ierr = VecCopy(_material->_u,_varFDPrev["u"]); CHKERRQ(ierr);
  if (_evolveTemperature == 1) { VecCopy(_varIm["Temp"], _varFDPrev["Temp"]); } // if solving the heat equation

  if (_hydraulicCoupling.compare("no")!=0 ) {
    VecCopy(_varIm["pressure"], _varFDPrev["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      VecCopy(_varQSEx["permeability"], _varFDPrev["permeability"]);
    }
  }

  // take 1 quasidynamic time step to compute variables at time n
  _inDynamic = 0;
  ierr = integrate_singleQDTimeStep(); CHKERRQ(ierr);
  _inDynamic = 1;

  // update varFD to reflect latest values
  ierr = VecCopy(_fault_qd->_slip,_varFD["slip"]); CHKERRQ(ierr);
  ierr = VecCopy(_fault_qd->_psi,_varFD["psi"]); CHKERRQ(ierr);
  ierr = VecCopy(_material->_u,_varFD["u"]); CHKERRQ(ierr);
  if (_evolveTemperature == 1) { ierr = VecCopy(_varIm["Temp"], _varFD["Temp"]); CHKERRQ(ierr); } // if solving the heat equation
  if (_hydraulicCoupling.compare("no")!=0 ) {
    VecCopy(_varIm["pressure"], _varFD["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      VecCopy(_varQSEx["permeability"], _varFD["permeability"]);
    }
    if (_hydraulicCoupling.compare("coupled")==0 ){
      // _fault_fd->setSNEff(_varFD["pressure"]);
      _fault_fd->setSNEff(_p->_p);
    }
  }

  // now change u to du
  ierr = VecAXPY(_varFD["u"],-1.0,_varFDPrev["u"]); CHKERRQ(ierr);
  ierr = VecCopy(_varFDPrev["u"],_u0); CHKERRQ(ierr);
  ierr = VecSet(_varFDPrev["u"],0.0); CHKERRQ(ierr);


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
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::integrate_singleQDTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::integrate_singleQDTimeStep()";
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



PetscErrorCode StrikeSlip_LinearElastic_qd_fd::writeStep1D(PetscInt stepCount, PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::writeStep1D";
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
    ierr = VecView(_regime1DVec, _viewer1D);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");        CHKERRQ(ierr);
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

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::writeStep2D(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::writeStep2D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer2D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_2D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer2D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer2D, PETSC_TRUE);CHKERRQ(ierr);

    // write time
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");        CHKERRQ(ierr);
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
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");        CHKERRQ(ierr);
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

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::writeSS(const int Ii)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::writeSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  bool needToDestroyJjSSVec = 0;
  if (_JjSSVec == NULL) {
    // initiate Vec to hold index Jj
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
    VecSetBlockSize(_JjSSVec, 1);
    PetscObjectSetName((PetscObject) _JjSSVec, "SS_index");
    VecSet(_JjSSVec,Ii);
    needToDestroyJjSSVec = 1;
  }
  else {
    VecSet(_JjSSVec,Ii);
  }

  if (_viewerSS == NULL) {
    // set up viewer for output of steady-state data
    string outFileName = _outputDir + "data_steadyState.h5";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewerSS);             CHKERRQ(ierr);
    ierr = PetscViewerSetType(_viewerSS, PETSCVIEWERBINARY);            CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewerSS);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = VecView(_JjSSVec, _viewerSS);                                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewerSS);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewerSS);                          CHKERRQ(ierr);

    ierr = _material->writeStep1D(_viewerSS);                           CHKERRQ(ierr);
    ierr = _fault_qd->writeStep(_viewerSS);                                CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr = _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

    ierr = _material->writeStep2D(_viewerSS);                           CHKERRQ(ierr);
    if (_thermalCoupling!="no") { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }
  }
  else {
    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewerSS);                 CHKERRQ(ierr);

    ierr = VecView(_JjSSVec, _viewerSS);                                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewerSS);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewerSS);                          CHKERRQ(ierr);

    ierr = _material->writeStep1D(_viewerSS);                           CHKERRQ(ierr);
    ierr = _fault_qd->writeStep(_viewerSS);                             CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { _p->writeStep(_viewerSS); }
    if (_evolveTemperature == 1) { ierr =  _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

    ierr = _material->writeStep2D(_viewerSS);                           CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }
  }

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::writeCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::writeCheckpoint";
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
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "cycleCount", PETSC_INT, &_cycleCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "phaseCount", PETSC_INT, &_phaseCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "inDynamic", PETSC_INT, &_inDynamic); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "allowed", PETSC_INT, &_allowed); CHKERRQ(ierr);
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

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::loadCheckpoint";
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


PetscErrorCode StrikeSlip_LinearElastic_qd_fd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  //~ if (_timeIntegrator.compare("IMEX")==0&& _quadImex_qd!=NULL) { ierr = _quadImex_qd->view(); }
  //~ if (_timeIntegrator.compare("RK32")==0 && _quadEx_qd!=NULL) { ierr = _quadEx_qd->view(); }

  _material->view(_integrateTime);
  _fault_qd->view(_integrateTime);
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_LinearElastic_qd_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent propagating the wave (s): %g\n",_propagateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in quasidynamic (s): %g\n",_qdTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in dynamic (s): %g\n",_dynTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time (s): %g\n",totRunTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",(_writeTime/_integrateTime)*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::writeContext";
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
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"forcingType = %s\n",_forcingType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"computeSSMomBal = %i\n",_computeSSMomBal);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"evolveTemperature = %i\n",_evolveTemperature);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeSSHeatEq = %i\n",_computeSSHeatEq);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"vL = %g\n",_vL);CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_qd = %i\n",_stride1D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_qd = %i\n",_stride2D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_fd = %i\n",_stride1D_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_fd = %i\n",_stride2D_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_fd_end = %i\n",_stride1D_fd_end);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_fd_end = %i\n",_stride2D_fd_end);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

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

  ierr = PetscViewerASCIIPrintf(viewer,"trigger_qd2fd = %.15e\n",_trigger_qd2fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"trigger_fd2qd = %.15e\n",_trigger_fd2qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_qd = %.15e\n",_limit_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_fd = %.15e\n",_limit_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_stride_fd = %.15e\n",_limit_stride_fd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"CFL = %.15e\n",_CFL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"deltaT_fd = %.15e\n",_deltaT_fd);CHKERRQ(ierr);


  // boundary conditions for momentum balance equation
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR_qd = %s\n",_qd_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT_qd = %s\n",_qd_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL_qd = %s\n",_qd_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB_qd = %s\n",_qd_bcBType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR_fd = %s\n",_fd_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT_fd = %s\n",_fd_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL_fd = %s\n",_fd_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB_fd = %s\n",_fd_bcBType.c_str());CHKERRQ(ierr);


  ierr = PetscViewerASCIIPrintf(viewer,"faultTypeScale = %g\n",_faultTypeScale);CHKERRQ(ierr);


  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  // write non-ascii context
  string outFileName = _outputDir + "data_context.h5";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewer_context); CHKERRQ(ierr);
  ierr = PetscViewerSetType(_viewer_context, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewer_context);CHKERRQ(ierr);

  _D->write(_viewer_context);
  _fault_qd->writeContext(_outputDir, _viewer_context);
  _material->writeContext(_outputDir, _viewer_context);
  if (_thermalCoupling.compare("no")!=0) { _he->writeContext(_outputDir, _viewer_context); }
  if (_hydraulicCoupling.compare("no")!=0) { _p->writeContext(_outputDir, _viewer_context); }
  if (_forcingType.compare("iceStream")==0) {
    ierr = PetscViewerHDF5PushGroup(_viewer_context, "/momBal");                 CHKERRQ(ierr);
    ierr = VecView(_forcingTermPlain, _viewer_context);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer_context);                             CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  _material->setRHS();

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) { VecAXPY(_material->_rhs,-1.0,_forcingTerm); }

  _material->computeU();
  _material->computeStresses();

  return ierr;
}


// fully dynamic: off-fault portion of the momentum balance equation
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::propagateWaves(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::propagateWaves";
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


// guess at the steady-state solution
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::solveSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initiate Vecs to hold index Jj
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
  VecSetBlockSize(_JjSSVec, 1);
  PetscObjectSetName((PetscObject) _JjSSVec, "SS_index");
  VecSet(_JjSSVec,0);

  // estimate steady-state conditions for fault, material based on strain rate
  _fault_qd->guessSS(_vL); // sets: slipVel, psi, tau
  loadVecFromInputFile(_fault_qd->_tauP,_inputDir,"tauSS");

  // output initial conditions, mostly for debugging purposes
  writeSS(0);

  // steady state momentum balance equation
  if (_computeSSMomBal == 1) {
    // set up KSP for steady-state solution
    Mat A;
    _material->_sbp->getA(A);
    _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS);

    // compute compute u that satisfies tau at left boundary
    ierr = VecSet(_material->_bcR,0.0); CHKERRQ(ierr);
    ierr = VecSet(_material->_bcT,0.0); CHKERRQ(ierr);
    ierr = VecSet(_material->_bcB,0.0); CHKERRQ(ierr);
    VecCopy(_fault_qd->_tauP,_material->_bcL);
    _material->setRHS();
    _material->computeU();
    _material->computeStresses();

    // update fault to contain correct stresses
    ierr = VecScatterBegin(*_body2fault, _material->_sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(*_body2fault, _material->_sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    VecCopy(_fault_qd->_tauQSP,_fault_qd->_tauP);
    VecCopy(_fault_qd->_tauQSP,_fault_qd->_strength);

    solveSSb();
    _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);

    KSPDestroy(&_material->_ksp);
  }

  // steady state temperature
  if (_computeSSHeatEq) {
    Vec T; VecDuplicate(_material->_sxy,&T);
    _he->computeSteadyStateTemp(_currTime,_fault_qd->_slipVel,_fault_qd->_tauP,NULL,NULL,T);
    VecDestroy(&T);
  }

  // update fault_fd to contain new steady-state fault_qd data as well
  VecCopy(_fault_qd->_psi,      _fault_fd->_psi);
  VecCopy(_fault_qd->_slipVel,  _fault_fd->_slipVel);
  VecCopy(_fault_qd->_slip,     _fault_fd->_slip);
  VecCopy(_fault_qd->_slip,     _fault_fd->_slip0);
  VecCopy(_fault_qd->_tauQSP,   _fault_fd->_tauQSP);
  VecCopy(_fault_qd->_tauP,     _fault_fd->_tauP);
  VecCopy(_fault_qd->_strength,  _fault_fd->_tau0);
  VecCopy(_fault_qd->_prestress,  _fault_fd->_prestress);
  if (_fault_qd->_stateLaw == "flashHeating") {
    VecCopy(_fault_qd->_T,  _fault_fd->_T);
    VecCopy(_fault_qd->_Tw,  _fault_fd->_Tw);
    VecCopy(_fault_qd->_Vw,  _fault_fd->_Vw);
  }

  // output final steady state results
  writeSS(1);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// update the boundary conditions based on new steady state u
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::solveSSb()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::solveSSb";
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

  if (_qd_bcRType=="remoteLoading") {
    // extract boundary data from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcRShift,_material->_bcR);
  }
  if (_qd_bcTType=="remoteLoading") {
    // extract R boundary from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcTShift,_material->_bcT);
  }
  if (_qd_bcBType=="remoteLoading") {
    // extract R boundary from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2B"], _material->_u, _material->_bcBShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2B"], _material->_u, _material->_bcBShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcBShift,_material->_bcB);
  }

  // extract L boundary from u to set slip, possibly _material->_bcL
  Vec uL; VecDuplicate(_material->_bcL,&uL);
  VecScatterBegin(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);

  if (_varQSEx.find("slip") != _varQSEx.end() ) { VecCopy(uL,_varQSEx["slip"]); }
  else {
    Vec slip;
    VecDuplicate(_material->_bcL,&slip);
    VecCopy(uL,slip);
    _varQSEx["slip"] = slip;
  }

  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    VecCopy(uL,_material->_bcL);
  }
  if (_qd_bcLType.compare("symmFault")==0) {
    VecScale(_varQSEx["slip"],2.0);
  }

  VecDestroy(&uL);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// constructs the body forcing term for an ice stream
// includes allocation of memory for this forcing term
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::constructIceStreamForcingTerm()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_LinearElastic_qd_fd::constructIceStreamForcingTerm";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

/*
  // matrix to map the value for the forcing term, which lives on the fault, to all other processors
  Mat MapV = NULL;
  MatCreate(PETSC_COMM_WORLD,&MapV);
  MatSetSizes(MapV,PETSC_DECIDE,PETSC_DECIDE,_D->_Ny*_D->_Nz,_D->_Nz);
  PetscInt NN = 0;
  VecGetLocalSize(_material->_mu,&NN);
  MatMPIAIJSetPreallocation(MapV,NN,NULL,NN,NULL);
  MatSeqAIJSetPreallocation(MapV,NN,NULL);
  MatSetUp(MapV);

  PetscScalar v=1.0;
  PetscInt Ii=0,Istart=0,Iend=0,Jj=0;
  MatGetOwnershipRange(MapV,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    Jj = Ii % _D->_Nz;
    MatSetValues(MapV,1,&Ii,1,&Jj,&v,INSERT_VALUES);
  }
  MatAssemblyBegin(MapV,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(MapV,MAT_FINAL_ASSEMBLY);
*/

  //~ // compute forcing term for momentum balance equation
  //~ // forcing = - tau_ss / Ly
  //~ Vec tauSS = NULL;
  //~ _fault_qd->computeTauRS(tauSS,_vL);
  //~ VecScale(tauSS,-1./_D->_Ly);

  //~ VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  //~ MatMult(MapV,tauSS,_forcingTerm);

  //~ MatDestroy(&MapV);
  //~ VecDestroy(&tauSS);

  // compute forcing term using scalar input
  VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,_forcingVal);
  PetscObjectSetName((PetscObject) _forcingTerm, "forcingTerm");
  VecDuplicate(_material->_u,&_forcingTermPlain); VecCopy(_forcingTerm,_forcingTermPlain);
  PetscObjectSetName((PetscObject) _forcingTermPlain, "forcingTermPlain");

  // alternatively, load forcing term from user input
  ierr = loadVecFromInputFile(_forcingTerm,_inputDir,"iceForcingTerm"); CHKERRQ(ierr);

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


PetscErrorCode StrikeSlip_LinearElastic_qd_fd::integrate_qd(int isFirstPhase)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::integrate_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  // update momentum balance equation boundary conditions
  ierr = _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType); CHKERRQ(ierr);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

  // ensure new memory isn't being spawned each time this function is called
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

  // calculate time used in integration
  _integrateTime += MPI_Wtime() - startTime;


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd_fd::integrate_fd(int isFirstPhase)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::integrate_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  // update momentum balance equation boundary conditions
  ierr = _material->changeBCTypes(_mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType); CHKERRQ(ierr);
  Mat A; _material->_sbp->getA(A);
  ierr = _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans); CHKERRQ(ierr);

    // ensure new memory isn't being spawned each time this function is called
  delete _quadWaveEx; _quadWaveEx = NULL;
  delete _quadWaveImex; _quadWaveImex = NULL;

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
  delete _quadWaveImex; _quadWaveImex = NULL;

  // calculate time used in integration
  _integrateTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// quasidynamic: purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::d_dt qd explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  if (_qd_bcTType=="remoteLoading") {
    ierr = VecSet(_material->_bcT,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcT,1.0,_material->_bcTShift);CHKERRQ(ierr);
  }
  if (_qd_bcBType=="remoteLoading") {
    ierr = VecSet(_material->_bcB,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcB,1.0,_material->_bcBShift);CHKERRQ(ierr);
  }

  ierr = _fault_qd->updateFields(time,varEx); CHKERRQ(ierr);

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end()) && _hydraulicCoupling.compare("no")!=0 ){
    ierr = _p->updateFields(time,varEx); CHKERRQ(ierr);
  }
  if (_hydraulicCoupling.compare("coupled")==0) {
    // _fault_qd->setSNEff(varEx.find("pressure")->second);
    ierr = _fault_qd->setSNEff(_p->_p); CHKERRQ(ierr);
  }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  // update fields on fault from other classes
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end() ) && _hydraulicCoupling.compare("no")!=0 ){
    _p->d_dt(time,varEx,dvarEx);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// quasidynamic: implicit/explicit time stepping
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::d_dt qd IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // 1. update BCs, state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  if (_qd_bcTType=="remoteLoading") {
    ierr = VecSet(_material->_bcT,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcT,1.0,_material->_bcTShift);CHKERRQ(ierr);
  }
  if (_qd_bcBType=="remoteLoading") {
    ierr = VecSet(_material->_bcB,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcB,1.0,_material->_bcBShift);CHKERRQ(ierr);
  }

  ierr = _fault_qd->updateFields(time,varEx); CHKERRQ(ierr);

  if ( _hydraulicCoupling.compare("no")!=0 ) {
    ierr = _p->updateFields(time,varEx,varImo); CHKERRQ(ierr);
  }

  // update temperature in momBal
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling.compare("coupled")==0) {
    ierr = _fault_qd->updateTemperature(varImo.find("Temp")->second); CHKERRQ(ierr);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    ierr = _fault_qd->setSNEff(_p->_p); CHKERRQ(ierr);
  }


  // 2. compute rates, and update implicitly integrated variables
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  if ( _hydraulicCoupling.compare("no")!=0 ) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
    // _p->d_dt(time,varEx,dvarEx);
  }

  // heat equation
  if (varIm.find("Temp") != varIm.end()) {
    Vec sxy,sxz,sdev;
    _material->getStresses(sxy,sxz,sdev);
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault_qd->_tauP;
    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,sdev,NULL,varIm["Temp"],Told,dt); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: purely explicit time stepping
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::d_dt fd explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // momentum balance equation except for fault boundary
  propagateWaves(time, deltaT, varNext, var, varPrev);

  // update fault
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);


  // update body u from fault u
  ierr = VecScatterBegin(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  // compute stresses
  VecCopy(varNext.find("u")->second, _material->_u);
  VecAXPY(_material->_u,1.0,_u0);
  _material->computeStresses();

  // update fault shear stress and quasi-static shear stress
  Vec sxy,sxz,sdev; _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  VecAXPY(_fault_fd->_tauQSP, 1.0, _fault_fd->_prestress);
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = eta_rad*slipVel + tauQSP

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

  if (_hydraulicCoupling.compare("no")!=0 ) {
    // Vec P = var.find("pressure")->second;
    // Vec dPdt;
    // VecDuplicate(P, &dPdt);
    // ierr = _p->dp_dt(time, P, dPdt); CHKERRQ(ierr);
    // VecWAXPY(varNext["pressure"], deltaT, dPdt, P);
    // _p->setPressure(varNext["pressure"]);
    // VecDestroy(&dPdt);
    VecCopy(var.find("pressure")->second, varNext["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      Vec V = _fault_fd->_slipVel;
      Vec K = var.find("permeability")->second;
      Vec dKdt;
      VecDuplicate(K, &dKdt);
      ierr = _p->dk_dt(time, V, K, dKdt); CHKERRQ(ierr);
      VecWAXPY(varNext["permeability"], deltaT, dKdt, K);
      _p->setPremeability(varNext["permeability"]);
      VecDestroy(&dKdt);
    }
  }

  // explicitly integrate heat equation using forward Euler
  if (_evolveTemperature == 1) {
    Vec V = _fault_fd->_slipVel;
    Vec tau = _fault_fd->_tauP;
    Vec Tn = var.find("Temp")->second;
    Vec dTdt; VecDuplicate(Tn,&dTdt);
    ierr = _he->d_dt(time,V,tau,NULL,NULL,Tn,dTdt); CHKERRQ(ierr);
    VecWAXPY(varNext["Temp"], deltaT, dTdt, Tn); // Tn+1 = deltaT * dTdt + Tn
    _he->setTemp(varNext["Temp"]); // keep heat equation T up to date
    VecDestroy(&dTdt);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: IMEX time stepping
PetscErrorCode StrikeSlip_LinearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev, map<string,Vec>& varIm, const map<string,Vec>& varImPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::d_dt fd imex";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // momentum balance equation except for fault boundary
  propagateWaves(time, deltaT, varNext, var, varPrev);

  // update fault
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);


  // update body u from fault u
  ierr = VecScatterBegin(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _fault_fd->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  // compute stresses
  VecCopy(varNext.find("u")->second, _material->_u);
  VecAXPY(_material->_u,1.0,_u0);
  _material->computeStresses();

  // update fault shear stress and quasi-static shear stress
  Vec sxy,sxz,sdev; _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_fd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  VecAXPY(_fault_fd->_tauQSP, 1.0, _fault_fd->_prestress);
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = eta_rad*slipVel + tauQSP

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



PetscErrorCode StrikeSlip_LinearElastic_qd_fd::timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd_fd::timeMonitor";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  // update to reflect current status
  _currTime = time;
  _deltaT = deltaT;
  VecSet(_time1DVec,time);
  VecSet(_dtime1DVec,_deltaT);
  VecSet(_regime1DVec,(int) _inDynamic);
  VecSet(_time2DVec,time);
  VecSet(_dtime2DVec,_deltaT);
  VecSet(_regime2DVec,(int) _inDynamic);

  if (_stepCount == stepCount && _stepCount != 0) { return ierr; } // don't write out the same step twice
  _stepCount = stepCount;

  if ( (_stride1D>0 &&_currTime == _maxTime) || (_stride1D>0 && stepCount % _stride1D == 0) ) {
    ierr = writeStep1D(_stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_viewer1D); CHKERRQ(ierr);
    if(_inDynamic){ ierr = _fault_fd->writeStep(_viewer1D); CHKERRQ(ierr); }
    else { ierr = _fault_qd->writeStep(_viewer1D); CHKERRQ(ierr); }
    if (_evolveTemperature == 1) { ierr =  _he->writeStep1D(_viewer1D); CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_viewer1D); CHKERRQ(ierr); }
  }

  if ( (_stride2D>0 &&_currTime == _maxTime) || (_stride2D>0 && stepCount % _stride2D == 0) ) {
    ierr = writeStep2D(_stepCount,time); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_viewer2D);CHKERRQ(ierr);
    if (_evolveTemperature == 1) { ierr =  _he->writeStep2D(_viewer2D);CHKERRQ(ierr); }
  }

  // checkpointing
  PetscInt strideChkpt = _strideChkpt_qd;
  if (_inDynamic) {strideChkpt = _strideChkpt_fd;}
  if ( _D->_saveChkpts== 1 && ((strideChkpt > 0 && stepCount % strideChkpt == 0) || (_currTime == _maxTime)) ) {
    ierr = writeCheckpoint();                                           CHKERRQ(ierr);
    ierr = _D->writeCheckpoint(_viewer_chkpt);                          CHKERRQ(ierr);
    ierr = _material->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault_qd->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault_fd->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    if (_quadEx_qd != NULL && !_inDynamic) { ierr = _quadEx_qd->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadImex_qd != NULL && !_inDynamic) { ierr = _quadImex_qd->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadWaveEx != NULL && _inDynamic) { ierr = _quadWaveEx->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    //~ if (_quadWaveImex != NULL) { ierr = _quadWaveImex->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeCheckpoint(_viewer_chkpt);  CHKERRQ(ierr); }
    if (_thermalCoupling.compare("no")!=0) { ierr = _he->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
  }

  if(_inDynamic){ if(checkSwitchRegime(_fault_fd)){ stopIntegration = 1; } }
  else { if(checkSwitchRegime(_fault_qd)){ stopIntegration = 1; } }

  #if VERBOSE > 0
    std::string regime = "quasidynamic";
    if(_inDynamic){ regime = "fully dynamic"; }
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s\n",stepCount,_currTime,_deltaT,regime.c_str());CHKERRQ(ierr);
    PetscScalar maxVel = 0;
    if(_inDynamic){ VecMax(_fault_fd->_slipVel,NULL,&maxVel); }
    else { VecMax(_fault_qd->_slipVel,NULL,&maxVel); }
    PetscScalar maxReq = 0;
    VecMax(_Req,NULL,&maxReq);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s | allowed = %i, maxVel = %.15e, maxReq = %.15e\n",stepCount,_currTime,_deltaT,regime.c_str(),_allowed,maxVel,maxReq);CHKERRQ(ierr);

    //~ PetscReal maxVel = 0;
    //~ if(_inDynamic){ ierr = VecMax(_fault_fd->_slipVel,NULL,&maxVel); CHKERRQ(ierr); }
    //~ if(!_inDynamic){ ierr = VecMax(_fault_qd->_slipVel,NULL,&maxVel); CHKERRQ(ierr); }
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s  |  allowed = %i, maxV = %e, limit_fd = %e\n",stepCount,_currTime,_deltaT,regime.c_str(),_allowed, maxVel, _limit_fd);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



