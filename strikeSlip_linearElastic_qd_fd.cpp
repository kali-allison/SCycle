#include "strikeSlip_linearElastic_qd_fd.hpp"

#define FILENAME "strikeSlip_linearElastic_qd_fd.cpp"

using namespace std;


strikeSlip_linearElastic_qd_fd::strikeSlip_linearElastic_qd_fd(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(1e-9),_isFault("true"),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _guessSteadyStateICs(0.),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),
  _deltaT(-1), _CFL(-1),
  _y(&D._y),_z(&D._z),
  _Fhat(NULL),_savedU(NULL),
  _timeIntegrator("RK43"),_timeControlType("PID"),
  _stride1D(1),_stride2D(1),
  _stride1D_qd(1),_stride2D_qd(1),_stride1D_dyn(1),_stride2D_dyn(1),_stride1D_dyn_long(1),_stride2D_dyn_long(1),
  _withFhat(1),
  _maxStepCount_dyn(2000),_maxStepCount_qd(1e8),_maxStepCount(1e6),
  _initTime(0),_currTime(0),_maxTime_dyn(1e15),_maxTime_qd(15),_minDeltaT(1e-3),_maxDeltaT(1e10),_maxTime(1e15),
  _inDynamic(false),_firstCycle(true),
  _stepCount(0),_atol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
  _debug(0), _localStep(0),_startOnDynamic(0),
  _timeV1D(NULL),_dtimeV1D(NULL),_timeV2D(NULL),_whichRegime(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),_miscTime(0),_dynTime(0), _qdTime(0),
  _allowed(false), _triggerqd2d(1e-3), _triggerd2qd(1e-3), _limit_qd(1e-8), _limit_dyn(1),_limit_stride_dyn(-1),
  _qd_bcRType("remoteLoading"),_qd_bcTType("freeSurface"),_qd_bcLType("symm_fault"),_qd_bcBType("freeSurface"),
  _dyn_bcRType("outGoingCharacteristics"),_dyn_bcTType("freeSurface"),_dyn_bcLType("outGoingCharacteristics"),_dyn_bcBType("outGoingCharacteristics"),
  _mat_dyn_bcRType("Neumann"),_mat_dyn_bcTType("Neumann"),_mat_dyn_bcLType("Neumann"),_mat_dyn_bcBType("Neumann"),
  _quadEx_qd(NULL),_quadImex_qd(NULL), _quadWaveEx(NULL),
  _fault_qd(NULL),_fault_fd(NULL), _material(NULL),_he(NULL),_p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::strikeSlip_linearElastic_qd_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();

  _fault_qd = new Fault_qd(D,D._scatters["body2L"]); // fault for quasidynamic problem
  _fault_fd = new Fault_fd(D, D._scatters["body2L"]); // fault for fully dynamic problem
  if (_thermalCoupling.compare("no")!=0) { // heat equation
    _he = new HeatEquation(D);
  }
  if (_thermalCoupling.compare("no")!=0 && _stateLaw.compare("flashHeating")==0) {
    Vec T; VecDuplicate(_D->_y,&T);
    _he->getTemp(T);
    _fault_qd->setThermalFields(T,_he->_k,_he->_c);
  }


  // pressure diffusion equation
  if (_hydraulicCoupling.compare("no")!=0) {
    _p = new PressureEq(D);
  }
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault_qd->setSNEff(_p->_p);
  }

  // initiate momentum balance equation
  parseBCs();
  if (_guessSteadyStateICs) { _material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,"Neumann",_mat_qd_bcBType); }
  else {_material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType); }
  _cs = _material->_cs;
  _rhoVec = _material->_rhoVec;
  _muVec = _material->_muVec;
  computePenaltyVectors();

  computeTimeStep(); // compute time step

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


strikeSlip_linearElastic_qd_fd::~strikeSlip_linearElastic_qd_fd()
{
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::~strikeSlip_linearElastic_qd_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

{
  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }
}

  //~ for (std::map <string,std::pair<PetscViewer,string> > it = _viewers.begin(); it!=_viewers.end(); it++ ) {
    //~ PetscViewerDestroy(&it->second);
  //~ }

  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_dtimeV1D);
  PetscViewerDestroy(&_timeV2D);
  PetscViewerDestroy(&_whichRegime);

  delete _quadImex_qd;    _quadImex_qd = NULL;
  delete _quadEx_qd;      _quadEx_qd = NULL;
  delete _material;    _material = NULL;
  delete _fault_qd;       _fault_qd = NULL;
  delete _fault_fd;       _fault_fd = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode strikeSlip_linearElastic_qd_fd::loadSettings(const char *file)
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

    if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hydraulicCoupling")==0) {
      _hydraulicCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("stateLaw")==0) {
      _stateLaw = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    else if (var.compare("guessSteadyStateICs")==0) {
      _guessSteadyStateICs = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) {
      _timeIntegrator = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("timeControlType")==0) {
      _timeControlType = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("stride1D")==0){ _stride1D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride1D_qd")==0){ _stride1D_qd = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D_qd")==0){ _stride2D_qd = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride1D_dyn")==0){ _stride1D_dyn = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D_dyn")==0){ _stride2D_dyn = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride1D_dyn_long")==0){ _stride1D_dyn_long = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D_dyn_long")==0){ _stride2D_dyn_long = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("withFhat")==0){ _withFhat = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxStepCount_dyn")==0) { _maxStepCount_dyn = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxStepCount_qd")==0) { _maxStepCount_qd = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxTime_qd")==0) { _maxTime_qd = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxTime_dyn")==0) { _maxTime_dyn = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("atol")==0) { _atol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("timeIntInds")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_timeIntInds);
    }

    else if (var.compare("vL")==0) { _vL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_dyn")==0) {
      _dyn_bcRType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcT_dyn")==0) {
      _dyn_bcTType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcL_dyn")==0) {
      _dyn_bcLType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcB_dyn")==0) {
      _dyn_bcBType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcR_qd")==0) {
      _qd_bcRType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcT_qd")==0) {
      _qd_bcTType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcL_qd")==0) {
      _qd_bcLType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcB_qd")==0) {
      _qd_bcBType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("triggerqd2d")==0) { _triggerqd2d = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("triggerd2qd")==0) { _triggerd2qd = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    else if (var.compare("deltaT")==0) { _deltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("CFL")==0) { _CFL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("limit_qd")==0) { _limit_qd = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("limit_dyn")==0) { _limit_dyn = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("limit_stride_dyn")==0) { _limit_stride_dyn = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("isFault")==0) { _isFault = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("initialConditions")==0) { _initialConditions = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("inputDir")==0) { _inputDir = line.substr(pos+_delim.length(),line.npos).c_str(); }

    else if (var.compare("debug")==0) { _debug = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("startOnDynamic")==0) { _startOnDynamic = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

  }
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
// Check that required fields have been set by the input file
PetscErrorCode strikeSlip_linearElastic_qd_fd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);
  if (_loadICs) { assert(_guessSteadyStateICs == 0); }

  assert(_thermalCoupling.compare("coupled")==0 ||
      _thermalCoupling.compare("uncoupled")==0 ||
      _thermalCoupling.compare("no")==0 );

  assert(_hydraulicCoupling.compare("coupled")==0 ||
      _hydraulicCoupling.compare("uncoupled")==0 ||
      _hydraulicCoupling.compare("no")==0 );

  assert(_timeIntegrator.compare("FEuler")==0 ||
      _timeIntegrator.compare("RK32")==0 ||
      _timeIntegrator.compare("RK43")==0 ||
      _timeIntegrator.compare("RK32_WBE")==0 ||
    _timeIntegrator.compare("RK43_WBE")==0 ||
      _timeIntegrator.compare("WaveEq")==0 );

  assert(_timeIntegrator.compare("FEuler")==0
    || _timeIntegrator.compare("RK32")==0
    || _timeIntegrator.compare("RK43")==0
    || _timeIntegrator.compare("RK32_WBE")==0
    || _timeIntegrator.compare("RK43_WBE")==0
    || _timeIntegrator.compare("WaveEq")==0);

  assert(_timeControlType.compare("P")==0 ||
         _timeControlType.compare("PI")==0 ||
         _timeControlType.compare("PID")==0 );

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {_initDeltaT = _minDeltaT; }
  assert(_maxStepCount_dyn >= 0);
  assert(_initTime >= 0);
  assert(_maxTime_dyn >= 0 && _maxTime_dyn>=_initTime);
  assert(_atol >= 1e-14);
  assert(_minDeltaT >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);

    // check boundary condition types for momentum balance equation
  assert(_qd_bcLType.compare("outGoingCharacteristics")==0 ||
    _qd_bcRType.compare("freeSurface")==0 ||
    _qd_bcRType.compare("tau")==0 ||
    _qd_bcRType.compare("remoteLoading")==0 ||
    _qd_bcRType.compare("symm_fault")==0 ||
    _qd_bcRType.compare("rigid_fault")==0 );

  assert(_qd_bcLType.compare("outGoingCharacteristics")==0 ||
    _qd_bcTType.compare("freeSurface")==0 ||
    _qd_bcTType.compare("tau")==0 ||
    _qd_bcTType.compare("remoteLoading")==0 ||
    _qd_bcTType.compare("symm_fault")==0 ||
    _qd_bcTType.compare("rigid_fault")==0 );

  assert(_qd_bcLType.compare("outGoingCharacteristics")==0 ||
    _qd_bcLType.compare("freeSurface")==0 ||
    _qd_bcLType.compare("tau")==0 ||
    _qd_bcLType.compare("remoteLoading")==0 ||
    _qd_bcLType.compare("symm_fault")==0 ||
    _qd_bcLType.compare("rigid_fault")==0 );

  assert(_qd_bcLType.compare("outGoingCharacteristics")==0 ||
    _qd_bcBType.compare("freeSurface")==0 ||
    _qd_bcBType.compare("tau")==0 ||
    _qd_bcBType.compare("remoteLoading")==0 ||
    _qd_bcBType.compare("symm_fault")==0 ||
    _qd_bcBType.compare("rigid_fault")==0 );

  if (_stateLaw.compare("flashHeating")==0) {
    assert(_thermalCoupling.compare("no")!=0);
  }

  if (_limit_dyn < _triggerqd2d){
    _limit_dyn = 10 * _triggerqd2d;
  }

  if (_limit_qd > _triggerd2qd){
    _limit_qd = _triggerqd2d / 10.0;
  }

  if (_limit_stride_dyn == -1){
    _limit_stride_dyn = _limit_dyn / 10.0;
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// parse boundary conditions
PetscErrorCode strikeSlip_linearElastic_qd_fd::parseBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::parseBCs()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_qd_bcRType.compare("symm_fault")==0 || _qd_bcRType.compare("rigid_fault")==0 || _qd_bcRType.compare("remoteLoading")==0) {
    _mat_qd_bcRType = "Dirichlet";
  }
  else if (_qd_bcRType.compare("freeSurface")==0 || _qd_bcRType.compare("tau")==0 || _qd_bcRType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcRType = "Neumann";
  }

  if (_qd_bcTType.compare("symm_fault")==0 || _qd_bcTType.compare("rigid_fault")==0 || _qd_bcTType.compare("remoteLoading")==0) {
    _mat_qd_bcTType = "Dirichlet";
  }
  else if (_qd_bcTType.compare("freeSurface")==0 || _qd_bcTType.compare("tau")==0 || _qd_bcTType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcTType = "Neumann";
  }

  if (_qd_bcLType.compare("symm_fault")==0 || _qd_bcLType.compare("rigid_fault")==0 || _qd_bcLType.compare("remoteLoading")==0) {
    _mat_qd_bcLType = "Dirichlet";
  }
  else if (_qd_bcLType.compare("freeSurface")==0 || _qd_bcLType.compare("tau")==0 || _qd_bcLType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcLType = "Neumann";
  }

  if (_qd_bcBType.compare("symm_fault")==0 || _qd_bcBType.compare("rigid_fault")==0 || _qd_bcBType.compare("remoteLoading")==0) {
    _mat_qd_bcBType = "Dirichlet";
  }
  else if (_qd_bcBType.compare("freeSurface")==0 || _qd_bcBType.compare("tau")==0 || _qd_bcBType.compare("outGoingCharacteristics")==0) {
    _mat_qd_bcBType = "Neumann";
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute allowed time step based on CFL condition and user input
PetscErrorCode strikeSlip_linearElastic_qd_fd::computeTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::computeTimeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // coefficient for CFL condition
  PetscScalar gcfl = 0.7071; // if order = 2
  if (_order == 4) { gcfl = 0.7071/sqrt(1.4498); }
  if (_order == 6) { gcfl = 0.7071/sqrt(2.1579); }


  // compute grid spacing in y and z
  Vec dy, dz;
  VecDuplicate(*_y,&dy);
  VecDuplicate(*_y,&dz);
  if (_D->_sbpType.compare("mfc_coordTrans")==0){
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatGetDiagonal(yq, dy); VecScale(dy,1.0/(_Ny-1));
    MatGetDiagonal(zr, dz); VecScale(dz,1.0/(_Nz-1));
  }
  else {
    VecSet(dy,_Ly/(_Ny-1.0));
    VecSet(dz,_Lz/(_Nz-1.0));
  }

  // compute time for shear wave to travel 1 dy or dz
  Vec ts_dy,ts_dz;
  VecDuplicate(*_y,&ts_dy);
  VecDuplicate(*_z,&ts_dz);
  VecPointwiseDivide(ts_dy,dy,_cs);
  VecPointwiseDivide(ts_dz,dz,_cs);
  PetscScalar min_ts_dy, min_ts_dz;
  VecMin(ts_dy,NULL,&min_ts_dy);
  VecMin(ts_dz,NULL,&min_ts_dz);

  // clean up memory usage
  VecDestroy(&dy);
  VecDestroy(&dz);
  VecDestroy(&ts_dy);
  VecDestroy(&ts_dz);

  // largest possible time step permitted by CFL condition
  PetscScalar max_deltaT = gcfl * min(min_ts_dy,min_ts_dz);


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

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute alphay and alphaz for use in time stepping routines
PetscErrorCode strikeSlip_linearElastic_qd_fd::computePenaltyVectors()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::computePenaltyVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar h11y, h11z;
  _material->_sbp->geth11(h11y, h11z);

  Vec alphay,alphaz;
  VecDuplicate(*_y, &alphay); VecSet(alphay,h11y);
  VecDuplicate(*_y, &alphaz); VecSet(alphaz,h11z);
  if(_D->_sbpType.compare("mfc_coordTrans")==0){
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
    if ( (Ii/_Nz == 0) && (_dyn_bcLType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11y; }
    if ( (Ii/_Nz == _Ny-1) && (_dyn_bcRType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11y; }
    if ( (Ii%_Nz == 0) && (_dyn_bcTType.compare("outGoingCharacteristics") == 0 )) { ay[Jj] += 0.5 / h11z; }
    if ( ((Ii+1)%_Nz == 0) && (_dyn_bcBType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11z; }
    Jj++;
  }
  VecRestoreArray(_ay,&ay);

  ierr = VecPointwiseMult(_ay, _ay, _cs);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}





PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate(){
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime0 = MPI_Wtime();

  integrate_qd();
  _qdTime += MPI_Wtime() - startTime0;
  double startTime1 = MPI_Wtime();
  integrate_dyn();
  _dynTime += MPI_Wtime() - startTime1;
  _integrateTime += MPI_Wtime() - startTime0;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}





bool strikeSlip_linearElastic_qd_fd::check_switch(const Fault* _fault){
  bool mustswitch = false;
  Vec absSlipVel;
  VecDuplicate(_fault->_slipVel, &absSlipVel);
  VecCopy(_fault->_slipVel, absSlipVel);
  PetscInt index;
  PetscScalar max_value;
  VecAbs(absSlipVel);
  VecMax(absSlipVel, &index, &max_value);

  #if VERBOSE > 0
    PetscPrintf(PETSC_COMM_WORLD, "max slipVel = %g\n", max_value);
  #endif

  if(_currTime > _maxTime || _stepCount > _maxStepCount){
    mustswitch = true;
    _maxStepCount_dyn = 0;
    _maxStepCount_qd = 0;
    _D->_numCycles = 0;
  }
  if(_inDynamic){
    if(!_allowed){
      if(max_value > _limit_dyn){
        _allowed = true;
      }
    }
    if (_allowed && max_value < _limit_stride_dyn){
      _stride1D = _stride1D_dyn_long;
      _stride2D = _stride2D_dyn_long;
      //~ PetscPrintf(PETSC_COMM_WORLD,"stride1D is now: %i\n",_stride1D);
    }
    if(_allowed && max_value < _triggerd2qd){
      mustswitch = true;
    }
  }
  else{
    if(!_allowed){
      if(max_value < _limit_qd){
        _allowed = true;
      }
    }
    if(_allowed && max_value > _triggerqd2d){
      mustswitch = true;
    }
  }
  VecDestroy(&absSlipVel);
  return mustswitch;
}


PetscErrorCode strikeSlip_linearElastic_qd_fd::reset_for_qd(){
  PetscErrorCode ierr = 0;
  // Force writing output
  if(_stepCount % _stride1D > 0){
    PetscInt stride1d, stride2d;
    stride1d = _stride1D;
    stride2d = _stride2D;
    _stride1D = 1;
    _stride2D = 1;
    timeMonitor(_currTime, _deltaT,_stepCount, _stride1D);
    _stride1D = stride1d;
    _stride2D = stride2d;
  }

  _localStep = 0;

  _allowed = false;
  // _limit = 1e-8;
  _varEx = _quadWaveEx->getVar();
  _firstCycle = false;
  _inDynamic = false;

  VecCopy(_fault_fd->_psi, _varEx["psi"]);
  VecCopy(_fault_fd->_psi, _fault_qd->_psi);
  VecCopy(_fault_fd->_slipVel, _fault_qd->_slipVel);

  // VecAXPY(_material->_u, 1.0, _savedU);
  VecCopy(_fault_fd->_slip, _varEx["slip"]);
  VecCopy(_varEx["slip"], _fault_qd->_slip);

  _fault_qd->_viewers.swap(_fault_fd->_viewers);
  _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);

  _varEx.erase("u");
  _varEx.erase("uPrev");
  return ierr;
}

//===========================================================================================================
// Quasi dynamic stuff
//===========================================================================================================


// initiate variables to be integrated in time
PetscErrorCode strikeSlip_linearElastic_qd_fd::initiateIntegrand_qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::initiateIntegrand_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Mat A;
  _material->_sbp->getA(A);
  _material->setupKSP(_material->_sbp,_material->_ksp,_material->_pc,A);

  if (_isMMS) { _material->setMMSInitialConditions(_initTime); }

  if(_firstCycle){
    VecSet(_material->_bcR,_vL*_initTime/2.0);

    Vec slip;
    VecDuplicate(_material->_bcL,&slip);
    VecSet(slip,0.);
    _varEx["slip"] = slip;

    Vec psi;
    VecDuplicate(_material->_bcL,&psi);
    VecSet(psi,0.);
    _varEx["psi"] = psi;

    if (_guessSteadyStateICs) { solveSS(); }
    _fault_qd->initiateIntegrand(_initTime,_varEx);

    if (_thermalCoupling.compare("no")!=0 ) {
      _he->initiateIntegrand(_initTime,_varEx,_varIm);
    }
    if (_hydraulicCoupling.compare("no")!=0 ) {
      _p->initiateIntegrand(_initTime,_varEx,_varIm);
    }

    if (_inputDir.compare("unspecified") != 0){

      ierr = loadFileIfExists_matlab(_inputDir+"u", _material->_u);
      if (ierr > 0){
          VecSet(_material->_u, 0.0);
      }

      ierr = loadFileIfExists_matlab(_inputDir + "psi", _varEx["psi"]);
      ierr = loadFileIfExists_matlab(_inputDir + "slipVel", _fault_qd->_slipVel);
      ierr = loadFileIfExists_matlab(_inputDir + "bcR", _material->_bcRShift);
      ierr = loadFileIfExists_matlab(_inputDir + "bcL", _material->_bcL);

      VecCopy(_material->_bcL, _varEx["slip"]);
      VecScale(_varEx["slip"], 2.0);
      VecCopy(_varEx["slip"], _fault_qd->_slip);
    }
  }
  ierr = _material->_sbp->setRhs(_material->_rhs,_material->_bcL,_material->_bcR,_material->_bcT,_material->_bcB);CHKERRQ(ierr);
  _stride1D = _stride1D_qd;
  _stride2D = _stride2D_qd;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




PetscErrorCode strikeSlip_linearElastic_qd_fd::writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_timeV1D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_time1D.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_dt1D.txt").c_str(),&_dtimeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_dT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"regime.txt").c_str(),&_whichRegime);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_whichRegime, "%i\n",_inDynamic);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_dT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_whichRegime, "%i\n",_inDynamic);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_timeV2D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode strikeSlip_linearElastic_qd_fd::view()
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"strikeSlip_linearElastic_qd_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent propagating the wave (s): %g\n",_propagateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in quasidynamic (s): %g\n",_qdTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in dynamic (s): %g\n",_dynTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time (s): %g\n",totRunTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",(_writeTime/_integrateTime)*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // output scalar fields
  std::string str = _outputDir + "mediator_context.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"vL = %g\n",_vL);CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_qd = %i\n",_stride1D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_qd = %i\n",_stride2D_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_dyn = %i\n",_stride1D_dyn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_dyn = %i\n",_stride2D_dyn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D_dyn = %i\n",_stride1D_dyn_long);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D_long = %i\n",_stride2D_dyn_long);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"minDeltaT = %.15e # (s)\n",_minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxDeltaT = %.15e # (s)\n",_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initDeltaT = %.15e # (s)\n",_initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"atol = %.15e\n",_atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"triggerqd2d = %.15e\n",_triggerqd2d);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"triggerd2qd = %.15e\n",_triggerd2qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_qd = %.15e\n",_limit_qd);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_dyn = %.15e\n",_limit_dyn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"limit_stride_dyn = %.15e\n",_limit_stride_dyn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"CFL = %.15e\n",_CFL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _material->writeContext(_outputDir);
  _fault_qd->writeContext(_outputDir);
  if (_thermalCoupling.compare("no")!=0) {
    _he->writeContext(_outputDir);
  }
  if (_hydraulicCoupling.compare("no")!=0) {
    _p->writeContext(_outputDir);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================

PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate_qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand_qd(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    if (!_firstCycle){
      delete _quadEx_qd;
      delete _quadEx_switch;
    }
    _quadEx_qd = new FEuler(_maxStepCount_qd,_maxTime_qd,_initDeltaT,_timeControlType);
    if(!_firstCycle){
      _quadEx_qd->_stepCount = _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_maxNumSteps = _maxStepCount_qd + _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_currT = _currTime;
    }
    _quadEx_switch = new FEuler(1,_deltaT,_deltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    if (!_firstCycle){
      delete _quadEx_qd;
      delete _quadEx_switch;
    }
    _quadEx_qd = new RK32(_maxStepCount_qd,_maxTime_qd,_initDeltaT,_timeControlType);
    if(!_firstCycle){
      _quadEx_qd->_stepCount = _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_maxNumSteps = _maxStepCount_qd + _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_currT = _currTime;
    }
    _quadEx_switch = new RK32(1,_deltaT,_deltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    if (!_firstCycle){
      delete _quadEx_qd;
      delete _quadEx_switch;
    }
    _quadEx_qd = new RK43(_maxStepCount_qd,_maxTime_qd,_initDeltaT,_timeControlType);
    if(!_firstCycle){
      _quadEx_qd->_stepCount = _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_maxNumSteps =_maxStepCount_qd +  _quadWaveEx->_stepCount + 1;
      _quadEx_qd->_currT = _currTime;
    }
    _quadEx_switch = new RK43(1,_deltaT,_deltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    if (!_firstCycle){
      delete _quadImex_qd;
      delete _quadImex_switch;
    }
    _quadImex_qd = new RK32_WBE(_maxStepCount_qd,_maxTime_qd,_initDeltaT,_timeControlType);
    if(!_firstCycle){
      _quadImex_qd->_stepCount = _quadWaveEx->_stepCount + 1;
      _quadImex_qd->_maxNumSteps = _maxStepCount_qd + _quadWaveEx->_stepCount + 1;
      _quadImex_qd->_currT = _currTime;
    }
    _quadImex_switch = new RK32_WBE(1,_deltaT,_deltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    if (!_firstCycle){
      delete _quadImex_qd;
      delete _quadImex_switch;
    }
    _quadImex_qd = new RK43_WBE(_maxStepCount_qd,_maxTime_qd,_initDeltaT,_timeControlType);
    if(!_firstCycle){
      _quadImex_qd->_stepCount = _quadWaveEx->_stepCount + 1;
      _quadImex_qd->_maxNumSteps = _maxStepCount_qd + _quadWaveEx->_stepCount + 1;
      _quadImex_qd->_currT = _currTime;
    }
    _quadImex_switch = new RK43_WBE(1,_deltaT,_deltaT,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex_qd->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex_qd->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex_qd->setTimeRange(_currTime,_maxTime_qd);
    ierr = _quadImex_qd->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadImex_qd->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);
    ierr = _quadImex_qd->setErrInds(_timeIntInds); // control which fields are used to select step size
    if (_maxStepCount_qd > 0 && _startOnDynamic != 1){
      ierr = _quadImex_qd->integrate(this);CHKERRQ(ierr);
    }
  }
  else {
    _quadEx_qd->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx_qd->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx_qd->setTimeRange(_currTime,_maxTime_qd);
    ierr = _quadEx_qd->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadEx_qd->setInitialConds(_varEx);CHKERRQ(ierr);
    ierr = _quadEx_qd->setErrInds(_timeIntInds); // control which fields are used to select step size
    if (_maxStepCount_qd > 0 && _startOnDynamic != 1){
      ierr = _quadEx_qd->integrate(this);CHKERRQ(ierr);
    }
  }
  _integrateTime = MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode strikeSlip_linearElastic_qd_fd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update rhs
  //~ if (_isMMS) { _material->setMMSBoundaryConditions(time); }
  _material->setRHS();
  //~ if (_isMMS) { _material->addRHS_MMSSource(time,_material->_rhs); }

  _material->computeU();
  _material->computeStresses();

  return ierr;
}

// guess at the steady-state solution
PetscErrorCode strikeSlip_linearElastic_qd_fd::solveSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // compute steady state stress on fault
  Vec tauSS = NULL;
  _fault_qd->computeTauRS(tauSS,_vL); // rate and state tauSS assuming velocity is vL

  if (_inputDir.compare("unspecified") != 0) {
    ierr = loadVecFromInputFile(tauSS,_inputDir,"tauSS"); CHKERRQ(ierr);
  }
  ierr = io_initiateWriteAppend(_viewers, "tau", tauSS, _outputDir + "SS_tau"); CHKERRQ(ierr);

  // compute compute u that satisfies tau at left boundary
  VecCopy(tauSS,_material->_bcL);
  _material->setRHS();
  _material->computeU();
  _material->computeStresses();


  // update fault to contain correct stresses
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = _fault_qd->setTauQS(sxy); CHKERRQ(ierr);

  // update boundary conditions, stresses
  solveSSb();
  _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);

  VecDestroy(&tauSS);

  // steady state temperature
  if (_thermalCoupling.compare("no")!=0) {
    ierr = writeVec(_he->_Tamb,_outputDir + "SS_T0"); CHKERRQ(ierr);
    _material->getStresses(sxy,sxz,sdev);
    Vec T; VecDuplicate(sxy,&T);
    _he->computeSteadyStateTemp(_currTime,_fault_qd->_slipVel,_fault_qd->_tauP,NULL,NULL,NULL,T);
    VecCopy(T,_he->_Tamb);
    ierr = writeVec(_he->_Tamb,_outputDir + "SS_TSS"); CHKERRQ(ierr);
    VecDestroy(&T);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// update the boundary conditions based on new steady state u
PetscErrorCode strikeSlip_linearElastic_qd_fd::solveSSb()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::solveSSb";
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

  if (_varEx.find("slip") != _varEx.end() ) { VecCopy(uL,_varEx["slip"]); }
  else {
    Vec slip;
    VecDuplicate(_material->_bcL,&slip);
    VecCopy(uL,slip);
    _varEx["slip"] = slip;
  }

  if (_qd_bcLType.compare("symm_fault")==0 || _qd_bcLType.compare("rigid_fault")==0 || _qd_bcLType.compare("remoteLoading")==0) {
    VecCopy(uL,_material->_bcL);
  }
  if (_qd_bcLType.compare("symm_fault")==0) {
    VecScale(_varEx["slip"],2.0);
  }

  VecDestroy(&uL);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::measureMMSError()
{
  PetscErrorCode ierr = 0;

  _material->measureMMSError(_currTime);
  //~ _he->measureMMSError(_currTime);
  //~ _p->measureMMSError(_currTime);

  return ierr;
}


// ================================================================================================
// Dynamic part
// ================================================================================================

PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate_dyn()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();
  if(_maxStepCount_dyn > 0){
    initiateIntegrand_dyn(); // put initial conditions into var for integration
    _stepCount = 0;

    // initialize time integrator
    if(_maxStepCount_qd > 0 || _startOnDynamic){
    if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
      if (!_firstCycle){
        delete _quadWaveImex;
      }
        _quadWaveImex = new OdeSolver_WaveEq_Imex(_maxStepCount_dyn,_currTime+_deltaT,_maxTime_dyn,_deltaT);
        _quadWaveImex->_stepCount = _quadImex_qd->_stepCount + 3;
        _quadWaveImex->_maxNumSteps = _maxStepCount_dyn + _quadImex_qd->_stepCount + 3;
        ierr = _quadWaveImex->setInitialConds(_varEx, _varIm);CHKERRQ(ierr);
        ierr = _quadWaveImex->integrate(this);CHKERRQ(ierr);
      }
      else{
        if (!_firstCycle){
          delete _quadWaveEx;
        }
        _quadWaveEx = new OdeSolver_WaveEq(_maxStepCount_dyn,_currTime+_deltaT,_maxTime_dyn,_deltaT);
        _quadWaveEx->_stepCount = _quadEx_qd->_stepCount + 3;
        _quadWaveEx->_maxNumSteps = _maxStepCount_dyn + _quadEx_qd->_stepCount + 3;
        ierr = _quadWaveEx->setInitialConds(_varEx);CHKERRQ(ierr);
        ierr = _quadWaveEx->integrate(this);CHKERRQ(ierr);
      }
    }
    else{
      if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
        if (!_firstCycle){
          delete _quadWaveImex;
        }
        _quadWaveImex = new OdeSolver_WaveEq_Imex(_maxStepCount_dyn,_currTime+_deltaT,_maxTime_dyn,_deltaT);
        _quadWaveImex->_stepCount = _quadImex_qd->_stepCount + 2;
        _quadWaveImex->_maxNumSteps = _maxStepCount_dyn + _quadImex_qd->_stepCount + 2;
        ierr = _quadWaveImex->setInitialConds(_varEx, _varIm);CHKERRQ(ierr);
        ierr = _quadWaveImex->integrate(this);CHKERRQ(ierr);
      }
      else{
        if (!_firstCycle){
          delete _quadWaveEx;
        }
        _quadWaveEx = new OdeSolver_WaveEq(_maxStepCount_dyn,_currTime+_deltaT,_maxTime_dyn,_deltaT);
        _quadWaveEx->_stepCount = _quadEx_qd->_stepCount + 2;
        _quadWaveEx->_maxNumSteps = _maxStepCount_dyn + _quadEx_qd->_stepCount + 2;
        ierr = _quadWaveEx->setInitialConds(_varEx);CHKERRQ(ierr);
        ierr = _quadWaveEx->integrate(this);CHKERRQ(ierr);
      }
    }

    _integrateTime += MPI_Wtime() - startTime;

    reset_for_qd();
  }
  _firstCycle = false;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



/*
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx){
  PetscErrorCode ierr = 0;
  if(_inDynamic){
    ierr = strikeSlip_linearElastic_qd_fd::d_dt_dyn(time,varEx,dvarEx);
  }
  else{
    ierr = strikeSlip_linearElastic_qd_fd::d_dt_qd(time,varEx,dvarEx);
  }
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time, const map<string,Vec>& varEx,map<string,Vec>& dvarEx){
  PetscErrorCode ierr = 0;
  if(!_inDynamic){
    ierr = strikeSlip_linearElastic_qd_fd::d_dt_qd(time,varEx,dvarEx);
  }
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt){
  PetscErrorCode ierr = 0;
  ierr = strikeSlip_linearElastic_qd_fd::d_dt_qd(time,varEx,dvarEx,varIm,varImo,dt);
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,map<string,Vec>& varImo){
  PetscErrorCode ierr = 0;
  ierr = strikeSlip_linearElastic_qd_fd::d_dt_dyn(time,varEx,dvarEx,varIm,varImo);
  return ierr;
}*/

// quasidynamic: purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt qd explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType.compare("symm_fault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,0.5);CHKERRQ(ierr);
  }
  else if (_qd_bcLType.compare("rigid_fault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
  }
  ierr = VecSet(_material->_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);

  if (_hydraulicCoupling.compare("coupled")==0 && varEx.find("pressure") != varEx.end() ) {
    _fault_qd->setSNEff(varEx.find("pressure")->second);
  }
  _fault_qd->updateFields(time,varEx);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->updateFields(time,varEx);
  }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->d_dt(time,varEx,dvarEx);
  }

  // update fields on fault from other classes
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = _fault_qd->setTauQS(sxy); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// quasidynamic: implicit/explicit time stepping
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt qd IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_qd_bcLType.compare("symm_fault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,0.5);CHKERRQ(ierr);
  }
  else if (_qd_bcLType.compare("rigid_fault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
  }
  ierr = VecSet(_material->_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);

  _fault_qd->updateFields(time,varEx);

  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->updateFields(time,varEx,varImo);
  }

  // update temperature in momBal
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling.compare("coupled")==0) {
    _fault_qd->updateTemperature(varImo.find("Temp")->second);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault_qd->setSNEff(_p->_p);
  }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);
  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
    // _p->d_dt(time,varEx,dvarEx);
  }

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = _fault_qd->setTauQS(sxy); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault_qd->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  // heat equation
  if (varIm.find("Temp") != varIm.end()) {
    //~ PetscPrintf(PETSC_COMM_WORLD,"Computing new steady state temperature at stepCount = %i\n",_stepCount);
    Vec sxy,sxz,sdev;
    _material->getStresses(sxy,sxz,sdev);
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault_qd->_tauP;
    Vec gVxy_t = NULL;
    Vec gVxz_t = NULL;
    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,sdev,gVxy_t,gVxz_t,varIm["Temp"],Told,dt); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, map<string,Vec>& var, map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt fd explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  double startPropagation = MPI_Wtime();

  // compute D2u = (Dyy+Dzz)*u
  Vec D2u, temp;
  VecDuplicate(*_y, &D2u);
  VecDuplicate(*_y, &temp);
  Mat A; _material->_sbp->getA(A);
  ierr = VecAXPY(temp, 1.0, _Fhat); // !!! Fhat term
  ierr = _material->_sbp->Hinv(temp, D2u);
  VecDestroy(&temp);
  if(_D->_sbpType.compare("mfc_coordTrans")==0){
      Mat J,Jinv,qy,rz,yq,zr;
      ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
      Vec temp;
      VecDuplicate(D2u, &temp);
      MatMult(Jinv, D2u, temp);
      VecCopy(temp, D2u);
      VecDestroy(&temp);
  }
  _fault_fd->setGetBody2Fault(D2u,_fault_fd->_d2u,SCATTER_FORWARD); // set D2u to fault


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
  ierr = VecGetArrayRead(_rhoVec, &rho);

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
  ierr = VecRestoreArrayRead(_rhoVec, &rho);

  VecDestroy(&D2u);

_propagateTime += MPI_Wtime() - startPropagation;

  if (_initialConditions.compare("tau")==0) { _fault_fd->updateTau0(time); }
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);

  // update body u from fault u
  _fault_fd->setGetBody2Fault(varNext["u"], _fault_fd->_u, SCATTER_REVERSE); // update body u with newly computed fault u

  VecCopy(varNext.find("u")->second, _material->_u);
  _material->computeStresses();
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  _fault_fd->setGetBody2Fault(sxy,_fault_fd->_tauP,SCATTER_FORWARD); // update shear stress on fault
  VecAXPY(_fault_fd->_tauP, 1.0, _fault_fd->_tau0);
  VecCopy(_fault_fd->_tauP,_fault_fd->_tauQSP); // keep quasi-static shear stress updated as well


  if (_qd_bcLType.compare("symm_fault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,0.5);CHKERRQ(ierr);
  }
  else if (_qd_bcLType.compare("rigid_fault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
  }
  ierr = VecSet(_material->_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: IMEX time stepping
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, map<string,Vec>& var, map<string,Vec>& varPrev,
      map<string,Vec>& varIm,map<string,Vec>& varImo)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt fd imex";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  double startPropagation = MPI_Wtime();

  // compute D2u = (Dyy+Dzz)*u
  Vec D2u, temp;
  VecDuplicate(*_y, &D2u);
  VecDuplicate(*_y, &temp);
  Mat A; _material->_sbp->getA(A);
  ierr = VecAXPY(temp, 1.0, _Fhat); // !!! Fhat term
  ierr = _material->_sbp->Hinv(temp, D2u);
  VecDestroy(&temp);
  if(_D->_sbpType.compare("mfc_coordTrans")==0){
      Mat J,Jinv,qy,rz,yq,zr;
      ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
      Vec temp;
      VecDuplicate(D2u, &temp);
      MatMult(Jinv, D2u, temp);
      VecCopy(temp, D2u);
      VecDestroy(&temp);
  }
  _fault_fd->setGetBody2Fault(D2u,_fault_fd->_d2u,SCATTER_FORWARD); // set D2u to fault


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
  ierr = VecGetArrayRead(_rhoVec, &rho);

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
  ierr = VecRestoreArrayRead(_rhoVec, &rho);

  VecDestroy(&D2u);

_propagateTime += MPI_Wtime() - startPropagation;

  if (_initialConditions.compare("tau")==0) { _fault_fd->updateTau0(time); }
  ierr = _fault_fd->d_dt(time,_deltaT,varNext,var,varPrev);CHKERRQ(ierr);

  // update body u from fault u
  _fault_fd->setGetBody2Fault(varNext["u"], _fault_fd->_u, SCATTER_REVERSE); // update body u with newly computed fault u

  VecCopy(varNext.find("u")->second, _material->_u);
  _material->computeStresses();
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  _fault_fd->setGetBody2Fault(sxy,_fault_fd->_tauP,SCATTER_FORWARD); // update shear stress on fault
  VecAXPY(_fault_fd->_tauP, 1.0, _fault_fd->_tau0);
  VecCopy(_fault_fd->_tauP,_fault_fd->_tauQSP); // keep quasi-static shear stress updated as well

  if (_qd_bcLType.compare("symm_fault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,0.5);CHKERRQ(ierr);
  }
  else if (_qd_bcLType.compare("rigid_fault")==0) {
    ierr = VecCopy(_fault_fd->_slip,_material->_bcL);CHKERRQ(ierr);
  }
  ierr = VecSet(_material->_bcR,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);


    // heat equation
  if (varIm.find("Temp") != varIm.end()) {
    //~ PetscPrintf(PETSC_COMM_WORLD,"Computing new steady state temperature at stepCount = %i\n",_stepCount);
    Vec sxy,sxz,sdev;
    _material->getStresses(sxy,sxz,sdev);
    Vec V = _fault_fd->_slipVel;
    Vec tau = _fault_fd->_tauP;
    Vec gVxy_t = NULL;
    Vec gVxz_t = NULL;
    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,sdev,gVxy_t,gVxz_t,varIm["Temp"],Told,deltaT); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode strikeSlip_linearElastic_qd_fd::initiateIntegrand_dyn()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) { _material->setMMSInitialConditions(_currTime); }

  // For checking over the switching

  // Force writing output
  if(_stepCount % _stride1D > 0){
    PetscInt stride1d, stride2d;
    stride1d = _stride1D;
    stride2d = _stride2D;
    _stride1D = 1;
    _stride2D = 1;
    timeMonitor(_currTime, _deltaT,_stepCount, _stride1D);
    _stride1D = stride1d;
    _stride2D = stride2d;
  }

  Vec uPrev;
  VecDuplicate(_material->_u, &uPrev);
  VecCopy(_material->_u, uPrev);

  VecDuplicate(_material->_u, &_savedU);
  VecCopy(_material->_u, _savedU);

  Mat A;
  ierr = _material->_sbp->getA(A);
  VecDuplicate(_material->_u, &_Fhat);
  MatMult(A, _material->_u, _Fhat);
  VecAXPY(_Fhat, -1, _material->_rhs);

  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    VecCopy(_quadImex_qd->getVar()["psi"], _fault_fd->_psiPrev);

    _varEx = _quadImex_qd->getVar();
    _varIm = _quadImex_qd->getVarIm();

    _quadImex_switch->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex_switch->setTimeStepBounds(_deltaT,_deltaT);CHKERRQ(ierr);
    ierr = _quadImex_switch->setTimeRange(_currTime,_maxTime_qd);
    ierr = _quadImex_switch->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadImex_switch->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);
    ierr = _quadImex_switch->setErrInds(_timeIntInds); // control which fields are used to select step size
    if(_maxStepCount_qd > 0 && _startOnDynamic != 1){
    _quadImex_switch->_stepCount = _quadImex_qd->_stepCount + 1;
    _quadImex_switch->_maxNumSteps += _quadImex_qd->_stepCount + 1;
    }
    ierr = _quadImex_switch->integrate(this);CHKERRQ(ierr);
  }
  else {
    VecCopy(_quadEx_qd->getVar()["psi"], _fault_fd->_psiPrev);

    _varEx = _quadEx_qd->getVar();

    _quadEx_switch->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx_switch->setTimeStepBounds(_deltaT,_deltaT);CHKERRQ(ierr);
    ierr = _quadEx_switch->setTimeRange(_currTime,_maxTime_qd);
    ierr = _quadEx_switch->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadEx_switch->setInitialConds(_varEx);CHKERRQ(ierr);
    ierr = _quadEx_switch->setErrInds(_timeIntInds); // control which fields are used to select step size
    if(_maxStepCount_qd > 0 && _startOnDynamic != 1){
    _quadEx_switch->_stepCount = _quadEx_qd->_stepCount + 1;
    _quadEx_switch->_maxNumSteps += _quadEx_qd->_stepCount + 1;
    }
    ierr = _quadEx_switch->integrate(this);CHKERRQ(ierr);
  }

  VecDuplicate(*_z, &_varEx["uPrev"]); VecSet(_varEx["uPrev"],0.);
  VecDuplicate(*_z, &_varEx["u"]); VecSet(_varEx["u"], 0.0);
  VecCopy(uPrev, _varEx["uPrev"]);
  VecCopy(_material->_u, _varEx["u"]);
  VecDestroy(&uPrev);

  VecCopy(_fault_qd->_slip, _fault_fd->_slip0);

  _allowed = false;
  // _limit = 1.0;
  _inDynamic = true;
  _localStep = 0;

  _stride1D = _stride1D_dyn;
  _stride2D = _stride2D_dyn;

  _fault_qd->_viewers.swap(_fault_fd->_viewers);
  // _fault_fd->writeUOffset(_savedU, _firstCycle, _outputDir);

  _material->computeStresses();
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = _fault_fd->setTauQS(sxy); CHKERRQ(ierr);
  VecCopy(_fault_fd->_tauQSP, _fault_fd->_tau0);

  _material->changeBCTypes(_mat_dyn_bcRType,_mat_dyn_bcTType,_mat_dyn_bcLType,_mat_dyn_bcBType);

  _fault_fd->initiateIntegrand(_initTime,_varEx);
  Vec slip;
  VecDuplicate(_varEx["psi"], &slip); VecSet(slip,0.);
  _varEx["slip"] = slip;
  Vec dslip;
  VecDuplicate(_varEx["psi"], &dslip); VecSet(dslip,0.);
  _varEx["dslip"] = dslip;

  VecCopy(_fault_qd->_slipVel, _varEx["dslip"]);
  VecCopy(_quadEx_switch->getVar()["psi"], _fault_fd->_psi);

  VecAXPY(_varEx["u"], -1.0, _varEx["uPrev"]);
  VecSet(_varEx["uPrev"], 0.0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// TODO get rid of this
PetscErrorCode strikeSlip_linearElastic_qd_fd::timeMonitor(const PetscScalar time,const PetscScalar deltaT,const PetscInt stepCount,int& stopIntegration){
  PetscErrorCode ierr = 0;
  if(_inDynamic){
    ierr = strikeSlip_linearElastic_qd_fd::timeMonitor_fd(time,deltaT,stepCount,stopIntegration);
  }
  else{
    ierr = strikeSlip_linearElastic_qd_fd::timeMonitor_qd(time,deltaT,stepCount,stopIntegration);
  }
  _localStep += 1;
  return ierr;
}



PetscErrorCode strikeSlip_linearElastic_qd_fd::timeMonitor_qd(const PetscScalar time,const PetscScalar deltaT,const PetscInt stepCount,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::timeMonitor_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _currTime = time;
  _deltaT = deltaT;
  _stepCount = stepCount;

  PetscInt localStride1d, localStride2d;
  if (_localStep < _debug){
    localStride1d = 1;
    localStride2d = 1;
  }
  else{
    localStride1d = _stride1D;
    localStride2d = _stride2D;
  }

  if (_stride1D>0 && stepCount % localStride1d == 0) {
    ierr = writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault_qd->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr); }
  }

  if (_stride2D>0 &&  stepCount % localStride2d == 0) {
    ierr = writeStep2D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr); }
  }

  if(check_switch(_fault_qd)){
    if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
      _quadImex_qd->_maxNumSteps = 0;
    }
    else{
      _quadEx_qd->_maxNumSteps = 0;
    }
  }

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e quasidynamic\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::timeMonitor_fd(const PetscScalar time,const PetscScalar deltaT,const PetscInt stepCount,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::timeMonitor_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;
  _dT = _deltaT;

  PetscInt localStride1d, localStride2d;
  if (_localStep < _debug){
    localStride1d = 1;
    localStride2d = 1;
  }
  else{
    localStride1d = _stride1D;
    localStride2d = _stride2D;
  }

  if ( _stride1D > 0 && stepCount % localStride1d == 0) {
    ierr = writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault_fd->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
  }

  if ( _stride2D > 0 && stepCount % localStride2d == 0) {
    ierr = writeStep2D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
  }

  if(check_switch(_fault_fd)){
    _quadWaveEx->_maxNumSteps = 0;
  }

_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e fully dynamic %i %i\n",stepCount,_currTime,_stride1D,_stride1D_dyn_long);CHKERRQ(ierr);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




PetscErrorCode strikeSlip_linearElastic_qd_fd::view_dyn()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault_fd->view(_integrateTime);
  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Domain Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Nz: %i\n",_Nz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Ny: %i\n",_Ny);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of processors: %i\n",num_proc);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"strikeSlip_linearElastic_qd_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent propagating the wave (s): %g\n",_propagateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}


