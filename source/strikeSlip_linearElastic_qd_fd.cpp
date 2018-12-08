#include "strikeSlip_linearElastic_qd_fd.hpp"

#define FILENAME "strikeSlip_linearElastic_qd_fd.cpp"

using namespace std;


strikeSlip_linearElastic_qd_fd::strikeSlip_linearElastic_qd_fd(Domain&D)
: _D(&D),_delim(D._delim),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(1e-9),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _guessSteadyStateICs(0.),_forcingType("no"),_faultTypeScale(2.0),
  _cycleCount(0),_maxNumCycles(1e3),
  _deltaT(-1), _CFL(-1),
  _y(&D._y),_z(&D._z),
  _inDynamic(false),_allowed(false),
  _trigger_qd2fd(1e-3), _trigger_fd2qd(1e-3), _limit_qd(10*_vL), _limit_fd(1e-1),_limit_stride_fd(-1),_u0(NULL),
  _timeIntegrator("RK43"),_timeControlType("PID"),
  _stride1D(10),_stride2D(10),
  _stride1D_qd(10),_stride2D_qd(10),_stride1D_fd(10),_stride2D_fd(10),_stride1D_fd_end(10),_stride2D_fd_end(10),
  _maxStepCount(1e8),
  _initTime(0),_currTime(0),_minDeltaT(1e-3),_maxDeltaT(1e10),_maxTime(1e15),
  _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
  _timeV1D(NULL),_dtimeV1D(NULL),_timeV2D(NULL),_regime1DV(NULL), _regime2DV(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),_miscTime(0),_dynTime(0), _qdTime(0),
  _qd_bcRType("remoteLoading"),_qd_bcTType("freeSurface"),_qd_bcLType("symmFault"),_qd_bcBType("freeSurface"),
  _fd_bcRType("outGoingCharacteristics"),_fd_bcTType("freeSurface"),_fd_bcLType("symmFault"),_fd_bcBType("outGoingCharacteristics"),
  _mat_fd_bcRType("Neumann"),_mat_fd_bcTType("Neumann"),_mat_fd_bcLType("Neumann"),_mat_fd_bcBType("Neumann"),
  _quadEx_qd(NULL),_quadImex_qd(NULL), _quadWaveEx(NULL),
  _fault_qd(NULL),_fault_fd(NULL), _material(NULL),_he(NULL),_p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::strikeSlip_linearElastic_qd_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  parseBCs();

  _body2fault = &(D._scatters["body2L"]);
  _fault_qd = new Fault_qd(D,D._scatters["body2L"],_faultTypeScale); // fault for quasidynamic problem
  _fault_fd = new Fault_fd(D, D._scatters["body2L"],_faultTypeScale); // fault for fully dynamic problem
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
    // _fault_fd->setSNEff(_p->_p);
  }

  // initiate momentum balance equation

  if (_guessSteadyStateICs) { _material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,"Neumann",_mat_qd_bcBType); }
  else {_material = new LinearElastic(D,_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType); }
  computePenaltyVectors();

  // body forcing term for ice stream
  _forcingTerm = NULL;
  if (_forcingType.compare("iceStream")==0) { constructIceStreamForcingTerm(); }

  computeTimeStep(); // compute fully dynamic time step size

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


  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_dtimeV1D);
  PetscViewerDestroy(&_timeV2D);
  PetscViewerDestroy(&_regime1DV);
  PetscViewerDestroy(&_regime2DV);
  VecDestroy(&_u0);
  VecDestroy(&_ay);
  VecDestroy(&_forcingTerm);

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
    else if (var.compare("forcingType")==0) { _forcingType = rhs.c_str(); }

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


    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof(rhs.c_str() ); }
    else if (var.compare("initTime")==0) {
      _initTime = atof(rhs.c_str() );
      _currTime = _initTime;
    }
    else if (var.compare("maxTime")==0) { _maxTime = atof(rhs.c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof(rhs.c_str() ); }
    else if (var.compare("timeStepTol")==0) { _timeStepTol = atof(rhs.c_str() ); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }
    else if (var.compare("scale")==0) { loadVectorFromInputFile(rhsFull,_scale); }
    else if (var.compare("normType")==0) { _normType = rhs.c_str(); }

    else if (var.compare("vL")==0) { _vL = atof(rhs.c_str() ); }

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

    else if (var.compare("inputDir")==0) { _inputDir = rhs.c_str(); }

    else if (var.compare("maxNumCycles")==0) { _maxNumCycles = atoi(rhs.c_str() ); }

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
    assert(_thermalCoupling.compare("no")!=0);
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
  if (_D->_order == 4) { gcfl = 0.7071/sqrt(1.4498); }
  if (_D->_order == 6) { gcfl = 0.7071/sqrt(2.1579); }


  // compute grid spacing in y and z
  Vec dy, dz;
  VecDuplicate(*_y,&dy);
  VecDuplicate(*_y,&dz);
  if (_D->_sbpType.compare("mfc_coordTrans")==0){
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





PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime_integrateTime = MPI_Wtime();

  // first cycle
  initiateIntegrands();

  // if start with quasidynamic phase
  {
double startTime_qd = MPI_Wtime();
    _allowed = false;
    _inDynamic = false;
    prepare_fd2qd();
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




// returns true if it's time to switch from qd to fd, or fd to qd, or if
// the maximum time or step count has been reached
bool strikeSlip_linearElastic_qd_fd::checkSwitchRegime(const Fault* _fault)
{
  bool mustSwitch = false;

  // if using max slip velocity as switching criteria
  //~ Vec absSlipVel;
  //~ VecDuplicate(_fault->_slipVel, &absSlipVel);
  //~ VecCopy(_fault->_slipVel, absSlipVel);
  //~ PetscScalar maxV;
  //~ VecAbs(absSlipVel);
  //~ VecMax(absSlipVel, NULL, &maxV);
  //~ VecDestroy(&absSlipVel);

  // if using R = eta*V / tauQS
  Vec R; VecDuplicate(_fault->_slipVel,&R);
  VecPointwiseMult(R,_fault_qd->_eta_rad,_fault->_slipVel);
  VecPointwiseDivide(R,R,_fault->_tauQSP);
  PetscScalar maxV;
  VecMax(R,NULL,&maxV);
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
  if (_inDynamic && _allowed && maxV < _limit_stride_fd) {
    _stride1D = _stride1D_fd_end;
    _stride2D = _stride2D_fd_end;
  }

  //~ if(_inDynamic){
    //~ if(!_allowed){
      //~ if(maxV > _limit_fd){
        //~ _allowed = true;
      //~ }
    //~ }
    //~ if (_allowed && maxV < _limit_stride_fd){
      //~ _stride1D = _stride1D_fd_end;
      //~ _stride2D = _stride2D_fd_end;
    //~ }
    //~ if(_allowed && maxV < _trigger_fd2qd){
      //~ mustSwitch = true;
    //~ }
  //~ }
  //~ else{
    //~ if(!_allowed){
      //~ if(maxV < _limit_qd){
        //~ _allowed = true;
      //~ }
    //~ }
    //~ if(_allowed && maxV > _trigger_qd2fd){
      //~ mustSwitch = true;
    //~ }
  //~ }
  return mustSwitch;
}


// initiate varQSEx, varIm, and varFD
// includes computation of steady-state initial conditions if necessary
// should only be called once before the 1st earthquake cycle
PetscErrorCode strikeSlip_linearElastic_qd_fd::initiateIntegrands()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::initiateIntegrands()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initiate integrand for QD

  // LinearElastic does not set up its KSP, so must set it up here
  Mat A; _material->_sbp->getA(A);
  _material->setupKSP(_material->_sbp,_material->_ksp,_material->_pc,A);

  Vec slip;
  VecDuplicate(_material->_bcL,&slip);
  VecCopy(_material->_bcL,slip);
  if (_qd_bcLType.compare("symmFault")==0) {
    VecScale(slip,2.0);
  }
  ierr = loadVecFromInputFile(slip,_inputDir,"slip"); CHKERRQ(ierr);
  _varQSEx["slip"] = slip;

  if (_guessSteadyStateICs) { solveSS(); }

  VecCopy(_varQSEx["slip"],_fault_qd->_slip);
  _fault_qd->initiateIntegrand(_initTime,_varQSEx);


  // initiate integrand for varIm
  if (_thermalCoupling.compare("no")!=0 ) {
    _he->initiateIntegrand(_initTime,_varQSEx,_varIm);
  }
  if (_hydraulicCoupling.compare("no")!=0 ) {
    _p->initiateIntegrand(_initTime,_varQSEx,_varIm);
  }


  // initiate integrand for FD:
  // ensure fault_fd == fault_qd
  VecCopy(_fault_qd->_psi,      _fault_fd->_psi);
  VecCopy(_fault_qd->_slipVel,  _fault_fd->_slipVel);
  VecCopy(_fault_qd->_slip,     _fault_fd->_slip);
  VecCopy(_fault_qd->_slip,     _fault_fd->_slip0);
  VecCopy(_fault_qd->_tauQSP,   _fault_fd->_tauQSP);
  VecCopy(_fault_qd->_tauP,     _fault_fd->_tauP);
  VecCopy(_fault_qd->_strength,  _fault_fd->_tau0);
  VecCopy(_fault_qd->_prestress,  _fault_fd->_prestress);

  // add psi and slip to varFD
  _fault_fd->initiateIntegrand(_initTime,_varFD); // adds psi and slip

  // add u
  VecDuplicate(_material->_u, &_varFD["u"]); VecCopy(_material->_u,_varFD["u"]);
  VecDuplicate(_material->_u, &_u0); VecSet(_u0,0.0);

  // if solving the heat equation, add temperature to varFD
  if (_thermalCoupling.compare("no")!=0 ) { VecDuplicate(_varIm["Temp"], &_varFD["Temp"]); VecCopy(_varIm["Temp"], _varFD["Temp"]); }
  if (_hydraulicCoupling.compare("no")!=0 ) {
    VecDuplicate(_varIm["pressure"], &_varFD["pressure"]);
    VecCopy(_varIm["pressure"], _varFD["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      VecDuplicate(_varQSEx["permeability"], &_varFD["permeability"]);
      VecCopy(_varQSEx["permeability"], _varFD["permeability"]);
    }
  }

   // copy varFD into varFDPrev
  for (map<string,Vec>::iterator it = _varFD.begin(); it != _varFD.end(); it++ ) {
    VecDuplicate(_varFD[it->first],&_varFDPrev[it->first]); VecCopy(_varFD[it->first],_varFDPrev[it->first]);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// move from a fully dynamic phase to a quasidynamic phase
PetscErrorCode strikeSlip_linearElastic_qd_fd::prepare_fd2qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::prepare_fd2qd()";
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
  if (_thermalCoupling.compare("no")!=0 ) { VecCopy(_varFD["Temp"],_varIm["Temp"]); } // if solving the heat equation
  if (_hydraulicCoupling.compare("no")!=0 ) {
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
  VecCopy(_fault_fd->_psi,       _fault_qd->_psi);
  VecCopy(_fault_fd->_slipVel,   _fault_qd->_slipVel);
  VecCopy(_fault_fd->_slip,      _fault_qd->_slip);
  VecCopy(_fault_fd->_tauP,      _fault_qd->_tauP);
  VecCopy(_fault_fd->_tauQSP,    _fault_qd->_tauQSP);
  VecCopy(_fault_fd->_strength,  _fault_qd->_strength);

  // update viewers to keep IO consistent
  _fault_fd->_viewers.swap(_fault_qd->_viewers);

  // update momentum balance equation boundary conditions
  _material->changeBCTypes(_mat_qd_bcRType,_mat_qd_bcTType,_mat_qd_bcLType,_mat_qd_bcBType);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// move from a quasidynamic phase to a fully dynamic phase
PetscErrorCode strikeSlip_linearElastic_qd_fd::prepare_qd2fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::prepare_qd2fd()";
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
  VecCopy(_fault_qd->_slip,_varFDPrev["slip"]);
  VecCopy(_fault_qd->_psi,_varFDPrev["psi"]);
  VecCopy(_material->_u,_varFDPrev["u"]);
  if (_thermalCoupling.compare("no")!=0 ) { VecCopy(_varIm["Temp"], _varFDPrev["Temp"]); } // if solving the heat equation
  if (_hydraulicCoupling.compare("no")!=0 ) {
    VecCopy(_varIm["pressure"], _varFDPrev["pressure"]);
    if ((_p->_permSlipDependent).compare("yes")==0) {
      VecCopy(_varQSEx["permeability"], _varFDPrev["permeability"]);
    }
    // if (_hydraulicCoupling.compare("coupled")==0 ){
    //   _fault_fd->setSNEff(_varFD["pressure"]);
    // }
  }

  // take 1 quasidynamic time step to compute variables at time n
  _inDynamic = 0;
  integrate_singleQDTimeStep();
  _inDynamic = 1;

  // update varFD to reflect latest values
  VecCopy(_fault_qd->_slip,_varFD["slip"]);
  VecCopy(_fault_qd->_psi,_varFD["psi"]);
  VecCopy(_material->_u,_varFD["u"]);
  if (_thermalCoupling.compare("no")!=0 ) { VecCopy(_varIm["Temp"], _varFD["Temp"]); } // if solving the heat equation
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
  VecAXPY(_varFD["u"],-1.0,_varFDPrev["u"]);
  VecCopy(_varFDPrev["u"],_u0);
  VecSet(_varFDPrev["u"],0.0);


  // update fault internal variables
  VecCopy(_fault_qd->_psi,       _fault_fd->_psi);
  VecCopy(_fault_qd->_slipVel,   _fault_fd->_slipVel);
  VecCopy(_fault_qd->_slip,      _fault_fd->_slip);
  VecCopy(_fault_qd->_slip,      _fault_fd->_slip0);
  VecCopy(_fault_qd->_strength,  _fault_fd->_strength);
  VecCopy(_fault_qd->_strength,  _fault_fd->_tau0);
  VecCopy(_fault_qd->_tauP,      _fault_fd->_tauP);
  VecCopy(_fault_qd->_tauQSP,    _fault_fd->_tauQSP);
  VecAXPY(_fault_fd->_tau0,1.0,_fault_fd->_prestress); // add prestress to tau0 so don't have to do this each time step
  _fault_fd->_viewers.swap(_fault_qd->_viewers);

  // update momentum balance equation boundary conditions
  _material->changeBCTypes(_mat_fd_bcRType,_mat_fd_bcTType,_mat_fd_bcLType,_mat_fd_bcBType);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// take 1 quasidynamic time step to set up varFDPrev and varFD
PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate_singleQDTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate_singleQDTimeStep()";
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
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_deltaT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"regime1D.txt").c_str(),&_regime1DV);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_regime1DV, "%i\n",_inDynamic);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_deltaT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_regime1DV, "%i\n",_inDynamic);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"regime2D.txt").c_str(),&_regime2DV);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_regime2DV, "%i\n",_inDynamic);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_regime2DV, "%i\n",_inDynamic);CHKERRQ(ierr);
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

  //~ if (_timeIntegrator.compare("IMEX")==0&& _quadImex_qd!=NULL) { ierr = _quadImex_qd->view(); }
  //~ if (_timeIntegrator.compare("RK32")==0 && _quadEx_qd!=NULL) { ierr = _quadEx_qd->view(); }

  _material->view(_integrateTime);
  _fault_qd->view(_integrateTime);
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }

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
  ierr = PetscViewerASCIIPrintf(viewer,"forcingType = %s\n",_forcingType.c_str());CHKERRQ(ierr);

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

  _material->writeContext(_outputDir);
  _fault_qd->writeContext(_outputDir);
  if (_thermalCoupling.compare("no")!=0) {
    _he->writeContext(_outputDir);
  }
  if (_hydraulicCoupling.compare("no")!=0) {
    _p->writeContext(_outputDir);
  }

  if (_forcingType.compare("iceStream")==0) {
    ierr = writeVec(_forcingTerm,_outputDir + "momBal_forcingTerm"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode strikeSlip_linearElastic_qd_fd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  _material->setRHS();

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) { VecAXPY(_material->_rhs,1.0,_forcingTerm); }

  _material->computeU();
  _material->computeStresses();

  return ierr;
}


// fully dynamic: off-fault portion of the momentum balance equation
PetscErrorCode strikeSlip_linearElastic_qd_fd::propagateWaves(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::propagateWaves";
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
  if(_D->_sbpType.compare("mfc_coordTrans")==0){
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
  ierr = VecGetArrayRead(_material->_rhoVec, &rho);

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
  ierr = VecRestoreArrayRead(_material->_rhoVec, &rho);

  VecDestroy(&D2u);

_propagateTime += MPI_Wtime() - startPropagation;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
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
  //~ ierr = _fault_qd->setTauQS(sxy); CHKERRQ(ierr); // new
  ierr = VecScatterBegin(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault_qd->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

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
PetscErrorCode strikeSlip_linearElastic_qd_fd::constructIceStreamForcingTerm()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "strikeSlip_linearElastic_qd_fd::constructIceStreamForcingTerm";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

// matrix to map the value for the forcing term, which lives on the fault, to all other processors
  Mat MapV;
  MatCreate(PETSC_COMM_WORLD,&MapV);
  MatSetSizes(MapV,PETSC_DECIDE,PETSC_DECIDE,_D->_Ny*_D->_Nz,_D->_Nz);
  MatSetFromOptions(MapV);
  MatMPIAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL,_D->_Ny*_D->_Nz,NULL);
  MatSeqAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL);
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

  // compute forcing term for momentum balance equation
  // forcing = - tau_ss / Ly
  Vec tauSS = NULL;
  _fault_qd->computeTauRS(tauSS,_vL);
  VecScale(tauSS,-1./_D->_Ly);

  VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  MatMult(MapV,tauSS,_forcingTerm);

  MatDestroy(&MapV);
  VecDestroy(&tauSS);

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
  if (_material->_sbpType.compare("mfc_coordTrans")==0) {
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


PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate_qd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  OdeSolver      *quadEx = NULL; // explicit time stepping
  OdeSolverImex  *quadImex = NULL; // implicit time stepping

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    quadEx = new FEuler(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    quadEx = new RK32(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    quadEx = new RK43(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    quadImex = new RK32_WBE(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    quadImex = new RK43_WBE(_maxStepCount,_maxTime,_deltaT_fd,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  // integrate
  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    quadImex->setTolerance(_timeStepTol);CHKERRQ(ierr);
    quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);
    quadImex->setTimeRange(_currTime,_maxTime);
    quadImex->setInitialStepCount(_stepCount);
    quadImex->setInitialConds(_varQSEx,_varIm);
    quadImex->setToleranceType(_normType);
    quadImex->setErrInds(_timeIntInds,_scale);

    ierr = quadImex->integrate(this); CHKERRQ(ierr);

    std::map<string,Vec> varOut = quadImex->_varEx;
    for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
      VecCopy(varOut[it->first],_varQSEx[it->first]);
    }
  }
  else {
    quadEx->setTolerance(_timeStepTol);CHKERRQ(ierr);
    quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);
    quadEx->setTimeRange(_currTime,_maxTime);
    quadEx->setInitialStepCount(_stepCount);
    quadEx->setToleranceType(_normType);
    quadEx->setInitialConds(_varQSEx);
    quadEx->setErrInds(_timeIntInds,_scale);

    ierr = quadEx->integrate(this); CHKERRQ(ierr);
    std::map<string,Vec> varOut = quadEx->_var;
    for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
      VecCopy(varOut[it->first],_varQSEx[it->first]);
    }
  }

  delete quadEx;
  delete quadImex;


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode strikeSlip_linearElastic_qd_fd::integrate_fd()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::integrate_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  OdeSolver_WaveEq          *quadWaveEx;
  //~ OdeSolver_WaveEq_Imex     *quadWaveImex;

  // initialize time integrator
  quadWaveEx = new OdeSolver_WaveEq(_maxStepCount,_currTime,_maxTime,_deltaT_fd);
  quadWaveEx->setInitialConds(_varFD);
  quadWaveEx->setInitialStepCount(_stepCount);

  ierr = quadWaveEx->integrate(this);CHKERRQ(ierr);

  std::map<string,Vec> varOut = quadWaveEx->_var;
  for (map<string,Vec>::iterator it = varOut.begin(); it != varOut.end(); it++ ) {
    VecCopy(varOut[it->first],_varFD[it->first]);
  }

  delete quadWaveEx;
  //~ delete quadWaveImex;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}




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
  if (_qd_bcLType.compare("symmFault")==0 || _qd_bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_qd_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  _fault_qd->updateFields(time,varEx);

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end()) && _hydraulicCoupling.compare("no")!=0 ){
    _p->updateFields(time,varEx);
  }
  if (_hydraulicCoupling.compare("coupled")==0) {
    // _fault_qd->setSNEff(varEx.find("pressure")->second);
    _fault_qd->setSNEff(_p->_p);
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
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt qd IMEX";
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

  _fault_qd->updateFields(time,varEx);

  if ( _hydraulicCoupling.compare("no")!=0 ) {
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
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt fd explicit";
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
  // update shear stress tauP: tau = tauQS - eta_rad * slipVel
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = eta_rad*slipVel + tauQSP


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
  if (_thermalCoupling.compare("no")!=0) {
    Vec V = _fault_fd->_slipVel;
    Vec tau = _fault_fd->_tauP;
    Vec Tn = var.find("Temp")->second;
    Vec dTdt; VecDuplicate(Tn,&dTdt);
    ierr = _he->d_dt(time,V,tau,NULL,NULL,NULL,Tn,dTdt); CHKERRQ(ierr);
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
PetscErrorCode strikeSlip_linearElastic_qd_fd::d_dt(const PetscScalar time, const PetscScalar deltaT,
      map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev,
      map<string,Vec>& varIm, const map<string,Vec>& varImPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::d_dt fd imex";
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
  // update shear stress: tau = tauQS - eta_rad * slipVel
  VecPointwiseMult(_fault_fd->_tauP,_fault_qd->_eta_rad,_fault_fd->_slipVel);
  VecAYPX(_fault_fd->_tauP,-1.0,_fault_fd->_tauQSP); // tauP = -tauP + tauQSP = eta_rad*slipVel + tauQSP

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



PetscErrorCode strikeSlip_linearElastic_qd_fd::timeMonitor(const PetscScalar time,const PetscScalar deltaT,const PetscInt stepCount,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "strikeSlip_linearElastic_qd_fd::timeMonitor";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _currTime = time;
  _deltaT = deltaT;
  if (_stepCount == stepCount && _stepCount != 0) { return ierr; } // don't write out the same step twice
  _stepCount = stepCount;

  if (_currTime == _maxTime || (_stride1D>0 && stepCount % _stride1D == 0)) {
    ierr = writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    if(_inDynamic){ ierr = _fault_fd->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
    else { ierr = _fault_qd->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr); }
  }

  if (_currTime == _maxTime || (_stride2D>0 &&  stepCount % _stride2D == 0)) {
    ierr = writeStep2D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr); }
  }

  if(_inDynamic){ if(checkSwitchRegime(_fault_fd)){ stopIntegration = 1; } }
  else { if(checkSwitchRegime(_fault_qd)){ stopIntegration = 1; } }

  #if VERBOSE > 0
    std::string regime = "quasidynamic";
    if(_inDynamic){ regime = "fully dynamic"; }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e %s\n",stepCount,_currTime,_deltaT,regime.c_str());CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



