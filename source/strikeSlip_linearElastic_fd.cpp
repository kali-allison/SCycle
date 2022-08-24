#include "strikeSlip_linearElastic_fd.hpp"

#define FILENAME "StrikeSlip_LinearElastic_fd.cpp"

using namespace std;


StrikeSlip_LinearElastic_fd::StrikeSlip_LinearElastic_fd(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz), _Ly(D._Ly),_Lz(D._Lz),
  _deltaT(-1), _CFL(-1),_y(&D._y),_z(&D._z),_alphay(NULL),
  _inputDir(D._inputDir),_outputDir(D._outputDir),_vL(1e-9),
  _initialConditions("u"),_guessSteadyStateICs(0),_faultTypeScale(2.0),
  _maxStepCount(1e8), _stride1D(1),_stride2D(1),
  _initTime(0),_currTime(0),_maxTime(1e15),
  _time1DVec(NULL), _dtime1DVec(NULL),_time2DVec(NULL), _dtime2DVec(NULL),
  _stepCount(0),
  _viewer_context(NULL),_viewer1D(NULL),_viewer2D(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),
  _startTime(MPI_Wtime()),_miscTime(0), _propagateTime(0),
  _bcRType("outGoingCharacteristics"),_bcTType("freeSurface"),_bcLType("symmFault"),_bcBType("outGoingCharacteristics"),
  _mat_bcRType("Neumann"),_mat_bcTType("Neumann"),_mat_bcLType("Neumann"),_mat_bcBType("Neumann"),
  _quadWaveEx(NULL),_fault(NULL),_material(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::StrikeSlip_LinearElastic_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();

  // determine if material is symmetric about the fault, or if one side is rigid
  _faultTypeScale = 2.0;
  if (_bcLType.compare("rigidFault")==0 ) { _faultTypeScale = 1.0; }

  _body2fault = &(D._scatters["body2L"]);
  _fault = new Fault_fd(D, D._scatters["body2L"],_faultTypeScale); // fault
  _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);
  _cs = _material->_cs;
  _rho = _material->_rho;
  _mu = _material->_mu;
  computePenaltyVectors();

  computeTimeStep(); // compute time step

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_LinearElastic_fd::~StrikeSlip_LinearElastic_fd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::~StrikeSlip_LinearElastic_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _var.begin(); it!=_var.end(); it++ ) {
    VecDestroy(&it->second);
  }
  //~ for (it = _varNext.begin(); it!=_varNext.end(); it++ ) {
    //~ VecDestroy(&it->second);
  //~ }
  for (it = _varPrev.begin(); it!=_varPrev.end(); it++ ) {
    VecDestroy(&it->second);
  }

  VecDestroy(&_time1DVec);
  VecDestroy(&_dtime1DVec);
  VecDestroy(&_time2DVec);
  VecDestroy(&_dtime2DVec);
  VecDestroy(&_ay);

  PetscViewerDestroy(&_viewer1D);
  PetscViewerDestroy(&_viewer2D);
  PetscViewerDestroy(&_viewer_context);

  delete _quadWaveEx;      _quadWaveEx = NULL;
  delete _material;        _material = NULL;
  delete _fault;           _fault = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_LinearElastic_fd::loadSettings(const char *file)
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


    if (var.compare("stride1D")==0){ _stride1D = (int)atof( rhs.c_str() ); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof( rhs.c_str() ); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( rhs.c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( rhs.c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( rhs.c_str() ); }
    else if (var.compare("deltaT")==0) { _deltaT = atof( rhs.c_str() ); }
    else if (var.compare("CFL")==0) { _CFL = atof( rhs.c_str() ); }

    else if (var.compare("initialConditions")==0) { _initialConditions = rhs.c_str(); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }

    else if (var.compare("vL")==0) { _vL = atof( rhs.c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_fd")==0) { _bcRType = rhs.c_str(); }
    else if (var.compare("momBal_bcT_fd")==0) { _bcTType = rhs.c_str(); }
    else if (var.compare("momBal_bcL_fd")==0) { _bcLType = rhs.c_str(); }
    else if (var.compare("momBal_bcB_fd")==0) { _bcBType = rhs.c_str(); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_LinearElastic_fd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);

  // check boundary condition types for momentum balance equation
  assert(_bcRType.compare("freeSurface")==0 || _bcRType.compare("outGoingCharacteristics")==0);
  assert(_bcLType.compare("symmFault")==0 || _bcRType.compare("rigidFault")==0);
  assert(_bcTType.compare("freeSurface")==0 || _bcTType.compare("outGoingCharacteristics")==0);
  assert(_bcBType.compare("freeSurface")==0 || _bcBType.compare("outGoingCharacteristics")==0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode StrikeSlip_LinearElastic_fd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) { _material->setMMSInitialConditions(_initTime); }

  _fault->initiateIntegrand(_initTime,_var);

  VecDuplicate(*_z, &_var["u"]); VecSet(_var["u"], 0.0);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// monitoring function for explicit integration
PetscErrorCode StrikeSlip_LinearElastic_fd::timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::timeMonitor for explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _deltaT = deltaT;
  _stepCount = stepCount;
  _currTime = time;

  if ( (_stride1D > 0 && _currTime == _maxTime) || (_stride1D > 0 && stepCount % _stride1D == 0) ) {
    ierr = writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_viewer1D); CHKERRQ(ierr);
    ierr = _fault->writeStep(_viewer1D); CHKERRQ(ierr);
  }

  if ( (_stride2D>0 &&_currTime == _maxTime) || (_stride2D>0 && stepCount % _stride2D == 0)) {
    ierr = writeStep2D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_viewer2D);CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_LinearElastic_fd::writeStep1D(PetscInt stepCount, PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update Vecs to reflect current time and time step
  VecSet(_time1DVec,time);
  VecSet(_dtime1DVec,_deltaT);

  if (_viewer1D == NULL ) {
    // initiate viewer
    string outFileName = outputDir + "data_1D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewer1D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer1D, PETSC_TRUE);CHKERRQ(ierr);

    // write time
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);               CHKERRQ(ierr);
    ierr = VecView(_time1DVec, _viewer1D);                           CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                       CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);               CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer1D);              CHKERRQ(ierr);
    ierr = VecView(_time1DVec, _viewer1D);                           CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                       CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_fd::writeStep2D(PetscInt stepCount, PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update Vecs to reflect current time and time step
  VecSet(_time2DVec,time);
  VecSet(_dtime2DVec,_deltaT);

  if (_viewer2D == NULL ) {
    // initiate viewer
    string outFileName = outputDir + "data_2D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewer2D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer2D, PETSC_TRUE);CHKERRQ(ierr);

    // write time
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);               CHKERRQ(ierr);
    ierr = VecView(_time2DVec, _viewer2D);                           CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                       CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");               CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);               CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer2D);              CHKERRQ(ierr);
    ierr = VecView(_time2DVec, _viewer2D);                           CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                       CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_LinearElastic_fd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault->view(_integrateTime);
  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Domain Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Nz: %i\n",_Nz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Ny: %i\n",_Ny);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of processors: %i\n",num_proc);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_LinearElastic_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent propagating the wave (s): %g\n",_propagateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_fd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::writeContext";
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

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"deltaT = %.15e # (s)\n",_deltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  // write non-ascii context
  string outFileName = _outputDir + "data_context.h5";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewer_context); CHKERRQ(ierr);
  ierr = PetscViewerSetType(_viewer_context, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewer_context);CHKERRQ(ierr);

  _D->write(_viewer_context);
  _fault->writeContext(_outputDir, _viewer_context);
  _material->writeContext(_outputDir, _viewer_context);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================
PetscErrorCode StrikeSlip_LinearElastic_fd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  _quadWaveEx = new OdeSolver_WaveEq(_maxStepCount,_initTime,_maxTime,_deltaT);
  ierr = _quadWaveEx->setInitialConds(_var);CHKERRQ(ierr);

  ierr = _quadWaveEx->integrate(this);CHKERRQ(ierr);

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// purely explicit time stepping// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_LinearElastic_fd::d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // momentum balance equation except for fault boundary
  propagateWaves(time, deltaT, varNext, var, varPrev);

  if (_initialConditions.compare("tau")==0) { _fault->updatePrestress(time); }
  ierr = _fault->d_dt(time,_deltaT,varNext,var,varPrev); CHKERRQ(ierr);

  // update body u from fault u
  ierr = VecScatterBegin(*_body2fault, _fault->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, _fault->_u, varNext["u"], INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  VecCopy(varNext.find("u")->second, _material->_u);
  _material->computeStresses();
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  VecAXPY(_fault->_tauP, 1.0, _fault->_prestress);
  VecCopy(_fault->_tauP,_fault->_tauQSP); // keep quasi-static shear stress updated as well

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// fully dynamic: off-fault portion of the momentum balance equation
PetscErrorCode StrikeSlip_LinearElastic_fd::propagateWaves(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::propagateWaves";
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
  ierr = VecScatterBegin(*_body2fault, D2u, _fault->_d2u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, D2u, _fault->_d2u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);



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
  ierr = VecGetArrayRead(_rho, &rho);

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
  ierr = VecRestoreArrayRead(_rho, &rho);

  VecDestroy(&D2u);


_propagateTime += MPI_Wtime() - startPropagation;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute allowed time step based on CFL condition and user input
// deltaT <= * gcfl * min(dy/cs, dz/cs)
PetscErrorCode StrikeSlip_LinearElastic_fd::computeTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::computeTimeStep";
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
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
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
PetscErrorCode StrikeSlip_LinearElastic_fd::computePenaltyVectors()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_fd::computePenaltyVectors";
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
  VecScatterBegin(_D->_scatters["body2L"], alphay, _fault->_alphay, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], alphay, _fault->_alphay, INSERT_VALUES, SCATTER_FORWARD);
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

    if ( (Ii/_Nz == _Ny-1) && (_bcRType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11y; }
    if ( (Ii%_Nz == 0) && (_bcTType.compare("outGoingCharacteristics") == 0 )) { ay[Jj] += 0.5 / h11z; }
    if ( ((Ii+1)%_Nz == 0) && (_bcBType.compare("outGoingCharacteristics") == 0) ) { ay[Jj] += 0.5 / h11z; }

    if ( (Ii/_Nz == 0) && ( _bcLType.compare("outGoingCharacteristics") == 0 ||
        _bcLType.compare("symmFault") == 0 || _bcLType.compare("rigidFault") == 0 ) )
    { ay[Jj] += 0.5 / h11y; }
    Jj++;
  }
  VecRestoreArray(_ay,&ay);

  ierr = VecPointwiseMult(_ay, _ay, _cs);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}
