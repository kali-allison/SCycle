#include "strikeSlip_linearElastic_dyn.hpp"

#define FILENAME "strikeSlip_linearElastic_dyn.cpp"

using namespace std;


StrikeSlip_LinearElastic_dyn::StrikeSlip_LinearElastic_dyn(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),
  _deltaT(1e-3), 
  _y(*(&(D._y))),_z(*(&(D._z))),
  _alphay(D._alphay), _alphaz(D._alphaz),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(1e-9),
  _isFault("true"),
  _stride1D(1),_stride2D(1),_maxStepCount(1e8),
  _initTime(0),_currTime(0),_maxTime(1e15),
  _stepCount(0),_atol(1e-8),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),
  _bcRType("outGoingCharacteristics"),_bcTType("freeSurface"),_bcLType("outGoingCharacteristics"),_bcBType("outGoingCharacteristics"),
  _quadWaveEx(NULL),
  _fault(NULL),_material(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::StrikeSlip_LinearElastic_dyn()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();

  _fault = new NewFault_dyn(D, D._scatters["body2L"]); // fault

  assert(_bcBType.compare("freeSurface")==0 || _bcBType.compare("outGoingCharacteristics")==0);
  assert(_bcTType.compare("freeSurface")==0 || _bcTType.compare("outGoingCharacteristics")==0);
  assert(_bcRType.compare("freeSurface")==0 || _bcRType.compare("outGoingCharacteristics")==0);
  assert(_bcLType.compare("freeSurface")==0 || _bcLType.compare("outGoingCharacteristics")==0);
  _mat_bcBType = "Neumann";
  _mat_bcTType = "Neumann";
  _mat_bcRType = "Neumann";
  _mat_bcLType = "Neumann";

  _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);
  _cs = *(&(_material->_cs));
  _rhoVec = *(&(_material->_rhoVec));
  _muVec = *(&(_material->_muVec));

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_LinearElastic_dyn::~StrikeSlip_LinearElastic_dyn()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::~StrikeSlip_LinearElastic_dyn()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }

  delete _quadWaveEx;      _quadWaveEx = NULL;
  delete _material;    _material = NULL;
  delete _fault;       _fault = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_LinearElastic_dyn::loadSettings(const char *file)
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

    if (var.compare("stride1D")==0){ _stride1D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("deltaT")==0) { _deltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("atol")==0) { _atol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("isFault")==0) { _isFault = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("timeIntInds")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_timeIntInds);
    }

    else if (var.compare("vL")==0) { _vL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_dyn")==0) {
      _bcRType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcT_dyn")==0) {
      _bcTType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcL_dyn")==0) {
      _bcLType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcB_dyn")==0) {
      _bcBType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_LinearElastic_dyn::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_loadICs) { assert(_guessSteadyStateICs == 0); }

  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_stride1D >= 1);
  assert(_stride2D >= 1);
  assert(_atol >= 1e-14);

    // check boundary condition types for momentum balance equation
  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcRType.compare("freeSurface")==0 ||
    _bcRType.compare("tau")==0 ||
    _bcRType.compare("remoteLoading")==0 ||
    _bcRType.compare("symm_fault")==0 ||
    _bcRType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcTType.compare("freeSurface")==0 ||
    _bcTType.compare("tau")==0 ||
    _bcTType.compare("remoteLoading")==0 ||
    _bcTType.compare("symm_fault")==0 ||
    _bcTType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcLType.compare("freeSurface")==0 ||
    _bcLType.compare("tau")==0 ||
    _bcLType.compare("remoteLoading")==0 ||
    _bcLType.compare("symm_fault")==0 ||
    _bcLType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcBType.compare("freeSurface")==0 ||
    _bcBType.compare("tau")==0 ||
    _bcBType.compare("remoteLoading")==0 ||
    _bcBType.compare("symm_fault")==0 ||
    _bcBType.compare("rigid_fault")==0 );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode StrikeSlip_LinearElastic_dyn::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) { _material->setMMSInitialConditions(_initTime); }

  _fault->initiateIntegrand(_initTime,_varEx);

  Vec slip;
  VecDuplicate(_varEx["psi"], &slip); VecSet(slip,0.);
  _varEx["slip"] = slip;
  
  VecDuplicate(_z, &_varEx["uPrev"]); VecSet(_varEx["uPrev"],0.);
  VecDuplicate(_z, &_varEx["u"]); VecSet(_varEx["u"], 0.0);

    PetscScalar *u, *uPrev, *y, *z;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_varEx["u"],&Istart,&Iend);
    VecGetArray(_varEx["u"],&u);
    VecGetArray(_varEx["uPrev"],&uPrev);
    VecGetArray(_y, &y);
    VecGetArray(_z, &z);

    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      u[Jj] = 10 * exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
      uPrev[Jj] = 10 *exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
      Jj++;
    }
    VecRestoreArray(_y,&y);
    VecRestoreArray(_z,&z);
    VecRestoreArray(_varEx["u"],&u);
    VecRestoreArray(_varEx["uPrev"],&uPrev);

    // Create matrix _ay
    VecDuplicate(_y, &_ay);
    VecSet(_ay, 0.0);

    PetscScalar *yy, *zz, *ay;
    VecGetOwnershipRange(_y,&Istart,&Iend);
    VecGetArray(_ay,&ay);
    VecGetArray(_y, &yy);
    VecGetArray(_z, &zz);
    Jj = 0;

    PetscScalar dy,dz;
    if (_D->_sbpType.compare("mfc_coordTrans")==0) { dy = 1./(_Ny-1); dz = 1./(_Nz-1); }
    else { dy = _Ly/(_Ny-1); dz = _Lz/(_Nz-1); }

    for (Ii=Istart;Ii<Iend;Ii++) {
      ay[Jj] = 0;
      PetscScalar tol;
      if (dy < dz){tol = dy / 10000;}
      else{tol = dz / 10000;}
      if (abs(yy[Jj]) < tol && _bcLType.compare("outGoingCharacteristics") == 0){ay[Jj] += 0.5 / _alphay;}
      if (abs(yy[Jj] - _Ly) < tol && _bcRType.compare("outGoingCharacteristics") == 0){ay[Jj] += 0.5 / _alphay;}
      if (abs(zz[Jj]) < tol && _bcTType.compare("outGoingCharacteristics") == 0){ay[Jj] += 0.5 / _alphaz;}
      if (abs(zz[Jj] - _Lz && _bcBType.compare("outGoingCharacteristics") == 0) < tol){ay[Jj] += 0.5 / _alphaz;}
      Jj++;
    }
    VecRestoreArray(_y,&yy);
    VecRestoreArray(_z,&zz);
    VecRestoreArray(_ay,&ay);

    ierr = VecPointwiseMult(_ay, _ay, _cs);

  _fault->initiateIntegrand_dyn(_varEx, _rhoVec);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// monitoring function for explicit integration
PetscErrorCode StrikeSlip_LinearElastic_dyn::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::timeMonitor for explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  if ( stepCount % _stride1D == 0) {
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
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

// monitoring function for IMEX integration
PetscErrorCode StrikeSlip_LinearElastic_dyn::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::timeMonitor for IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  if ( stepCount % _stride1D == 0) {
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
  }

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_LinearElastic_dyn::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault->view(_integrateTime);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_LinearElastic_dyn Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_dyn::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::writeContext";
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
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"deltaT = %.15e # (s)\n",_deltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"atol = %.15e\n",_atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _material->writeContext(_outputDir);
  _fault->writeContext(_outputDir);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================
PetscErrorCode StrikeSlip_LinearElastic_dyn::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_dyn::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  _quadWaveEx = new OdeSolver_WaveEq(_maxStepCount,_initTime,_maxTime,_deltaT);
  ierr = _quadWaveEx->setInitialConds(_varEx);CHKERRQ(ierr);

  ierr = _quadWaveEx->integrate(this);CHKERRQ(ierr);

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// purely explicit time stepping// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_LinearElastic_dyn::d_dt(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // ierr = _material->_sbp->setRhs(_material->_rhs,_material->_bcL,_material->_bcR,_material->_bcT,_material->_bcB);CHKERRQ(ierr);
  Mat A;
  ierr = _material->_sbp->getA(A);

  // Update the laplacian
  Vec Laplacian, temp;
  VecDuplicate(_y, &Laplacian);
  VecDuplicate(_y, &temp);
  ierr = MatMult(A, varEx["u"], temp);
  ierr = _material->_sbp->Hinv(temp, Laplacian);
  ierr = VecCopy(Laplacian, dvarEx["u"]);
  VecDestroy(&temp);

  // Apply the time step
  Vec uNext, correction, previous, ones;

  VecDuplicate(varEx["u"], &ones);
  VecDuplicate(varEx["u"], &correction);
  VecSet(ones, 1.0);
  VecSet(correction, 0.0);
  ierr = VecAXPY(correction, _deltaT, _ay);
  ierr = VecAXPY(correction, -1.0, ones);

  VecDuplicate(varEx["u"], &previous);
  VecSet(previous, 0.0);
  ierr = VecPointwiseMult(previous, correction, varEx["uPrev"]);

  VecDuplicate(varEx["u"], &uNext);
  VecSet(uNext, 0.0);
  ierr = VecAXPY(uNext, pow(_deltaT, 2), dvarEx["u"]);
  ierr = VecPointwiseDivide(uNext, uNext, _rhoVec);

  ierr = VecAXPY(uNext, 2, varEx["u"]);
  ierr = VecAXPY(uNext, 1, previous);
  ierr = VecAXPY(correction, 2, ones);
  ierr = VecPointwiseDivide(uNext, uNext, correction);

  ierr = VecCopy(varEx["u"], varEx["uPrev"]);
  ierr = VecCopy(uNext, varEx["u"]);
  VecDestroy(&uNext);
  VecDestroy(&ones);
  VecDestroy(&correction);
  VecDestroy(&previous);

  if (_isFault.compare("true") == 0){
  ierr = _fault->d_dt(time,varEx,dvarEx, _deltaT);CHKERRQ(ierr);
}
  VecCopy(varEx["u"], _material->_u);

  return ierr;
}
