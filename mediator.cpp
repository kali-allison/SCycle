#include "mediator.hpp"

#define FILENAME "mediator.cpp"

using namespace std;


Mediator::Mediator(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _bcLTauQS(0),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(D._vL),_problemType(D._problemType),
  _momBalType("static"),_bulkDeformationType(D._bulkDeformationType),_thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _initialU("gaussian"),
  _timeIntegrator(D._timeIntegrator),
  _stride1D(D._stride1D),_stride2D(D._stride2D),_maxStepCount(D._maxStepCount),
  _initTime(D._initTime),_currTime(_initTime),_maxTime(D._maxTime),
  _minDeltaT(D._minDeltaT),_maxDeltaT(D._maxDeltaT),
  _stepCount(0),_atol(D._atol),_initDeltaT(D._initDeltaT),_timeIntInds(D._timeIntInds),
  _fss_T(0.1),_fss_EffVisc(0.25),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),
  _quadEx(NULL),_quadImex(NULL),_quadWaveEq(NULL),
  _fault(NULL),_momBal(NULL),_he(NULL),_p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Mediator::Mediator()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _startTime = MPI_Wtime();

  loadSettings(D._file);
  checkInput();

  _he = new HeatEquation(D); // heat equation
  _fault = new SymmFault(D,*_he); // fault

  // pressure diffusion equation
  if (_hydraulicCoupling.compare("no")!=0) {
    _p = new PressureEq(D);
  }
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }

  // initiate momentum balance equation
  if (_bulkDeformationType.compare("linearElastic")==0) { _momBal = new LinearElastic(D,*_he); }
  else if (_bulkDeformationType.compare("powerLaw")==0) { _momBal = new PowerLaw(D,*_he); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


Mediator::~Mediator()
{
  #if VERBOSE > 1
    std::string funcName = "Mediator::~Mediator()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (map<string,std::pair<PetscViewer,string> >::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first].first);
  }

  if (_varSS.find("Temp") != _varSS.end()) { VecDestroy(&_varSS["Temp"]); }
  if (_varSS.find("v") != _varSS.end()) { VecDestroy(&_varSS["v"]); }
  if (_varSS.find("gVxy_t") != _varSS.end()) { VecDestroy(&_varSS["gVxy_t"]); }
  if (_varSS.find("gVxz_t") != _varSS.end()) { VecDestroy(&_varSS["gVxz_t"]); }
  if (_varSS.find("tau") != _varSS.end()) { VecDestroy(&_varSS["tau"]); }

  delete _quadImex;    _quadImex = NULL;
  delete _quadEx;      _quadEx = NULL;
  delete _quadWaveEq;  _quadWaveEq = NULL;
  delete _momBal;      _momBal = NULL;
  delete _fault;       _fault = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode Mediator::loadSettings(const char *file)
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
    else if (var.compare("timeControlType")==0) {
      _timeControlType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("initialU")==0) {
      _initialU = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBalType")==0) {
      _momBalType = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // for steady state iteration
    else if (var.compare("fss_T")==0) {
      _fss_T = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("fss_EffVisc")==0) {
      _fss_EffVisc = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode Mediator::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

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

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode Mediator::initiateIntegrand_qs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::initiateIntegrand_qs()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ if (!_loadICs && _bulkDeformationType.compare("linearElastic")==0) { solveSS_linEl(); }
  //~ if (!_loadICs && _bulkDeformationType.compare("powerLaw")==0) { solveSS_pl(); }

  _momBal->initiateIntegrand_qs(_initTime,_varEx);
  _fault->initiateIntegrand(_initTime,_varEx);

  if (_thermalCoupling.compare("no")!=0 ) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  if (_hydraulicCoupling.compare("no")!=0 ) {
     _p->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// initiate variables to be integrated in time for steady state iteration
PetscErrorCode Mediator::initiateIntegrand_ss()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::initiateIntegrand_ss()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _momBal->initiateIntegrand_qs(_initTime,_varEx);
  _fault->initiateIntegrand(_initTime,_varEx);

  if (_thermalCoupling.compare("no")!=0 ) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  if (_hydraulicCoupling.compare("no")!=0 ) {
     _p->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// initiate variables to be integrated in time for fully dynamic wave equation
// this means constant time step
PetscErrorCode Mediator::initiateIntegrand_dyn()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::initiateIntegrand_dyn()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // _fault->initiateIntegrand(_initTime,_varEx);
  _momBal->initiateIntegrand_dyn(_initTime, _varEx);

  //~ if (_thermalCoupling.compare("no")!=0 ) {
     //~ _he->initiateIntegrand(_initTime,_varEx,_varIm);
  //~ }

  //~ if (_hydraulicCoupling.compare("no")!=0 ) {
     //~ _p->initiateIntegrand(_initTime,_varEx,_varIm);
  //~ }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// monitoring function for explicit integration
PetscErrorCode Mediator::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::timeMonitor for explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  // stopping criteria for time integration
  if (_problemType.compare("steadyStateIts")==0) {
  //~ if (_stepCount > 5) { stopIntegration = 1; } // basic test
    PetscScalar maxVel; VecMax(dvarEx.find("slip")->second,NULL,&maxVel);
    if (maxVel < 1.2e-9 && _stepCount > 500) { stopIntegration = 1; }
  }


  if ( stepCount % _stride1D == 0) {
    ierr = _momBal->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _momBal->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
  }

  if (stepCount % 50 == 0) {
    PetscScalar maxTimeStep_tot, maxDeltaT_momBal = 0.0;
    _momBal->computeMaxTimeStep(maxDeltaT_momBal);
    maxTimeStep_tot = min(_maxDeltaT,maxDeltaT_momBal);
    if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
        _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
    }
    else if (_momBalType.compare("dynamic") != 0) { _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr); }
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
PetscErrorCode Mediator::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "Mediator::timeMonitor for IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  // stopping criteria for time integration
  if (_problemType.compare("steadyStateIts")==0) {
  //~ if (_stepCount > 5) { stopIntegration = 1; } // basic test
    PetscScalar maxVel; VecMax(dvarEx.find("slip")->second,NULL,&maxVel);
    if (maxVel < 1.2e-9 && _stepCount > 500) { stopIntegration = 1; }
  }


  if ( stepCount % _stride1D == 0) {
    ierr = _momBal->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr); }
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr); }
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _momBal->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr); }
  }

  if (stepCount % 50 == 0) {
    PetscScalar maxTimeStep_tot, maxDeltaT_momBal = 0.0;
    _momBal->computeMaxTimeStep(maxDeltaT_momBal);
    maxTimeStep_tot = min(_maxDeltaT,maxDeltaT_momBal);
    if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
        _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
    }
    else if (_momBalType.compare("dynamic") != 0) { _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr); }
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


PetscErrorCode Mediator::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  if (_timeIntegrator.compare("IMEX")==0&& _quadImex!=NULL) { ierr = _quadImex->view(); }
  if (_timeIntegrator.compare("RK32")==0 && _quadEx!=NULL) { ierr = _quadEx->view(); }

  _momBal->view(_integrateTime);
  _fault->view(_integrateTime);
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Mediator Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode Mediator::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // output scalar fields
  std::string str = _outputDir + "mediator_context.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"problemType = %s\n",_problemType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);

  // if steady state oscillation
  if (_problemType.compare("steadyStateIts")==0) {
    ierr = PetscViewerASCIIPrintf(viewer,"f_T = %.15e\n",_fss_T);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"f_EffVisc = %.15e\n",_fss_EffVisc);CHKERRQ(ierr);
  }

  PetscViewerDestroy(&viewer);

  _momBal->writeContext(_outputDir);
   _he->writeContext(_outputDir);
  _fault->writeContext(_outputDir);

  if (_hydraulicCoupling.compare("no")!=0) {
    _p->writeContext(_outputDir);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode Mediator::solveSS_pl()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::solveSS_pl";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ PetscScalar H = 10; // seismogenic depth
  //~ PetscScalar ess_t = _vL*0.5/H; // guess at steady state strain rate
  PetscScalar ess_t = 1e-7; // guess at steady state strain rate

  // compute steady state stress on fault
  Vec tauRS = NULL,tauVisc = NULL,tauSS=NULL;
  _fault->getTauRS(tauRS,_vL); // rate and state tauSS assuming velocity is vL
  _momBal->getTauVisc(tauVisc,ess_t); // tau visc from steady state strain rate

  // tauSS = min(tauRS,tauVisc)
  VecDuplicate(tauRS,&tauSS);
  VecPointwiseMin(tauSS,tauRS,tauVisc);
  //~ VecCopy(tauRS,tauSS);


  if (_inputDir.compare("unspecified") != 0) {
    ierr = loadVecFromInputFile(tauSS,_inputDir,"tauSS"); CHKERRQ(ierr);
  }
  ierr = io_initiateWriteAppend(_viewers, "SS_tauSS", tauSS, _outputDir + "SS_tauSS"); CHKERRQ(ierr);

  // first, set up _varSS
  _varSS["tau"] = tauSS;
  _momBal->initiateVarSS(_varSS);
  _fault->initiateVarSS(_varSS);

  // don't iterate on effective viscosity
  //~ ierr = _momBal->updateSSa(_varSS); CHKERRQ(ierr);

  // loop over effective viscosity
  Vec effVisc_old; VecDuplicate(_varSS["effVisc"],&effVisc_old);
  Vec temp; VecDuplicate(_varSS["effVisc"],&temp); VecSet(temp,0.);
  double err = 1e10;
  int Ii = 0;
  while (Ii < 50 && err > 1e-3) {
    VecCopy(_varSS["effVisc"],effVisc_old);
    _momBal->updateSSa(_varSS); // compute v, viscous strain rates
    // update effective viscosity: accepted viscosity = (1-f)*(old viscosity) + f*(new viscosity):
    //~ VecScale(_varSS["effVisc"],_fss_EffVisc);
    //~ VecAXPY(_varSS["effVisc"],1.-_fss_EffVisc,effVisc_old);
    // update effective viscosity: log10(accepted viscosity) = (1-f)*log10(old viscosity) + f*log10(new viscosity):
      MyVecLog10AXPBY(temp,1.-_fss_EffVisc,effVisc_old,_fss_EffVisc,_varSS["effVisc"]);
      VecCopy(temp,_varSS["effVisc"]);

    PetscScalar len;
    VecNorm(effVisc_old,NORM_2,&len);
    err = computeNormDiff_L2_scaleL2(effVisc_old,_varSS["effVisc"]);
    PetscPrintf(PETSC_COMM_WORLD,"    inner loop: %i %e\n",Ii,err);
    Ii++;
  }
  VecDestroy(&effVisc_old);
  VecDestroy(&temp);

  // solve for gVxy, gVxz, u, bcL and bcR
  ierr = _momBal->updateSSb(_varSS); CHKERRQ(ierr);

  Vec sxy,sxz,sdev;
  ierr = _momBal->getStresses(sxy,sxz,sdev);
  ierr = _fault->setTauQS(sxy,sxz); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// if solving linear elastic steady state problem
PetscErrorCode Mediator::solveSS_linEl()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar H = 10; // seismogenic depth
  PetscScalar ess_t = _vL*0.5/H; // steady state strain rate

  // compute steady state stress on fault
  Vec tauRS = NULL,tauVisc = NULL,tauSS = NULL;
  _fault->getTauRS(tauRS,_vL); // rate and state tauSS assuming velocity is vL
  _momBal->getTauVisc(tauVisc,ess_t); // tau visc from steady state strain rate

  // tauSS = min(tauRS,tauVisc)
  VecDuplicate(tauRS,&tauSS);
  VecPointwiseMin(tauSS,tauRS,tauVisc);
  //~ VecCopy(tauRS,tauSS);
  VecDestroy(&tauVisc);
  VecDestroy(&tauRS);

  if (_inputDir.compare("unspecified") != 0) {
    ierr = loadVecFromInputFile(tauSS,_inputDir,"tauSS"); CHKERRQ(ierr);
  }
  ierr = io_initiateWriteAppend(_viewers, "tau", tauSS, _outputDir + "SS_tau"); CHKERRQ(ierr);

  // first, set up _varSS
  _varSS["tau"] = tauSS;
  _momBal->initiateVarSS(_varSS);
  _fault->initiateVarSS(_varSS);

  ierr = _momBal->updateSSa(_varSS); CHKERRQ(ierr);
  // solve for gVxy, gVxz, u, bcL and bcR
  ierr = _momBal->updateSSb(_varSS); CHKERRQ(ierr);

  Vec sxy,sxz,sdev;
  ierr = _momBal->getStresses(sxy,sxz,sdev);
  ierr = _fault->setTauQS(sxy,sxz); CHKERRQ(ierr);

  VecDestroy(&_varSS["tau"]);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode Mediator::writeSS(const int Ii, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::writeSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (Ii == 0) {
    ierr = io_initiateWriteAppend(_viewers, "slipVel", _varSS["slipVel"], outputDir + "SS_slipVel"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "tau", _varSS["tau"], outputDir + "SS_tau"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "effVisc", _varSS["effVisc"], outputDir + "SS_effVisc"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gVxy_t", _varSS["gVxy_t"], outputDir + "SS_gVxy_t"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gVxz_t", _varSS["gVxz_t"], outputDir + "SS_gVxz_t"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "sxy", _varSS["sxy"], outputDir + "SS_sxy"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "sxz", _varSS["sxz"], outputDir + "SS_sxz"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gxy", _varSS["gxy"], outputDir + "SS_gxy"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "gxz", _varSS["gxz"], outputDir + "SS_gxz"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "u", _varSS["u"], outputDir + "SS_u"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "v", _varSS["v"], outputDir + "SS_v"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "Temp", _varSS["Temp"], outputDir + "SS_Temp"); CHKERRQ(ierr);

  }
  else {
    ierr = VecView(_varSS["slipVel"],_viewers["slipVel"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["tau"],_viewers["tau"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["effVisc"],_viewers["effVisc"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["gVxy_t"],_viewers["gVxy_t"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["gVxz_t"],_viewers["gVxz_t"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["sxy"],_viewers["sxy"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["sxz"],_viewers["sxz"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["gxy"],_viewers["gxy"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["gxz"],_viewers["gxz"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["u"],_viewers["u"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["v"],_viewers["v"].first); CHKERRQ(ierr);
    ierr = VecView(_varSS["Temp"],_viewers["Temp"].first); CHKERRQ(ierr);

  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================
PetscErrorCode Mediator::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  if (_problemType.compare("dynamic")==0) { integrate_dyn(); }
  else if (_problemType.compare("quasidynamic")==0) { integrate_qs(); }
  else if (_problemType.compare("quasidynamic_and_dynamic")==0) { integrate_qs(); } // not yet supported
  else if (_problemType.compare("steadyStateIts")==0) { integrate_SS(); }

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Mediator::integrate_SS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::integrate_SS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  std::string baseOutDir = _outputDir;

  // initial guess
  solveSS_pl();
  Vec T; VecDuplicate(_varSS["effVisc"],&T);
  _varSS["Temp"] = T;
  _he->getTemp(_varSS["Temp"]);
  // update temperature
  {
    Vec sxy=NULL,sxz=NULL,sdev = NULL;
    _momBal->getStresses(sxy,sxz,sdev);
    Vec gVxy_t = _varSS["gVxy_t"];
    Vec gVxz_t = _varSS["gVxy_t"];
    _he->computeSteadyStateTemp(_currTime,NULL,NULL,sdev,gVxy_t,gVxz_t,_varSS["Temp"]);
    _momBal->updateTemperature(_varSS["Temp"]);
  }

  VecCopy(_fault->_tauQSP,_varSS["tau"]);
  ierr = io_initiateWriteAppend(_viewers, "effVisc", _varSS["effVisc"], _outputDir + "SS_effVisc"); CHKERRQ(ierr);
  ierr = io_initiateWriteAppend(_viewers, "Temp", _varSS["Temp"], _outputDir + "SS_Temp"); CHKERRQ(ierr);
  ierr = io_initiateWriteAppend(_viewers, "Q", _he->_Q, _outputDir + "SS_Q"); CHKERRQ(ierr);

  PetscInt Jj = 0;
  _currTime = _initTime;
  Vec T_old; VecDuplicate(_varSS["Temp"],&T_old); VecSet(T_old,0.);
  PetscInt maxItCount = min((int) _maxStepCount, (int)1e4);
  while (Jj < maxItCount) {
    PetscPrintf(PETSC_COMM_WORLD,"Jj = %i, _stepCount = %i\n",Jj,_stepCount);

    // create output path with Jj appended on end
    char buff[5];
    sprintf(buff,"%04d",Jj);
    _outputDir = baseOutDir + string(buff) + "_";
    PetscPrintf(PETSC_COMM_WORLD,"baseDir = %s\n\n",_outputDir.c_str());


    // integrate to find the approximate steady state shear stress on the fault
    _momBal->initiateIntegrand_qs(_initTime,_varEx);
    _fault->initiateIntegrand(_initTime,_varEx);
    _stepCount = 0;
    _currTime = _initTime;

    PetscScalar maxTemp; VecMax(_varSS["Temp"],NULL,&maxTemp);

    _quadEx = new RK32(2e4,_maxTime,_initDeltaT,_timeControlType);
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    _quadEx->setTimeRange(_initTime,_maxTime);
    _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);
    _quadEx->setErrInds(_timeIntInds); // control which fields are used to select step size
    _quadEx->integrate(this);CHKERRQ(ierr);


    // compute steady state conditions
    VecCopy(_fault->_tauP,_varSS["tau"]);
    delete _quadEx; _quadEx = NULL;
    map<string,Vec>::iterator it;
    for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      VecDestroy(&it->second);
    }

    // loop over effective viscosity
    Vec effVisc_old; VecDuplicate(_varSS["effVisc"],&effVisc_old);
    Vec temp; VecDuplicate(_varSS["effVisc"],&temp); VecSet(temp,0.);
    double err = 1e10;
    int Ii = 0;
    while (Ii < 50 && err > 1e-3) {
      VecCopy(_varSS["effVisc"],effVisc_old);
      _momBal->updateSSa(_varSS); // compute steady state: v, gVij_t, sij, effVisc
      MyVecLog10AXPBY(temp, 1.-_fss_EffVisc, effVisc_old, _fss_EffVisc, _varSS["effVisc"]);
      VecCopy(temp,_varSS["effVisc"]);
      err = computeNormDiff_L2_scaleL2(effVisc_old,_varSS["effVisc"]);
      PetscPrintf(PETSC_COMM_WORLD,"    eff visc loop: %i %e\n",Ii,err);
      Ii++;
    }
    VecDestroy(&effVisc_old);
    VecDestroy(&temp);

    // update temperature, with damping: Tnew = (1-f)*Told + f*Tnew
    if (_thermalCoupling.compare("coupled")==0) {
      Vec sxy=NULL,sxz=NULL,sdev = NULL;
      _momBal->getStresses(sxy,sxz,sdev);
      Vec gVxy_t = _varSS["gVxy_t"];
      Vec gVxz_t = _varSS["gVxy_t"];
      VecCopy(_varSS["Temp"],T_old);
      _he->computeSteadyStateTemp(_currTime,NULL,NULL,sdev,gVxy_t,gVxz_t,_varSS["Temp"]);
      VecScale(_varSS["Temp"],_fss_T);
      VecAXPY(_varSS["Temp"],1.-_fss_T,T_old);
      _momBal->updateTemperature(_varSS["Temp"]);
    }

    ierr = _momBal->updateSSb(_varSS); CHKERRQ(ierr);
    {
      Vec sxy,sxz,sdev;
      ierr = _momBal->getStresses(sxy,sxz,sdev);
      ierr = _fault->setTauQS(sxy,sxz); CHKERRQ(ierr);
    }

    VecCopy(_fault->_tauP,_varSS["tau"]);
    writeSS(Jj,baseOutDir);
    //~ ierr = VecView(_varSS["tau"],_viewers["tauSS"].first); CHKERRQ(ierr);
    //~ ierr = VecView(_varSS["effVisc"],_viewers["effVisc"].first); CHKERRQ(ierr);
    //~ ierr = VecView(_varSS["Temp"],_viewers["Temp"].first); CHKERRQ(ierr);
    Jj++;
  }
  VecDestroy(&T_old);


  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode Mediator::integrate_qs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::integrate_qs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand_qs(); // put initial conditions into var for integration
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
    _quadImex->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);
    ierr = _quadImex->setErrInds(_timeIntInds); // control which fields are used to select step size

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else {
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);
    ierr = _quadEx->setErrInds(_timeIntInds); // control which fields are used to select step size

    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Mediator::integrate_dyn()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::integrate_dyn";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand_dyn(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  _quadWaveEq = new OdeSolver_WaveEq(_maxStepCount,_initTime,_maxTime,_initDeltaT);
  ierr = _quadWaveEq->setInitialConds(_varEx);CHKERRQ(ierr);

  ierr = _quadWaveEq->integrate(this);CHKERRQ(ierr);

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode Mediator::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update fields based on varEx, varIm
  _momBal->updateFields(time,varEx);
  _fault->updateFields(time,varEx);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->updateFields(time,varEx);
  }

  // compute rates
  ierr = _momBal->d_dt(time,varEx,dvarEx); CHKERRQ(ierr);
  if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
    _p->d_dt(time,varEx,dvarEx);
  }

  // update fields on fault from other classes
  Vec sxy,sxz,sdev;
  ierr = _momBal->getStresses(sxy,sxz,sdev);
  ierr = _fault->setTauQS(sxy,sxz); CHKERRQ(ierr);

  if (_hydraulicCoupling.compare("coupled")==0) { _fault->setSNEff(_p->_p); }

  // rates for fault
  ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  return ierr;
}

// Wave equation
PetscErrorCode Mediator::d_dt_WaveEq(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT)
{
  PetscErrorCode ierr = 0;
  ierr = _momBal->d_dt_WaveEq(time,varEx,dvarEx, _deltaT); CHKERRQ(ierr);
  // ierr = _fault->d_dt_WaveEq(time,varEx,dvarEx, _deltaT);

  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode Mediator::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // update state of each class from integrated variables varEx and varImo
  _momBal->updateFields(time,varEx);
  _fault->updateFields(time,varEx);

  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->updateFields(time,varEx,varImo);
  }

  // update temperature in momBal
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling.compare("coupled")==0) {
    _momBal->updateTemperature(varImo.find("Temp")->second);
    _fault->setTemp(varImo.find("Temp")->second);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }


  // compute rates
  ierr = _momBal->d_dt(time,varEx,dvarEx); CHKERRQ(ierr);
  if ( varImo.find("pressure") != varImo.end() || varEx.find("pressure") != varEx.end()) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
    // _p->d_dt(time,varEx,dvarEx);
  }

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _momBal->getStresses(sxy,sxz,sdev);
  ierr = _fault->setTauQS(sxy,sxz); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  // heat equation
  if (varIm.find("Temp") != varIm.end()) {
    //~ PetscPrintf(PETSC_COMM_WORLD,"Computing new steady state temperature at stepCount = %i\n",_stepCount);
    Vec sxy,sxz,sdev;
    _momBal->getStresses(sxy,sxz,sdev);
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault->_tauP;
    Vec gVxy_t = dvarEx.find("gVxy")->second;
    Vec gVxz_t = dvarEx.find("gVxz")->second;
    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,sdev,gVxy_t,gVxz_t,varIm["Temp"],Told,dt); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// Outputs data at each time step.
PetscErrorCode Mediator::debug(const PetscScalar time,const PetscInt stepCount,
                         const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;


#endif
  return ierr;
}

PetscErrorCode Mediator::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // _momBal->measureMMSError(_currTime);

  //~ _he->measureMMSError(_currTime);
  _p->measureMMSError(_currTime);

  return ierr;
}



