#include "earth.hpp"

using namespace std;

//================= constructor and destructor ========================

Earth::Earth(const char* inputFile1, const char* inputFile2)
: _domain1(inputFile1), _domain2(inputFile2),
  _slider1(_domain1),_slider2(_domain2),
  _vL(_domain1._vp),
  _timeIntegrator(_domain1._timeIntegrator),
  _strideLength(_domain1._strideLength),_maxStepCount(_domain1._maxStepCount),
  _initTime(_domain1._initTime),_currTime(_initTime),
  _maxTime(_domain1._maxTime),_minDeltaT(_domain1._minDeltaT),_maxDeltaT(_domain1._maxDeltaT),
  _stepCount(0),_atol(_domain1._atol),_initDeltaT(_domain1._initDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in earth.cpp.\n");
#endif

  assert(_domain1._order == _domain2._order);
  assert(_domain1._Nz == _domain2._Nz);
  assert(_domain1._initTime == _domain2._initTime);

  _domain1.write();
  _domain2.write();

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcL);
  VecSetSizes(_bcL,PETSC_DECIDE,_domain1._Nz);
  VecSetFromOptions(_bcL);     PetscObjectSetName((PetscObject) _bcL, "_bcL");
  VecSet(_bcL,0.0);
  VecDuplicate(_bcL,&_bcR);
  _bcRShift = _slider1._fault.getBcRShift();
  PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,_vL*_initTime/2.0);
  VecAXPY(_bcR,1.0,_bcRShift);

  VecCreate(PETSC_COMM_WORLD,&_tauMod);
  VecSetSizes(_tauMod,PETSC_DECIDE,_domain1._Nz);
  VecSetFromOptions(_tauMod);     PetscObjectSetName((PetscObject) _tauMod, "_tauMod");
  VecSet(_tauMod,0.0);

  // set up initial conditions on sliders
  VecSet(_slider1._bcF,0.0);
  VecSet(_slider1._bcR,0.0); // slider 2 is initially locked by friction
  VecSet(_slider2._bcF,0.0); // bcF holds slip from equilibrium, not distance from slider1
  VecSet(_slider2._bcR,_vL*_initTime/2.0); // slider 2 is loaded by plate tectonics
  VecAXPY(_slider2._bcR,1.0,_bcRShift);

  _slider1.resetInitialConds();
  _slider2.resetInitialConds();


  if (_timeIntegrator.compare("FEuler")==0) {
    _quadrature = new FEuler(_maxStepCount,_maxTime,_initDeltaT,_domain1._timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadrature = new RK32(_maxStepCount,_maxTime,_initDeltaT,_domain1._timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
  }

  // set up initial conditions for integration (shallow copy)
  _var.push_back(_slider1._fault._var[0]);
  _var.push_back(_slider1._fault._var[1]);
  _var.push_back(_slider2._fault._var[0]);
  _var.push_back(_slider2._fault._var[1]);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in earth.cpp.\n");
#endif
}

Earth::~Earth()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in earth.cpp.\n");
#endif


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in earth.cpp.\n");
#endif
}

PetscErrorCode Earth::view()
{
  PetscErrorCode ierr = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n===============================\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"===============================\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing Earth OdeSolver Summary:\n");CHKERRQ(ierr);
  ierr = _quadrature->view();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n===============================\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"===============================\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing Earth slider1 Summary:\n");CHKERRQ(ierr);
  ierr = _slider1.view();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n===============================\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"===============================\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing Earth slider2 Summary:\n");CHKERRQ(ierr);
  ierr = _slider2.view();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n===============================\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"===============================\n\n");CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Runtime Summary:\n");CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}

//===================== private member functions =======================


//~PetscErrorCode Earth::d_dt(PetscScalar const time,const vector<Vec>& var,vector<Vec>& dvar)
PetscErrorCode Earth::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting d_dt in earth.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  // !!!TO DO: think about the factor of 1/2 thing!

  // slip on each slider is determined by faultDisp, which _quadrature just integrated
  ierr = VecCopy(*varBegin,_slider1._bcF);CHKERRQ(ierr);
  ierr = VecScale(_slider1._bcF,0.5);CHKERRQ(ierr);

  ierr = VecCopy(*(varBegin+2),_slider2._bcF);CHKERRQ(ierr);
  ierr = VecScale(_slider2._bcF,0.5);CHKERRQ(ierr);

  ierr = VecCopy(_slider2._bcF,_slider1._bcR);CHKERRQ(ierr); // slider 2's movement stretches slider 1's spring

  ierr = VecSet(_slider2._bcR,_vL*time/2.0);CHKERRQ(ierr); // slider 2 is loaded by plate tectonics
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);

  // slider one's force balance is determined entirely from displacement BCs
  ierr = _slider1.d_dt(time,varBegin,varBegin+2,dvarBegin,dvarBegin+2);CHKERRQ(ierr);

  // slider 2's force balance also has the force from slider 1
  ierr = VecCopy(_slider1._fault._tau,_tauMod);CHKERRQ(ierr);
  ierr = VecScale(_tauMod,-1.0);CHKERRQ(ierr);
  ierr = _slider2.d_dt(time,varBegin+2,varEnd,dvarBegin+2,dvarEnd,_tauMod);CHKERRQ(ierr);
  //~ierr = _slider2.d_dt(time,varBegin+2,varEnd,dvarBegin+2,dvarEnd);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending d_dt in earth.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

//~PetscErrorCode Earth::timeMonitor(const PetscReal time, const PetscInt stepCount,
                 //~const vector<Vec>& var,const vector<Vec>& dvar)
PetscErrorCode Earth::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;

  if ( stepCount % _strideLength == 0) {
    _stepCount++;
    _currTime = time;
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep();CHKERRQ(ierr);
    //~ierr = _slider1.timeMonitor(time,stepCount,var,dvar);CHKERRQ(ierr);
    ierr = _slider1.timeMonitor(time,stepCount,varBegin,varBegin+2,dvarBegin,dvarBegin+2);CHKERRQ(ierr);
    ierr = _slider2.timeMonitor(time,stepCount,varBegin+2,varEnd,dvarBegin+2,dvarEnd);CHKERRQ(ierr);
  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Earth::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting integrate in earth.cpp\n");CHKERRQ(ierr);
#endif
  //~double startTime = MPI_Wtime();

  // call odeSolver routine integrate here
  //~_slider1.integrate();

  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_var, 4);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  //~_integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending integrate in earth.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Earth::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeStep in earth.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  //~double startTime = MPI_Wtime();


  if (_stepCount==0) {
    ierr = _slider1.writeStep();CHKERRQ(ierr);
    ierr = _slider2.writeStep();CHKERRQ(ierr);
  }
  //~else {
  //~}
  //~ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  //~ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);

  //~_writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in earth.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}
