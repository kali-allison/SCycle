/***********************************************************************
 *
 * Contains functions to test the odeSolver routines.
 *
 **********************************************************************/
#include "testOdeSolver.hpp"

TestOdeSolver::TestOdeSolver()
: _f(NULL),_quadrature(NULL),
  _timeViewer(NULL), _fViewer(NULL),
  _strideLength(1),_maxStepCount(3),
  _initTime(0.0),_currTime(_initTime),_maxTime(10),
  _minDeltaT(10),_maxDeltaT(1e4),
  _stepCount(0),_atol(1e-7),_initDeltaT(1)
{
  PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::TestOdeSolver in testOdeSolver.cpp\n");

  VecCreate(PETSC_COMM_WORLD,&_f);
  VecSetSizes(_f,PETSC_DECIDE,5);
  VecSetFromOptions(_f);     PetscObjectSetName((PetscObject) _f, "_f");
  VecSet(_f,0.0);

  _var.push_back(_f);

  _quadrature = new RK32(_maxStepCount,_maxTime,_initDeltaT,"P");

  PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::TestOdeSolver in testOdeSolver.cpp\n");
}

TestOdeSolver::~TestOdeSolver()
{
  PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::~TestOdeSolver in testOdeSolver.cpp\n");

  VecDestroy(&_f);

  PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::~TestOdeSolver in testOdeSolver.cpp\n");
}



PetscErrorCode TestOdeSolver::integrate()
{
    PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  // call odeSolver routine integrate here
  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_var);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode TestOdeSolver::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
        it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  VecSet(*dvarBegin,0.0);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}






PetscErrorCode TestOdeSolver::timeMonitor(const PetscReal time,const PetscInt stepCount,
                       const_it_vec varBegin,const_it_vec varEnd,
                       const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::timeMonitor in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

    std::string outputDir = "./data/";
    _stepCount++;
    _currTime = time;
    writeStep();



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::timeMonitor in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode TestOdeSolver::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::writeStep in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

    std::string outputDir = "./data/";

  if (_stepCount == 0) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"f").c_str(),FILE_MODE_WRITE,
                                 &_fViewer);CHKERRQ(ierr);
    ierr = VecView(_f,_fViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_fViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"f").c_str(),
                                   FILE_MODE_APPEND,&_fViewer);CHKERRQ(ierr);
    _stepCount++;
  }
  else {
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);
  ierr = VecView(_f,_fViewer);CHKERRQ(ierr);
}


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::writeStep in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode TestOdeSolver::debug(const PetscReal time,const PetscInt stepCount,
                       const_it_vec varBegin,const_it_vec varEnd,
                       const_it_vec dvarBegin,const_it_vec dvarEnd, const char *stage)
{
  PetscErrorCode ierr = 0;
//~#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TestOdeSolver::debug in lithosphere.cpp\n");CHKERRQ(ierr);
//~#endif
//~
//~
//~#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TestOdeSolver::debug in lithosphere.cpp\n");CHKERRQ(ierr);
//~#endif
  return ierr;
}
