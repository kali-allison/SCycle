#include <petscts.h>
#include <iostream>
#include <string>
#include "odeSolver.h"

using namespace std;


//================= constructors and destructor ========================

TimeSolver::TimeSolver(PetscInt maxNumSteps,string solverType)
:_initT(0),_finalT(0),_currT(0),_deltaT(0),_minDeltaT(1e-14),_maxDeltaT(1e-14),
_atol(1.0e-9),_reltol(1.0e-9),
_maxNumSteps(maxNumSteps),_stepCount(0),_numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0),
_solverType(solverType),
_var(NULL),_dvar(NULL),_lenVar(0),_userContext(NULL)
{
  _rhsFunc = &tempRhsFunc;
  _timeMonitor = &tempTimeMonitor;
}

TimeSolver::TimeSolver(PetscScalar finalT,PetscInt maxNumSteps,string solverType)
:_initT(0),_finalT(finalT),_currT(0),_deltaT(finalT/maxNumSteps),_minDeltaT(1e-14),_maxDeltaT(finalT/10.0),
_atol(1.0e-9),_reltol(1.0e-9),
_maxNumSteps(maxNumSteps),_stepCount(0),_solverType(solverType),
_var(NULL),_dvar(NULL),_lenVar(0),_userContext(NULL)
{
  _rhsFunc = &tempRhsFunc;
  _timeMonitor = &tempTimeMonitor;
}

TimeSolver::~TimeSolver()
{
  if (_dvar!=0) {
    for (int ind=0;ind<_lenVar;ind++) {
      VecDestroy(&_dvar[ind]);
    }
    delete[] _dvar;
  }
}

//================= modify data members ================================

PetscErrorCode TimeSolver::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
  _initT = initT;
  _currT = initT;
  _finalT = finalT;
  _maxDeltaT = max((_finalT - initT)/10.0,_deltaT);
  return 0;
}

PetscErrorCode TimeSolver::setStepSize(const PetscReal deltaT)
{
  _deltaT = deltaT;
  return 0;
}

PetscErrorCode TimeSolver::setTolerance(const PetscReal tol)
{
  _atol = tol;
  _reltol = tol;
  return 0;
}

PetscErrorCode TimeSolver::setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*))
{
  _rhsFunc = rhsFunc;
  return 0;
}

PetscErrorCode TimeSolver::setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*))
{
  _timeMonitor = timeMonitor;
  return 0;
}

PetscErrorCode TimeSolver::setUserContext(void * userContext)
{
  _userContext = userContext;
  return 0;
}

PetscErrorCode TimeSolver::setInitialConds(Vec *var, const int lenVar)
{
  PetscErrorCode ierr = 0;
  PetscScalar    zero=0.0;

  _var = var;
  _lenVar = lenVar;

  _dvar = new Vec[_lenVar];
  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    ierr = VecSet(_dvar[ind],zero);CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode TimeSolver::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  return 0;
}

//================= output useful info =========================

PetscErrorCode TimeSolver::viewSolver()
{
  PetscErrorCode ierr;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"TimeSolver summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   solver type: %s\n",
                     (_solverType).c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time interval: %g to %g\n",
                     _initT,_finalT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   permitted step size range: [%g,%g]\n",
                     _minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total number of steps taken: %i/%i\n",
                     _stepCount,_maxNumSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   final time reached: %g\n",
                     _currT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   tolerance: %g\n",
                     _atol);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of rejected steps: %i\n",
                     _numRejectedSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times min step size enforced: %i\n",
                     _numMinSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times max step size enforced: %i\n",
                     _numMaxSteps);CHKERRQ(ierr);
  return 0;
}

// Outputs data at each time step.
PetscErrorCode TimeSolver::debugMyCode(const PetscReal time,const PetscInt steps,const Vec *var,const char *str)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    val;

  ierr= VecGetOwnershipRange(var[0],&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(var[0],1,&Istart,&val);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"  %s: %i %g %g\n",str,steps,time,val);CHKERRQ(ierr);

  return ierr;
}

//================= perform actual integration =========================

PetscErrorCode TimeSolver::runTimeSolver()
{
  PetscErrorCode ierr;

  if (_solverType.compare("FEULER")==0) {
    ierr = odeFEULER();CHKERRQ(ierr);
  }
  else if (_solverType.compare("RK32")==0) {
    ierr = odeRK32();CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Solver type not understood\n");CHKERRQ(ierr);
  }

  return 0;
}

PetscErrorCode TimeSolver::odeFEULER()
{

  PetscErrorCode ierr = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    ierr = _rhsFunc(_currT,_lenVar,_var,_dvar,_userContext);CHKERRQ(ierr);
    for (int varInd=0;varInd<_lenVar;varInd++) {
      ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // in = in + deltaT*dwdt
    }
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = _timeMonitor(_currT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode TimeSolver::odeRK32()
{
  PetscErrorCode ierr=0;
  int            attemptCount=0;
  PetscReal      err[_lenVar],totErr=0.0;
  PetscInt       size;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // containers for intermediate results
  Vec *varHalfdT,*dvarHalfdT,*vardT,*dvardT,*var2nd,*dvar2nd,*var3rd,*errVec;
  varHalfdT = new Vec[_lenVar]; dvarHalfdT = new Vec[_lenVar];
  vardT = new Vec[_lenVar];     dvardT = new Vec[_lenVar];
  var2nd = new Vec[_lenVar];    dvar2nd = new Vec[_lenVar];
  var3rd = new Vec[_lenVar];
  errVec  = new Vec[_lenVar];
  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&varHalfdT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&vardT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&var2nd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&var3rd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&dvarHalfdT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&dvardT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&dvar2nd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&var3rd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&errVec[ind]);CHKERRQ(ierr);
  }

  // set initial condition
  ierr = _rhsFunc(_currT,_lenVar,_var,_dvar,_userContext);CHKERRQ(ierr);
  //~ierr = _timeMonitor(_initT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;

    while (1) { // repeat until time step is acceptable

      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(varHalfdT[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
      }
      ierr = _rhsFunc(_currT+0.5*_deltaT,_lenVar,varHalfdT,dvarHalfdT,_userContext);

      // stage 2: integrate fields to _currT + _deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(vardT[ind],-_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(vardT[ind],2*_deltaT,dvarHalfdT[ind]);CHKERRQ(ierr);
      }
      ierr = _rhsFunc(_currT+_deltaT,_lenVar,vardT,dvardT,_userContext);

      // 2nd and 3rd order update
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(var2nd[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var2nd[ind],0.5*_deltaT,dvardT[ind]);CHKERRQ(ierr);

        ierr = VecWAXPY(var3rd[ind],_deltaT/6.0,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var3rd[ind],2*_deltaT/3.0,dvarHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var3rd[ind],_deltaT/6.0,dvardT[ind]);CHKERRQ(ierr);
      }

      // calculate error
      totErr = 0.0;
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(errVec[ind],-1.0,var2nd[ind],var3rd[ind]);CHKERRQ(ierr);

        // error based on max norm
        //~ierr = VecNorm(errVec[ind],NORM_INFINITY,&err[ind]);CHKERRQ(ierr);
        //~if (err[ind]>totErr) { totErr=err[ind]; }

        // error based on weighted 2 norm
        VecDot(errVec[ind],errVec[ind],&err[ind]);
        VecGetSize(errVec[ind],&size);
        totErr += err[ind]/size;
      }
      totErr = sqrt(totErr);

      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"  attemptCount =  %d, totErr = %g, currT = %g,_deltaT=%g\n",
                         //~attemptCount,totErr,_currT,_deltaT);CHKERRQ(ierr);

      if (totErr<_atol) { break; }
      // else: step is unacceptable, so modify time step

      _deltaT = min(_maxDeltaT,0.9*_deltaT*pow(_atol/totErr,1.0/3.0));
      _deltaT = max(_minDeltaT,_deltaT);
      if (_minDeltaT == _deltaT) {
        _numMinSteps++;
        break;
      }
      else if (_maxDeltaT == _deltaT) {
        _numMaxSteps++;
        break;
      }

      _numRejectedSteps++;
    }
    //~debugMyCode(_currT,_stepCount,_var,"var");
    _currT = _currT+_deltaT;

    // accept 3rd order solution as update
    for (int ind=0;ind<_lenVar;ind++) {
      ierr = VecCopy(var3rd[ind],_var[ind]);CHKERRQ(ierr);
      ierr = VecCopy(dvardT[ind],_dvar[ind]);CHKERRQ(ierr);
    }
    if (totErr!=0.0) {
      _deltaT=min(_maxDeltaT,0.9*_deltaT*pow(_atol/totErr,1.0/3.0));
      _deltaT = max(_minDeltaT,_deltaT);
      if (_minDeltaT == _deltaT) {
        _numMinSteps++;
        break;
      }
      else if (_maxDeltaT == _deltaT) {
        _numMaxSteps++;
        break;
      }
    }

    ierr = _timeMonitor(_currT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %e %i\n",_stepCount,_currT,attemptCount);CHKERRQ(ierr);
  }

  // destruct temporary containers
  for (int ind=0;ind<_lenVar;ind++) {
    VecDestroy(&varHalfdT[ind]); VecDestroy(&dvarHalfdT[ind]);
    VecDestroy(&vardT[ind]);     VecDestroy(&dvardT[ind]);
    VecDestroy(&var2nd[ind]);
    VecDestroy(&var3rd[ind]);
    VecDestroy(&errVec[ind]);
  }
  delete[] varHalfdT; delete[] dvarHalfdT;
  delete[] vardT;     delete[] dvardT;
  delete[] var2nd;
  delete[] var3rd;
  delete[] errVec;


  return ierr;
}

//================= placehold functions ================================
PetscErrorCode tempRhsFunc(const PetscReal time, const int lenVar, Vec *var, Vec *dvar, void*userContext)
{
  PetscErrorCode ierr = 0;
  SETERRQ(PETSC_COMM_WORLD,1,"rhsFunc not defined.\n");
  return ierr;
}

PetscErrorCode tempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext)
{
  PetscErrorCode ierr = 0;
  return ierr;
}
