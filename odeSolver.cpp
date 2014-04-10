#include <petscts.h>
#include <iostream>
#include <string>
#include "odeSolver.h"
#include "userContext.h"

using namespace std;


//================= constructors and destructor ========================

OdeSolver::OdeSolver(PetscInt maxNumSteps,string solverType)
:_initT(0),_finalT(0),_currT(0),_deltaT(0),_minDeltaT(1e-14),_maxDeltaT(1e-14),
_atol(1.0e-9),_reltol(1.0e-9),
_maxNumSteps(maxNumSteps),_stepCount(0),_numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0),
_solverType(solverType),_sourceFile(""),
_var(NULL),_dvar(NULL),_lenVar(0),_userContext(NULL)
{
  _rhsFunc = &tempRhsFunc;
  _timeMonitor = &tempTimeMonitor;
}

OdeSolver::OdeSolver(PetscScalar finalT,PetscInt maxNumSteps,string solverType)
:_initT(0),_finalT(finalT),_currT(0),_deltaT(finalT/maxNumSteps),_minDeltaT(1e-14),_maxDeltaT(finalT/10.0),
_atol(1.0e-9),_reltol(1.0e-9),
_maxNumSteps(maxNumSteps),_stepCount(0),
_solverType(solverType),_sourceFile(""),
_var(NULL),_dvar(NULL),_lenVar(0),_userContext(NULL)
{
  _rhsFunc = &tempRhsFunc;
  _timeMonitor = &tempTimeMonitor;
}

OdeSolver::~OdeSolver()
{
  if (_dvar!=0) {
    for (int ind=0;ind<_lenVar;ind++) {
      VecDestroy(&_dvar[ind]);
    }
    delete[] _dvar;
  }
}

//================= modify data members ================================

PetscErrorCode OdeSolver::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
  _initT = initT;
  _currT = initT;
  _finalT = finalT;
  _maxDeltaT = max((_finalT - initT)/10.0,_deltaT);
  return 0;
}

PetscErrorCode OdeSolver::setStepSize(const PetscReal deltaT)
{
  _deltaT = deltaT;
  return 0;
}

PetscErrorCode OdeSolver::setTolerance(const PetscReal tol)
{
  _atol = tol;
  _reltol = tol;
  return 0;
}

PetscErrorCode OdeSolver::setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*))
{
  _rhsFunc = rhsFunc;
  return 0;
}

PetscErrorCode OdeSolver::setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*))
{
  _timeMonitor = timeMonitor;
  return 0;
}

PetscErrorCode OdeSolver::setUserContext(void * userContext)
{
  _userContext = userContext;
  return 0;
}

PetscErrorCode OdeSolver::setInitialConds(Vec *var, const int lenVar)
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

PetscErrorCode OdeSolver::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  return 0;
}

PetscErrorCode OdeSolver::setSourceFile(const std::string sourceFile)
{
  _sourceFile = sourceFile;
  return 0;
}

//================= output useful info =========================

PetscErrorCode OdeSolver::viewSolver()
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
PetscErrorCode OdeSolver::debug(const PetscReal time,const PetscInt steps,const Vec *var,const Vec *dvar,const char *str)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    gRval,uVal,psiVal,velVal,dQVal;
  UserContext    *D = (UserContext*) _userContext;
  PetscScalar k = D->G/2/D->Ly;

  ierr= VecGetOwnershipRange(var[0],&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(var[0],1,&Istart,&uVal);CHKERRQ(ierr);
  ierr = VecGetValues(var[1],1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(dvar[0],&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(dvar[0],1,&Istart,&velVal);CHKERRQ(ierr);
  ierr = VecGetValues(dvar[1],1,&Istart,&dQVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(D->gR,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(D->gR,1,&Istart,&gRval);CHKERRQ(ierr);

  PetscScalar tauVal;
  ierr = VecGetValues(D->tau,1,&Istart,&tauVal);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"tau = %e\n",tauVal);CHKERRQ(ierr);

  if (steps == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s | %-15s %-15s %-15s | %-15s %-15s %-15s\n",
                       "Step","Stage","gR","D","Q","VL","V","dQ");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %6s | %.9e %.9e %.9e | %.9e %.9e %.9e\n",
                     _stepCount,str,2*gRval*k,uVal,psiVal,D->vp/2,velVal,dQVal);CHKERRQ(ierr);


  return ierr;
}

//================= perform actual integration =========================

PetscErrorCode OdeSolver::runOdeSolver()
{
  PetscErrorCode ierr;

  if (_solverType.compare("FEULER")==0) {
    ierr = odeFEULER();CHKERRQ(ierr);
  }
  else if (_solverType.compare("MANUAL")==0) {
    ierr = odeMANUAL();CHKERRQ(ierr);
  }
  else if (_solverType.compare("RK32")==0) {
    ierr = odeRK32();CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Solver type not understood\n");CHKERRQ(ierr);
  }

  return 0;
}

PetscErrorCode OdeSolver::odeFEULER()
{

  PetscErrorCode ierr = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    ierr = _rhsFunc(_currT,_lenVar,_var,_dvar,_userContext);CHKERRQ(ierr);
    for (int varInd=0;varInd<_lenVar;varInd++) {
      ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // var = var + deltaT*dvar
    }
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = _timeMonitor(_currT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode OdeSolver::odeMANUAL()
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  Vec            timeVec;
  PetscInt       Ii,Istart,Iend,innerCount=0;
  PetscScalar    newTime;

  // this code only works on 1 processor right now!!!
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1) {
    SETERRQ(PETSC_COMM_SELF,1,"The MANUAL ode algorithm only works on 1 processor");
    abort();
  }

  VecCreate(PETSC_COMM_WORLD,&timeVec);CHKERRQ(ierr);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,_sourceFile.c_str(),FILE_MODE_READ,&viewer);
  ierr = VecLoad(timeVec,viewer);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(timeVec,&Istart,&Iend);
  Ii = Istart+1;
  while (Ii<Iend && _stepCount < _maxNumSteps) {

    if (innerCount == 0) {
      ierr = VecGetValues(timeVec,1,&Ii,&newTime);CHKERRQ(ierr);
     _deltaT = newTime - _currT;
     _deltaT = (newTime - _currT)*0.25;
      Ii++;
      innerCount=4;
    }
    else { innerCount--; }

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ii=%i currTime=%g newTime=%g deltaT=%g\n",
                       Ii,_currT,newTime,_deltaT);CHKERRQ(ierr);

    ierr = _rhsFunc(_currT,_lenVar,_var,_dvar,_userContext);CHKERRQ(ierr);
    for (int varInd=0;varInd<_lenVar;varInd++) {
      ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // var = var + deltaT*dvar
    }
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = _timeMonitor(_currT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode OdeSolver::odeRK32()
{
  PetscErrorCode ierr=0;
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
  //~debug(_currT,_stepCount,_var,_dvar,"IC");
  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;

    while (1) { // repeat until time step is acceptable
      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(varHalfdT[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
      }
      ierr = _rhsFunc(_currT+0.5*_deltaT,_lenVar,varHalfdT,dvarHalfdT,_userContext);
      //~debug(_currT+0.5*_deltaT,_stepCount,varHalfdT,dvarHalfdT,"t+dt/2");

      // stage 2: integrate fields to _currT + _deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(vardT[ind],-_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(vardT[ind],2*_deltaT,dvarHalfdT[ind]);CHKERRQ(ierr);
      }
      ierr = _rhsFunc(_currT+_deltaT,_lenVar,vardT,dvardT,_userContext);
      //~debug(_currT+_deltaT,_stepCount,vardT,dvardT,"t+dt");

      // 2nd and 3rd order update
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(var2nd[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var2nd[ind],0.5*_deltaT,dvardT[ind]);CHKERRQ(ierr);

        ierr = VecWAXPY(var3rd[ind],_deltaT/6.0,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var3rd[ind],2*_deltaT/3.0,dvarHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(var3rd[ind],_deltaT/6.0,dvardT[ind]);CHKERRQ(ierr);
      }
      //~debug(_currT+_deltaT,_stepCount,var2nd,dvardT,"Y2");
      //~debug(_currT+_deltaT,_stepCount,var3rd,dvardT,"Y3");

      // calculate error
      totErr = 0.0;
      //~for (int ind=0;ind<_lenVar;ind++) {
        int ind = 0;
        ierr = VecWAXPY(errVec[ind],-1.0,var2nd[ind],var3rd[ind]);CHKERRQ(ierr);

        // error based on max norm
        ierr = VecNorm(errVec[ind],NORM_INFINITY,&err[ind]);CHKERRQ(ierr);
        //~if (err[ind]>totErr) { totErr=err[ind]; }

        // error based on weighted 2 norm
        //~VecDot(errVec[ind],errVec[ind],&err[ind]);
        //~VecGetSize(errVec[ind],&size);
        //~totErr += err[ind]/size;
      //~}
      //~totErr = sqrt(totErr);
      totErr = err[ind];
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"totErr=%7e\n",totErr);CHKERRQ(ierr);

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
        //~break;
      }

      _numRejectedSteps++;
    }
    _currT = _currT+_deltaT;

    // accept 3rd order solution as update
    for (int ind=0;ind<_lenVar;ind++) {
      ierr = VecCopy(var3rd[ind],_var[ind]);CHKERRQ(ierr);
    }
    ierr = _rhsFunc(_currT,_lenVar,_var,_dvar,_userContext);
    //~debug(_currT+_deltaT,_stepCount,_var,_dvar,"Y3 F");

    if (totErr!=0.0) {
      _deltaT=min(_maxDeltaT,0.9*_deltaT*pow(_atol/totErr,1.0/3.0));
      _deltaT = max(_minDeltaT,_deltaT);
      if (_minDeltaT == _deltaT) {
        _numMinSteps++;
      }
      else if (_maxDeltaT == _deltaT) {
        _numMaxSteps++;
      }
    }

    ierr = _timeMonitor(_currT,_stepCount,_var,_lenVar,_userContext);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %e\n",_stepCount,_currT);CHKERRQ(ierr);
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
