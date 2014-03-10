#include <petscts.h>
#include <iostream>
#include <string>
#include "userContext.h"
#include "rateAndState.h"
//~ #include "debuggingFuncs.h"
#include "linearSysFuncs.h"
#include "timeStepping.h"
#include "timeSolver.h"

using namespace std;

PetscErrorCode func(const PetscReal t,const int lenVar,Vec *var,Vec *dvar,void *ctx)
{

  PetscErrorCode ierr = 0;

  for (int ind=0;ind<lenVar;ind++) {
    ierr = VecCopy(var[ind],dvar[ind]);CHKERRQ(ierr);
    ierr= VecScale(dvar[ind],-t);CHKERRQ(ierr);
  }

  //~ ierr = VecCopy(*in,*out);CHKERRQ(ierr);
  //~ ierr = VecScale(*out,-t);CHKERRQ(ierr);

   return ierr;
}

int runTests(UserContext *D)
{

  PetscErrorCode ierr = 0;
  PetscInt Ii=1;
  PetscScalar val;

  VecSet(D->tau,3.14e2);
  computeSlipVel(D);
  VecGetValues(D->V,1,&Ii,&val);
  PetscPrintf(PETSC_COMM_WORLD,"V(1)=%e\n",val);

  return ierr;
}

int runEqCycle(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ny=5, Nz=7, order=2;
  PetscBool      loadMat = PETSC_FALSE;
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  // Solution for my own timeSolver routines
  //~UserContext D = UserContext(order,Ny,Nz,"dataTimeSolver/");
  UserContext D = UserContext(order,Ny,Nz,"data/");
  ierr = setParameters(&D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(&D);CHKERRQ(ierr);
  ierr = setLinearSystem(&D,loadMat);CHKERRQ(ierr);
  if (!loadMat) { ierr = D.writeOperators();CHKERRQ(ierr); }
  //~ierr = setInitialTimeStep(&D);CHKERRQ(ierr);
  //~ierr = D.writeInitialStep();CHKERRQ(ierr);
  ierr = ComputeRHS(&D);CHKERRQ(ierr);

  //~KSPCreate(PETSC_COMM_WORLD,&D.ksp);
  //~KSPSetType(D.ksp,KSPPREONLY);
  //~KSPSetOperators(D.ksp,D.A,D.A,SAME_PRECONDITIONER);
  //~KSPGetPC(D.ksp,&D.pc);
  //~PCSetType(D.pc,PCLU);
  //~KSPSetUp(D.ksp);
  //~KSPSetFromOptions(D.ksp);
  //~double startTime = MPI_Wtime();
  //~double endTime = MPI_Wtime();
  //~PetscInt its;
  //~startTime = MPI_Wtime();
  //~for (int count=0;count<10;count++) {
    //~KSPSolve(D.ksp,D.rhs,D.uhat);
    //~ierr = KSPGetIterationNumber(D.ksp,&its);CHKERRQ(ierr);
  //~}
  //~endTime = MPI_Wtime();
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"its = %d, time =%g\n",its,(endTime-startTime)/10.);CHKERRQ(ierr);
  //~ierr = KSPView(D.ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"About to start integrating ODE\n");CHKERRQ(ierr);

  TimeSolver ts = TimeSolver(D.maxStepCount,"RK32");
  ierr = ts.setInitialConds(D.var,2);CHKERRQ(ierr);
  ierr = ts.setTimeRange(D.initTime,D.maxTime);CHKERRQ(ierr);
  ierr = ts.setTolerance(D.atol);CHKERRQ(ierr);
  ierr = ts.setStepSize(D.initDeltaT);CHKERRQ(ierr);
  ierr = ts.setTimeStepBounds(D.minDeltaT,D.maxDeltaT);CHKERRQ(ierr);
  ierr = ts.setRhsFunc(rhsFunc);CHKERRQ(ierr);
  ierr = ts.setUserContext(&D);CHKERRQ(ierr);
  ierr = ts.setTimeMonitor(timeMonitor);CHKERRQ(ierr);

  double timeBeforeIntegration = MPI_Wtime();
  ierr = ts.runTimeSolver();CHKERRQ(ierr);
  double timeAfterIntegration = MPI_Wtime();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"integration time: %f\n",timeAfterIntegration-timeBeforeIntegration);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeTauTime = %g\n",D.computeTauTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeVelTime = %g\n",D.computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"kspTime = %g\n",D.kspTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeRhsTime = %g\n",D.computeRhsTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"agingLawTime = %g\n",D.agingLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"rhsTime = %g\n",D.rhsTime);CHKERRQ(ierr);

  ierr = ts.viewSolver();CHKERRQ(ierr);



  return 0;
}

int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  runEqCycle(argc,args);

  PetscFinalize();

  return 0;
}

