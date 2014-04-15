#include <petscts.h>
#include <iostream>
#include <string>
#include "userContext.h"
#include "init.hpp"
#include "rateAndState.h"
#include "rootFindingScalar.h"
 #include "debuggingFuncs.hpp"
#include "linearSysFuncs.h"
#include "timeStepping.h"
#include "odeSolver.h"

using namespace std;

// For preconditioner experimentation
int linearSolveTests(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ny=5, Nz=7, order=2;
  PetscBool      loadMat = PETSC_FALSE;
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = writeRateAndState(D);CHKERRQ(ierr);

  // SBP operators and penalty terms
  D.alphaF = -13.0/D.dy;
  D.alphaR = -13.0/D.dy;
  D.alphaS = -1.0;
  D.alphaD = -1.0;
  D.beta   = 1.0;

  // set boundary data to match constant tectonic plate motion
  ierr = VecSet(D.gF,0.0);CHKERRQ(ierr);
  ierr = VecSet(D.gS,0.0);CHKERRQ(ierr);
  ierr = VecSet(D.gD,0.0);CHKERRQ(ierr);
  ierr = VecSet(D.gR,D.vp*D.initTime/2.0);CHKERRQ(ierr);

  if (loadMat) { ierr = loadOperators(D);CHKERRQ(ierr); }
  else { ierr = createOperators(D);CHKERRQ(ierr);}
  ierr = ComputeRHS(D);CHKERRQ(ierr);

  ierr = KSPSetType(D.ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(D.ksp,D.A,D.A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(D.ksp,&D.pc);CHKERRQ(ierr);

  // use direct solve (LU)
  //~ierr = KSPSetType(D.ksp,KSPPREONLY);CHKERRQ(ierr);
  //~ierr = KSPSetOperators(D.ksp,D.A,D.A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  //~ierr = KSPGetPC(D.ksp,&D.pc);CHKERRQ(ierr);
  //~ierr = PCSetType(D.pc,PCLU);CHKERRQ(ierr);

  // use preconditioning from HYPRE
  //~ierr = PCSetType(D.pc,PCHYPRE);CHKERRQ(ierr);
  //~ierr = PCHYPRESetType(D.pc,"boomeramg");CHKERRQ(ierr);
  //~ierr = KSPSetTolerances(D.ksp,D.kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  //~ierr = PCFactorSetLevels(D.pc,4);CHKERRQ(ierr);

  // use direct LU from MUMPS
  PCSetType(D.pc,PCLU);
  PCFactorSetMatSolverPackage(D.pc,MATSOLVERMUMPS);
  PCFactorSetUpMatSolverPackage(D.pc);


  ierr = KSPSetUp(D.ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(D.ksp);CHKERRQ(ierr);
  PetscInt its,maxCount=10;
  double startTime = MPI_Wtime();
  for (int count=0;count<maxCount;count++) {
    ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);
  }
  ierr = KSPGetIterationNumber(D.ksp,&its);CHKERRQ(ierr);
  double endTime = MPI_Wtime();
  ierr = KSPView(D.ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nits = %d, time =%g\n",its,(endTime-startTime)/maxCount);CHKERRQ(ierr);


  return ierr;
}


int runTests(int argc,char **args)
{

  PetscErrorCode ierr = 0;
  PetscInt       Ny=5, Nz=7, order=2;
  PetscBool      loadMat = PETSC_FALSE;
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  //~PetscScalar Hinvy[Ny];
  //~ierr = SBPopsArrays(2,Ny,0.5,Hinvy);CHKERRQ(ierr);
  //~ierr = printMyArray(Hinvy, Ny);CHKERRQ(ierr);

  //~PetscInt rows[Ny];
  //~for (PetscInt ind=0;ind<Ny;ind++){
    //~rows[ind]=ind;
  //~}
  //~PetscInt Ii,Istart,Iend;

  /* Try making things faster with arrays!!!*/
  Mat Iy_Hinvz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_Hinvz,PETSC_DECIDE,PETSC_DECIDE,Ny*Nz,Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_Hinvz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_Hinvz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_Hinvz);CHKERRQ(ierr);
  //~ierr = MatGetOwnershipRange(Iy_Hinvz,&Istart,&Iend);CHKERRQ(ierr);
  //~ierr = MatSetValues(Iy_Hinvz,1,&Istart,Ny,rows,Hinvy,INSERT_VALUES);CHKERRQ(ierr);
  //~for (Ii=Istart;Ii<Nz;Ii++) {
    //~ierr = MatSetValues(Hinvy_Iz_e0y_Iz,1,&Ii,1,&Ii,&(Hinvy[0]),INSERT_VALUES);CHKERRQ(ierr);
  //~}
  //~ierr = MatAssemblyBegin(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatAssemblyEnd(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //~ierr = MatView(Iy_Hinvz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  return ierr;
}

int runEqCycle(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ny=5, Nz=7, order=2;
  PetscBool      loadMat = PETSC_FALSE;

  // allow command line user input to override defaults
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = writeRateAndState(D);CHKERRQ(ierr);
  ierr = setLinearSystem(D,loadMat);CHKERRQ(ierr);
  if (!loadMat) { ierr = D.writeOperators();CHKERRQ(ierr); }
  ierr = setInitialTimeStep(D);CHKERRQ(ierr);
  ierr = D.writeInitialStep();CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"About to start integrating ODE\n");CHKERRQ(ierr);

  OdeSolver ts = OdeSolver(D.maxStepCount,"RK32");
  ierr = ts.setInitialConds(D.var,2);CHKERRQ(ierr);
  ierr = ts.setTimeRange(D.initTime,D.maxTime);CHKERRQ(ierr);
  ierr = ts.setTolerance(D.atol);CHKERRQ(ierr);
  ierr = ts.setStepSize(D.initDeltaT);CHKERRQ(ierr);
  ierr = ts.setTimeStepBounds(D.minDeltaT,D.maxDeltaT);CHKERRQ(ierr);
  ierr = ts.setRhsFunc(rhsFunc);CHKERRQ(ierr);
  ierr = ts.setUserContext(&D);CHKERRQ(ierr);
  ierr = ts.setTimeMonitor(timeMonitor);CHKERRQ(ierr);


  double timeBeforeIntegration = MPI_Wtime();
  ierr = ts.runOdeSolver();CHKERRQ(ierr);
  double timeAfterIntegration = MPI_Wtime();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"integration time: %f\n",timeAfterIntegration-timeBeforeIntegration);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeTauTime = %g\n",D.computeTauTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeVelTime = %g\n",D.computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"kspTime = %g\n",D.kspTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeRhsTime = %g\n",D.computeRhsTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"agingLawTime = %g\n",D.agingLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"rhsTime = %g\n",D.rhsTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"rootIts = %i\n",D.rootIts);CHKERRQ(ierr);

  ierr = ts.viewSolver();CHKERRQ(ierr);

  return 0;
}

int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  runEqCycle(argc,args);
  //~ierr = linearSolveTests(argc,args);CHKERRQ(ierr);
  //~runTests(argc,args);

  PetscFinalize();

  return ierr;
}

