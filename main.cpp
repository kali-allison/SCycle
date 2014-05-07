#include <petscts.h>
#include <iostream>
#include <sstream>
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
  PetscInt       Ny=10, Nz=10, order=2;
  PetscBool      loadMat = PETSC_FALSE;
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  //~ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  //~ierr = writeRateAndState(D);CHKERRQ(ierr);

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


PetscViewer outviewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(D.outFileRoot + "Aorder2").c_str(),
                               FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(D.A,outviewer);CHKERRQ(ierr);

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

  PCFactorSetUseInPlace(D.pc);
  ierr = KSPSetUp(D.ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(D.ksp);CHKERRQ(ierr);

  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);

  //~Mat F;
  //~ierr = PCFactorGetMatrix(D.pc,&F);CHKERRQ(ierr);
  //~ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);




  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(D.outFileRoot + "Forder2").c_str(),
                               FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(D.A,outviewer);CHKERRQ(ierr);

  ierr = KSPView(D.ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PCView(D.pc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);




  /*
  PetscInt its,maxCount=10;
  double startTime = MPI_Wtime();
  for (int count=0;count<maxCount;count++) {
    ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);
  }
  ierr = KSPGetIterationNumber(D.ksp,&its);CHKERRQ(ierr);
  double endTime = MPI_Wtime();
  ierr = KSPView(D.ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nits = %d, time =%g\n",its,(endTime-startTime)/maxCount);CHKERRQ(ierr);
  */

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

//~int screwDislocation(int argc,char **args)
int screwDislocation(PetscInt Ny,PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       order=4;
  PetscBool      loadMat = PETSC_FALSE;
  PetscViewer    viewer;
  PetscScalar    u,y,z;


  // set up the problem context
  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = writeRateAndState(D);CHKERRQ(ierr);
  ierr = setLinearSystem(D,loadMat);CHKERRQ(ierr);
  ierr = D.writeOperators();CHKERRQ(ierr);

  // set boundary conditions
  ierr = VecSet(D.gS,0.0);CHKERRQ(ierr); // surface
  ierr = VecSet(D.gD,0.0);CHKERRQ(ierr); // depth
  ierr = VecSet(D.gR,0.5);CHKERRQ(ierr); // remote

  // fault
  PetscInt Ii,Istart,Iend, N1 = D.H/D.dz;
  ierr = VecGetOwnershipRange(D.gF,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<N1) { ierr = VecSetValue(D.gF,Ii,0.0,INSERT_VALUES); }
    else { ierr = VecSetValue(D.gF,Ii,0.5,INSERT_VALUES); }
  }
  ierr = VecAssemblyBegin(D.gF);CHKERRQ(ierr);  ierr = VecAssemblyEnd(D.gF);CHKERRQ(ierr);

  // compute analytic surface displacement
  Vec anal;
  ierr = VecDuplicate(D.surfDisp,&anal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(anal,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-D.Nz*(Ii/D.Nz);
    y = D.dy*(Ii/D.Nz);
    u = (1.0/PETSC_PI)*atan(D.dy*Ii/D.H);
    ierr = VecSetValues(anal,1,&Ii,&u,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(anal);CHKERRQ(ierr);  ierr = VecAssemblyEnd(anal);CHKERRQ(ierr);

  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/anal",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(anal,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/gR",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(D.gR,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/gF",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(D.gF,viewer);CHKERRQ(ierr);

  ierr = ComputeRHS(D);CHKERRQ(ierr); // assumes gS and gD are 0
  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);

  // pull out surface displacement
  PetscInt ind;
  ierr = VecGetOwnershipRange(D.uhat,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-D.Nz*(Ii/D.Nz);
    ind = Ii/D.Nz;
    if (z == 0) {
      ierr = VecGetValues(D.uhat,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValues(D.surfDisp,1,&ind,&u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(D.surfDisp);CHKERRQ(ierr);  ierr = VecAssemblyEnd(D.surfDisp);CHKERRQ(ierr);

  std::ostringstream fileName;
  fileName << "data/surfDisp" << order << "Ny" << Ny;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.str().c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/surfDisp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(D.surfDisp,viewer);CHKERRQ(ierr);


  // compute error
  PetscScalar maxErr[2] = {0.0,0.0};
  Vec diff;
  VecDuplicate(D.surfDisp,&diff);CHKERRQ(ierr);

  ierr = VecWAXPY(diff,-1.0,D.surfDisp,anal);CHKERRQ(ierr);
  ierr = VecAbs(diff);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_1_AND_2,maxErr);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %g %g %g %g %.9g %.9g,\n",
                     D.Ny,D.Nz,D.dy,D.dz,D.Ly,D.Lz,maxErr[0],maxErr[1]);CHKERRQ(ierr);


  return ierr;
}

// performs MMS test for SBP operator convergence
int sbpConvergence(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  Vec            uAnal;
  PetscScalar    u,y,z,err,n=7.0;
  PetscInt       Ii,Istart,Iend;
  PetscViewer    viewer;

  PetscInt       order=2,Ny=301, Nz=301;
  // allow command line user input to override defaults
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);

  // set up the problem context
  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = setLinearSystem(D,PETSC_FALSE);CHKERRQ(ierr);

  // set boundary conditions
  ierr = VecSet(D.gS,0.0);CHKERRQ(ierr); // surface
  ierr = VecSet(D.gD,0.0);CHKERRQ(ierr); // depth

  // fault:
  ierr = VecGetOwnershipRange(D.gR,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = D.dz*(Ii-D.Nz*(Ii/D.Nz));
    y = D.dy*(Ii/D.Nz);
    u = cos(n*PETSC_PI*(z-D.Lz)/D.Lz);
    ierr = VecSetValue(D.gR,Ii,u,INSERT_VALUES);

    u = u*cosh( (double)n*PETSC_PI*D.Ly/D.Lz );
    ierr = VecSetValue(D.gF,Ii,u,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(D.gF);CHKERRQ(ierr); ierr = VecAssemblyBegin(D.gR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.gF);CHKERRQ(ierr);   ierr = VecAssemblyEnd(D.gR);CHKERRQ(ierr);

  VecDuplicate(D.uhat,&uAnal);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(uAnal,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = D.dz*(Ii-D.Nz*(Ii/D.Nz));
    y = D.dy*(Ii/D.Nz);
    u = cos(n*PETSC_PI*(z-D.Lz)/D.Lz)*cosh( (double)n*PETSC_PI*(y-D.Ly)/D.Lz );
    ierr = VecSetValue(uAnal,Ii,u,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(uAnal);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uAnal);CHKERRQ(ierr);

  // output
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(D.outFileRoot+"uAnal").c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(uAnal,viewer);CHKERRQ(ierr);


  ierr = ComputeRHS(D);CHKERRQ(ierr);

  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);

  PetscScalar uhat;
  ierr = VecGetOwnershipRange(uAnal,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(uAnal,1,&Ii,&u);
    ierr = VecGetValues(D.uhat,1,&Ii,&uhat);
    err += (u-uhat)*(u-uhat);
  }
  err = sqrt(err/((double) Ny*Nz));

  //~ierr = VecAXPY(uAnal,-1.0,D.uhat);CHKERRQ(ierr);
  //~ierr = VecNorm(uAnal,NORM_2,&err);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %i %e %e %.9e\n",
                     order,Ny,Nz,D.dy,D.dz,err);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(D.outFileRoot+"uhat").c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(D.uhat,viewer);CHKERRQ(ierr);


  return ierr;
}

int runEqCycle(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ny=301, Nz=301, order=4;
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
  double timeBeforeOpCreation = MPI_Wtime();
  ierr = setLinearSystem(D,loadMat);CHKERRQ(ierr);
  double timeAfterOpCreation = MPI_Wtime();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"linear op time: %f\n",timeAfterOpCreation-timeBeforeOpCreation);CHKERRQ(ierr);
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


  //~ierr = sbpConvergence(argc,args);CHKERRQ(ierr);// perform MMS

  // compare screw dislocation with numerics
  PetscInt Ny=721, Nz=241;
  //~for (Nz=401;Nz<802;Nz+=200) {
    for (Ny=241;Ny<1442;Ny+=200) {
      screwDislocation(Ny,Nz);
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"%i ",Nz);CHKERRQ(ierr);
    }
    //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  //~}
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);


  //~runEqCycle(argc,args);
  //~ierr = linearSolveTests(argc,args);CHKERRQ(ierr);
  //~runTests(argc,args);



  PetscFinalize();

  return ierr;
}

