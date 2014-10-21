#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>

#include "domain.hpp"
//~#include "odeSolver.hpp"
#include "lithosphere.hpp"



using namespace std;
/*
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
*/



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

  //~return ierr;
//~}

/*
int runTests(int argc,char **args)
{

  PetscErrorCode ierr = 0;

  Domain domain("init.txt");
  domain.write();

  Lithosphere lith(domain);

  Fault fault(domain);
  VecView(fault._vel,PETSC_VIEWER_STDOUT_WORLD);

  //~VecSet(fault._var[0],
  VecView(fault._vel,PETSC_VIEWER_STDOUT_WORLD);



  return ierr;
}

//~int screwDislocation(int argc,char **args)
int screwDislocation(PetscInt Ny,PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       order=4;
  PetscBool      loadMat = PETSC_FALSE;
  PetscViewer    viewer;
  PetscScalar    u,z;


  // set up the problem context
  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = D.writeRateAndState();CHKERRQ(ierr);
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
    //~y = D.dy*(Ii/D.Nz);
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
  ierr = VecNorm(diff,NORM_2,&maxErr[0]);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_INFINITY,&maxErr[1]);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %g %g %g %g %.9g %.9g\n",
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
  //~PetscViewer    viewer;
  PetscBool      loadMat = PETSC_FALSE;

  PetscInt       order=2,Ny=76, Nz=76;
  // allow command line user input to override defaults
  ierr = PetscOptionsGetInt(NULL,"-order",&order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-loadMat",&loadMat,NULL);CHKERRQ(ierr);

  // set up the problem context
  UserContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = setLinearSystem(D,loadMat);CHKERRQ(ierr);
  if (!loadMat) {ierr = D.writeOperators();CHKERRQ(ierr);}

  // set boundary conditions
  ierr = VecSet(D.gS,0.0);CHKERRQ(ierr); // surface
  ierr = VecSet(D.gD,0.0);CHKERRQ(ierr); // depth
  ierr = VecGetOwnershipRange(D.gR,&Istart,&Iend); // fault and remote boundary conditions
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
    u = cos(n*PETSC_PI*(z-D.Lz)/D.Lz)*cosh( (double) n*PETSC_PI*(y-D.Ly)/D.Lz );
    ierr = VecSetValue(uAnal,Ii,u,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(uAnal);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uAnal);CHKERRQ(ierr);

  // output
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(D.outFileRoot+"uAnal").c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(uAnal,viewer);CHKERRQ(ierr);


  ierr = ComputeRHS(D);CHKERRQ(ierr);

  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);

  ierr = VecAXPY(uAnal,-1.0,D.uhat);CHKERRQ(ierr); //overwrites 1st arg with sum
  ierr = VecNorm(uAnal,NORM_2,&err);
  //~ierr = VecNorm(D.uhat,NORM_2,&err);
  err = err/sqrt( (double) D.Ny*D.Nz );

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %i %e %e %.9e\n",
                     D.order,D.Ny,D.Nz,D.dy,D.dz,err);CHKERRQ(ierr);

  return ierr;
}*/

int noSlip(int argc,char **args)
{
  PetscErrorCode ierr = 0;

  Domain domain("init.txt");

  // set domain._mu differently
  PetscInt       Ii;
  PetscScalar    v,y,z;
  Vec muVec;
  PetscInt *muInds;
  PetscScalar *_muArr;
  ierr = PetscMalloc(domain._Ny*domain._Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);
  ierr = PetscMalloc(domain._Ny*domain._Nz*sizeof(PetscScalar),&_muArr);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,domain._Ny*domain._Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);

  PetscScalar r = 0;
  PetscScalar rbar = 0.25*domain._width*domain._width;
  PetscScalar rw = 1+0.5*domain._width/domain._depth;
  for (Ii=0;Ii<domain._Ny*domain._Nz;Ii++) {
    z = domain._dz*(Ii-domain._Nz*(Ii/domain._Nz));
    y = domain._dy*(Ii/domain._Nz);
    r=y*y+(0.25*domain._width*domain._width/domain._depth/domain._depth)*z*z;

    //~v = 0.5*(_muOut-_muIn)*(tanh((double)(r-rbar)/rw)+1) + _muIn;
    if (y<=10) { v = domain._muIn; }
    else { v = domain._muOut; }

    _muArr[Ii] = v;
    muInds[Ii] = Ii;
  }
  ierr = VecSetValues(muVec,domain._Ny*domain._Nz,muInds,_muArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);
  ierr = MatDiagonalSet(domain._mu,muVec,INSERT_VALUES);CHKERRQ(ierr);

  VecDestroy(&muVec);
  PetscFree(muInds);



  domain.write();
  SbpOps sbp(domain);

  // set up ksp
  KSP ksp;
  PC  pc;
  KSPCreate(PETSC_COMM_WORLD,&ksp);
  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
  ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,domain._kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n!!ksp type: HYPRE boomeramg\n\n");CHKERRQ(ierr);


  // boundary conditions
  Vec bcF,bcS,bcR,bcD;
  VecCreate(PETSC_COMM_WORLD,&bcF);
  VecSetSizes(bcF,PETSC_DECIDE,domain._Nz);
  VecSetFromOptions(bcF);     PetscObjectSetName((PetscObject) bcF, "bcF");
  VecSet(bcF,0.0);
  VecDuplicate(bcF,&bcR); PetscObjectSetName((PetscObject) bcR, "bcR");
  VecSet(bcR,1.0);

  VecCreate(PETSC_COMM_WORLD,&bcS);
  VecSetSizes(bcS,PETSC_DECIDE,domain._Ny);
  VecSetFromOptions(bcS);     PetscObjectSetName((PetscObject) bcS, "bcS");
  VecSet(bcS,0.0);
  VecDuplicate(bcS,&bcD); PetscObjectSetName((PetscObject) bcD, "bcD");
  VecSet(bcD,0.0);


  // set rhs and uhat vectors
  Vec rhs,uhat;
  VecCreate(PETSC_COMM_WORLD,&rhs);
  VecSetSizes(rhs,PETSC_DECIDE,domain._Ny*domain._Nz);
  VecSetFromOptions(rhs);
  sbp.setRhs(rhs,bcF,bcR,bcS,bcD);
  VecDuplicate(rhs,&uhat); PetscObjectSetName((PetscObject) uhat, "uhat");
  VecSet(uhat,21.0);

  ierr = KSPSolve(ksp,rhs,uhat);CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/uhat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(uhat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/rhs",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(rhs,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // compute shear stress
  Vec sigma_xy;
  VecDuplicate(rhs,&sigma_xy);
  MatMult(sbp._Dy_Iz,uhat,sigma_xy);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/sigma_xy",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(sigma_xy,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  return ierr;
}

// perform MMS in space
int mmsSpace(int argc,char **args)
{
  PetscErrorCode ierr = 0;

  Domain domain("init.txt");

  //~PetscMPIInt localRank;
  //~MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
  //~domain.view(localRank);

  SbpOps sbp(domain);

  return ierr;
}



int runEqCycle(int argc,char **args)
{
  PetscErrorCode ierr = 0;

  Domain domain("init.txt");
  domain.write();

  Lithosphere lith(domain);


  ierr = lith.writeStep();CHKERRQ(ierr);
  ierr = lith.integrate();CHKERRQ(ierr);
  ierr = lith.view();CHKERRQ(ierr);


  return ierr;
}

int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  /*
  PetscInt Ny = 401, Nz = 401;
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nz",&Nz,NULL);CHKERRQ(ierr);
  screwDislocation(Ny,Nz);
  */

  //~ierr = sbpConvergence(argc,args);CHKERRQ(ierr);// perform MMS

  // compare screw dislocation with numerics
  /*
  PetscInt Ny=721, Nz=241;
  for (Nz=241;Nz<482;Nz+=120) {
    for (Ny=241;Ny<1442;Ny+=120) {
      screwDislocation(Ny,Nz);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  */


  //~runEqCycle(argc,args);
  mmsSpace(argc,args);
  //~noSlip(argc,args);

  //~testDebugFuncs();



  PetscFinalize();

  return ierr;
}

