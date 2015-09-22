#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "genFuncs.hpp"
#include "spmat.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "fault.hpp"
#include "linearElastic.hpp"
#include "maxwellViscoelastic.hpp"





using namespace std;


int rewriteMatlabIO()
{
  PetscErrorCode ierr = 0;
  PetscScalar v = 0;
  Vec            vec;

  VecCreate(PETSC_COMM_WORLD,&vec);
  VecSetSizes(vec,PETSC_DECIDE,6);
  VecSetFromOptions(vec);
  PetscObjectSetName((PetscObject) vec, "vec");
  VecSet(vec,1.0);

  PetscInt Ii,Istart,Iend;

  VecGetOwnershipRange(vec,&Istart,&Iend);
  for(Ii=Istart;Ii<Iend;Ii++)
  {
    v = Ii + 1;
    VecSetValue(vec,Ii,v,INSERT_VALUES);
  }
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);

  // set up viewer
  PetscViewer vw;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vec",FILE_MODE_WRITE,&vw);
  VecView(vec,vw);
  PetscViewerDestroy(&vw);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vec",FILE_MODE_APPEND,&vw);

  for (int i=2;i<8;i++)
  {
    //~v = (double) i;
    VecScale(vec,2.0);
    VecView(vec,vw);
  }


  VecDestroy(&vec);
  PetscViewerDestroy(&vw);


  return ierr;
}



int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  //~Domain domain(inputFile,5,4);
  //~domain.write();


  //~SbpOps sbp(domain,*domain._muArrPlus,domain._muP);
  //~MatView(domain._muP,PETSC_VIEWER_STDOUT_WORLD);
  //~MatView(sbp._muxDy_Iz,PETSC_VIEWER_STDOUT_WORLD);

  //~SymmFault fault(domain);
  SymmLinearElastic lith(domain);



  return ierr;
}



/* This has not been rewritten since the post-quals refactoring
int screwDislocation(PetscInt Ny,PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       order=4;
  PetscBool      loadMat = PETSC_FALSE;
  PetscViewer    viewer;
  PetscScalar    u,z;


  // set up the problem context
  IntegratorContext D(order,Ny,Nz,"data/");
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
*/


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();
  //~SymmMaxwellViscoelastic *lith;
  //~lith = new SymmMaxwellViscoelastic(domain);

  //~LinearElastic *lith;
  //~if (domain._problemType.compare("symmetric")==0) {
    //~lith = new SymmLinearElastic(domain);
  //~}
  //~else {
    //~lith = new FullLinearElastic(domain);
  //~}

  //~PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  //~ierr = lith->writeStep();CHKERRQ(ierr);
  //~ierr = lith->integrate();CHKERRQ(ierr);
  //~ierr = lith->view();CHKERRQ(ierr);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "init.txt"; }

  runEqCycle(inputFile);

  //~const char* inputFile2;
  //~if (argc > 2) {inputFile2 = args[2]; }
  //~else { inputFile2 = inputFile; }
  //~coupledSpringSliders(inputFile, inputFile2);


  //~runTests(inputFile);
  //~testDMDA_ScatterToVec();
  //~testDMDA();
  //~testMatShell();

  //~// MMS test (compare with answers produced by Matlab file by same name)
  //~PetscPrintf(PETSC_COMM_WORLD,"MMS:\n%5s %5s %5s %20s %20s\n",
             //~"order","Ny","Nz","log2(||u-u^||)","log2(||tau-tau^||)");
  //~PetscInt Ny=21;
  //~for (Ny=21;Ny<82;Ny=(Ny-1)*2+1)
  //~{
    //~mmsSpace(inputFile,Ny,Ny); // perform MMS
  //~}
//~
  // check for critical grid point spacing
  //~PetscInt Ny=251; // crit for order=2 is 417
  //~for (Ny=51;Ny<1002;Ny+=50)
  //~{
    //~PetscPrintf(PETSC_COMM_WORLD,"Ny=%i\n",Ny);
    //~critSpacing(inputFile,Ny,Ny);
  //~}



  PetscFinalize();
  return ierr;
}

