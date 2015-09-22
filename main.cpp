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


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();
  SymmMaxwellViscoelastic *lith;
  lith = new SymmMaxwellViscoelastic(domain);

  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  ierr = lith->writeStep();CHKERRQ(ierr);
  ierr = lith->integrate();CHKERRQ(ierr);
  ierr = lith->view();CHKERRQ(ierr);
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


  //~runTests(inputFile);

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

