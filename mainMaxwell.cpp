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
#include "powerLaw.hpp"



using namespace std;


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  //~Domain domain(inputFile,5,4);
  d.write();


  //~SbpOps sbp(domain,*domain._muArrPlus,domain._muP);
  //~MatView(domain._muP,PETSC_VIEWER_STDOUT_WORLD);

  //~SymmFault fault(domain);
  //~fault.writeContext(domain._outputDir);
  //~ SymmLinearElastic lith(domain);
  //~ lith.writeStep1D();
  //~ lith.writeStep2D();
  //~ lith.integrate();
  //~ lith.view();

  SymmMaxwellViscoelastic max(d);
  max.writeStep1D();
  max.writeStep2D();
  max.integrate();


  return ierr;
}

int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-3s %-10s %-10s %-22s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","errL2u","log2(errL2u)","errL2gxy","log2(errL2gxy)",
             "errL2gxz","log2(errL2gxz)");
  //~ for(PetscInt Ny=11;Ny<12;Ny=(Ny-1)*2+1)
  for(PetscInt Ny=51;Ny<202;Ny=(Ny-1)*2+1)
  {
    Domain domain(inputFile,Ny,Ny);
    //~ Domain domain(inputFile,Ny,1);
    domain.write();
    SymmMaxwellViscoelastic *obj;
    obj = new SymmMaxwellViscoelastic(domain);

    //~ ierr = obj->writeStep1D();CHKERRQ(ierr);
    //~ ierr = obj->writeStep2D();CHKERRQ(ierr);
    ierr = obj->integrate();CHKERRQ(ierr);
    ierr = obj->measureMMSError();CHKERRQ(ierr);
  }

  return ierr;
}


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();

  SymmMaxwellViscoelastic *obj;
  obj = new SymmMaxwellViscoelastic(domain);

  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  ierr = obj->integrate();CHKERRQ(ierr);
  //~ ierr = obj->view();CHKERRQ(ierr);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "init.txt"; }

  {
    Domain domain(inputFile);
    if (!domain._shearDistribution.compare("mms")) { runMMSTests(inputFile); }
    else { runEqCycle(inputFile); }
  }

  //~ runTests(inputFile);

  PetscFinalize();
  return ierr;
}

