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
  domain.write();


  //~SbpOps sbp(domain,*domain._muArrPlus,domain._muP);
  //~MatView(domain._muP,PETSC_VIEWER_STDOUT_WORLD);
  //~MatView(sbp._muxDy_Iz,PETSC_VIEWER_STDOUT_WORLD);

  SymmFault fault(domain);
  fault.writeContext(domain._outputDir);
  //~SymmLinearElastic lith(domain);



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
  ierr = obj->writeStep();CHKERRQ(ierr);
  ierr = obj->integrate();CHKERRQ(ierr);
  ierr = obj->view();CHKERRQ(ierr);

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
    if (!domain._shearDistribution.compare("mms"))
    {
      for(PetscInt Ny=11;Ny<642;Ny=(Ny-1)*2+1)
      {
        Domain domain(inputFile,Ny,Ny);
        domain.write();
        SymmMaxwellViscoelastic *obj;
        obj = new SymmMaxwellViscoelastic(domain);

        ierr = obj->writeStep();CHKERRQ(ierr);
        ierr = obj->integrate();CHKERRQ(ierr);
        ierr = obj->measureMMSError();CHKERRQ(ierr);
      }
    }
    else
    {
      runEqCycle(inputFile);
    }
  }


  //~runTests(inputFile);

  PetscFinalize();
  return ierr;
}

