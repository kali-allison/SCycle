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



using namespace std;

int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();

  LinearElastic *obj;
  if (domain._problemType.compare("symmetric")==0) {
    obj = new SymmLinearElastic(domain);
  }
  else {
    obj = new FullLinearElastic(domain);
  }

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
  else { inputFile = "test.in"; }

  {
    Domain domain(inputFile);
    if (!domain._shearDistribution.compare("mms"))
    {
      for(PetscInt Ny=21;Ny<322;Ny=(Ny-1)*2+1)
      {
        Domain domain(inputFile,Ny,Ny);
        domain.write();

        LinearElastic *obj;
        if (domain._problemType.compare("symmetric")==0) {
          obj = new SymmLinearElastic(domain);
        }
        else {
          obj = new FullLinearElastic(domain);
        }

        ierr = obj->writeStep();CHKERRQ(ierr);
        ierr = obj->integrate();CHKERRQ(ierr);

        obj->measureMMSError();
      }
    }
    else
    {
      runEqCycle(inputFile);
    }
  }


  PetscFinalize();
  return ierr;
}

