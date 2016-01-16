#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "genFuncs.hpp"
#include "spmat.hpp"
#include "domain.hpp"
//~#include "sbpOps.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_c.hpp"
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
  ierr = obj->writeStep1D();CHKERRQ(ierr);
  ierr = obj->writeStep2D();CHKERRQ(ierr);
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
      PetscPrintf(PETSC_COMM_WORLD,"%-3s %-10s %-10s %-22s %-10s %-22s\n",
                 "Ny","dy","err2","log2(err2)","errH","log2(errH)");
      for(PetscInt Ny=11;Ny<162;Ny=(Ny-1)*2+1)
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

        ierr = obj->writeStep1D();CHKERRQ(ierr);
        ierr = obj->writeStep2D();CHKERRQ(ierr);
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

