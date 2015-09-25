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


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "test.in"; }

  {
    for(PetscInt Ny=21;Ny<322;Ny=(Ny-1)*2+1) {
      Domain domain(inputFile,Ny,Ny);
      domain.write();

      LinearElastic *lith;
      if (domain._problemType.compare("symmetric")==0) {
        lith = new SymmLinearElastic(domain);
      }
      else {
        lith = new FullLinearElastic(domain);
      }

      //~PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
      ierr = lith->writeStep();CHKERRQ(ierr);
      ierr = lith->integrate();CHKERRQ(ierr);
      //~ierr = lith->view();CHKERRQ(ierr);

      lith->measureMMSError();

    }
  }


  PetscFinalize();
  return ierr;
}

