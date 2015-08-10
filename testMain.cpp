#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "testOdeSolver.hpp"
#include "odeSolver.hpp"
#include "testOdeSolver.hpp"

int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  TestOdeSolver *trial;
  trial = new TestOdeSolver();

  trial->writeStep();

  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  trial->integrate();


  PetscFinalize();
  return ierr;
}
