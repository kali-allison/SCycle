#include <petscts.h>


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Hello\n");

  PetscFinalize();
  return ierr;
}
