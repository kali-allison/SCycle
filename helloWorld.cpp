#include <petscts.h>


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;


  PetscFinalize();
  return ierr;
}
