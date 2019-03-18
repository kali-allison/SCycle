#include "genFuncs.hpp"

using namespace std;

int main(int argc, char **args) {
  PetscErrorCode ierr = 0;

  ierr = PetscInitialize(&argc, &args, NULL, NULL); CHKERRQ(ierr);

  // directory and filename
  const string outputDir = "/home/yyy910805/scycle/tests/";
  const string filename = "momBal_bcL_ckpt";

  Vec x;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, 1);
  VecSetFromOptions(x);
  loadVecFromInputFile(x, outputDir, filename);

  VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  
  return 0;
}

