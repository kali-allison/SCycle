#include "genFuncs.hpp"

#include <petscvec.h>
#include <iostream>
#include <map>

using namespace std;

int main(int argc, char **argv) {
  Vec x, y;
  PetscInt n = 2.0;
  string outputDir = "/home/yyy910805/scycle/tests/";
  map<string, Vec> M;
  PetscErrorCode ierr = 0;
  
  PetscInitialize(&argc, &argv, NULL, NULL);
  ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
  ierr = VecSet(x, 1.0); CHKERRQ(ierr);

  ierr = VecDuplicate(x, &y); CHKERRQ(ierr);
  ierr = VecSet(y, 2.0); CHKERRQ(ierr);
  
  M["test"] = x;

  ierr = writeVec(y, outputDir + "VecTestMap"); CHKERRQ(ierr);
  
  ierr = loadVecFromInputFile(M["test"], outputDir, "VecTestMap"); CHKERRQ(ierr);

  ierr = VecView(M["test"], PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&x);
  VecDestroy(&y);
  M.clear();
  cout << M.size() << "\n";
  
  PetscFinalize();

  return ierr;
}
