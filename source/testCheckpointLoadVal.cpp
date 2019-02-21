#include "genFuncs.hpp"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <petscdmda.h>
#include <petscts.h>
#include <petscviewer.h>

using namespace std;

int main(int argc, char **args) {
  PetscErrorCode ierr = 0;

  ierr = PetscInitialize(&argc, &args, NULL, NULL); CHKERRQ(ierr);

  // directory and filename
  const string outputDir = "/home/yyy910805/";
  const string filename = "_test";
  PetscScalar value = 0;

  loadValueFromCheckpoint(outputDir, filename, value);
  ierr = PetscPrintf(PETSC_COMM_WORLD, " %f\n", value); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  
  return 0;
}
