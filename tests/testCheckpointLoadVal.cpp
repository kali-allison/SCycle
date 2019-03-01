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
  PetscScalar value = 50.5;

  const string checkpointFile = outputDir + filename;
  PetscViewer viewer1, viewer2, viewer3;
  int fd;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, checkpointFile.c_str(), FILE_MODE_WRITE, &viewer1);
  PetscViewerBinaryGetDescriptor(viewer1, &fd);
  PetscBinaryWrite(fd, &value, 1, PETSC_SCALAR, PETSC_FALSE);
  PetscViewerDestroy(&viewer1);

  value = 100.5;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, checkpointFile.c_str(), FILE_MODE_WRITE, &viewer3);
  PetscViewerBinaryGetDescriptor(viewer3, &fd);
  PetscBinaryWrite(fd, &value, 1, PETSC_SCALAR, PETSC_FALSE);
  PetscViewerDestroy(&viewer3);
  
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, checkpointFile.c_str(), FILE_MODE_READ, &viewer2);
  PetscViewerBinaryGetDescriptor(viewer2, &fd);
  PetscBinaryRead(fd, &value, 1, PETSC_SCALAR);
  PetscViewerDestroy(&viewer2);
  
  ierr = PetscPrintf(PETSC_COMM_WORLD, " %f\n", value); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  
  return 0;
}

