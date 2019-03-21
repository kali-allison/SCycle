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
  const string filename = "time.txt";
  const string timeFile = outputDir + filename;
  PetscScalar time = 1.5;

  PetscViewer _timeV1D;
  PetscViewerCreate(PETSC_COMM_WORLD, &_timeV1D);
  PetscViewerSetType(_timeV1D, PETSCVIEWERASCII);
  PetscViewerFileSetMode(_timeV1D, FILE_MODE_WRITE);
  PetscViewerFileSetName(_timeV1D, timeFile.c_str());
  ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n", time);CHKERRQ(ierr);
  time = 2.5;
  ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
  PetscViewerDestroy(&_timeV1D);

  // PetscViewer _timeV1D_;
  // PetscViewerCreate(PETSC_COMM_WORLD, &_timeV1D_);
  // PetscViewerSetType(_timeV1D_, PETSCVIEWERASCII);
  // PetscViewerFileSetMode(_timeV1D_, FILE_MODE_APPEND);
  // PetscViewerFileSetName(_timeV1D_, timeFile.c_str());
  // time = 2.5;
  // ierr = PetscViewerASCIIPrintf(_timeV1D_, "%.15e\n",time);CHKERRQ(ierr);
  // PetscViewerDestroy(&_timeV1D_);
  
  ierr = PetscFinalize(); CHKERRQ(ierr);
  
  return 0;
}
