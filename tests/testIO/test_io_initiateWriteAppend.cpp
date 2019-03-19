#include "genFuncs.hpp"

using namespace std;


int main(int argc, char **args) {
  PetscErrorCode ierr = 0;

  ierr = PetscInitialize(&argc, &args, NULL, NULL); CHKERRQ(ierr);
  
  // directory and filename
  const string outputDir = "/home/yyy910805/scycle/tests/testIO/";
  const string filename = "test_io";

  Vec x;
  PetscScalar alpha = 1;
  PetscInt n = 3;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, n);
  VecSetFromOptions(x);
  VecSet(x, alpha);

  // viewer map
  map<string, pair<PetscViewer, string>> vwL;
  string file = outputDir + filename;
  io_initiateWriteAppend(vwL, filename, x, file);

  // read vector from input file
  Vec y;
  VecDuplicate(x, &y);
  VecSet(y, 0.0);
  
  loadVecFromInputFile(y, outputDir, filename);
  printf("Value of y from io_initiateWriteAppend:\n");
  VecView(y, PETSC_VIEWER_STDOUT_SELF);

  // destroy viewer
  PetscViewerDestroy(&vwL[filename].first);

  //===========================================================================
  // test appendVecToOutput
  PetscScalar beta = 2;
  VecSet(x, beta);  
  initiate_appendVecToOutput(vwL, filename, x, file);

  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, file.c_str(), FILE_MODE_READ, &viewer);
  VecLoad(y, viewer);
  printf("First value of y from initiate_appendVecToOutput:\n");
  VecView(y, PETSC_VIEWER_STDOUT_SELF);
  printf("Second value of y from initiate_appendVecToOutput:\n");
  VecLoad(y, viewer);
  VecView(y, PETSC_VIEWER_STDOUT_SELF);

  PetscViewerDestroy(&viewer);
  
  VecDestroy(&x);
  VecDestroy(&y);

  PetscFinalize();
  
  return 0;
}
