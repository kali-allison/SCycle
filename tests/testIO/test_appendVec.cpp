#include "genFuncs.hpp"

using namespace std;

int main(int argc, char **argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
  PetscErrorCode ierr = 0;

  Vec x;
  PetscInt n = 1;
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, n);
  VecSetFromOptions(x);
  VecSet(x,1.0);

  string key = "test";
  string filename = "/home/yyy910805/scycle/tests/test_appendVec";
  map<string, pair<PetscViewer,string>> vwL;
  
  // initiate doesn't do any writing
  //  io_initiateWriteAppend(vwL, key, x, filename);
  //VecView(x, vwL[key].first);
  //VecView(x, vwL[key].first);

  // now should have appended another 3 vectors
  initiate_appendVecToOutput(vwL, key, x, filename);
  VecView(x, vwL[key].first);
  VecView(x, vwL[key].first);
  VecView(x, vwL[key].first);
    
  VecDestroy(&x);
  vwL.clear();
  
  PetscFinalize();
  return ierr;
}
