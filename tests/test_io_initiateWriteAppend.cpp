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

// check if file exists
bool doesFileExist(const string fileName)
{
  std::ifstream infile(fileName.c_str());
  return infile.good();
};


// loads a PETSc Vec from a binary file
// Note: memory for out MUST be allocated before calling this function
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName, bool& fileExists)
{
  PetscErrorCode ierr = 0;
    #if VERBOSE > 1
  string funcName = "loadFieldsFromFiles";
  string fileName = "genFuncs.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"  Attempting to load: %s%s\n",inputDir.c_str(),fieldName.c_str());CHKERRQ(ierr);
    #endif

  string vecSourceFile = inputDir + fieldName;

  fileExists = doesFileExist(vecSourceFile);
  if (fileExists) {
    PetscPrintf(PETSC_COMM_WORLD,"Note: Loading Vec from file: %s\n",vecSourceFile.c_str());
    PetscViewer inv;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecLoad(out,inv);CHKERRQ(ierr);
    PetscViewerPopFormat(inv);
    PetscViewerDestroy(&inv);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: File not found: %s\n",vecSourceFile.c_str());
  }

    #if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
    #endif
  return ierr;
}

// loads a PETSc Vec from a binary file
// Note: memory for out MUST be allocated before calling this function
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName)
{
  PetscErrorCode ierr = 0;
  bool fileExists = 0;
  ierr = loadVecFromInputFile(out,inputDir,fieldName,fileExists); CHKERRQ(ierr);
  return ierr;
}


// append vector into binary file
PetscErrorCode io_initiateWriteAppend(map<string, pair<PetscViewer,string>>& vwL, const string key, const Vec& vec, const string file)
{
  PetscErrorCode ierr = 0;

  // initiate viewer
  PetscViewer vw;
  vwL[key].first = vw;
  vwL[key].second = file;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file.c_str(), FILE_MODE_WRITE, &vw); CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(vw, PETSC_VIEWER_BINARY_MATLAB); CHKERRQ(ierr);
  ierr = VecView(vec, vw); CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(vw); CHKERRQ(ierr);
  PetscViewerDestroy(&vw);

  // open in append mode after we finish initializing, so future VecView calls will directly append to this file
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file.c_str(), FILE_MODE_APPEND, &vwL[key].first); CHKERRQ(ierr);

  return ierr;
}


int main(int argc, char **args) {
  PetscErrorCode ierr = 0;

  ierr = PetscInitialize(&argc, &args, NULL, NULL); CHKERRQ(ierr);
  
  // directory and filename
  const string outputDir = "/home/yyy910805/";
  const string filename = "_test";
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
  VecCreate(PETSC_COMM_WORLD, &y);
  VecSetSizes(y, PETSC_DECIDE, 3);
  VecSetFromOptions(y);
  
  loadVecFromInputFile(y, outputDir, filename);
  VecView(y, PETSC_VIEWER_STDOUT_SELF);

  VecDestroy(&x);
  VecDestroy(&y);

  // destroy viewers
  map<string, pair<PetscViewer, string>>::iterator it;
  for (it = vwL.begin(); it != vwL.end(); it++) {
    PetscViewerDestroy(&(vwL[it->first].first));
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);
  
  return 0;
}
