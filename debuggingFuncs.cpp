#include <petscts.h>
#include <string>
#include "odeSolver.h"
#include "userContext.h"
#include "debuggingFuncs.hpp"

using namespace std;

PetscErrorCode checkMatrix(Mat * mat,string fileLoc,string name,UserContext *D)
{
  PetscErrorCode ierr = 0;
  Mat            debugMat;
  PetscViewer    inviewer;
  PetscBool      debugBool = PETSC_FALSE;
  PetscInt       rowSizeDebugMat,colSizeDebugMat,rowSizeMat,colSizeMat;

  string matSourceFile = fileLoc + name;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inviewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,matSourceFile.c_str(),FILE_MODE_READ,&inviewer);
  ierr = PetscViewerSetFormat(inviewer,PETSC_VIEWER_BINARY_MATLAB);

  ierr = MatCreate(PETSC_COMM_WORLD,&debugMat);CHKERRQ(ierr);
  ierr = MatLoad(debugMat,inviewer);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(debugMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(debugMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetSize(debugMat,&rowSizeDebugMat,&colSizeDebugMat);CHKERRQ(ierr);
  ierr = MatGetSize(*mat,&rowSizeMat,&colSizeMat);CHKERRQ(ierr);
  if ( rowSizeDebugMat == rowSizeMat && colSizeDebugMat == colSizeMat) {
    ierr = MatEqual(debugMat,*mat,&debugBool);CHKERRQ(ierr);
    //~if (debugBool==PETSC_FALSE) {
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Trying MatEqualVals on %s\n",name.c_str());CHKERRQ(ierr);
      //~ierr = MatEqualVals(&debugMat,mat,&debugBool,D);CHKERRQ(ierr);
    //~}
  }
  else{
    ierr = PetscPrintf(PETSC_COMM_WORLD,"   wrong size!!!\n");CHKERRQ(ierr);
  }

#if VERBOSE == 1
  if (debugBool==PETSC_FALSE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Checking %s, and debugBool is %d.\n",matSourceFile.c_str(),debugBool);CHKERRQ(ierr);
  }
#elif VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Checking %s, and debugBool is %d.\n",matSourceFile.c_str(),debugBool);CHKERRQ(ierr);
#endif

#if VERBOSE > 2
  if (debugBool==PETSC_FALSE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing matrix %s from Matlab output:\n",matSourceFile.c_str());
    ierr = MatView(debugMat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing matrix %s created by Petsc:\n",name.c_str());
    ierr = MatView(*mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
#endif

  //~ ierr = MatDestroy(&debugMat);CHKERRQ(ierr);
  //~ free(matSourceFile);
  //  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  return 0;
}

/*
 * Determines if two matrices have equal contents (even if the
 * preallocated structure is different).
 */
PetscErrorCode MatEqualVals(Mat * A, Mat * B,PetscBool *flg,UserContext *D)
{
  PetscErrorCode ierr;
  PetscInt       rowSizeA,colSizeA,rowSizeB,colSizeB;
  PetscInt       Ii,Istart,Iend,indA,indB,ncolsA,ncolsB;
  *flg = PETSC_TRUE;
  PetscInt const *constColsA;
  PetscScalar const *constValsA;
  PetscInt const *constColsB;
  PetscScalar const *constValsB;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting MatEqualVals in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif

  ierr = MatGetSize(*A,&rowSizeA,&colSizeA);CHKERRQ(ierr);
  ierr = MatGetSize(*B,&rowSizeB,&colSizeB);CHKERRQ(ierr);
  if ( rowSizeA != rowSizeB && colSizeA != colSizeB) {
    *flg=PETSC_FALSE;
    return ierr;
  }

  ierr = MatGetOwnershipRange(*A,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {

    ierr = MatGetRow(*A,Ii,&ncolsA,&constColsA,&constValsA);CHKERRQ(ierr);
    ierr = MatGetRow(*B,Ii,&ncolsB,&constColsB,&constValsB);CHKERRQ(ierr);

    indA=0;indB=0;
    while (indA<ncolsA && indB<ncolsB && *flg!=PETSC_FALSE) {
      if ( abs(constValsA[indA])<1e-19 ) { indA++; }
      else if ( abs(constValsB[indB])<1e-19 ) { indB++; }
      else if ( constColsA[indA]==constColsB[indB] && abs(constValsA[indA]-constValsB[indB])<1e-19 ) {
      //~if ( constColsA[indA]==constColsB[indB] && constValsA[indA]==constValsB[indB] ) {
        indA++;indB++;
        *flg=PETSC_TRUE;
      }
      else {
        *flg=PETSC_FALSE;
        ierr=PetscPrintf(PETSC_COMM_WORLD,"A(%d,%d)=%g,B(%d,%d)=%g\n",
          Ii,indA,constValsA[indA],Ii,indB,constValsB[indB]);
        }
    }
    ierr = MatRestoreRow(*A,Ii,&ncolsA,&constColsA,&constValsA);CHKERRQ(ierr);
    ierr = MatRestoreRow(*B,Ii,&ncolsB,&constColsB,&constValsB);CHKERRQ(ierr);
  }
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending MatEqualVals in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N)
{
  PetscErrorCode ierr;
  PetscInt       Ii;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[");CHKERRQ(ierr);
  for (Ii=0;Ii<N-1;Ii++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ",myArray[Ii]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%f]\n",myArray[N-1]);CHKERRQ(ierr);
  return 0;
}

