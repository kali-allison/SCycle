#include "debuggingFuncs.hpp"

//~PetscErrorCode checkMatrix(Mat * mat,string fileLoc,string name,UserContext *D)
PetscErrorCode checkMatrix(Mat * mat,string fileLoc,string name)
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


  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing matrix %s from Matlab output:\n",matSourceFile.c_str());
  //~ierr = MatView(debugMat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Printing matrix %s created by Petsc:\n",name.c_str());
  //~ierr = MatView(*mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = MatGetSize(debugMat,&rowSizeDebugMat,&colSizeDebugMat);CHKERRQ(ierr);
  ierr = MatGetSize(*mat,&rowSizeMat,&colSizeMat);CHKERRQ(ierr);
  if ( rowSizeDebugMat == rowSizeMat && colSizeDebugMat == colSizeMat) {
    ierr = MatEqual(debugMat,*mat,&debugBool);CHKERRQ(ierr);
    if (debugBool==PETSC_FALSE) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Trying MatEqualVals on %s\n",name.c_str());CHKERRQ(ierr);
      ierr = MatEqualVals(&debugMat,mat,&debugBool);CHKERRQ(ierr);
    }
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
//~PetscErrorCode MatEqualVals(Mat * A, Mat * B,PetscBool *flg,UserContext *D)
PetscErrorCode MatEqualVals(Mat * A, Mat * B,PetscBool *flg)
{
  PetscErrorCode ierr;
  PetscScalar    tol = 1e-13;

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
      tol = pow(0.1, 13) * max(abs(constValsA[indA]), abs(constValsB[indB])); // corresponds to match up to # of sig figs = 13
    //~while (indA<ncolsA && indB<ncolsB ) {
      if ( abs(constValsA[indA])<1e-13 ) { indA++; }
      else if ( abs(constValsB[indB])<1e-13 ) { indB++; }
      //~else if ( constColsA[indA]==constColsB[indB] &&
                //~abs(constValsA[indA]-constValsB[indB])<tol*constValsA[indA] ) {
      else if ( constColsA[indA]==constColsB[indB] && abs(constValsA[indA]-constValsB[indB])<tol ) {
      //~if ( constColsA[indA]==constColsB[indB] && constValsA[indA]==constValsB[indB] ) {
        indA++;indB++;
        //~*flg=PETSC_TRUE;
      }
      else {
        *flg=PETSC_FALSE;
        ierr=PetscPrintf(PETSC_COMM_WORLD,"A(%d,%d)=%.15e,B(%d,%d)=%.15e\n",
          Ii,indA,constValsA[indA],Ii,indB,constValsB[indB]);
          indA++;indB++;
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

// prints 1d array
PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N)
{
  PetscErrorCode ierr;
#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting printMyArray in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[");CHKERRQ(ierr);
  for (Ii=0;Ii<N-1;Ii++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%f, ",myArray[Ii]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%f]\n",myArray[N-1]);CHKERRQ(ierr);

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending printMyArray in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// prints 1d array representing 2d matrix
PetscErrorCode printMy2DArray(PetscScalar *myArray, PetscInt Nrow, PetscInt Ncol)
{
  PetscErrorCode ierr;
#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting printMy2DArray in debuggingFuncs.c\n\n\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Jj;

  for (Ii=0;Ii<Nrow;Ii++)
  {
    for (Jj=0;Jj<Ncol-1;Jj++)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%f ",myArray[Ii*Nrow+Jj]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%f\n",myArray[Ii*Nrow+Jj]);CHKERRQ(ierr);
  }

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending printMy2DArray in debuggingFuncs.c\n\n\n");CHKERRQ(ierr);
#endif
  return ierr;
}




PetscErrorCode testDebugFuncs()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting testDebugFuncs in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif
  PetscInt nRows=3,nCols=4;
  PetscInt Ii,Istart,Iend,Jj,Jstart,Jend;
  PetscScalar v;
  PetscBool flag;

  Mat A;
  MatCreate(PETSC_COMM_WORLD,&A); PetscObjectSetName((PetscObject) A, "A");
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,nRows,nCols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A,&Jstart,&Jend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    for (Jj=Jstart;Jj<Jend;Jj++) {
      v = (Ii+1)*(Jj+1);
      ierr = MatSetValues(A,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  Mat B;
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr); PetscObjectSetName((PetscObject) B, "B");

  // checking that equal matrices are found to be equal
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Equal Matrices are evaluated as equal\n");CHKERRQ(ierr);
  ierr = MatEqualVals(&A,&B,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting true, got: %i\n",flag);CHKERRQ(ierr);

  // checking that matrices w/ differing first entries are found unequal
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Entry (0,0) differs\n");CHKERRQ(ierr);
  Ii = 0; Jj = 0; v = 5;
  ierr = MatSetValues(B,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatEqualVals(&A,&B,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting false, got: %i\n",flag);CHKERRQ(ierr);


  // checking that matrices w/ additional differing entries are found to be unequal
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Entry (1,2) differs\n");CHKERRQ(ierr);
  Ii = 0; Jj = 0; v = 1;
  ierr = MatSetValues(B,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  Ii = 1; Jj = 2; v = 5;
  ierr = MatSetValues(B,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatEqualVals(&A,&B,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting false, got: %i\n",flag);CHKERRQ(ierr);


  // checking that entries equal to 0 are viewed the same as nonexistant entries
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n\nNew Matrices C and D defined:\n\n\n");CHKERRQ(ierr);
  Ii = 1; Jj = 2; v = 6;
  ierr = MatSetValues(B,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  Ii = 0; Jj = 0; v = -1;
  ierr = MatSetValues(B,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  Mat C;
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&C);CHKERRQ(ierr); PetscObjectSetName((PetscObject) C, "C");
  ierr = MatAXPY(C,1.0,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  Mat D;
  MatCreate(PETSC_COMM_WORLD,&D); PetscObjectSetName((PetscObject) D, "D");
  ierr = MatSetSizes(D,PETSC_DECIDE,PETSC_DECIDE,nRows,nCols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D,&Istart,&Iend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(D,&Jstart,&Jend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    for (Jj=Jstart;Jj<Jend;Jj++) {
      if (Ii==0 && Jj==0) {}
    else {
      v = 2*(Ii+1)*(Jj+1);
      ierr = MatSetValues(D,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr); }
    }
  }
  ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Entry (0,0)=0 vs does not exist\n");CHKERRQ(ierr);
  ierr = MatEqualVals(&C,&D,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting true, got: %i\n",flag);CHKERRQ(ierr);


  // checking that the tolerance in MatEqualVals works the way I think it does
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Tolerance Check on entry (1,2), diff < tol\n");CHKERRQ(ierr);
  Ii = 1; Jj = 2; v = 12 + 1e-14;
  ierr = MatSetValues(D,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatEqualVals(&C,&D,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting true, got: %i\n",flag);CHKERRQ(ierr);


  // checking that the tolerance in MatEqualVals works the way I think it does
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTEST: Tolerance Check on entry (1,2), diff > tol (barely)\n");CHKERRQ(ierr);
  Ii = 1; Jj = 2; v = 12 - 1e-10;
  ierr = MatSetValues(D,1,&Ii,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatEqualVals(&C,&D,&flag);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"expecting false, got: %i\n",flag);CHKERRQ(ierr);





#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending testDebugFuncs in debuggingFuncs.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}

