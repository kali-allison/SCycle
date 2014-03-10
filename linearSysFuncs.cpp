#include <petscts.h>
#include <string>
#include "userContext.h"
//~ #include "debuggingFuncs.h"
#include "linearSysFuncs.h"


PetscErrorCode setLinearSystem(UserContext *D, const PetscBool loadMat)
{
  PetscErrorCode ierr = 0;

  // SBP operators and penalty terms
  D->alphaF = -13.0/D->dy;
  D->alphaR = -13.0/D->dy;
  D->alphaS = -1.0;
  D->alphaD = -1.0;
  D->beta   = 1.0;

  // set boundary data to match constant tectonic plate motion
  ierr = VecSet(D->gF,0.0);CHKERRQ(ierr);
  ierr = VecSet(D->gS,0.0);CHKERRQ(ierr);
  ierr = VecSet(D->gD,0.0);CHKERRQ(ierr);
  ierr = VecSet(D->gR,D->vp*D->initTime/2.0);CHKERRQ(ierr);

  if (loadMat) { ierr = loadOperators(D);CHKERRQ(ierr); }
  else { ierr = createOperators(D);CHKERRQ(ierr);}

  // use direct solve (LU)
  KSPSetType(D->ksp,KSPPREONLY);
  KSPSetOperators(D->ksp,D->A,D->A,SAME_PRECONDITIONER);
  KSPGetPC(D->ksp,&D->pc);
  PCSetType(D->pc,PCLU);

  // use GMRES with preconditioning
  //~KSPSetType(D->ksp,KSPGMRES);
  //~KSPGMRESSetRestart(D->ksp,100);
  //~KSPSetOperators(D->ksp,D->A,D->A,SAME_PRECONDITIONER);
  //~KSPGetPC(D->ksp,&D->pc);
  //~PCSetType(D->pc,PCHYPRE);
  //~PCHYPRESetType(D->pc,"boomeramg");
  //~KSPSetTolerances(D->ksp,D->kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  //~PCFactorSetLevels(D->pc,4);

  KSPSetUp(D->ksp);
  KSPSetFromOptions(D->ksp);
  ierr = ComputeRHS(D);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode ComputeRHS(UserContext *D)
{
  PetscErrorCode ierr = 0;

  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function ComputeRHS in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  /* rhs =  D.alphaF*p.G*D.Hinvy_Iz_e0y_Iz +... */
  ierr = MatMult(D->Hinvy_Iz_e0y_Iz,D->gF,D->rhs);CHKERRQ(ierr);
  ierr = VecScale(D->rhs,D->alphaF*D->G);CHKERRQ(ierr);

  /* + D.beta*D.G*D.Hinvy_Iz_BySy_Iz_e0y_Iz*gF + ... */
  Vec temp;
  ierr = VecDuplicate(D->rhs,&temp);CHKERRQ(ierr);
  ierr = MatMult(D->Hinvy_Iz_BySy_Iz_e0y_Iz,D->gF,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D->rhs,D->beta*D->G,temp);CHKERRQ(ierr);

  /* + D.alphaR*D.G*D.Hinvy_Iz_eNy_Iz*gR + ... */
  ierr = MatMult(D->Hinvy_Iz_eNy_Iz,D->gR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D->rhs,D->alphaR*D->G,temp);CHKERRQ(ierr);

  /* + D.beta*D.G*D.Hinvy_Iz_BySy_Iz_eNy_Iz*gR + ... */
  ierr = MatMult(D->Hinvy_Iz_BySy_Iz_eNy_Iz,D->gR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D->rhs,D->beta*D->G,temp);CHKERRQ(ierr);

  //~ /* - D.alphaS*M.IyHinvz_Iye0z*gS + ... */
  //~ ierr = MatMult(D->IyHinvz_Iye0z,D->gS,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(D->rhs,D->alphaS,temp);CHKERRQ(ierr);
//~
  //~ /* + D.alphaD*M.IyHinvz_IyeNz*gD */
  //~ ierr = MatMult(D->IyHinvz_IyeNz,D->gD,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(D->rhs,D->alphaD,temp);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function ComputeRHS in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  double endTime = MPI_Wtime();
  D->computeRhsTime = D->computeRhsTime + (endTime-startTime);

  return ierr;
}


PetscErrorCode loadOperators(UserContext *D)
{
  PetscErrorCode  ierr;
  PetscViewer     fd;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function loadOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  int size;
  MatType matType;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  if (size > 1) {matType = MATMPIAIJ;}
  else {matType = MATSEQAIJ;}

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"A",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Dy_Iz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->Dy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(D->Dy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->Dy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Iz_e0y_Iz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->Hinvy_Iz_e0y_Iz);CHKERRQ(ierr);
  ierr = MatSetType(D->Hinvy_Iz_e0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->Hinvy_Iz_e0y_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Iz_BySy_Iz_e0y_Iz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->Hinvy_Iz_BySy_Iz_e0y_Iz);CHKERRQ(ierr);
  ierr = MatSetType(D->Hinvy_Iz_BySy_Iz_e0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->Hinvy_Iz_BySy_Iz_e0y_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Iz_eNy_Iz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->Hinvy_Iz_eNy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(D->Hinvy_Iz_eNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->Hinvy_Iz_eNy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Iz_BySy_Iz_eNy_Iz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->Hinvy_Iz_BySy_Iz_eNy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(D->Hinvy_Iz_BySy_Iz_eNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->Hinvy_Iz_BySy_Iz_eNy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"IyHinvz_Iye0z",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->IyHinvz_Iye0z);CHKERRQ(ierr);
  ierr = MatSetType(D->IyHinvz_Iye0z,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->IyHinvz_Iye0z,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"IyHinvz_IyeNz",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&D->IyHinvz_IyeNz);CHKERRQ(ierr);
  ierr = MatSetType(D->IyHinvz_IyeNz,matType);CHKERRQ(ierr);
  ierr = MatLoad(D->IyHinvz_IyeNz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function loadOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

    return ierr;
}

/*
 * Creates the matrices in struct D
 */
PetscErrorCode createOperators(UserContext *D)
{

  PetscErrorCode  ierr = 0;
  PetscScalar     v;
  PetscInt        Ii,J,Istart,Iend,ncols,indx,*cols,Jj;
  Mat             D2y,Sy,D2z,Sz,temp,Dy,Dz;

  PetscInt const *constCols;
  PetscScalar const *constVals;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function createOperators in linearSysFuncs.c.\n");
  CHKERRQ(ierr);
#endif

  PetscMalloc(D->Nz*sizeof(PetscInt),&cols);

  ierr = SBPoperators(D->order,D->Ny-1,&D->Hinvy,&Dy,&D2y,&Sy);CHKERRQ(ierr);
  ierr = SBPoperators(D->order,D->Nz-1,&D->Hinvz,&Dz,&D2z,&Sz);CHKERRQ(ierr);

#if DEBUG > 1
  ierr = checkMatrix(&Dy,D->debugFolder,"Dy",D);CHKERRQ(ierr);
  ierr = checkMatrix(&D->Hinvy,D->debugFolder,"Hinvy",D);CHKERRQ(ierr);
  ierr = checkMatrix(&D->Hinvz,D->debugFolder,"Hinvz",D);CHKERRQ(ierr);
  ierr = checkMatrix(&D2y,D->debugFolder,"D2y",D);CHKERRQ(ierr);
  ierr = checkMatrix(&Sy,D->debugFolder,"Sy",D);CHKERRQ(ierr);
  ierr = checkMatrix(&Dz,D->debugFolder,"Dz",D);CHKERRQ(ierr);
  ierr = checkMatrix(&D2z,D->debugFolder,"D2z",D);CHKERRQ(ierr);
  ierr = checkMatrix(&Sz,D->debugFolder,"Sz",D);CHKERRQ(ierr);
#endif

  /* Scaling (why not do this while initializing each matrix?) */
  D->D2y = D2y; /* note that further changes to D2y will also change D.D2y */
  D->D2z = D2z;
  ierr = MatScale(D2y,1.0/D->dy/D->dy);CHKERRQ(ierr);
  ierr = MatScale(D2z,1.0/D->dz/D->dz);CHKERRQ(ierr);
  ierr = MatScale(Sy,1.0/D->dy);CHKERRQ(ierr);
  ierr = MatScale(Sz,1.0/D->dz);CHKERRQ(ierr);
  ierr = MatScale(Dy,1.0/D->dy);CHKERRQ(ierr);
  ierr = MatScale(Dz,1.0/D->dz);CHKERRQ(ierr);
  ierr = MatScale(D->Hinvy,1.0/D->dy);CHKERRQ(ierr);
  ierr = MatScale(D->Hinvz,1.0/D->dz);CHKERRQ(ierr);


  /* G*kron(Dy,Iz) */
  //Mat Dy_Iz;
  //ierr = MatCreate(PETSC_COMM_WORLD,&Dy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(D->Dy_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D->Dy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D->Dy_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D->Dy_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D->Dy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Dy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(Dy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=0;J<ncols;J++) { cols[J]=constCols[J]*D->Nz;}
    for (J=Ii*D->Nz;J<(Ii+1)*D->Nz;J++) {
      ierr = MatSetValues(D->Dy_Iz,1,&J,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
      for (Jj=0;Jj<ncols;Jj++) { cols[Jj]=cols[Jj]+1;}
    }
    ierr = MatRestoreRow(Dy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D->Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D->Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->Dy_Iz,D->debugFolder,"Dy_Iz",D);CHKERRQ(ierr);
#endif
ierr = MatScale(D->Dy_Iz,D->G);CHKERRQ(ierr);

  /* kron(D2y,Iz) */
  Mat D2y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&D2y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(D2y_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D2y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D2y_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D2y_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D2y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2y,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D2y,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=0;J<ncols;J++) { cols[J]=constCols[J]*D->Nz;}
    for (J=Ii*D->Nz;J<(Ii+1)*D->Nz;J++) {
      ierr = MatSetValues(D2y_Iz,1,&J,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
      for (Jj=0;Jj<ncols;Jj++) { cols[Jj]=cols[Jj]+1;}
    }
    ierr = MatRestoreRow(D2y,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D2y_Iz,D->debugFolder,"D2y_Iz",D);CHKERRQ(ierr);
#endif

  /* kron(Iy,D2z) */
  Mat Iy_D2z;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_D2z);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_D2z,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_D2z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_D2z,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_D2z,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_D2z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2z,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D2z,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (Jj=Ii;Jj<D->Ny*D->Nz;Jj=Jj+D->Nz) {
      for (J=0;J<ncols;J++) { cols[J] = constCols[J]+(Jj/D->Nz)*D->Nz; }
      ierr = MatSetValues(Iy_D2z,1,&Jj,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(D2z,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_D2z,D->debugFolder,"Iy_D2z",D);CHKERRQ(ierr);
#endif


  Mat D2yplusD2z;
  ierr = MatDuplicate(D2y_Iz,MAT_COPY_VALUES,&(D2yplusD2z));CHKERRQ(ierr);
  v=1.0;
  ierr = MatAXPY(D2yplusD2z,v,Iy_D2z,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 1
  checkMatrix(&D2yplusD2z,D->debugFolder,"D2yplusD2z",D);CHKERRQ(ierr);
#endif

//~ ierr = MatView(D2yplusD2z,PETSC_VIEWER_STDOUT_WORLD);

//~ Mat tempTempMat;
//~ PetscBool bool;
//~ ierr = MatConvert(D2yplusD2z, MATSAME,MAT_INITIAL_MATRIX,&tempTempMat);
//~ ierr = MatEqualVals(&D2yplusD2z,&tempTempMat,&bool,D);CHKERRQ(ierr);
//~
//~ ierr = MatScale(tempTempMat,-1.0);CHKERRQ(ierr);
//~ ierr = MatEqualVals(&D2yplusD2z,&tempTempMat,&bool,D);CHKERRQ(ierr);
//~
//~ checkMatrix(&D2yplusD2z,D->debugFolder,"D2yplusD2z",D);CHKERRQ(ierr);

  /* kron(Hinvy,Iz) */
  Mat Hinvy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_Iz,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D->Hinvy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D->Hinvy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=Ii*D->Nz;J<(Ii+1)*D->Nz;J++) {
      ierr = MatSetValues(Hinvy_Iz,1,&J,1,&J,constVals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(D->Hinvy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Hinvy_Iz,D->debugFolder,"Hinvy_Iz",D);CHKERRQ(ierr);
#endif

  /* kron(Iy,Hinvz) */
  Mat Iy_Hinvz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_Hinvz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_Hinvz,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_Hinvz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D->Hinvz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D->Hinvz,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=Ii;J<D->Ny*D->Nz;J=J+D->Nz) {
      ierr = MatSetValues(Iy_Hinvz,1,&J,1,&J,constVals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(D->Hinvz,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_Hinvz,D->debugFolder,"Iy_Hinvz",D);CHKERRQ(ierr);
#endif

  /* kron(E0y,Iz)
     where E0y(1,1) = 1, 0 else
     */
  Mat E0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&E0y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(E0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(E0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(E0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(E0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(E0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(E0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<D->Nz) { ierr = MatSetValues(E0y_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(E0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(E0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&E0y_Iz,D->debugFolder,"upperE0y_Iz",D);CHKERRQ(ierr);
#endif

  /* kron(ENy,Iz)
   * where ENy(Nz,Nz) = 1, 0 else
   */
  Mat ENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&ENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(ENy_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(ENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(ENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;J=D->Ny*D->Nz;
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>J-D->Nz-1 && Ii<J) { ierr = MatSetValues(ENy_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(ENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&ENy_Iz,D->debugFolder,"upperENy_Iz",D);CHKERRQ(ierr);
#endif

//~ ierr = MatView(ENy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//~ checkMatrix(&ENy_Iz,D->debugFolder,"ENy_Iz",D);CHKERRQ(ierr);

  /* kron(Iy,E0z)
   * where E0y(0,0) = 1, 0 else
   */
  Mat Iy_E0z;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_E0z);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_E0z,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_E0z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_E0z,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_E0z,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_E0z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_E0z,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=Ii/D->Nz;J=Ii-J*D->Nz;
    if (J==0) { ierr = MatSetValues(Iy_E0z,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(Iy_E0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_E0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_E0z,D->debugFolder,"upperIy_E0z",D);CHKERRQ(ierr);
#endif

  /* kron(Iy,ENz)
   * where ENy(Nz,Nz) = 1, 0 else
   */
  Mat Iy_ENz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_ENz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_ENz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_ENz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_ENz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_ENz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_ENz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_ENz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/D->Nz;J=Ii+1-J*D->Nz;
    if (J==0) { ierr = MatSetValues(Iy_ENz,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(Iy_ENz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_ENz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_ENz,D->debugFolder,"upperIy_ENz",D);CHKERRQ(ierr);
#endif

  /* kron(e0y,Iz)
   * where e0y(0) = 1, 0 else
   */
  Mat e0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&e0y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(e0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(e0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(e0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(e0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(e0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(e0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<D->Nz) { ierr = MatSetValues(e0y_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(e0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(e0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&e0y_Iz,D->debugFolder,"lowere0y_Iz",D);CHKERRQ(ierr);
#endif

  /* kron(eNy,Iz)
   * where eNy(Ny) = 1, 0 else
   */
  Mat eNy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&eNy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(eNy_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(eNy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(eNy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(eNy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(eNy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(eNy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  v = 1.0;indx = 0;
  J = D->Nz*D->Ny - D->Nz - 1;
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>J) {
      indx = Ii-J-1;
      ierr = MatSetValues(eNy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(eNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(eNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&eNy_Iz,D->debugFolder,"lowereNy_Iz",D);CHKERRQ(ierr);
#endif


  /* kron(Iy,eNz)
   * where eNz(Nz,Nz) = 1, 0 else
   */
  Mat Iy_eNz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_eNz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_eNz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_eNz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_eNz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_eNz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_eNz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_eNz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;indx=0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/D->Nz;J=Ii+1-J*D->Nz;
    if (J==0) {
      indx = Ii/D->Nz;
      ierr = MatSetValues(Iy_eNz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_eNz,D->debugFolder,"lowerIy_eNz",D);CHKERRQ(ierr);
#endif

  /* kron(Iy,e0z)
   * where e0z(0,0) = 1, 0 else
   */
  Mat Iy_e0z;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_e0z);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_e0z,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_e0z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_e0z,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_e0z,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_e0z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_e0z,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;indx=0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/D->Nz;J=Ii-indx*D->Nz;
    if (J==0) {
      ierr = MatSetValues(Iy_e0z,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_e0z,D->debugFolder,"lowerIy_e0z",D);CHKERRQ(ierr);
#endif

  /* kron(By*Sy,Iz)
   * Note than I changed the sign on Sy's 1st row instead of multiplying by By
   */
  Mat BySy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&BySy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(BySy_Iz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(BySy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(BySy_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(BySy_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(BySy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Sy,&Istart,&Iend);CHKERRQ(ierr);

  if (Istart==0) {
    ierr = MatGetRow(Sy,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for(indx=0;indx<ncols;indx++) {
      J=constCols[indx]*D->Nz;
      v=constVals[indx];
      for (Ii=0;Ii<D->Nz;Ii++) {
        ierr = MatSetValues(BySy_Iz,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        J++;
      }
    }
    ierr = MatRestoreRow(Sy,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }

  Ii=D->Ny-1;
  if (Ii>=Istart && Ii<Iend) {
    ierr = MatGetRow(Sy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for(indx=0;indx<ncols;indx++) {
      J= D->Ny*D->Nz - D->Nz*(D->Ny-constCols[indx]);
      v=constVals[indx];
      for (Ii=D->Ny*D->Nz-D->Nz;Ii<D->Nz*D->Ny;Ii++) {
        ierr = MatSetValues(BySy_Iz,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        J++;
      }
    }
    ierr = MatRestoreRow(Sy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(BySy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(BySy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&BySy_Iz,D->debugFolder,"BySy_Iz",D);CHKERRQ(ierr);
#endif

  /* kron(Iy,Bz*Sz)
   * Note than I changed the sign on Sz's 1st row instead of multiplying by Bz
   */
  Mat Iy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_BzSz,PETSC_DECIDE,PETSC_DECIDE,D->Ny*D->Nz,D->Ny*D->Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_BzSz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_BzSz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Sz,&Istart,&Iend);CHKERRQ(ierr);

  /* rank 0 processor does all initialization for 1st row of BzSz, communicates to others */
  if (Istart==0) {
    ierr = MatGetRow(Sz,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (Ii=0;Ii<D->Ny*D->Nz;Ii=Ii+D->Nz) {
      for (indx=0;indx<ncols;indx++) {
        cols[indx] = constCols[indx]+Ii;
      }
      ierr = MatSetValues(Iy_BzSz,1,&Ii,ncols,cols,constVals,INSERT_VALUES);
    }
    ierr = MatRestoreRow(Sz,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }

  /* lowest rank processor does all initialization for last row of BzSz, communicates to others */
  Ii=D->Nz-1;
  if (Ii>=Istart && Ii<Iend) {
    ierr = MatGetRow(Sz,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (Ii=D->Ny*D->Nz-1;Ii>=0;Ii=Ii-D->Nz) {
      for (indx=0;indx<ncols;indx++) {
        cols[indx] = Ii-D->Nz+constCols[indx]+1;
      }
      ierr = MatSetValues(Iy_BzSz,1,&Ii,ncols,cols,constVals,INSERT_VALUES);
    }
    ierr = MatRestoreRow(Sz,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Iy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_BzSz,D->debugFolder,"Iy_BzSz",D);CHKERRQ(ierr);
#endif


  /* Now all the multiplication terms!! */
  ierr = MatMatMult(Iy_Hinvz,Iy_e0z,MAT_INITIAL_MATRIX,1.0,&(D->IyHinvz_Iye0z));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&(D->IyHinvz_Iye0z),D->debugFolder,"IyHinvz_Iye0z",D);CHKERRQ(ierr);
#endif

  ierr = MatMatMult(Iy_Hinvz,Iy_eNz,MAT_INITIAL_MATRIX,1.0,&(D->IyHinvz_IyeNz));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->IyHinvz_IyeNz,D->debugFolder,"IyHinvz_IyeNz",D);CHKERRQ(ierr);
#endif

  ierr = MatMatMult(Hinvy_Iz,e0y_Iz,MAT_INITIAL_MATRIX,1.0,&(D->Hinvy_Iz_e0y_Iz));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->Hinvy_Iz_e0y_Iz,D->debugFolder,"Hinvy_Iz_e0y_Iz",D);CHKERRQ(ierr);
#endif

  Mat tempMat;
  ierr = MatTranspose(BySy_Iz,MAT_REUSE_MATRIX,&BySy_Iz);CHKERRQ(ierr);
  ierr = MatMatMult(Hinvy_Iz,BySy_Iz,MAT_INITIAL_MATRIX,1.0,&tempMat);CHKERRQ(ierr);
  ierr = MatMatMult(tempMat,e0y_Iz,MAT_INITIAL_MATRIX,1.0,&(D->Hinvy_Iz_BySy_Iz_e0y_Iz));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->Hinvy_Iz_BySy_Iz_e0y_Iz,D->debugFolder,"Hinvy_Iz_BySy_Iz_e0y_Iz",D);CHKERRQ(ierr);
#endif

  ierr = MatMatMult(Hinvy_Iz,eNy_Iz,MAT_INITIAL_MATRIX,1.0,&(D->Hinvy_Iz_eNy_Iz));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->Hinvy_Iz_eNy_Iz,D->debugFolder,"Hinvy_Iz_eNy_Iz",D);CHKERRQ(ierr);
#endif

  ierr = MatMatMult(tempMat,eNy_Iz,MAT_INITIAL_MATRIX,1.0,&(D->Hinvy_Iz_BySy_Iz_eNy_Iz));CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->Hinvy_Iz_BySy_Iz_eNy_Iz,D->debugFolder,"Hinvy_Iz_BySy_Iz_eNy_Iz",D);CHKERRQ(ierr);
#endif

//~ ierr = PetscPrintf(PETSC_COMM_WORLD,"D.G = %g\n",D->G);CHKERRQ(ierr);

  /* Compute A */
  // A = D.G*D2yplusD2z + alphaF*D.G*Hinvy_Iz*E0y_Iz
  ierr = MatMatMult(Hinvy_Iz,E0y_Iz,MAT_INITIAL_MATRIX,1.0,&(D->A));CHKERRQ(ierr);
  ierr = MatScale(D->A,D->alphaF*D->G);CHKERRQ(ierr);
  ierr = MatAXPY(D->A,D->G,D2yplusD2z,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"Astage1",D);CHKERRQ(ierr);
#endif

  // + beta*Hinvy_Iz*(D.G*BySy_Iz)^T*E0y_Iz + ...
  ierr = MatScale(BySy_Iz,D->G);CHKERRQ(ierr);
  ierr = MatMatMult(Hinvy_Iz,BySy_Iz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(temp,E0y_Iz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatAYPX(D->A,D->beta,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"Astage2",D);CHKERRQ(ierr);
#endif

  // + alphaR*D.G*Hinvy_Iz*ENy_Iz + ...
  ierr = MatMatMult(Hinvy_Iz,ENy_Iz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatAXPY(D->A,D->alphaR*D->G,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"Astage3",D);CHKERRQ(ierr);
#endif

  // + beta*Hinvy_Iz*(D.G*BySy_Iz)^T*ENy_Iz + ...
  ierr = MatMatMult(Hinvy_Iz,BySy_Iz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(temp,ENy_Iz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatAXPY(D->A,1.0,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"Astage4",D);CHKERRQ(ierr);
#endif

  //~ ierr = MatView(D->A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // + alphaS*Iy_Hinvz*Iy_E0z*D.G*Iy_BzSz + ...
  ierr = MatMatMult(Iy_Hinvz,Iy_E0z,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatScale(Iy_BzSz,D->G);CHKERRQ(ierr);
  ierr = MatMatMult(temp,Iy_BzSz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatScale(temp,D->alphaS);CHKERRQ(ierr);
  ierr = MatAXPY(D->A,1.0,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"Astage5",D);CHKERRQ(ierr);
#endif

  // + alphaD*Iy_Hinvz*Iy_ENz*D.G*Iy_BzSz
  ierr = MatMatMult(Iy_Hinvz,Iy_ENz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(temp,Iy_BzSz,MAT_INITIAL_MATRIX,1.0,&temp);CHKERRQ(ierr);
  //~ ierr = MatAXPY(D->A,D->alphaD*D->G,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D->A,D->alphaD,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

#if DEBUG > 0
  checkMatrix(&D->A,D->debugFolder,"A",D);CHKERRQ(ierr);
#endif

  ierr = PetscFree(cols);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function createOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  return 0;
};

PetscErrorCode SBPoperators(PetscInt ORDER, PetscInt N, Mat *PinvMat, Mat *D, Mat *D2, Mat *S)
{
  PetscErrorCode ierr;
  PetscScalar    v,*stencil;
  PetscInt       Ii,J,Istart,Iend,*cols,ncols;
  Mat            Q;
  //Mat            debugMat;
  //PetscBool      debugBool;

  PetscInt const *constCols;
  PetscScalar const *constVals;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function SBPoperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  /* Create matrix operators: PinvMat, Q, D2, S */
  ierr = MatCreate(PETSC_COMM_WORLD,PinvMat);CHKERRQ(ierr);
  ierr = MatSetSizes(*PinvMat,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*PinvMat);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*PinvMat,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*PinvMat,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*PinvMat);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Q);CHKERRQ(ierr);
  ierr = MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Q);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Q,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Q,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Q);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,D2);CHKERRQ(ierr);
  ierr = MatSetSizes(*D2,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*D2);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*D2,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*D2,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*D2);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,S);CHKERRQ(ierr);
  ierr = MatSetSizes(*S,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*S);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*S,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*S,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(*S);CHKERRQ(ierr);

  PetscMalloc(16*sizeof(PetscInt),&cols);
  PetscMalloc(16*sizeof(PetscScalar),&stencil);

  switch ( ORDER ) {
    case 2:
      /*
      PinvArray[0]=2.0;
      PinvArray[N]=2.0;
      for (Ii=1;Ii<N;Ii++) {
        PinvArray[Ii]=1.0;
      }
       */

      ierr = MatGetOwnershipRange(Q,&Istart,&Iend); CHKERRQ(ierr);
      for (Ii=Istart;Ii<Iend;Ii++) {
        if (Ii<N) { v=0.5;J=Ii+1; ierr=MatSetValues(Q,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
        if (Ii>0) { v=-0.5;J=Ii-1; ierr=MatSetValues(Q,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      }
      if (Istart==0) { v=-0.5; ierr=MatSetValues(Q,1,&Istart,1,&Istart,&v,INSERT_VALUES);CHKERRQ(ierr); }
      if (Iend==N+1) { v=0.5;Ii=N; ierr=MatSetValues(Q,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }

      ierr = MatGetOwnershipRange(*PinvMat,&Istart,&Iend); CHKERRQ(ierr);
      for (Ii=Istart;Ii<Iend;Ii++) {
        v=1.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Istart==0) { v=2.0; ierr=MatSetValues(*PinvMat,1,&Istart,1,&Istart,&v,INSERT_VALUES);CHKERRQ(ierr); }
      if (Iend==N+1) { v=2.0;Ii=Iend-1; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }

      ierr = MatGetOwnershipRange(*D2,&Istart,&Iend);CHKERRQ(ierr);
      stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
      for (Ii=Istart;Ii<Iend;Ii++) {
        if (Ii>0 && Ii<N) {
          cols[0] = Ii-1;cols[1]=Ii;cols[2]=Ii+1;
          v=1.0;ierr=MatSetValues(*D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (Istart==0) {
        cols[0]=Istart;cols[1]=Istart+1;cols[2]=Istart+2;
        ierr=MatSetValues(*D2,1,&Istart,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=N;
        cols[2]=Ii;cols[1]=Ii-1;cols[0]=Ii-2;
        ierr=MatSetValues(*D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }

      ierr = MatGetOwnershipRange(*S,&Istart,&Iend);CHKERRQ(ierr);
      if (Istart==0) {
        cols[0]=0;cols[1]=1;cols[2]=2;
        stencil[0]=1.5;stencil[1]=-2;stencil[2]=0.5; // -1 * row from paper
        ierr=MatSetValues(*S,1,&Istart,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=Iend-1;
        cols[2]=N;cols[1]=N-1;cols[0]=N-2;
        stencil[0]=0.5;stencil[1]=-2;stencil[2]=1.5;
        ierr=MatSetValues(*S,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }

      break;

    case 4:
      if (N<8) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      /*PinvArray[0]=48.0/17.0;PinvArray[1]=48.0/59.0;PinvArray[2]=48.0/43.0;PinvArray[3]=48.0/49.0;
      PinvArray[N-3]=48.0/49.0;PinvArray[N-2]=48.0/43.0;PinvArray[N-1]=48.0/59.0;PinvArray[N]=48.0/17.0;
      for (Ii=4;Ii<N-3;Ii++) {
        PinvArray[Ii]=1.0;
      }*/

      ierr = MatGetOwnershipRange(Q,&Istart,&Iend); CHKERRQ(ierr);
      if (Istart==0) {
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;
        stencil[0]=-1.0/2.0;stencil[1]=59.0/96.0;stencil[2]=-1.0/12.0;stencil[3]=-1.0/32.0;
        ierr=MatSetValues(Q,1,&Istart,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=1;
        cols[0]=0;cols[1]=2;stencil[0]=-59.0/96.0;stencil[1]=59.0/96.0;
        ierr=MatSetValues(Q,1,&Ii,2,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=2;
        cols[0]=0;cols[1]=1;cols[2]=3;cols[3]=4;
        stencil[0]=1.0/12.0;stencil[1]=-59.0/96.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/12.0;
        ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=3;
        cols[0]=0;cols[1]=2;cols[2]=4;cols[3]=5;
        stencil[0]=1.0/32.0;stencil[1]=-59.0/96.0;stencil[2]=2.0/3.0;stencil[3]=-1.0/12.0;
        ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=N;
        cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        stencil[0]=1.0/32.0;stencil[1]=1.0/12.0;stencil[2]=-59.0/96.0;stencil[3]=1.0/2.0;
        ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N-1;
        cols[0]=N-2;cols[1]=N;
        stencil[0]=-59.0/96.0;stencil[1]=59.0/96.0;
        ierr=MatSetValues(Q,1,&Ii,2,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N-2;
        cols[0]=N-4;cols[1]=N-3;cols[2]=N-1;cols[3]=N;
        stencil[0]=1.0/12.0;stencil[1]=-59.0/96.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/12.0;
        ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N-3;
        cols[0]=N-5;cols[1]=N-4;cols[2]=N-2;cols[3]=N;
        stencil[0]=1.0/12.0;stencil[1]=-2.0/3.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/32.0;
        ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      for (Ii=Istart;Ii<Iend;Ii++) {
        if (Ii>3 && Ii<N-3) {
          cols[0]=Ii-2;cols[1]=Ii-1;cols[2]=Ii+1;cols[3]=Ii+2;
          stencil[0]=1.0/12.0;stencil[1]=-2.0/3.0;stencil[2]=2.0/3.0;stencil[3]=-1.0/12.0;
          ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      ierr = MatGetOwnershipRange(*PinvMat,&Istart,&Iend); CHKERRQ(ierr);
      for (Ii=Istart;Ii<Iend;Ii++) {
        v=1.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Istart==0) {
        Ii=Istart;v=48.0/17.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=Istart+1;v=48.0/59.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=Istart+2;v=48.0/43.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=Istart+3;v=48.0/49.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=N;v=48.0/17.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=N-1;v=48.0/59.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=N-2;v=48.0/43.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        Ii=N-3;v=48.0/49.0; ierr=MatSetValues(*PinvMat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }


      ierr = MatGetOwnershipRange(*D2,&Istart,&Iend);CHKERRQ(ierr);
      for (Ii=Istart;Ii<Iend;Ii++) {
        if (Ii>3 && Ii<N-3) {
          cols[0]=Ii-2;cols[1]=Ii-1;cols[2]=Ii;cols[3]=Ii+1;cols[4]=Ii+2;
          stencil[0]=-1.0/12.0;stencil[1]=4.0/3.0;stencil[2]=-5.0/2.0;stencil[3]=4.0/3.0;stencil[4]=-1.0/12.0;
          ierr=MatSetValues(*D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (Istart==0) {
        Ii=Istart;
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        stencil[0]=2.0;stencil[1]=-5.0;stencil[2]=4.0;stencil[3]=-1.0;
        ierr=MatSetValues(*D2,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=Istart+1;
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
        ierr=MatSetValues(*D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=Istart+2;
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        stencil[0]=-4.0/43.0;stencil[1]=59.0/43.0;stencil[2]=-110.0/43.0;stencil[3]=59.0/43;stencil[4]=-4.0/43.0;
        ierr=MatSetValues(*D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=Istart+3;
        cols[0]=0;cols[1]=2;cols[2]=3;cols[3]=4;cols[4]=5;
        stencil[0]=-1.0/49.0;stencil[1]=59.0/49.0;stencil[2]=-118.0/49.0;stencil[3]=64.0/49.0;stencil[4]=-4.0/49.0;
        ierr=MatSetValues(*D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=N-3;
        cols[0]=N-5;cols[1]=N-4;cols[2]=N-3;cols[3]=N-2;cols[4]=N;
        stencil[0]=-4.0/49.;stencil[1]=64.0/49.0;stencil[2]=-118.0/49.0;stencil[3]=59.0/49.0;stencil[4]=-1.0/49.0;
        ierr=MatSetValues(*D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N-2;
        cols[0]=N-4;cols[1]=N-3;cols[2]=N-2;cols[3]=N-1;cols[4]=N;
        stencil[0]=-4.0/43.0;stencil[1]=59.0/43.0;stencil[2]=-110.0/43.0;stencil[3]=59.0/43.0;stencil[4]=-4.0/43.0;
        ierr=MatSetValues(*D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N-1;
        cols[0]=N-2;cols[1]=N-1;cols[2]=N;
        stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
        ierr=MatSetValues(*D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        Ii=N;
        cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        stencil[0]=-1.0;stencil[1]=4.0;stencil[2]=-5.0;stencil[3]=2.0;
        ierr=MatSetValues(*D2,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }

      ierr = MatGetOwnershipRange(*S,&Istart,&Iend);CHKERRQ(ierr);
      if (Istart==0) {
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;
        stencil[0]=11.0/6.0;stencil[1]=-3.0;stencil[2]=3.0/2.0;stencil[3]=-1.0/3.0; // -1 * row from paper
        ierr=MatSetValues(*S,1,&Istart,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=Iend-1;
        cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        stencil[0]=-1.0/3.0;stencil[1]=3.0/2.0;stencil[2]=-3.0;stencil[3]=11.0/6.0;
        ierr=MatSetValues(*S,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }

      break;

    case 6:
      if (N<12) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      /* Initialize values of Pinv */
      /*PinvArray[0] = 43200.0/13649.0;  PinvArray[N-5] = PinvArray[0];
      PinvArray[1] = 8640.0/12013.0;   PinvArray[N-4] = PinvArray[1];
      PinvArray[2] = 4320.0/2711.0;    PinvArray[N-3] = PinvArray[2];
      PinvArray[3] = 4320.0/5359.0;    PinvArray[N-2] = PinvArray[3];
      PinvArray[4] = 8640.0/7877.0;    PinvArray[N-1] = PinvArray[4];
      PinvArray[5] = 43200.0/43801.0;  PinvArray[N]   = PinvArray[5];
      for (Ii=6;Ii<N-5;Ii++) {
        PinvArray[Ii]=1.0;
      }*/

      /* Initialize values of S */
      ierr = MatGetOwnershipRange(*S,&Istart,&Iend);CHKERRQ(ierr);
      if (Istart==0) {
        cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        stencil[0]=25.0/12.0;stencil[1]=-4.0;stencil[2]=3.0;stencil[3]=-4.0/3.0;stencil[4]=-0.25; // -1 * row from paper
        ierr=MatSetValues(*S,1,&Istart,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (Iend==N+1) {
        Ii=Iend-1;
        cols[0]=N-4;cols[1]=N-3;cols[2]=N-2;cols[3]=N-1;cols[4]=N;
        stencil[0]=0.25;stencil[1]=-4.0/3.0;stencil[2]=3.0;stencil[3]=-4.0;stencil[4]=25.0/12.0;
        ierr=MatSetValues(*S,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      }

      /* Initialize values of D2 */
      // this is going to be hard to extract from the paper and what I have now

      break;


    default:
      SETERRQ(PETSC_COMM_WORLD,1,"ORDER not understood.");
      break;
  }

  ierr = MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*PinvMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*D2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*PinvMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*D2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMatMult(*PinvMat,Q,MAT_INITIAL_MATRIX,1.0,D);CHKERRQ(ierr);
  ierr = MatSetOption(*D,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);
  ierr = MatSetFromOptions(*D);CHKERRQ(ierr);
  // D(1,:) S(1,:); D(end,:) = S(end,:)
  ierr = MatGetOwnershipRange(*S,&Istart,&Iend);CHKERRQ(ierr);
  if (Istart == 0) {
    ierr = MatGetRow(*S,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=0;J<ncols;J++) { stencil[J]=-1.0*constVals[J];}
    ierr = MatSetValues(*D,1,&Istart,ncols,constCols,stencil,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(*S,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  Ii=N;
  if (Ii>=Istart && Ii<Iend) {
    ierr = MatGetRow(*S,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    ierr = MatSetValues(*D,1,&Ii,ncols,constCols,constVals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(*S,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(stencil);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function SBPoperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  return 0;
}

