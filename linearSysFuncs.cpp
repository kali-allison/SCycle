#include <petscts.h>
#include <string>
#include "odeSolver.h"
#include "userContext.h"
#include "debuggingFuncs.hpp"
#include "linearSysFuncs.h"


PetscErrorCode setLinearSystem(UserContext &D, const PetscBool loadMat)
{
  PetscErrorCode ierr = 0;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function setLinearSystem in linearSysFuncs.cpp.\n");CHKERRQ(ierr);
#endif

  // SBP operators and penalty terms
  D.alphaF = -13.0/D.dy;
  D.alphaR = -13.0/D.dy;
  D.alphaS = -1.0;
  D.alphaD = -1.0;
  D.beta   = 1.0;

  if (loadMat) { ierr = D.loadOperators();CHKERRQ(ierr); }
  else { ierr = createOperators(D);CHKERRQ(ierr);}

  ierr = KSPSetType(D.ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(D.ksp,D.A,D.A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(D.ksp,&D.pc);CHKERRQ(ierr);

  // use PETSc's direct LU - only available on 1 processor!!!
  ierr = PCSetType(D.pc,PCLU);CHKERRQ(ierr);

  // use HYPRE
  //~ierr = PCSetType(D.pc,PCHYPRE);CHKERRQ(ierr);
  //~ierr = PCHYPRESetType(D.pc,"boomeramg");CHKERRQ(ierr);
  //~ierr = KSPSetTolerances(D.ksp,D.kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  //~ierr = PCFactorSetLevels(D.pc,4);CHKERRQ(ierr);

  // use direct LU from MUMPS
  //~PCSetType(D.pc,PCLU);
  //~PCFactorSetMatSolverPackage(D.pc,MATSOLVERMUMPS);
  //~PCFactorSetUpMatSolverPackage(D.pc);

  ierr = KSPSetUp(D.ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(D.ksp);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function setLinearSystem in linearSysFuncs.cpp.\n");CHKERRQ(ierr);
#endif

  return ierr;
}

PetscErrorCode ComputeRHS(UserContext &D)
{
  PetscErrorCode ierr = 0;

  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function ComputeRHS in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

/* rhs =  D.alphaF*mu*D.Hinvy_Iz_e0y_Iz*gF +... */
  ierr = MatMult(D.Hinvy_Izxe0y_Iz,D.gF,D.rhs);CHKERRQ(ierr);
  ierr = VecScale(D.rhs,D.alphaF);CHKERRQ(ierr);

  /* + D.beta*mu*D.Hinvy_Iz_BySy_Iz_e0y_Iz*gF + ... */
  Vec temp;
  ierr = VecDuplicate(D.rhs,&temp);CHKERRQ(ierr);
  ierr = MatMult(D.Hinvy_IzxBySy_IzTxe0y_Iz,D.gF,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D.rhs,D.beta,temp);CHKERRQ(ierr);

  /* + D.alphaR*mu*D.Hinvy_Iz_eNy_Iz*gR + ... */
  ierr = MatMult(D.Hinvy_IzxeNy_Iz,D.gR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D.rhs,D.alphaR,temp);CHKERRQ(ierr);

  /* + D.beta*mu*D.Hinvy_Iz_BySy_Iz_eNy_Iz*gR + ... */
  ierr = MatMult(D.Hinvy_IzxBySy_IzTxeNy_Iz,D.gR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(D.rhs,D.beta,temp);CHKERRQ(ierr);

  //~ /* - D.alphaS*M.IyHinvz_Iye0z*gS + ... */
  //~ ierr = MatMult(D.IyHinvz_Iye0z,D.gS,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(D.rhs,D.alphaS,temp);CHKERRQ(ierr);
//~
  //~ /* + D.alphaD*M.IyHinvz_IyeNz*gD */
  //~ ierr = MatMult(D.IyHinvz_IyeNz,D.gD,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(D.rhs,D.alphaD,temp);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function ComputeRHS in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  double endTime = MPI_Wtime();
  D.computeRhsTime = D.computeRhsTime + (endTime-startTime);

  return ierr;
}


PetscErrorCode createOperators(UserContext &D)
{
  PetscErrorCode  ierr = 0;
  PetscScalar     v;
  PetscInt        Ii,J,Istart,Iend,ncols,indx,*cols,Jj;
  Mat             D2y,Sy,D2z,Sz,Dy,Dz;

  PetscInt const *constCols;
  PetscScalar const *constVals;
  PetscScalar *vals;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function createOperators in linearSysFuncs.c.\n");
  CHKERRQ(ierr);
#endif

  PetscMalloc(D.Nz*sizeof(PetscInt),&cols);
  PetscMalloc(D.Nz*sizeof(PetscInt),&vals);


  ierr = SBPoperators(D.order,D.Ny-1,&D.Hinvy,&Dy,&D2y,&Sy);CHKERRQ(ierr);
  ierr = SBPoperators(D.order,D.Nz-1,&D.Hinvz,&Dz,&D2z,&Sz);CHKERRQ(ierr);

#if DEBUG > 1
  ierr = checkMatrix(&Dy,D.debugFolder,"Dy",&D);CHKERRQ(ierr);
  ierr = checkMatrix(&D2y,D.debugFolder,"D2y",&D);CHKERRQ(ierr);
  ierr = checkMatrix(&Dz,D.debugFolder,"Dz",&D);CHKERRQ(ierr);
  ierr = checkMatrix(&D2z,D.debugFolder,"D2z",&D);CHKERRQ(ierr);
#endif

  /* Scaling (why not do this while initializing each matrix?) */
  D.D2y = D2y;
  D.D2z = D2z;
  ierr = MatScale(D2y,1.0/D.dy/D.dy);CHKERRQ(ierr);
  ierr = MatScale(D2z,1.0/D.dz/D.dz);CHKERRQ(ierr);
  ierr = MatScale(Dy,1.0/D.dy);CHKERRQ(ierr);

  /* mu*kron(Dy,Iz) */
  ierr = MatSetSizes(D.Dy_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Dy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Dy_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Dy_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Dy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Dy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(Dy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=0;J<ncols;J++) { cols[J]=constCols[J]*D.Nz;}
    for (J=Ii*D.Nz;J<(Ii+1)*D.Nz;J++) {
      ierr = MatSetValues(D.Dy_Iz,1,&J,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
      for (Jj=0;Jj<ncols;Jj++) { cols[Jj]=cols[Jj]+1;}
    }
    ierr = MatRestoreRow(Dy,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D.Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Dy_Iz,D.debugFolder,"Dy_Iz",&D);CHKERRQ(ierr);
#endif
ierr = MatMatMult(D.mu,D.Dy_Iz,MAT_INITIAL_MATRIX,1.0,&D.Dy_Iz);CHKERRQ(ierr);


  /* kron(D2y,Iz) */
  Mat D2y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&D2y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(D2y_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D2y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D2y_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D2y_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D2y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2y,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D2y,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (J=0;J<ncols;J++) { cols[J]=constCols[J]*D.Nz;}
    for (J=Ii*D.Nz;J<(Ii+1)*D.Nz;J++) {
      ierr = MatSetValues(D2y_Iz,1,&J,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
      for (Jj=0;Jj<ncols;Jj++) { cols[Jj]=cols[Jj]+1;}
    }
    ierr = MatRestoreRow(D2y,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D2y_Iz,D.debugFolder,"D2y_Iz",&D);CHKERRQ(ierr);
#endif

  /* kron(Iy,D2z) */
  Mat Iy_D2z;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_D2z);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_D2z,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_D2z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_D2z,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_D2z,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_D2z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2z,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = MatGetRow(D2z,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    for (Jj=Ii;Jj<D.Ny*D.Nz;Jj=Jj+D.Nz) {
      for (J=0;J<ncols;J++) { cols[J] = constCols[J]+(Jj/D.Nz)*D.Nz; }
      ierr = MatSetValues(Iy_D2z,1,&Jj,ncols,cols,constVals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(D2z,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_D2z,D.debugFolder,"Iy_D2z",&D);CHKERRQ(ierr);
#endif

  Mat D2yplusD2z;
  ierr = MatDuplicate(D2y_Iz,MAT_COPY_VALUES,&(D2yplusD2z));CHKERRQ(ierr);
  ierr = MatAXPY(D2yplusD2z,1.0,Iy_D2z,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D2yplusD2z,D.debugFolder,"D2yplusD2z",&D);CHKERRQ(ierr);
#endif

  PetscScalar HinvyArr[D.Ny], SyArr[D.Ny*2];
  PetscScalar HinvzArr[D.Nz], SzArr[D.Nz*2];
  PetscInt Sylen=0, Szlen=0;
  ierr = SBPopsArrays(D.order,D.Ny,1/D.dy,HinvyArr,SyArr,&Sylen);CHKERRQ(ierr);
  ierr = SBPopsArrays(D.order,D.Nz,1/D.dz,HinvzArr,SzArr,&Szlen);CHKERRQ(ierr);

  double startArrLinOps = MPI_Wtime(); //!!!!!!!!!!

  // for producing rhs vector
  // Hinvy_Izxe0y_Iz = kron(Hinvy,Iz)*kron(e0y,Iz)
  ierr = MatSetSizes(D.Hinvy_Izxe0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Hinvy_Izxe0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Hinvy_Izxe0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Hinvy_Izxe0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<D.Nz;Ii++) {
    ierr = MatSetValues(D.Hinvy_Izxe0y_Iz,1,&Ii,1,&Ii,&(HinvyArr[0]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D.Hinvy_Izxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Hinvy_Izxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,D.Hinvy_Izxe0y_Iz,MAT_INITIAL_MATRIX,1.0,&D.Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Hinvy_Izxe0y_Iz,D.debugFolder,"Hinvy_Izxe0y_Iz",&D);CHKERRQ(ierr);
#endif

  // Hinvy_IzxeNy_Iz = kron(Hinvy,Iz)*kron(eNy,Iz)
  ierr = MatSetSizes(D.Hinvy_IzxeNy_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Hinvy_IzxeNy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Hinvy_IzxeNy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Hinvy_IzxeNy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  J = D.Ny*D.Nz - D.Nz;
  for (Ii=Iend-1;Ii>=D.Ny*D.Nz-D.Nz;Ii--) {
    indx = Ii-J;
    ierr = MatSetValues(D.Hinvy_IzxeNy_Iz,1,&Ii,1,&indx,&(HinvyArr[D.Ny-1]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D.Hinvy_IzxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Hinvy_IzxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,D.Hinvy_IzxeNy_Iz,MAT_INITIAL_MATRIX,1.0,&D.Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Hinvy_IzxeNy_Iz,D.debugFolder,"Hinvy_IzxeNy_Iz",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_e0z = kron(Iz,Hiinvz)*kron(Iy,e0z)
  ierr = MatSetSizes(D.Iy_HinvzxIy_e0z,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Iy_HinvzxIy_e0z,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Iy_HinvzxIy_e0z,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Iy_HinvzxIy_e0z,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;indx=0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/D.Nz;J=Ii-indx*D.Nz;
    if (J==0) {
      ierr = MatSetValues(D.Iy_HinvzxIy_e0z,1,&Ii,1,&indx,&HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(D.Iy_HinvzxIy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Iy_HinvzxIy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Iy_HinvzxIy_e0z,D.debugFolder,"Iy_HinvzxIy_e0z",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_eNz = kron(Iy,Hinvz)*kron(Iy,eNz)
  ierr = MatSetSizes(D.Iy_HinvzxIy_eNz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Iy_HinvzxIy_eNz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Iy_HinvzxIy_eNz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Iy_HinvzxIy_eNz,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;indx=0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/D.Nz;J=Ii+1-J*D.Nz;
    if (J==0) {
      indx = Ii/D.Nz;
      ierr = MatSetValues(D.Iy_HinvzxIy_eNz,1,&Ii,1,&indx,&HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(D.Iy_HinvzxIy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Iy_HinvzxIy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Iy_HinvzxIy_eNz,D.debugFolder,"Iy_HinvzxIy_eNz",&D);CHKERRQ(ierr);
#endif


  // Hinvy_IzxBySy_IzTxe0y_Iz = kron(Hinvy,Iz)*kron(BySy,Iz)^T*kron(e0y,Iz)
  ierr = MatSetSizes(D.Hinvy_IzxBySy_IzTxe0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Hinvy_IzxBySy_IzTxe0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Hinvy_IzxBySy_IzTxe0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Hinvy_IzxBySy_IzTxe0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<D.Nz*Sylen;Ii++) {
    indx = Ii-(Ii/D.Nz)*D.Nz;
    v = HinvyArr[Ii/D.Nz]*SyArr[Ii/D.Nz];
    ierr = MatSetValues(D.Hinvy_IzxBySy_IzTxe0y_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(D.Hinvy_IzxBySy_IzTxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Hinvy_IzxBySy_IzTxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,D.Hinvy_IzxBySy_IzTxe0y_Iz,MAT_INITIAL_MATRIX,1.0,&D.Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Hinvy_IzxBySy_IzTxe0y_Iz,D.debugFolder,"Hinvy_IzxBySy_IzTxe0y_Iz",&D);CHKERRQ(ierr);
#endif

  // Hinvy_IzxBySy_IzTxeNy_Iz = kron(Hinvy,Iz)*kron(BySy,Iz)^T*kron(eNy,Iz)
  ierr = MatSetSizes(D.Hinvy_IzxBySy_IzTxeNy_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.Hinvy_IzxBySy_IzTxeNy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.Hinvy_IzxBySy_IzTxeNy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D.Hinvy_IzxBySy_IzTxeNy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>=Iend-D.Nz*Sylen) {
      indx = Ii-(Ii/D.Nz)*D.Nz;
      v = HinvyArr[Ii/D.Nz]*SyArr[2*Sylen-1-(Iend-1-Ii)/D.Nz];
      ierr = MatSetValues(D.Hinvy_IzxBySy_IzTxeNy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(D.Hinvy_IzxBySy_IzTxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D.Hinvy_IzxBySy_IzTxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,D.Hinvy_IzxBySy_IzTxeNy_Iz,MAT_INITIAL_MATRIX,1.0,&D.Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.Hinvy_IzxBySy_IzTxeNy_Iz,D.debugFolder,"Hinvy_IzxBySy_IzTxeNy_Iz",&D);CHKERRQ(ierr);
#endif


  // for producing matrix A
  // Hinvy_IzxE0y_Iz = kron(Hinvy,Iz)*kron(E0y,Iz)
  Mat Hinvy_IzxE0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxE0y_Iz);
  ierr = MatSetSizes(Hinvy_IzxE0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxE0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxE0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxE0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<D.Nz;Ii++) {
    ierr = MatSetValues(Hinvy_IzxE0y_Iz,1,&Ii,1,&Ii,&(HinvyArr[0]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,Hinvy_IzxE0y_Iz,MAT_INITIAL_MATRIX,1.0,&Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Hinvy_IzxE0y_Iz,D.debugFolder,"Hinvy_IzxUE0y_Iz",&D);CHKERRQ(ierr);
#endif

  // Hinvy_IzxENy_Iz = kron(Hinvy,Iz)*kron(ENy,Iz)
  Mat Hinvy_IzxENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxENy_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  J = D.Ny*D.Nz - D.Nz;
  for (Ii=Iend-1;Ii>=D.Ny*D.Nz-D.Nz;Ii--) {
    ierr = MatSetValues(Hinvy_IzxENy_Iz,1,&Ii,1,&Ii,&(HinvyArr[D.Ny-1]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(D.mu,Hinvy_IzxENy_Iz,MAT_INITIAL_MATRIX,1.0,&Hinvy_IzxENy_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Hinvy_IzxENy_Iz,D.debugFolder,"Hinvy_IzxUENy_Iz",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_E0z = kron(Iy,Hinvz)*kron(Iy,E0z)
  Mat Iy_HinvzxIy_E0z;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_E0z);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_E0z,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_E0z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_E0z,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_E0z,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_E0z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_E0z,&Istart,&Iend);CHKERRQ(ierr);
  v=1.0;indx=0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/D.Nz;J=Ii-indx*D.Nz;
    if (J==0) {
      ierr = MatSetValues(Iy_HinvzxIy_E0z,1,&Ii,1,&Ii,&HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_E0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_E0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvzxIy_E0z,D.debugFolder,"Iy_HinvzxIy_UE0z",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_ENz = kron(Iy,Hinvz)*kron(Iy,ENz)
  Mat Iy_HinvzxIy_ENz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_ENz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_ENz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_ENz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_ENz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_ENz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_ENz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_ENz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/D.Nz;J=Ii+1-J*D.Nz;
    if (J==0) {
      indx = Ii/D.Nz;
      ierr = MatSetValues(Iy_HinvzxIy_ENz,1,&Ii,1,&Ii,&HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_ENz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_ENz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvzxIy_ENz,D.debugFolder,"Iy_HinvzxIy_UENz",&D);CHKERRQ(ierr);
#endif

  // Hinvy_IzxBySy_IzTxE0y_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(E0z,Iz)
  Mat Hinvy_IzxBySy_IzTxE0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxBySy_IzTxE0y_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxBySy_IzTxE0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxBySy_IzTxE0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxBySy_IzTxE0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<D.Nz*Sylen;Ii++) {
    indx = Ii-(Ii/D.Nz)*D.Nz;
    v = D.muArr[indx]*HinvyArr[Ii/D.Nz]*SyArr[Ii/D.Nz];
    ierr = MatSetValues(Hinvy_IzxBySy_IzTxE0y_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Hinvy_IzxBySy_IzTxE0y_Iz,D.debugFolder,"Hinvy_IzxBySy_IzTxUE0y_Iz",&D);CHKERRQ(ierr);
#endif


  // Hinvy_IzxBySy_IzTxENy_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(ENz,Iz)
  Mat Hinvy_IzxBySy_IzTxENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxBySy_IzTxENy_Iz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxBySy_IzTxENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxBySy_IzTxENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxBySy_IzTxENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>=Iend-D.Nz*Sylen) {
      indx = (D.Ny-1)*D.Nz + Ii-(Ii/D.Nz)*D.Nz;
      v = D.muArr[indx]*HinvyArr[Ii/D.Nz]*SyArr[2*Sylen-1-(Iend-1-Ii)/D.Nz];
      ierr = MatSetValues(Hinvy_IzxBySy_IzTxENy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Hinvy_IzxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Hinvy_IzxBySy_IzTxENy_Iz,D.debugFolder,"Hinvy_IzxBySy_IzTxUENy_Iz",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvxIy_E0zxIy_BzSz = kron(Iy,Hinvz)*kron(Iy,E0z)*mu*kron(Iy,BzSz)
  Mat Iy_HinvxIy_E0zxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvxIy_E0zxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvxIy_E0zxIy_BzSz,Szlen,NULL,Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvxIy_E0zxIy_BzSz,Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvxIy_E0zxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/D.Nz;J=Ii-indx*D.Nz;
    if (J==0) {
      ierr = MatGetRow(D.mu,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
      for (indx=0;indx<Szlen;indx++) {
        cols[indx]=Ii+indx;
        vals[indx]=constVals[0]*HinvzArr[J]*SzArr[indx];
      }
      ierr = MatSetValues(Iy_HinvxIy_E0zxIy_BzSz,1,&Ii,Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D.mu,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvxIy_E0zxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvxIy_E0zxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvxIy_E0zxIy_BzSz,D.debugFolder,"Iy_HinvxIy_E0zxIy_BzSz",&D);CHKERRQ(ierr);
#endif


  // Iy_HinvxIy_ENzxIy_BzSz = kron(Iy,Hinvz)*kron(Iy,ENz)*mu*kron(Iy,BzSz)
  Mat Iy_HinvxIy_ENzxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvxIy_ENzxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvxIy_ENzxIy_BzSz,Szlen,NULL,Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvxIy_ENzxIy_BzSz,Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvxIy_ENzxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/D.Nz;J=Ii+1-J*D.Nz;
    if (J==0) {
      ierr = MatGetRow(D.mu,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
      for (indx=0;indx<Szlen;indx++) {
        cols[indx]=Ii-Szlen+1+indx;
        vals[indx]=constVals[0]*HinvzArr[J]*SzArr[Sylen+indx];
      }
      ierr = MatSetValues(Iy_HinvxIy_ENzxIy_BzSz,1,&Ii,Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D.mu,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvxIy_ENzxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvxIy_ENzxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvxIy_ENzxIy_BzSz,D.debugFolder,"Iy_HinvxIy_ENzxIy_BzSz",&D);CHKERRQ(ierr);
#endif

  /* Compute A */
  // A = mu*D2yplusD2z + alphaF*mu*Hinvy_Iz*E0y_Iz...
  ierr = MatMatMult(D.mu,D2yplusD2z,MAT_INITIAL_MATRIX,1.0,&D.A);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.A,D.debugFolder,"Astage1",&D);CHKERRQ(ierr);
#endif

  ierr = MatAXPY(D.A,D.alphaF,Hinvy_IzxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.A,D.debugFolder,"Astage2",&D);CHKERRQ(ierr);
#endif
  // + beta*Hinvy_Iz*(mu*BySy_Iz)^T*E0y_Iz + ...
  //~ierr = MatAYPX(D.A,D.beta,Hinvy_IzxBySy_IzTxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D.A,D.beta,Hinvy_IzxBySy_IzTxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&D.A,D.debugFolder,"Astage3",&D);CHKERRQ(ierr);
#endif

  // + alphaR*mu*Hinvy_Iz*ENy_Iz + ...
  ierr = MatAXPY(D.A,D.alphaR,Hinvy_IzxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // + beta*Hinvy_Iz*(mu*BySy_Iz)^T*ENy_Iz + ...
  ierr = MatAXPY(D.A,D.beta,Hinvy_IzxBySy_IzTxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // + alphaS*Iy_Hinvz*Iy_E0z*D.G*Iy_BzSz + ...
  ierr = MatAXPY(D.A,D.alphaS,Iy_HinvxIy_E0zxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // + alphaD*Iy_Hinvz*Iy_ENz*D.G*Iy_BzSz
  ierr = MatAXPY(D.A,D.alphaD,Iy_HinvxIy_ENzxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

#if DEBUG > 0
  checkMatrix(&D.A,D.debugFolder,"A",&D);CHKERRQ(ierr);
#endif

  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  D.fullLinOps = MPI_Wtime() - startTime;
  D.arrLinOps = MPI_Wtime() - startArrLinOps;

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
      SETERRQ(PETSC_COMM_WORLD,1,"SBP order not understood.");
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


PetscErrorCode SBPopsArrays(PetscInt ORDER, PetscInt N,PetscScalar scale, PetscScalar *Hinv,PetscScalar *S,PetscInt *Slen)
{
  PetscErrorCode ierr = 0;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function SBPopsArrays in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  switch ( ORDER ) {
    case 2:
      std::fill_n(Hinv, N, scale);
      Hinv[0] = 2.0*scale;
      Hinv[N-1] = 2.0*scale;

      S[0]=1.5*scale;S[1]=-2*scale;S[2]=0.5*scale;
      S[3]=0.5*scale;S[4]=-2*scale;S[5]=1.5*scale;
      *Slen = 3;

      break;

    case 4:
      if (N<8) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      std::fill_n(Hinv,N,scale);
      Hinv[0] = scale*48.0/17.0;
      Hinv[1] = scale*48.0/59.0;
      Hinv[2] = scale*48.0/43.0;
      Hinv[3] = scale*48.0/49.0;
      Hinv[N-4] = scale*48.0/49.0;
      Hinv[N-3] = scale*48.0/43.0;
      Hinv[N-2] = scale*48.0/59.0;
      Hinv[N-1] = scale*48.0/17.0;

      S[0]=11.0/6.0*scale;S[1]=-3.0*scale;S[2]=1.5*scale;S[3]=-scale/3.0;
      S[4]=-scale/3.0;S[5]=1.5*scale;S[6]=-3.0*scale;S[7]=11.0/6.0*scale;
      *Slen = 4;

      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,1,"ORDER not understood.");
      break;
  }



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function SBPoperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  return ierr;
}
