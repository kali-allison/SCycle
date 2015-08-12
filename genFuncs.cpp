#include "genFuncs.hpp"

// Print out a vector with 15 significant figures.
void printVec(Vec vec)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v;
  VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec,1,&Ii,&v);
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsDiff(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Ii,&v1);
    VecGetValues(vec2,1,&Ii,&v2);
    v = v1 - v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsSum(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Ii,&v1);
    VecGetValues(vec2,1,&Ii,&v2);
    v = v1 + v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}


// Write vec to the file loc in binary format.
// Note that due to a memory problem in PETSc, looping over this many
// times will result in an error.
PetscErrorCode writeVec(Vec vec,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}


// Print all entries of 2D DMDA global vector to stdout, including which
// processor each entry lives on, and the corresponding subscripting
// indices
PetscErrorCode printf_DM_2d(const Vec gvec, const DM dm)
{
    PetscErrorCode ierr = 0;
#if VERBOSE > 2
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::printf_DM_2d in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  DMDAGetCorners(dm,&mStart,&nStart,0,&m,&n,0);

  PetscScalar **gxArr;
  DMDAVecGetArray(dm,gvec,&gxArr);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      PetscPrintf(PETSC_COMM_SELF,"%i: gxArr[%i][%i] = %g\n",
        rank,j,i,gxArr[j][i]);
    }
  }
  DMDAVecRestoreArray(dm,gvec,&gxArr);

#if VERBOSE > 2
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::printf_DM_2d in fault.cpp.\n");
#endif
  return ierr;
}



