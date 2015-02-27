#include "genFuncs.hpp"

void printVec(Vec vec)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v;
  VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec,1,&Istart,&v);
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

void printVecsDiff(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Istart,&v1);
    VecGetValues(vec2,1,&Istart,&v2);
    v = v1 - v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

void printVecsSum(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Istart,&v1);
    VecGetValues(vec2,1,&Istart,&v2);
    v = v1 + v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}
