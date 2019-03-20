#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "genFuncs.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"
#include "spmat.hpp"

using namespace std;

PetscErrorCode setFields(const int Ny, const int Nz, const double Ly, const double Lz, Vec& _y, Vec& _z)
{
  PetscErrorCode ierr = 0;

  Vec _q, _r;
  PetscScalar _dq = 1.0/(Ny-1.);
  PetscScalar _dr = 1.0/(Nz-1.);

  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,Ny*Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);

  VecDuplicate(_y,&_z); PetscObjectSetName((PetscObject) _z, "z");
  VecDuplicate(_y,&_q); PetscObjectSetName((PetscObject) _q, "q");
  VecDuplicate(_y,&_r); PetscObjectSetName((PetscObject) _r, "r");

  // construct coordinate transform
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar *y,*z,*q,*r;
  VecGetArray(_y,&y);
  VecGetArray(_z,&z);
  VecGetArray(_q,&q);
  VecGetArray(_r,&r);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    q[Jj] = _dq*(Ii/Nz);
    r[Jj] = _dr*(Ii-Nz*(Ii/Nz));

    z[Jj] = r[Jj]*Lz;
    double bCoordTrans = 5.0;
    y[Jj] = Ly * sinh(bCoordTrans*q[Jj])/sinh(bCoordTrans);

    Jj++;
  }
  VecRestoreArray(_y,&y);
  VecRestoreArray(_z,&z);
  VecRestoreArray(_q,&q);
  VecRestoreArray(_r,&r);

  VecDestroy(&_q);
  VecDestroy(&_r);
  return ierr;
}


int m7()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;
  PetscScalar dy = Ly/(Ny-1.0), dz = 1/(Nz-1.0);

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // make coordinates
  Vec _y, _z;
  setFields(Ny, Nz, Ly, Lz, _y, _z);

  SbpOps_m_varGrid *sbp = new SbpOps_m_varGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setGrid(&_y,&_z);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  TempMats_m_constGrid tempMats(order,Ny,dy,Nz,dz,sbp->_compatibilityType);
  Spmat D3z(Nz,Nz);
  Spmat D4z(Nz,Nz);
  Spmat C3z(Nz,Nz);
  Spmat C4z(Nz,Nz);
  sbp_Spmat4(Nz,1/dz,D3z,D4z,C3z,C4z);

  Mat Iy_D3z = NULL;
  kronConvert(tempMats._Iy,D3z,Iy_D3z,6,0);

  int ii = 0;
  while (ii < 100)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Multiply Iy_D3z * Iy_D3z...", ii);
    Mat A;
    MatMatMult(Iy_D3z,Iy_D3z,MAT_INITIAL_MATRIX,1,&A);
    MatDestroy(&A);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  MatDestroy(&Iy_D3z);
  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&mu);
  delete sbp;

  return ierr;
}

int m6()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;
  PetscScalar dy = Ly/(Ny-1.0), dz = 1/(Nz-1.0);

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // make coordinates
  Vec _y, _z;
  setFields(Ny, Nz, Ly, Lz, _y, _z);

  SbpOps_m_varGrid *sbp = new SbpOps_m_varGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setGrid(&_y,&_z);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  TempMats_m_constGrid tempMats(order,Ny,dy,Nz,dz,sbp->_compatibilityType);
  Spmat D3z(Nz,Nz);
  Spmat D4z(Nz,Nz);
  Spmat C3z(Nz,Nz);
  Spmat C4z(Nz,Nz);
  sbp_Spmat4(Nz,1/dz,D3z,D4z,C3z,C4z);

  Mat Iy_Hz = NULL;
  kronConvert(tempMats._Iy,tempMats._Hz,Iy_Hz,1,0);


  int ii = 0;
  while (ii < 100)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Multiply Iy_Hz * Iy_Hz...", ii);
    Mat A;
    MatMatMult(Iy_Hz,Iy_Hz,MAT_INITIAL_MATRIX,1,&A);
    MatDestroy(&A);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  MatDestroy(&Iy_Hz);
  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&mu);
  delete sbp;


  return ierr;
}

int m5()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;
  PetscScalar dy = Ly/(Ny-1.0), dz = 1/(Nz-1.0);

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // make coordinates
  Vec _y, _z;
  setFields(Ny, Nz, Ly, Lz, _y, _z);

  SbpOps_m_varGrid *sbp = new SbpOps_m_varGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setGrid(&_y,&_z);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  TempMats_m_constGrid tempMats(order,Ny,dy,Nz,dz,sbp->_compatibilityType);
  Spmat D3z(Nz,Nz);
  Spmat D4z(Nz,Nz);
  Spmat C3z(Nz,Nz);
  Spmat C4z(Nz,Nz);
  sbp_Spmat4(Nz,1/dz,D3z,D4z,C3z,C4z);

  int ii = 0;
  while (ii < 100)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Creating and destroying Iy_D3z...", ii);
    Mat Iy_D3z = NULL;
    kronConvert(tempMats._Iy,D3z,Iy_D3z,6,0);
    MatDestroy(&Iy_D3z);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&mu);
  delete sbp;


  return ierr;
}

int m4()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;
  PetscScalar dy = Ly/(Ny-1.0), dz = 1/(Nz-1.0);

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // make coordinates
  Vec _y, _z;
  setFields(Ny, Nz, Ly, Lz, _y, _z);

  SbpOps_m_varGrid *sbp = new SbpOps_m_varGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setGrid(&_y,&_z);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  TempMats_m_constGrid tempMats(order,Ny,dy,Nz,dz,sbp->_compatibilityType);

  int ii = 0;
  while (ii < 100)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Creating and destroying Iy_Hz...", ii);
    Mat Iy_Hz = NULL;
    kronConvert(tempMats._Iy,tempMats._Hz,Iy_Hz,1,0);
    MatDestroy(&Iy_Hz);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&mu);
  delete sbp;


  return ierr;
}


int m3()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // make coordinates
  Vec _y, _z;
  setFields(Ny, Nz, Ly, Lz, _y, _z);

  SbpOps_m_varGrid *sbp = new SbpOps_m_varGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setGrid(&_y,&_z);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  int ii = 0;
  while (1)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Calling SbpOps_m_varGrid::updateVarCoeff...", ii);
    sbp->updateVarCoeff(mu);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&mu);
  delete sbp;


  return ierr;
}

int m2()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  // construct matrices for the first time
  SbpOps_m_constGrid *sbp = new SbpOps_m_constGrid(order,Ny,Nz,Ly,Lz,mu);
  sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
  sbp->setMultiplyByH(1);
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  int ii = 0;
  while (1)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Calling updateVarCoeff...", ii);
    sbp->updateVarCoeff(mu);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
    ii++;
  }

  VecDestroy(&mu);
  delete sbp;


  return ierr;
}


int m1()
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"Initiating context.\n");

  PetscInt order = 4;
  PetscInt Ny = 1, Nz = 102;
  PetscScalar Ly = 50, Lz = 60;

  // make variable coefficient
  Vec mu;
  VecCreate(PETSC_COMM_WORLD,&mu);
  VecSetSizes(mu,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(mu);
  PetscObjectSetName((PetscObject) mu, "mu");
  VecSet(mu,30.0);

  int ii = 0;
  while (1)
  {
    PetscPrintf(PETSC_COMM_WORLD,"%i: Creating matrices...", ii);
    SbpOps *sbp;
    sbp = new SbpOps_m_constGrid(order,Ny,Nz,Ly,Lz,mu);

    sbp->setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann");
    sbp->setMultiplyByH(1);
    sbp->setDeleteIntermediateFields(1);
    sbp->computeMatrices(); // actually create the matrices

    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");

    delete sbp;
    ii++;
  }

  VecDestroy(&mu);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  // These are the functions we want to work, but they leak memory.
  // For tests, I especially want m2 to work.
  // m3 should be the same in terms of memory behavior, it's there as an example.
  //~ m1(); // construct/destruct entire SbpOps_m_constGrid
  //~ m2(); // updateVariableCoefficient
  //~ m3(); // updateVariableCoefficient with variable grid spacing

  // To figure out why m1 - m3 fail, these are the tests I ran.
  // I'm pretty sure the problem is m5, but you should double check that.
  // m4(); // create Mat Iy_Hz; this one doesn't leak
   m5(); // create Mat Iy_D3z; leaks memory

  // I'm also pretty sure this leaks, and I have no idea why that would be.
  // It's somewhat analogous to creating the full A matrix.
  // m6(); // multiply Iy_Hz*Iy_Hz; shouldn't leak (can't remember if I tested this)
  // m7(); // multiply Iy_D3z*Iy_D3z; leaks

  PetscFinalize();
  return ierr;
}
