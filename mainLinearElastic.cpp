#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "genFuncs.hpp"
#include "spmat.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_sc.hpp"
#include "fault.hpp"
#include "linearElastic.hpp"



using namespace std;

int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-3s %-10s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","L2u","log2(L2u)","L2sigmaxy","log2(L2sigmaxy)");
  for(PetscInt Ny=11;Ny<162;Ny=(Ny-1)*2+1)
  {
    Domain domain(inputFile,Ny,Ny);
    domain.write();

    LinearElastic *obj;
    if (domain._problemType.compare("symmetric")==0) {
      obj = new SymmLinearElastic(domain);
    }
    else {
      obj = new FullLinearElastic(domain);
    }

    ierr = obj->writeStep1D();CHKERRQ(ierr);
    ierr = obj->writeStep2D();CHKERRQ(ierr);

    obj->measureMMSError();
  }

  return ierr;
}


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;
  PetscInt N = 11,dof=1;
  PetscScalar dz = 1.0;

  DM da;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,dof,2,NULL,&da);
  PetscInt zn,zS,dim;
  DMDAGetGhostCorners(da, &zS, 0, 0, &zn, 0, 0);
  DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);

  Vec f;
  DMCreateGlobalVector(da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_test,N,dz,da);
  //~printVec(f,D._da);
  VecView(f,PETSC_VIEWER_STDOUT_WORLD);

  Vec dfdz;
  VecDuplicate(f,&dfdz); PetscObjectSetName((PetscObject) dfdz, "dfdz");
  VecSet(dfdz,0.0);

  Mat mat;
  MatCreate(PETSC_COMM_WORLD,&mat);
  PetscInt zE = zS + zn;
  MatSetSizes(mat,zn,zn,PETSC_DETERMINE,PETSC_DETERMINE);
  MatSetFromOptions(mat);
  PetscInt diag=5,offDiag=5;
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);


  // stuff necessary to use MatSetValuesStencil (which takes global natural ordering)
  PetscInt dims[1] = {zn};
  PetscInt starts[1] = {zS};
  MatSetStencil(mat,dim,dims,starts,1);
  ISLocalToGlobalMapping map;
  DMGetLocalToGlobalMapping(da,&map);
  MatSetLocalToGlobalMapping(mat,map,map);
  MatSetStencil(mat,dim,dims,starts,1);

  // create 1st derivative matrix (2nd order accuracy)
  MatStencil row,col[2];
  PetscScalar v[2] = {0,0};
  for (PetscInt zI = zS; zI < zE; zI++) {
    row.i = zI;
    if (zI > 0 && zI < N-1) {
      col[0].i = zI-1; v[0] = -0.5/dz;
      col[1].i = zI+1; v[1] = 0.5/dz;
      MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
    }
    else if (zI == 0) {
      col[0].i = zI; v[0] = -1.0/dz;
      col[1].i = zI+1; v[1] = 1.0/dz;
      MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
    }
    else if (zI == N-1) {
      col[0].i = zI-1; v[0] = -1.0/dz;
      col[1].i = zI; v[1] = 1.0/dz;
      MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
    }

  }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
  MatView(mat,PETSC_VIEWER_STDOUT_WORLD);

  MatMult(mat,f,dfdz);
  VecView(dfdz,PETSC_VIEWER_STDOUT_WORLD);


  return ierr;
}


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();

  LinearElastic *obj;
  if (domain._problemType.compare("symmetric")==0) {
    obj = new SymmLinearElastic(domain);
  }
  else {
    obj = new FullLinearElastic(domain);
  }
  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  ierr = obj->writeStep1D();CHKERRQ(ierr);
  ierr = obj->writeStep2D();CHKERRQ(ierr);
  ierr = obj->integrate();CHKERRQ(ierr);
  ierr = obj->view();CHKERRQ(ierr);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "test.in"; }

  //~{
    //~Domain domain(inputFile);
    //~if (!domain._shearDistribution.compare("mms")) { runMMSTests(inputFile); }
    //~else { runEqCycle(inputFile); }
  //~}

  runTests(inputFile);


  PetscFinalize();
  return ierr;
}

