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
  PetscInt N = 21,dof=1;
  PetscScalar dz = 1.0;

  DM da;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,dof,2,NULL,&da);
  PetscInt zn,zS,dim;
  DMDAGetCorners(da, &zS, 0, 0, &zn, 0, 0);
  PetscInt zE = zS + zn;
  DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);

  Vec f;
  DMCreateGlobalVector(da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_test,N,dz,da);
  //~VecView(f,PETSC_VIEWER_STDOUT_WORLD);

  Vec dfdz;
  VecDuplicate(f,&dfdz); PetscObjectSetName((PetscObject) dfdz, "dfdz");
  VecSet(dfdz,0.0);

  Mat mat;
  MatCreate(PETSC_COMM_WORLD,&mat);
  DMDAGetCorners(da, 0, 0, 0, &zn, 0, 0);
  MatSetSizes(mat,dof*zn,dof*zn,PETSC_DETERMINE,PETSC_DETERMINE);
  MatSetFromOptions(mat);
  PetscInt diag=5,offDiag=5;
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);


  // stuff necessary to use MatSetValuesStencil (which takes global natural ordering)
  ISLocalToGlobalMapping map;
  DMGetLocalToGlobalMapping(da,&map);
  MatSetLocalToGlobalMapping(mat,map,map);
  DMDAGetGhostCorners(da, &zS, 0, 0, &zn, 0, 0);
  PetscInt dims = zn;
  PetscInt starts = zS;
  MatSetStencil(mat,dim,&dims,&starts,1); // be sure to set with DMDAGetGhostCorners!!

  // create 1st derivative matrix (2nd order accuracy)
  PetscMPIInt localRank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
  DMDAGetCorners(da, &zS, 0, 0, &zn, 0, 0);
  MatStencil row,col[2];
  PetscScalar v[2] = {0.0, 0.0};
  for (PetscInt zI = zS; zI < zS + zn; zI++) {
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
      PetscPrintf(PETSC_COMM_WORLD,"here!!\n");
      col[0].i = zI-1; v[0] = -1.0/dz;
      col[1].i = zI; v[1] = 1.0/dz;
      MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
    }
  }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
  MatView(mat,PETSC_VIEWER_STDOUT_WORLD);

  //~PetscMPIInt localRank;
  //~MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
  //~DMDALocalInfo info;
  //~DMDAGetLocalInfo(da,&info);
  //~PetscPrintf(PETSC_COMM_SELF,"[%i] da | zS = %i, zN = %i, zE = %i, info.sw=%i\n",localRank,zS,zn,zE,info.sw);
  //~PetscInt rg,cg,rl,cl;
  //~MatGetSize(mat,&rg,&cg);
  //~MatGetSize(mat,&rl,&cl);
  //~PetscPrintf(PETSC_COMM_SELF,"[%i] mat size | local: %i x %i, global: %i x %i\n",localRank,rl,cl,rg,cg);
  //~PetscInt vg,vl;
  //~VecGetSize(f,&vg);
  //~VecGetLocalSize(f,&vl);
  //~PetscPrintf(PETSC_COMM_SELF,"[%i] vec size | local: %i, global: %i\n",localRank,vl,vg);


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

