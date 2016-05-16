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


int runTests1D()
{
  PetscErrorCode ierr = 0;
  PetscInt N = 21,dof=1;
  PetscScalar dz = 1.0;

  DM da;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,dof,2,NULL,&da);
  PetscInt zn,zS,dim;
  DMDAGetCorners(da, &zS, 0, 0, &zn, 0, 0);
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

  MatMult(mat,f,dfdz);
  VecView(dfdz,PETSC_VIEWER_STDOUT_WORLD);

  return ierr;
}


int runTests2D()
{
  PetscErrorCode ierr = 0;
  PetscInt Ny = 5, Nz = 6,dof=1,sw=2; // degrees of freedom, stencil width
  PetscScalar dy =1.0, dz = 1.0;

  DM da;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nz,Ny,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);
  PetscInt yn,yS,zn,zS,dim;
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);

  Vec f;
  DMCreateGlobalVector(da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_test,Nz,dy,dz,da);
  VecView(f,PETSC_VIEWER_STDOUT_WORLD);

  Vec dfdz;
  VecDuplicate(f,&dfdz); PetscObjectSetName((PetscObject) dfdz, "dfdz");
  VecSet(dfdz,0.0);

  Mat mat;
  MatCreate(PETSC_COMM_WORLD,&mat);
  DMDAGetCorners(da, 0, 0, 0, &zn, &yn, 0);
  MatSetSizes(mat,dof*zn*yn,dof*zn*yn,PETSC_DETERMINE,PETSC_DETERMINE); // be sure to set with DMDAGetCorners!!
  MatSetFromOptions(mat);
  PetscInt diag=Ny*Nz,offDiag=Ny*Nz;
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);


  // stuff necessary to use MatSetValuesStencil (which takes global natural ordering)
  ISLocalToGlobalMapping map;
  DMGetLocalToGlobalMapping(da,&map);
  MatSetLocalToGlobalMapping(mat,map,map);
  DMDAGetGhostCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt dims[2] = {zn,yn};
  PetscInt starts[2] = {zS,yS};
  MatSetStencil(mat,dim,dims,starts,1); // be sure to set with DMDAGetGhostCorners!!

  // create Dz matrix (2nd order accuracy)
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  MatStencil row,col[2];
  PetscScalar v[2] = {0.0,0.0};
  //~for (PetscInt yI = yS; yI < yS + yn; yI++) {
    //~for (PetscInt zI = zS; zI < zS + zn; zI++) {
      //~row.i = zI; row.j = yI;

      //~// interior
      //~if (zI > 0 && zI < Nz-1) {
        //~col[0].i = zI-1; col[0].j = yI; v[0] = -0.5/dz;
        //~col[1].i = zI+1; col[1].j = yI; v[1] = 0.5/dz;
        //~MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      //~}
      //~else if (zI == 0) { // boundary
        //~col[0].i = zI; col[0].j = yI; v[0] = -1.0/dz;
        //~col[1].i = zI+1; col[1].j = yI; v[1] = 1.0/dz;
        //~MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      //~}
      //~else if (zI == Nz-1) { // boundary
        //~col[0].i = zI-1; col[0].j = yI; v[0] = -1.0/dz;
        //~col[1].i = zI; col[1].j = yI; v[1] = 1.0/dz;
        //~MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      //~}
    //~}
  //~}
  //~MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  //~MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
  //~MatView(mat,PETSC_VIEWER_STDOUT_WORLD);

  // create Dy matrix (2nd order accuracy)
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  for (PetscInt yI = yS; yI < yS + yn; yI++) {
    for (PetscInt zI = zS; zI < zS + zn; zI++) {
      row.i = zI; row.j = yI;

      // interior
      if (yI > 0 && yI < Ny-1) {
        col[0].i = zI; col[0].j = yI-1; v[0] = -0.5/dy;
        col[1].i = zI; col[1].j = yI+1; v[1] = 0.5/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
      else if (yI == 0) { // boundary
        col[0].i = zI; col[0].j = yI; v[0] = -1.0/dy;
        col[1].i = zI; col[1].j = yI+1; v[1] = 1.0/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
      else if (yI == Ny-1) { // boundary
        col[0].i = zI; col[0].j = yI-1; v[0] = -1.0/dy;
        col[1].i = zI; col[1].j = yI; v[1] = 1.0/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
  MatView(mat,PETSC_VIEWER_STDOUT_WORLD);

  MatMult(mat,f,dfdz);
  VecView(dfdz,PETSC_VIEWER_STDOUT_WORLD);

  return ierr;
}


int runTimingTest(const char * inputFile)
{
  PetscErrorCode ierr = 0;
  Domain d(inputFile);

  PetscInt Ny = d._Ny, Nz = d._Nz, dof=1, sw=1; // degrees of freedom, stencil width
  PetscScalar dy =d._dy, dz = d._dz;

  DM da;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nz,Ny,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);
  PetscInt yn,yS,zn,zS,dim;
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, 0,0,0,0,0,0);

  Vec f;
  DMCreateGlobalVector(da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_test,Nz,dy,dz,da);

  Vec dfdz;
  VecDuplicate(f,&dfdz); PetscObjectSetName((PetscObject) dfdz, "dfdz");
  VecSet(dfdz,0.0);

  Mat mat;
  MatCreate(PETSC_COMM_WORLD,&mat);
  DMDAGetCorners(da, 0, 0, 0, &zn, &yn, 0);
  MatSetSizes(mat,dof*zn*yn,dof*zn*yn,PETSC_DETERMINE,PETSC_DETERMINE); // be sure to set with DMDAGetCorners!!
  MatSetFromOptions(mat);
  PetscInt diag=2,offDiag=2;
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);

  // stuff necessary to use MatSetValuesStencil (which takes global natural ordering)
  ISLocalToGlobalMapping map;
  DMGetLocalToGlobalMapping(da,&map);
  MatSetLocalToGlobalMapping(mat,map,map);
  DMDAGetGhostCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt dims[2] = {zn,yn};
  PetscInt starts[2] = {zS,yS};
  MatSetStencil(mat,dim,dims,starts,1); // be sure to set with DMDAGetGhostCorners!!

  // create Dz matrix (2nd order accuracy)
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  MatStencil row,col[2];
  PetscScalar v[2] = {0.0,0.0};
  // create Dy matrix (2nd order accuracy)
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  for (PetscInt yI = yS; yI < yS + yn; yI++) {
    for (PetscInt zI = zS; zI < zS + zn; zI++) {
      row.i = zI; row.j = yI;

      // interior
      if (yI > 0 && yI < Ny-1) {
        col[0].i = zI; col[0].j = yI-1; v[0] = -0.5/dy;
        col[1].i = zI; col[1].j = yI+1; v[1] = 0.5/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
      else if (yI == 0) { // boundary
        col[0].i = zI; col[0].j = yI; v[0] = -1.0/dy;
        col[1].i = zI; col[1].j = yI+1; v[1] = 1.0/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
      else if (yI == Ny-1) { // boundary
        col[0].i = zI; col[0].j = yI-1; v[0] = -1.0/dy;
        col[1].i = zI; col[1].j = yI; v[1] = 1.0/dy;
        MatSetValuesStencil(mat,1,&row,2,col,v,INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

  // print data about run
  PetscMPIInt np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  PetscPrintf(PETSC_COMM_WORLD,"np = %i, Ny = %i\n",np,d._Ny);

  { double startTime = MPI_Wtime();
  for (int ind=0; ind < 99; ind++) { MatMult(mat,f,dfdz); }
  double endTime = MPI_Wtime() - startTime;
  PetscPrintf(PETSC_COMM_WORLD,"DMDA mat time: %.9e\n",endTime); }

  // perform stencil-based derivative
  { SbpOps_sc sbp_sc(d,d._muVecP,"Neumann","Dirichlet","Neumann","Dirichlet","yz");
  double startTime = MPI_Wtime();
  for (int ind=0; ind < 99; ind++) { sbp_sc.Dy(f,dfdz); }
  double endTime = MPI_Wtime() - startTime;
  PetscPrintf(PETSC_COMM_WORLD,"stencil time: %.9e\n",endTime); }


  // non-DMDA matrix method
  Vec g;
  VecCreate(PETSC_COMM_WORLD,&g);
  VecSetSizes(g,PETSC_DECIDE,d._Ny*d._Nz);
  VecSetFromOptions(g);     PetscObjectSetName((PetscObject) g, "g");
  VecSet(g,0.0);
  mapToVec(g,MMS_test,Nz,dy,dz);

  Vec dgdz;
  VecDuplicate(g,&dgdz); PetscObjectSetName((PetscObject) dgdz, "dgdz");
  VecSet(dgdz,0.0);

  { SbpOps_c sbp_c(d,d._muVecP,"Neumann","Dirichlet","Neumann","Dirichlet","yz");
  double startTime = MPI_Wtime();
  for (int ind=0; ind < 99; ind++) { sbp_c.Dy(g,dgdz); }
  double endTime = MPI_Wtime() - startTime;
  PetscPrintf(PETSC_COMM_WORLD,"non DMDA mat time: %.9e\n",endTime); }

  VecDestroy(&f);
  VecDestroy(&dfdz);
  VecDestroy(&g);
  VecDestroy(&dgdz);
  PetscPrintf(PETSC_COMM_WORLD,"========================================\n\n\n");
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
  //~ ierr = obj->view();CHKERRQ(ierr);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "test.in"; }

  {
    Domain domain(inputFile);
    if (!domain._shearDistribution.compare("mms")) { runMMSTests(inputFile); }
    else { runEqCycle(inputFile); }
  }

  //~runTests1D();
  //~runTests2D();
  //~runTimingTest(inputFile);


  PetscFinalize();
  return ierr;
}

