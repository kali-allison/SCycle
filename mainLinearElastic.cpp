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


/*int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain D(inputFile);
  mapToVec(D._muVP,MMS_mu,D._Nz,D._dy,D._dz,D._da);

  SbpOps_sc s(D,*D._muArrPlus,D._muP);

  Vec f;
  DMCreateGlobalVector(D._da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_uA,D._Nz,D._dy,D._dz,5,D._da);
  //~mapToVec(f,MMS_test,D._Nz,D._dy,D._dz,D._da);
  //~printVec(f,D._da);
  //~VecView(f,PETSC_VIEWER_STDOUT_WORLD);

  Vec dmdag;
  VecDuplicate(f,&dmdag); PetscObjectSetName((PetscObject) dmdag, "dmdag");
  VecSet(dmdag,0.0);

  s.Dy(f,dmdag);
  writeVec(dmdag,"data/dmdag");
  //~VecView(dmdag,PETSC_VIEWER_STDOUT_WORLD);


  // try to create matrix derivative
  Spmat Dy(D._Ny,D._Ny);
  Dy(0,0,-1.0/D._dy);Dy(0,1,1.0/D._dy); // first row
  for (int Ii=1;Ii<D._Ny-1;Ii++) {
    Dy(Ii,Ii-1,-0.5/D._dy);
    Dy(Ii,Ii+1,0.5/D._dy);
  }
  Dy(D._Ny-1,D._Ny-1,1.0/D._dy);Dy(D._Ny-1,D._Ny-2,-1.0/D._dy); // last row

  //~kronConvertDMDA(const Spmat& left,const Spmat& right,Mat& mat,PetscInt diag,PetscInt offDiag,DM dm)
  Mat mat;
  MatCreate(PETSC_COMM_WORLD,&mat);
  PetscInt zn,yn,yS,zS;
  DMDAGetGhostCorners(D._da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt zE = zS + zn;
  PetscInt yE = yS + yn;
  MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,yn*zn,yn*zn);

  MatSetFromOptions(mat);
  PetscInt diag=1,offDiag=2;
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);

  //~AO ao;
  //~DMDAGetAO(D._da,&ao);


  // stuff necessary to use MatSetValuesStencil (which takes global natural ordering)
  PetscInt dims[2] = {zn, yn};
  PetscInt starts[2] = {zS, yS};
  MatSetStencil(mat,2,dims,starts,1);
  ISLocalToGlobalMapping map;
  DMGetLocalToGlobalMapping(D._da,&map);
  MatSetLocalToGlobalMapping(mat,map,map); // do the 2 map arguments need to be different??

  // try to only set 1 value in matrix
  MatStencil row,col;
  //~row.i = 0; row.j = 0; row.k = 10; row.c = 10; // I think c and k are useless
  //~col.i = 0; col.j = 0; col.k = 10; col.c = 10;
  PetscScalar v = 1;
  //~MatSetValuesStencil(mat,1,&row,1,&col,&v,INSERT_VALUES);

  //~MatStencil row,col[2];
  //~PetscScalar v[2];
  DMDAGetCorners(D._da, &zS, &yS, 0, &zn, &yn, 0);

  PetscPrintf(PETSC_COMM_WORLD,"zS = %i, zn = %i,  yS = %i, yn = %i\n",zS,zn,yS,yn);
    for (PetscInt yI = yS; yI < yE; yI++) {
      for (PetscInt zI = zS; zI < zE; zI++) {
        row.i = yI;
        col.i = zI;
        v = yI;
        MatSetValuesStencil(mat,1,&row,1,&col,&v,INSERT_VALUES);
      }
    }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
  MatView(mat,PETSC_VIEWER_STDOUT_WORLD);


  // iterate over local matrix and set values using "local" grid numbering
  // MatSetValuesStencil


  //~// compare the effect of matrix derivatives and stencils on DMDA Vecs
  //~Vec matg;
  //~VecDuplicate(f,&matg); PetscObjectSetName((PetscObject) matg, "matg");
  //~VecSet(matg,0.0);

  //~m.Dy(f,matg);
  //~writeVec(matg,"data/matg");
  //~VecView(dmdag,PETSC_VIEWER_STDOUT_WORLD);

  return ierr;
}
*/

int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  //~Domain D(inputFile);

  //~Vec f;
  //~VecCreate(PETSC_COMM_WORLD,&f);
  //~VecSetSizes(f,PETSC_DECIDE,D._Ny*D._Nz);
  //~VecSetFromOptions(f);
  //~mapToVec(f,MMS_mu,D._Nz,D._dy,D._dz);

  //~SbpOps_sc s(D,*D._muArrPlus,D._muP);

  //~Vec dfdy;
  //~VecDuplicate(f,&dfdy);
  //~VecSet(dfdy,0.0);
  //~s.Dy(f,dfdy);

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

  {
    Domain domain(inputFile);
    if (!domain._shearDistribution.compare("mms")) { runMMSTests(inputFile); }
    else { runEqCycle(inputFile); }
  }

  //~runTests(inputFile);


  PetscFinalize();
  return ierr;
}

