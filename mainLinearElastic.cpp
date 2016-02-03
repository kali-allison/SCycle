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

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-10s %-10s %-22s\n",
             "Ny","dy","err2","log2(err2)");
  for(PetscInt Ny=11;Ny<82;Ny=(Ny-1)*2+1)
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

  Domain D(inputFile);
  mapToVec(D._muVP,MMS_mu,D._Nz,D._dy,D._dz,D._da);

  SbpOps_sc s(D,*D._muArrPlus,D._muP);
  SbpOps_c m(D,*D._muArrPlus,D._muP);

  Vec f;
  DMCreateGlobalVector(D._da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_uA,D._Nz,D._dy,D._dz,5,D._da);
  //~mapToVec(f,MMS_test,D._Nz,D._dy,D._dz,D._da);
  printVec(f,D._da);
  //~VecView(f,PETSC_VIEWER_STDOUT_WORLD);

  Vec dmdag;
  VecDuplicate(f,&dmdag); PetscObjectSetName((PetscObject) dmdag, "dmdag");
  VecSet(dmdag,0.0);

  s.Dy(f,dmdag);
  VecView(dmdag,PETSC_VIEWER_STDOUT_WORLD);

  // compare the effect of matrix derivatives and stencils on DMDA Vecs
  Vec matg;
  VecDuplicate(f,&matg); PetscObjectSetName((PetscObject) matg, "matg");
  VecSet(matg,0.0);

  s.Dy(f,matg);
  VecView(dmdag,PETSC_VIEWER_STDOUT_WORLD);


  //~// try to develop a function to perform the array stuff
  //~PetscInt m_D1close=1,n_D1close=3;
  //~PetscScalar D1closS[1][3] = {{-1.5, 2.0, -0.5}};



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

