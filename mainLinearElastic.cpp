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

  //~// compare the effect of matrix derivatives and stencils on DMDA Vecs
  //~Vec matg;
  //~VecDuplicate(f,&matg); PetscObjectSetName((PetscObject) matg, "matg");
  //~VecSet(matg,0.0);

  //~m.Dy(f,matg);
  //~writeVec(matg,"data/matg");
  //~VecView(dmdag,PETSC_VIEWER_STDOUT_WORLD);






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

