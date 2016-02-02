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

  SbpOps_sc sbp(D,*D._muArrPlus,D._muP);

  Vec f;
  DMCreateGlobalVector(D._da,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);
  mapToVec(f,MMS_uA,D._Nz,D._dy,D._dz,5,D._da);
  //~mapToVec(f,MMS_test,D._Nz,D._dy,D._dz,D._da);
  //~printVec(f,D._da);

  Vec g;
  VecDuplicate(f,&g); PetscObjectSetName((PetscObject) g, "g");
  VecSet(g,0.0);

  //~sbp.Dy(f,g);
  //~VecView(g,PETSC_VIEWER_STDOUT_WORLD);

  //~sbp.muxDz(f,g);
  //~sbp.muxDy(f,g);
  //~VecView(g,PETSC_VIEWER_STDOUT_WORLD);

  //~sbp.Dzxmu(f,g);
  sbp.Dyxmu(f,g);
  VecView(g,PETSC_VIEWER_STDOUT_WORLD);

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

