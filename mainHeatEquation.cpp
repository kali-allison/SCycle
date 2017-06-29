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
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "linearElastic.hpp"
#include "heatequation.hpp"



using namespace std;

/*
int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-3s %-10s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","L2u","log2(L2u)","L2sigmaxy","log2(L2sigmaxy)");
  for(PetscInt Ny=11;Ny<2562;Ny=(Ny-1)*2+1)
  {
    Domain domain(inputFile,Ny,Ny);
    //~ Domain domain(inputFile,Ny,1);
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
}*/



int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();


  //~ SymmLinearElastic sle(domain);
  //~ sle.writeContext();
  //~ PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  //~ ierr = sle.integrate();CHKERRQ(ierr);
  //~ ierr = sle.view();CHKERRQ(ierr);

  HeatEquation he(domain);
  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  he.writeContext();
  ierr = he.integrate();CHKERRQ(ierr);
  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "test.in"; }

  runEqCycle(inputFile);

  //~ {
    //~ PetscMPIInt localRank;
    //~ MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: hi!\n", localRank);
    //~ Domain domain(inputFile);
    //~ if (!domain._shearDistribution.compare("mms")) { runMMSTests(inputFile); }
    //~ else { runEqCycle(inputFile); }
  //~ }


  PetscFinalize();
  return ierr;
}

