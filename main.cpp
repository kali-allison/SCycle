#include <petscts.h>
#include <petscviewerhdf5.h>
#include <string>
#include <petscdmda.h>

#include "genFuncs.hpp"
#include "spmat.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "fault.hpp"
#include "linearElastic.hpp"
#include "pressureEq.hpp"
#include "powerLaw.hpp"
#include "mediator.hpp"



using namespace std;


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  d.write();

  PressureEq p(d); // pressure equation
  //~ p.writeContext();
  //~ HeatEquation he(d); // heat equation

  return ierr;
}

int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-2s %-10s %-10s %-22s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","errL2u","log2(errL2u)","errL2gxy","log2(errL2gxy)",
             "errL2gxz","log2(errL2gxz)");
  for(PetscInt Ny=11;Ny<12;Ny=(Ny-1)*2+1)
  //~ for(PetscInt Ny=11;Ny<12;Ny=(Ny-1)*2+1)
  {
    Domain d(inputFile,Ny,Ny);
    //~ Domain d(inputFile,Ny,1);
    d.write();

    Mediator m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    ierr = m.integrate();CHKERRQ(ierr);
    ierr = m.measureMMSError();CHKERRQ(ierr);
  }

  return ierr;
}


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  d.write();

  Mediator m(d);
  ierr = m.writeContext(); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  //~ ierr = m.integrate(); CHKERRQ(ierr);
  ierr = m.integrate_SS(); CHKERRQ(ierr);
  //~ ierr = m.view();CHKERRQ(ierr);

  return ierr;
}


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  const char * inputFile;
  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "init.in"; }

  {
    Domain domain(inputFile);
    if (domain._isMMS) { runMMSTests(inputFile); }
    else { runEqCycle(inputFile); }
  }

  //~ runTests(inputFile);

  PetscFinalize();
  return ierr;
}

