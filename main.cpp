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
#include "powerLaw.hpp"
#include "pressureEq.hpp"
#include "heatEquation.hpp"
#include "linearElastic.hpp"
#include "powerLaw.hpp"
#include "iceStream_linearElastic_qd.hpp"
#include "strikeSlip_linearElastic_qd.hpp"
#include "strikeSlip_linearElastic_fd.hpp"
#include "strikeSlip_linearElastic_qd_fd.hpp"
#include "strikeSlip_powerLaw_qd.hpp"

using namespace std;


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  // d.write();

  //~ PressureEq p(d); // pressure equation
  //~ p.writeContext();
  HeatEquation he(d); // heat equation

  return ierr;
}

int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-2s %-10s %-10s %-22s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","errL2u","log2(errL2u)","errL2gxy","log2(errL2gxy)",
             "errL2gxz","log2(errL2gxz)");

for(PetscInt Ny=11;Ny<82;Ny=(Ny-1)*2+1)
  // for(PetscInt Ny=81;Ny<82;Ny=(Ny-1)*2+1)

  //~ for(PetscInt Ny=11;Ny<12;Ny=(Ny-1)*2+1)
  {
    Domain d(inputFile,Ny,Ny);
    //~ Domain d(inputFile,Ny,1);
    d.write();

    //~ Mediator m(d);
    //~ ierr = m.writeContext(); CHKERRQ(ierr);
    //~ ierr = m.integrate();CHKERRQ(ierr);
    //~ ierr = m.measureMMSError();CHKERRQ(ierr);
  }

  return ierr;
}


int runEqCycle(Domain& d)
{
  PetscErrorCode ierr = 0;

  //~ Domain d(inputFile);
  d.write();

  // solving linear elastic, quasi-dynamic simulation with a vertical strike-slip fault
  if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("linearElastic")==0 && d._momentumBalanceType.compare("quasidynamic")==0) {
    StrikeSlip_LinearElastic_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

  // solving linear elastic, dynamic simulation with a vertical strike-slip fault
  if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("linearElastic")==0 && d._momentumBalanceType.compare("dynamic")==0) {
    strikeSlip_linearElastic_fd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

  if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("linearElastic")==0 && d._momentumBalanceType.compare("quasidynamic_and_dynamic")==0) {
    strikeSlip_linearElastic_qd_fd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

  //~ if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("powerLaw")==0 && d._momentumBalanceType.compare("quasidynamic_and_dynamic")==0) {
    //~ StrikeSlip_PowerLaw_qd_fd m(d);
    //~ ierr = m.writeContext(); CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    //~ for (int i=0; i<d._numCycles; i++){
      //~ PetscPrintf(PETSC_COMM_WORLD, "Starting loop %i\n", i);
      //~ ierr = m.integrate(); CHKERRQ(ierr);
    //~ }
    //~ ierr = m.view();CHKERRQ(ierr);
  //~ }

  // solving viscoelastic, quasi-dynamic simulation with a vertical strike-slip fault
  if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("powerLaw")==0 && d._momentumBalanceType.compare("quasidynamic")==0) {
    StrikeSlip_PowerLaw_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

  // fixed point iteration for power-law viscoelastic simulation with a vertical strike-slip fault
  if (d._problemType.compare("strikeSlip")==0 && d._bulkDeformationType.compare("powerLaw")==0 && d._momentumBalanceType.compare("steadyStateIts")==0) {
    StrikeSlip_PowerLaw_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrateSS(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

  // solving linear elastic, quasi-dynamic simulation for an ice stream
  if (d._problemType.compare("iceStream")==0 && d._bulkDeformationType.compare("linearElastic")==0 && d._momentumBalanceType.compare("quasidynamic")==0) {
    IceStream_LinearElastic_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view();CHKERRQ(ierr);
  }

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
    Domain d(inputFile);
    if (d._isMMS) { runMMSTests(inputFile); }
    else { runEqCycle(d); }
  }

  //~ runTests(inputFile);

  PetscFinalize();
  return ierr;
}

