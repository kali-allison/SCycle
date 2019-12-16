
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
#include "strikeSlip_linearElastic_qd.hpp"
#include "strikeSlip_linearElastic_fd.hpp"
#include "strikeSlip_linearElastic_qd_fd.hpp"
#include "strikeSlip_powerLaw_qd.hpp"
#include "strikeSlip_powerLaw_qd_fd.hpp"

using namespace std;


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain D(inputFile);

  string outputDir = "";
  string filename = "test.txt";
  //~ PetscFileMode mode = D._outFileMode;
  PetscViewer viewer;
  string format = "%.15e\n";


  PetscViewerCreate(PETSC_COMM_WORLD,&viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetName(viewer, filename.c_str());
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);

  PetscScalar var = 0.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);

  var = 1.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);


  PetscViewerDestroy(&viewer);
  PetscViewerCreate(PETSC_COMM_WORLD,&viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_APPEND);
  PetscViewerFileSetName(viewer, filename.c_str());

  var = 2.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);


/*
  //~ ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, (outputDir + filename).c_str(), &viewer);
  //~ ierr = PetscViewerFileSetMode(viewer, mode); CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);

  PetscScalar var = 0.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);

  // ensure that viewer mode switches to append if it isn't that already
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_APPEND); CHKERRQ(ierr);

  var = 1.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, (outputDir + filename).c_str(), &viewer);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_APPEND); CHKERRQ(ierr);
  var = 2.0;
  ierr = PetscViewerASCIIPrintf(viewer, format.c_str(), var);CHKERRQ(ierr);
*/

  //~ VecScatter* _body2fault = &(D._scatters["body2L"]);
  //~ Fault_qd _fault_qd(D,D._scatters["body2L"],2.0); // fault for quasidynamic problem
  //~ Fault_fd _fault_fd(D, D._scatters["body2L"],2.0); // fault for fully dynamic problem
  //~ HeatEquation _he(D);
  //~ LinearElastic _material(D,"Dirichlet","Neumann","Dirichlet","Neumann");
  //~ PowerLaw _material(D,_he,"Dirichlet","Neumann","Dirichlet","Neumann");

  //~ strikeSlip_linearElastic_fd m(D);
  //~ strikeSlip_linearElastic_qd_fd m(D);
  //~ StrikeSlip_PowerLaw_qd m(D);
  //~ StrikeSlip_PowerLaw_qd_fd m(D);

  return ierr;
}


int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-2s %-10s %-10s %-22s %-10s %-22s %-10s %-22s\n", "ord","Ny","dy","errL2u","log2(errL2u)","errL2gxy","log2(errL2gxy)", "errL2gxz","log2(errL2gxz)");

for (PetscInt Ny = 11; Ny < 82; Ny = (Ny - 1) * 2 + 1)
  // for(PetscInt Ny=81;Ny<82;Ny=(Ny-1)*2+1)
  // for(PetscInt Ny=11;Ny<12;Ny=(Ny-1)*2+1)
  {
    Domain d(inputFile,Ny,Ny);
    // Domain d(inputFile,Ny,1);
    d.write();

    // Mediator m(d);
    // ierr = m.writeContext(); CHKERRQ(ierr);
    // ierr = m.integrate();CHKERRQ(ierr);
    // ierr = m.measureMMSError();CHKERRQ(ierr);
  }

  return ierr;
}


// calculate Green's functions and write to file "G"
// also write bcL and surfDisp into file
int computeGreensFunction(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  // create domain object and write scalar fields into file
  Domain d(inputFile);
  d.write();

  // create linear elastic object using domain (includes material properties) specifications
  LinearElastic le(d,"Dirichlet","Neumann","Dirichlet","Neumann");
  Mat A;
  le._sbp->getA(A);
  le.setupKSP(le._ksp,le._pc,A);

  // set up boundaries
  VecSet(le._bcT,0.0);
  VecSet(le._bcB,0.0);
  VecSet(le._bcR,0.0);

  // prepare matrix to hold greens function
  Mat G;
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,d._Ny,d._Nz,NULL,&G);
  MatSetUp(G);

  PetscInt *rows,*cols;
  PetscMalloc1(d._Ny,&rows);
  PetscMalloc1(d._Ny,&cols);
  PetscScalar *si;

  // loop over elements of bcL and compute corresponding entry of G
  PetscScalar v = 1.0;
  PetscInt Istart,Iend;
  VecGetOwnershipRange(le._bcL,&Istart,&Iend);

  for(PetscInt Ii = Istart;Ii < Iend;Ii++) {
    PetscPrintf(PETSC_COMM_WORLD,"Ii = %i\n",Ii);
    VecSet(le._bcL,0.0);
    VecSetValue(le._bcL,Ii,v,INSERT_VALUES);

    // solve for displacement
    ierr = le._sbp->setRhs(le._rhs,le._bcL,le._bcR,le._bcT,le._bcB); CHKERRQ(ierr);
    ierr = KSPSolve(le._ksp,le._rhs,le._u);
    ierr = le.setSurfDisp();

    // assign values to G
    VecGetArray(le._surfDisp,&si);
    for(PetscInt ind = 0; ind < d._Ny; ind++) {
      rows[ind]=ind;
    }
    for(PetscInt ind = 0; ind < d._Ny; ind++) {
      cols[ind]=Ii;
    }
    MatSetValues(G,d._Ny,rows,1,&Ii,si,INSERT_VALUES);
    MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);
    VecRestoreArray(le._bcL,&si);
  }

  // output greens function
  string filename;
  filename =  d._outputDir + "G";
  writeMat(G, filename);

/*
  // output testing stuff
  VecSet(le._bcL,0.0);
  VecSet(le._bcR,5.0);
  for(PetscInt Ii = Istart; Ii < Iend; Ii++) {
    v = ((PetscScalar) Ii+1)/((PetscScalar) d._Nz);
    VecSetValue(le._bcL,Ii,v,INSERT_VALUES);
  }

  // solve for displacement
  ierr = le._sbp->setRhs(le._rhs,le._bcL,le._bcR,le._bcT,le._bcB);CHKERRQ(ierr);
  ierr = KSPSolve(le._ksp,le._rhs,le._u);CHKERRQ(ierr);
  ierr = le.setSurfDisp();

  str =  d._outputDir + "bcL";
  writeVec(le._bcL,str.c_str());

  str =  d._outputDir + "surfDisp";
  writeVec(le._surfDisp,str.c_str());
  */

  // write left boundary condition and surface displacement into file
  filename =  d._outputDir + "bcL";
  writeVec(le._bcL, filename);
  filename =  d._outputDir + "surfDisp";
  writeVec(le._surfDisp, filename);

  // free memory
  MatDestroy(&G);
  PetscFree(rows);
  PetscFree(cols);
  return ierr;
}


// run different earthquake cycle scenarios depending on input
int runEqCycle(Domain& d)
{
  PetscErrorCode ierr = 0;

  // quasi-dynamic earthquake cycle simulation
  // with a vertical strike-slip fault, and linear elastic off-fault material
  if (d._bulkDeformationType.compare("linearElastic") == 0 && d._momentumBalanceType.compare("quasidynamic") == 0) {
    StrikeSlip_LinearElastic_qd m(d);
    if (d._ckptNumber < 1) { ierr = m.writeContext(); CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  // single fully dynamic earthquake simulation
  // with a vertical strike-slip fault, and linear elastic off-fault material
  if (d._bulkDeformationType.compare("linearElastic") == 0 && d._momentumBalanceType.compare("dynamic") == 0) {
    strikeSlip_linearElastic_fd m(d);
    //~ if (d._ckptNumber < 1) { ierr = m.writeContext(); CHKERRQ(ierr); }
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  // quasi-dynamic earthquake cycle simulation
  // with a vertical strike-slip fault, and power-law viscoelastic off-fault material
  if (d._bulkDeformationType.compare("powerLaw") == 0 && d._momentumBalanceType.compare("quasidynamic") == 0) {
    StrikeSlip_PowerLaw_qd m(d);
    //~ if (d._ckptNumber < 1) { ierr = m.writeContext(); CHKERRQ(ierr); }
    ierr = m.writeContext();
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  // fixed point iteration for power-law viscoelastic simulation with a vertical strike-slip fault
  if (d._bulkDeformationType.compare("powerLaw") == 0 && d._momentumBalanceType.compare("steadyStateIts") == 0) {
    StrikeSlip_PowerLaw_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrateSS(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  // earthquake cycle simulation, with fully dynamic earthquakes and quasi-dynamic interseismic periods
  // with a vertical strike-slip fault, and linear elastic off-fault material
  if (d._bulkDeformationType.compare("linearElastic") == 0 && d._momentumBalanceType.compare("quasidynamic_and_dynamic") == 0) {
    strikeSlip_linearElastic_qd_fd m(d);
    if (d._ckptNumber < 1) { ierr = m.writeContext(); CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  // earthquake cycle simulation, with fully dynamic earthquakes and quasi-dynamic interseismic periods
  // with a vertical strike-slip fault, and viscoelastic off-fault material
  if (d._bulkDeformationType.compare("powerLaw") == 0 && d._momentumBalanceType.compare("quasidynamic_and_dynamic") == 0) {
    StrikeSlip_PowerLaw_qd_fd m(d);
    if (d._ckptNumber < 1) { ierr = m.writeContext(); CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m.integrate(); CHKERRQ(ierr);
    ierr = m.view(); CHKERRQ(ierr);
  }

  return ierr;
}


int main(int argc,char **args)
{
  PetscErrorCode ierr = 0;
  PetscInitialize(&argc,&args,NULL,NULL);

  const char * inputFile;

  if (argc > 1) { inputFile = args[1]; }
  else { inputFile = "init.in"; }

  {
    Domain d(inputFile);
    if (d._isMMS) { runMMSTests(inputFile); }
    else { runEqCycle(d); }
    //~ computeGreensFunction(inputFile);
    //runTests(inputFile);
  }


  PetscFinalize();
  return ierr;
}
