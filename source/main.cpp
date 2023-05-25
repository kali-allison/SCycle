
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
#include "problemContext.hpp"
#include "strikeSlip_linearElastic_qd.hpp"
#include "strikeSlip_linearElastic_fd.hpp"
#include "strikeSlip_linearElastic_qd_fd.hpp"
#include "strikeSlip_powerLaw_qd.hpp"
#include "strikeSlip_powerLaw_qd_fd.hpp"

using namespace std;


int runTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain D(inputFile); // checked

  //~ VecScatter* _body2fault = &(D._scatters["body2L"]);
  //~ Fault_qd _fault_qd(D,D._scatters["body2L"],2.0); // fault for quasidynamic problem
  //~ Fault_fd _fault_fd(D, D._scatters["body2L"],2.0); // fault for fully dynamic problem
  //~ HeatEquation _he(D);
  //~ LinearElastic _material(D,"Dirichlet","Neumann","Dirichlet","Neumann");
  //~ PowerLaw _material(D,"Dirichlet","Neumann","Dirichlet","Neumann");

  //~ StrikeSlip_LinearElastic_fd m(D);
  //~ StrikeSlip_LinearElastic_qd_fd m(D);
  StrikeSlip_PowerLaw_qd m(D);
  //~ StrikeSlip_PowerLaw_qd_fd m(D);

  return ierr;
}

// generate data and write to file for future checkpoint experiment
int initiateFields(Vec& timeVec, Vec& solution, Vec& chkptIndex)
{
  PetscErrorCode ierr = 0;

  ierr = VecCreateMPI(PETSC_COMM_WORLD, 1, 1, &timeVec);CHKERRQ(ierr);
  ierr = VecSetBlockSize(timeVec, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) timeVec, "time");CHKERRQ(ierr);
  VecSet(timeVec,0.);

  ierr = VecCreate(PETSC_COMM_WORLD,&solution); CHKERRQ(ierr);
  ierr = VecSetSizes(solution,PETSC_DECIDE,5); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) solution, "solution");CHKERRQ(ierr);
  ierr = VecSetFromOptions(solution); CHKERRQ(ierr);
  VecSet(solution,0.);

  VecDuplicate(timeVec,&chkptIndex);
  VecSet(chkptIndex,0.);
  ierr = PetscObjectSetName((PetscObject) chkptIndex, "chkptIndex");CHKERRQ(ierr);


  return ierr;
}



// generate data and write to file for future checkpoint experiment
int runFirstStep()
{
  PetscErrorCode ierr = 0;

  //~ Domain D(inputFile);
  PetscPrintf(PETSC_COMM_WORLD,"Running first step.\n");

  // directory for output
  string outputDir = "/Users/kallison/scycle/data/";

  PetscScalar time = 0.;
  PetscInt    chkptIndex = 0; // for writing out to checkpoint file

  // prepare to output data
  PetscViewer viewer_checkpoint;
  string outFileName = outputDir + "checkpoint.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &viewer_checkpoint);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetBaseDimension2(viewer_checkpoint, PETSC_TRUE);CHKERRQ(ierr);

  PetscViewer viewer;
  outFileName = outputDir + "results.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);CHKERRQ(ierr);

  // generate data for simulation
  Vec timeVec, solution, chkptIndexVec;
  ierr = initiateFields(timeVec, solution, chkptIndexVec);CHKERRQ(ierr);
  ierr = VecSet(timeVec, time);                                          CHKERRQ(ierr);
  ierr = VecSet(solution, time);                                         CHKERRQ(ierr);

  // Write time and solution
  ierr = PetscViewerHDF5PushGroup(viewer, "/timeStepResults");           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = VecView(timeVec, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(solution, viewer);                                      CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                                CHKERRQ(ierr);

  // write checkpoint
  ierr = PetscViewerHDF5GetTimestep(viewer,&chkptIndex);                 CHKERRQ(ierr);
  ierr = VecSet(chkptIndexVec,chkptIndex);                               CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer_checkpoint, "/");               CHKERRQ(ierr);
  ierr = VecView(timeVec, viewer_checkpoint);                            CHKERRQ(ierr);
  ierr = VecView(solution, viewer_checkpoint);                           CHKERRQ(ierr);
  ierr = VecView(chkptIndexVec, viewer_checkpoint);                      CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer_checkpoint, "/time", "chkptTimeStep", PETSC_INT, &chkptIndex);
  ierr = PetscViewerHDF5PopGroup(viewer_checkpoint);                     CHKERRQ(ierr);

  // simulate writing out many time steps + occasional checkpointing
  for (int ii = 1; ii <31; ii++ )
  {
    time = (float) ii;
    VecSet(timeVec, time);
    VecSet(solution, time);
    //~ PetscPrintf(PETSC_COMM_WORLD,"ii = %i, time = %f\n",ii, time);

    PetscPrintf(PETSC_COMM_WORLD,"ii = %i, time = %0.f",ii, time);
    if (ii % 2 == 0) {
      // Write time and solution
      ierr = PetscViewerHDF5PushGroup(viewer, "/timeStepResults");       CHKERRQ(ierr);
      ierr = PetscViewerHDF5IncrementTimestep(viewer);                   CHKERRQ(ierr);
      ierr = VecView(timeVec, viewer);                                   CHKERRQ(ierr);
      ierr = VecView(solution, viewer);                                  CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);                            CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,", regular write");
    }

    if (ii % 5 == 0) {
      // write checkpoint
      PetscViewerFileSetMode(viewer_checkpoint,FILE_MODE_WRITE);
      ierr = PetscViewerHDF5GetTimestep(viewer,&chkptIndex);             CHKERRQ(ierr);
      ierr = VecSet(chkptIndexVec,chkptIndex);                           CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer_checkpoint, "/");           CHKERRQ(ierr);
      ierr = VecView(timeVec, viewer_checkpoint);                        CHKERRQ(ierr);
      ierr = VecView(solution, viewer_checkpoint);                       CHKERRQ(ierr);
      ierr = VecView(chkptIndexVec, viewer_checkpoint);                  CHKERRQ(ierr);
      ierr = PetscViewerHDF5WriteAttribute(viewer_checkpoint, "/time", "chkptTimeStep", PETSC_INT, &chkptIndex);
      ierr = PetscViewerHDF5PopGroup(viewer_checkpoint);                 CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,", chkptIndex = %i",chkptIndex);
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
  }

  PetscViewerDestroy(&viewer);
  PetscViewerDestroy(&viewer_checkpoint);
  VecDestroy(&timeVec);
  VecDestroy(&solution);
  VecDestroy(&chkptIndexVec);


  return ierr;
}

// try loading from checkpoint produced by runFirstStep
int runSecondStep()
{
  PetscErrorCode ierr = 0;

  //~ Domain D(inputFile);
  PetscPrintf(PETSC_COMM_WORLD,"Running second step.\n");

  // directory for output
  string outputDir = "/Users/kallison/scycle/data/";

  PetscInt chkptTimeStep;

  // load saved checkpoint data
  PetscViewer viewer_prev_checkpoint;
  string outFileName = outputDir + "checkpoint.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_READ, &viewer_prev_checkpoint);CHKERRQ(ierr);

  // initiate Vecs to put data into
  Vec timeVec, solution, chkptIndexVec;
  ierr = initiateFields(timeVec, solution, chkptIndexVec);               CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer_prev_checkpoint, "/");          CHKERRQ(ierr);
  ierr = VecLoad(timeVec,viewer_prev_checkpoint);                        CHKERRQ(ierr);
  ierr = VecLoad(solution,viewer_prev_checkpoint);                       CHKERRQ(ierr);
  ierr = VecLoad(chkptIndexVec,viewer_prev_checkpoint);                  CHKERRQ(ierr);
  //~ PetscErrorCode PetscViewerHDF5ReadAttribute(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *defaultValue, void *value)
  ierr = PetscViewerHDF5ReadAttribute(viewer_prev_checkpoint, "/time", "chkptTimeStep", PETSC_INT, NULL, &chkptTimeStep); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer_prev_checkpoint);                CHKERRQ(ierr);

  VecView(chkptIndexVec, PETSC_VIEWER_STDOUT_WORLD);
  VecView(timeVec, PETSC_VIEWER_STDOUT_WORLD);

  PetscPrintf(PETSC_COMM_WORLD,"chkptTimeStep = %i\n",chkptTimeStep);

  PetscViewerDestroy(&viewer_prev_checkpoint);
  VecDestroy(&timeVec);
  VecDestroy(&solution);
  VecDestroy(&chkptIndexVec);


  return ierr;
}

int testHDF5()
{
  PetscErrorCode ierr = 0;

  //~ Domain D(inputFile);
  PetscPrintf(PETSC_COMM_WORLD,"Hello!\n");

  // directory for output
  string outputDir = "/Users/kallison/scycle/data/";

  runFirstStep();

  runSecondStep();



  return ierr;
}


int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-2s %-10s %-10s %-22s %-10s %-22s %-10s %-22s\n", "ord","Ny","dy","errL2u","log2(errL2u)","errL2gxy","log2(errL2gxy)", "errL2gxz","log2(errL2gxz)");

for (PetscInt Ny = 21; Ny < 82; Ny = (Ny - 1) * 2 + 1)
  {
    Domain d(inputFile,Ny,Ny);
    // Domain d(inputFile,Ny,1);
    //~ d.write();

    StrikeSlip_LinearElastic_qd m(d);
    ierr = m.writeContext(); CHKERRQ(ierr);
    ierr = m.integrate(); CHKERRQ(ierr);

    ierr = m.view(); CHKERRQ(ierr);
    ierr = m.measureMMSError();CHKERRQ(ierr);
  }

  return ierr;
}


// calculate Green's function mapping fault slip to surface displacement
// written to file "G"
// also write bcL and surfDisp into file
int computeGreensFunction_fault(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  // create domain object and write scalar fields into file
  Domain d(inputFile);
  //~ d.write();
  PetscPrintf(PETSC_COMM_WORLD,"Running computeGreensFunction_fault\n");

  // create linear elastic object using domain (includes material properties) specifications
  LinearElastic le(d,"Dirichlet","Neumann","Dirichlet","Neumann");
  Mat A;
  le._sbp->getA(A);
  le.setupKSP(le._ksp,le._pc,A,le._linSolverSS);

  // set up boundaries
  VecSet(le._bcT,0.0);
  VecSet(le._bcB,0.0);
  VecSet(le._bcR,0.0);



  // prepare matrix to hold greens function
  Mat G;
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,d._Ny,d._Nz,NULL,&G);
  MatSetUp(G);

  PetscInt *rows;
  PetscMalloc1(d._Ny,&rows);
  PetscScalar const *si;

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
    VecGetArrayRead(le._surfDisp,&si);
    for(PetscInt ind = 0; ind < d._Ny; ind++) {
      rows[ind]=ind;
    }
    MatSetValues(G,d._Ny,rows,1,&Ii,si,INSERT_VALUES);
    MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);
    VecRestoreArrayRead(le._bcL,&si);
  }

  // output greens function
  string filename;
  filename =  d._outputDir + "G_fault";
  writeMat(G, filename);

  // free memory
  MatDestroy(&G);
  PetscFree(rows);
  return ierr;
}

// calculate Green's function to map viscous to surface displacement
// can be used to map viscous strain rate to surface velocity
int computeGreensFunction_offFault(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  // create domain object and write scalar fields into file
  Domain d(inputFile);
  PetscPrintf(PETSC_COMM_WORLD,"Running computeGreensFunction_offFault\n");

  // set up HDF5 file viewer
  PetscViewer viewer;
  string outFileName = d._outputDir + "G_offFault.h5";
  PetscFileMode outputFileMode = FILE_MODE_WRITE;
  PetscInt startIi = 0;
  // if file from pervious simulation exists, continue from where previous simulation left off
  bool fileExists = 0;
  fileExists = doesFileExist(outFileName);
  if (fileExists) {
    PetscPrintf(PETSC_COMM_WORLD,"File exists!\n");
    outputFileMode = FILE_MODE_APPEND;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), outputFileMode, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(viewer);                     CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer, "surfDisp", "Ii", PETSC_INT, NULL, &startIi); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"previous Ii = %i\n",startIi);
    startIi++;
    ierr = PetscViewerHDF5SetTimestep(viewer, startIi); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Ii = %i\n",startIi);
  }
  else {
    outputFileMode = FILE_MODE_WRITE;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), outputFileMode, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(viewer);                  CHKERRQ(ierr);
  }


  // create power law object
  PowerLaw pl(d,"Dirichlet","Neumann","Dirichlet","Neumann");
  HeatEquation he(d); // heat equation
  pl.updateTemperature(he._T);

  // set up KSP context
  Mat A;
  pl._sbp->getA(A);
  pl.setupKSP(pl._ksp,pl._pc,A,pl._linSolverTrans);

  // set up boundaries
  VecSet(pl._bcR,0.0);
  VecSet(pl._bcT,0.0);
  VecSet(pl._bcL,0.0);
  VecSet(pl._bcB,0.0);

  // initialize source terms
  Vec viscSource;
  VecDuplicate(pl._gVxy,&viscSource);
  VecSet(viscSource,0.0);

  // create small vec to store Ii in, to ensure consistency when job stops and is restarted
  Vec test;
  ierr = VecCreate(PETSC_COMM_WORLD,&test); CHKERRQ(ierr);
  ierr = VecSetSizes(test,PETSC_DECIDE,1); CHKERRQ(ierr);
  ierr = VecSetFromOptions(test); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) test, "test"); CHKERRQ(ierr);


  // loop over elements of viscous strains and compute corresponding entry of G
  PetscScalar v = 1.0;

  //~ for(PetscInt Ii = startIi; Ii < d._Ny*d._Nz*2;Ii++) {
  for(PetscInt Ii = startIi; Ii < startIi+10;Ii++) {
    PetscPrintf(PETSC_COMM_WORLD,"Ii = %i...",Ii);
    VecSet(test,Ii);
    VecSet(pl._gVxy,0.0);
    VecSet(pl._gVxz,0.0);

    // set just 1 element of either gVxy or gVxz to 1
    if (Ii < d._Ny*d._Nz) { // then in section of viscStrains that corresponds to gVxy
      VecSetValue(pl._gVxy,Ii,v,INSERT_VALUES);
      VecAssemblyBegin(pl._gVxy);
      VecAssemblyEnd(pl._gVxy);
    }
    else{  // then in section of viscStrains that corresponds to gVxz
      PetscInt Jj = Ii - d._Ny*d._Nz;
      VecSetValue(pl._gVxz,Jj,v,INSERT_VALUES);
      VecAssemblyBegin(pl._gVxz);
      VecAssemblyEnd(pl._gVxz);
    }

    // prepare linear system to solve for surface displacement
    // compute source terms to rhs: d/dy(mu*gVxy) + d/dz(mu*gVxz)
    ierr = pl.computeViscStrainSourceTerms(viscSource); CHKERRQ(ierr);

    // set up rhs vector
    pl.setRHS();
    ierr = VecAXPY(pl._rhs,1.0,viscSource); CHKERRQ(ierr);

    // solve for displacement (this function also updates surface displacement)
    ierr = pl.computeU(); CHKERRQ(ierr);

    // assign values to G
    //~ VecGetArrayRead(pl._surfDisp,&si);
    //~ MatSetValues(G,d._Ny,rows,1,&Ii,si,INSERT_VALUES);

    if (Ii > startIi) {
      ierr = PetscViewerHDF5IncrementTimestep(viewer);                  CHKERRQ(ierr);
    }
    ierr = VecView(pl._surfDisp, viewer);                               CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, "surfDisp", "Ii", PETSC_INT, &Ii); CHKERRQ(ierr);
    ierr = VecView(test, viewer);                                       CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);                                    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
  }

  // free memory
  VecDestroy(&test);
  PetscViewerDestroy(&viewer);
  return ierr;
}


// test Green's function to map viscous to surface displacement
// can be used to map viscous strain rate to surface velocity
int computeGreensFunction_test(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  // create domain object and write scalar fields into file
  Domain d(inputFile);
  PetscPrintf(PETSC_COMM_WORLD,"Running computeGreensFunction_test\n");

  // set up HDF5 file viewer
  PetscViewer viewer;
  string outFileName = d._outputDir + "G_test.h5";
  PetscFileMode outputFileMode = FILE_MODE_WRITE;
  PetscInt startIi = 0;
  // if file from pervious simulation exists, continue from where previous simulation left off
  bool fileExists = 0;
  fileExists = doesFileExist(outFileName);
  if (fileExists) {
    PetscPrintf(PETSC_COMM_WORLD,"File exists!\n");
    outputFileMode = FILE_MODE_APPEND;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), outputFileMode, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(viewer);                     CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer, "surfDisp", "Ii", PETSC_INT, NULL, &startIi); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"previous Ii = %i\n",startIi);
    startIi++;
    ierr = PetscViewerHDF5SetTimestep(viewer, startIi); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Ii = %i\n",startIi);
  }
  else {
    outputFileMode = FILE_MODE_WRITE;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), outputFileMode, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(viewer);                  CHKERRQ(ierr);
  }


  // create power law object
  PowerLaw pl(d,"Dirichlet","Neumann","Dirichlet","Neumann");
  //~ HeatEquation he(d); // heat equation
  //~ pl.updateTemperature(he._T);

  // set up KSP context
  Mat A;
  pl._sbp->getA(A);
  //~ pl.setupKSP(pl._ksp,pl._pc,A,pl._linSolverTrans);

  // set up boundaries
  VecSet(pl._bcR,0.0);
  VecSet(pl._bcT,0.0);
  VecSet(pl._bcL,0.0);
  VecSet(pl._bcB,0.0);

  // initialize source terms
  Vec viscSource;
  VecDuplicate(pl._gVxy,&viscSource);
  VecSet(viscSource,0.0);


  // loop over elements of viscous strains and compute corresponding entry of G

  //~ for(PetscInt Ii = startIi; Ii < d._Ny*d._Nz*2;Ii++) {
  //~ for(PetscInt Ii = startIi; Ii < 10;Ii++) {
  for(PetscInt Ii = startIi; Ii < startIi+10;Ii++) {
    PetscPrintf(PETSC_COMM_WORLD,"Ii = %i...",Ii);
    VecSet(pl._surfDisp,Ii);


    // prepare linear system to solve for surface displacement
    // compute source terms to rhs: d/dy(mu*gVxy) + d/dz(mu*gVxz)
    //~ ierr = pl.computeViscStrainSourceTerms(viscSource); CHKERRQ(ierr);

    // set up rhs vector
    //~ pl.setRHS();
    //~ ierr = VecAXPY(pl._rhs,1.0,viscSource); CHKERRQ(ierr);

    // solve for displacement (this function also updates surface displacement)
    //~ ierr = pl.computeU(); CHKERRQ(ierr);

    if (Ii > startIi) {
      ierr = PetscViewerHDF5IncrementTimestep(viewer);                 CHKERRQ(ierr);
    }
    ierr = VecView(pl._surfDisp, viewer);                               CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, "surfDisp", "Ii", PETSC_INT, &Ii); CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);                                    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"finished.\n");
  }

  // free memory
  PetscViewerDestroy(&viewer);
  return ierr;
}


// run different earthquake cycle scenarios depending on input
int runEqCycle(Domain& d)
{
  PetscErrorCode ierr = 0;
  // report number of processors
  PetscMPIInt numCores;
  MPI_Comm_size(PETSC_COMM_WORLD,&numCores);
  PetscPrintf(PETSC_COMM_WORLD,"Total number of processors: %i\n\n",numCores);

  if (d._bulkDeformationType == "linearElastic") {
    ProblemContext *m;
    if (d._momentumBalanceType == "quasidynamic") { m = new StrikeSlip_LinearElastic_qd(d); }
    if (d._momentumBalanceType == "dynamic") { m = new StrikeSlip_LinearElastic_fd(d); }
    if (d._momentumBalanceType == "quasidynamic_and_dynamic") { m = new StrikeSlip_LinearElastic_qd_fd(d); }

    if (d._restartFromChkpt == 0) { ierr = m->writeContext(); CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
    ierr = m->integrate(); CHKERRQ(ierr);
    ierr = m->view(); CHKERRQ(ierr);
    delete m;
  }

  if (d._bulkDeformationType == "powerLaw") {
    ProblemContext *m;
    if (d._momentumBalanceType == "quasidynamic") { m = new StrikeSlip_PowerLaw_qd(d); }
    if (d._momentumBalanceType == "quasidynamic_and_dynamic") { m = new StrikeSlip_PowerLaw_qd_fd(d); }

    if (d._restartFromChkpt == 0) { ierr = m->writeContext(); CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");

    if (d._systemEvolutionType == "steadyStateIts") { ierr = m->integrateSS(); CHKERRQ(ierr); }
    if (d._systemEvolutionType == "transient") {
      ierr = m->initiateIntegrand(); CHKERRQ(ierr);
      ierr = m->integrate(); CHKERRQ(ierr);
    }

    ierr = m->view(); CHKERRQ(ierr);
    delete m;
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
    else if (d._computeGreensFunction_fault) { computeGreensFunction_fault(inputFile); }
    else if (d._computeGreensFunction_offFault) { computeGreensFunction_offFault(inputFile); }
    else { runEqCycle(d); }
    //~ testHDF5();
    //~ runTests(inputFile);
  }


  PetscFinalize();
  return ierr;
}
