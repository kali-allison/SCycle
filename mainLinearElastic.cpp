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



using namespace std;

int runMMSTests(const char * inputFile)
{
  PetscErrorCode ierr = 0;
  //~ PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e %.4e % .15e\n",
              //~ _order,_Ny,_dy,err2uA,log2(err2uA),err2sigmaxy,log2(err2sigmaxy));
  PetscPrintf(PETSC_COMM_WORLD,"%-3s %-3s %-10s %-10s %-22s %-10s %-22s\n",
             "ord","Ny","dy","L2u","log2(L2u)","L2sigmaxy","log2(L2sigmaxy)");
  //~ for(PetscInt Ny=11;Ny<2562;Ny=(Ny-1)*2+1)
  for(PetscInt Ny=11;Ny<82;Ny=(Ny-1)*2+1)
  {
    Domain domain(inputFile,Ny,Ny);
    //~ Domain domain(inputFile,Ny,1);
    domain.write();

    LinearElastic *obj;
    if (domain._geometry.compare("symmetric")==0) {
      obj = new SymmLinearElastic(domain);
    }
    //~ else {
      //~ obj = new FullLinearElastic(domain);
    //~ }

    ierr = obj->writeStep1D();CHKERRQ(ierr);
    ierr = obj->writeStep2D();CHKERRQ(ierr);
    ierr = obj->integrate();CHKERRQ(ierr);
    obj->measureMMSError();
  }

  return ierr;
}


int testMemoryLeak(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  d.write();

  //~ HeatEquation he(d);
  //~ he.writeContext();
  //~ he.writeStep2D(0);
  //~ he.writeStep2D(1);

  //~ SymmFault fault(d,he);
  //~ fault.writeContext(d._outputDir);
  //~ fault.writeStep(d._outputDir,0);
  //~ fault.writeStep(d._outputDir,1);
  //~ fault.writeStep(d._outputDir,2);

  //~ SbpOps_fc sbp(d,d._muVecP,"Neumann","Dirichlet","Neumann","Dirichlet","yz");
  //~ SbpOps_fc sbp(d,d._muVecP,"Neumann","Neumann","Neumann","Neumann","yz");

  //~ SbpOps_fc_coordTrans sbp(d,d._muVecP,"Neumann","Neumann","Neumann","Neumann","yz");
  //~ SbpOps_fc_coordTrans sbp(d,d._muVecP,"Dirichlet","Dirichlet","Dirichlet","Dirichlet","yz");
  //~ SbpOps_fc_coordTrans sbp(d,d._muVecP,"Neumann","Dirichlet","Neumann","Dirichlet","yz");

  SymmLinearElastic sle(d);
  //~ sle.writeStep1D();
  //~ sle.writeStep2D();
  ierr = sle.integrate();CHKERRQ(ierr);


  //~ sle.writeStep1D();
  //~ sle.writeStep2D();
  //~ ierr = sle.integrate();CHKERRQ(ierr);
  return ierr;
}

int computeGreensFunction(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain d(inputFile);
  d.write();
  //~ _sbpP = new SbpOps_fc_coordTrans(d,d._muVecP,"Neumann","Dirichlet","Neumann","Dirichlet","yz");
  SymmLinearElastic sle(d);

  // set up boundaries
  VecSet(sle._bcTP,0.0);
  VecSet(sle._bcBP,0.0);
  VecSet(sle._bcRP,0.0);

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
  VecGetOwnershipRange(sle._bcLP,&Istart,&Iend);
  for(PetscInt Ii=Istart;Ii<Iend;Ii++) {
    VecSet(sle._bcLP,0.0);
    VecSetValue(sle._bcLP,Ii,v,INSERT_VALUES);

    // solve for displacement
    ierr = sle._sbpP->setRhs(sle._rhsP,sle._bcLP,sle._bcRP,sle._bcTP,sle._bcBP);CHKERRQ(ierr);
    ierr = KSPSolve(sle._kspP,sle._rhsP,sle._uP);CHKERRQ(ierr);
    ierr = sle.setSurfDisp();

    // assign values to G
    //~ _surfDispPlus
    VecGetArray(sle._surfDispPlus,&si);
    for(PetscInt ind=0;ind<d._Ny;ind++) { rows[ind]=ind; }
    for(PetscInt ind=0;ind<d._Ny;ind++) { cols[ind]=Ii; }
    MatSetValues(G,d._Ny,rows,1,&Ii,si,INSERT_VALUES);
    MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);
    VecRestoreArray(sle._bcLP,&si);
  }

  // output greens function
  std::string str;

  str =  d._outputDir + "G";
  writeMat(G,str.c_str());



  // output testing stuff
  VecSet(sle._bcLP,0.0);
  VecSet(sle._bcRP,5.0);
  for(PetscInt Ii=Istart;Ii<Iend;Ii++) {
    v = ((PetscScalar) Ii+1)/((PetscScalar) d._Nz);
    //~ if (Ii == 0) { v=1.0; } else { v=0.0; }
    VecSetValue(sle._bcLP,Ii,v,INSERT_VALUES);
  }
  // solve for displacement
  ierr = sle._sbpP->setRhs(sle._rhsP,sle._bcLP,sle._bcRP,sle._bcTP,sle._bcBP);CHKERRQ(ierr);
  ierr = KSPSolve(sle._kspP,sle._rhsP,sle._uP);CHKERRQ(ierr);
  ierr = sle.setSurfDisp();
  //~ VecView(sle._bcLP,PETSC_VIEWER_STDOUT_WORLD);

  str =  d._outputDir + "bcL";
  writeVec(sle._bcLP,str.c_str());

  str =  d._outputDir + "surfDisp";
  writeVec(sle._surfDispPlus,str.c_str());


  MatDestroy(&G);
  return ierr;
}


int runEqCycle(const char * inputFile)
{
  PetscErrorCode ierr = 0;

  Domain domain(inputFile);
  domain.write();

  // if want to switch between full and symmetric problems
  //~ LinearElastic *obj;
  //~ if (domain._problemType.compare("symmetric")==0) {
    //~ obj = new SymmLinearElastic(domain);
  //~ }
  //~ else {
    //~ obj = new FullLinearElastic(domain);
  //~ }

  SymmLinearElastic sle(domain);
  PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  ierr = sle.integrate();CHKERRQ(ierr);
  //~ ierr = sle.view();CHKERRQ(ierr);
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
    //~ PetscMPIInt localRank;
    //~ MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: hi!\n", localRank);
    Domain domain(inputFile);
    if (domain._isMMS) { runMMSTests(inputFile); }
    else { runEqCycle(inputFile); }
  }

  //~ testMemoryLeak(inputFile);

  //~ computeGreensFunction(inputFile);

  //~runTests1D();
  //~runTests2D();
  //~runTimingTest(inputFile);


  PetscFinalize();
  return ierr;
}

