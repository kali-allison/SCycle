#include <petscts.h>
#include <iostream>
#include <string>
#include "userContext.h"
//~ #include "debuggingFuncs.h"
#include "linearSysFuncs.h"
#include "rateAndState.h"
#include "rootFindingScalar.h"
#include "timeStepping.h"

using namespace std;

PetscErrorCode setInitialTimeStep(UserContext& D)
{
  PetscErrorCode ierr = 0;

  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);
  ierr = computeTau(D);CHKERRQ(ierr);
  //~ierr = initSlipVel(D);CHKERRQ(ierr);
  ierr = computeSlipVel(D);CHKERRQ(ierr);

  ierr = VecCopy(D.gF,D.faultDisp);CHKERRQ(ierr);
  ierr = VecScale(D.faultDisp,2.0);CHKERRQ(ierr);
  ierr = VecSet(D.gR,D.vp*D.initTime/2.0);CHKERRQ(ierr);

  ierr = ComputeRHS(D);CHKERRQ(ierr);

  return ierr;
}

/*
 * Computes shear stress on fault (nodes 1:D.Nz of slip vector)
 */
PetscErrorCode computeTau(UserContext& D)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting computeTau in timeStepping.c\n");CHKERRQ(ierr);
#endif

  // compute initial shear stress (MPa)
  Vec sigma_xy;
  ierr = VecDuplicate(D.uhat,&sigma_xy);CHKERRQ(ierr);
  ierr = MatMult(D.Dy_Iz,D.uhat,sigma_xy);
  ierr = VecDuplicate(D.gF,&D.tau);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<D.Nz) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(D.tau,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(D.tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.tau);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending computeTau in timeStepping.c\n");CHKERRQ(ierr);
#endif

  double endTime = MPI_Wtime();
  D.computeTauTime = D.computeTauTime + (endTime-startTime);

  return 0;
}

/*
 * Computes slip approximate slip velocity using linear equation.
 * Only used to create inital guess for velocity.
 */
PetscErrorCode initSlipVel(UserContext& D)
{
  PetscErrorCode ierr = 0;
  Vec            temp1,temp2,temp3,temp4;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting initSlipVel in timeStepping.c\n");CHKERRQ(ierr);
#endif

  //~ ierr = VecCreate(PETSC_COMM_WORLD,V);CHKERRQ(ierr);
  //~ ierr = VecDuplicate(D.gF,D.V);CHKERRQ(ierr);
  ierr = VecDuplicate(D.gF,&temp1);CHKERRQ(ierr);
  ierr = VecDuplicate(D.gF,&temp2);CHKERRQ(ierr);
  ierr = VecDuplicate(D.gF,&temp3);CHKERRQ(ierr);
  ierr = VecDuplicate(D.gF,&temp4);CHKERRQ(ierr);

  ierr = VecPointwiseMult(temp1,D.s_NORM,D.a);CHKERRQ(ierr); // temp = s_NORM.*a
  ierr = VecPointwiseDivide(temp1,D.tau,temp1);CHKERRQ(ierr); // temp = tau./(s_NORM.*a)

  ierr = VecCopy(temp1,temp2);CHKERRQ(ierr);

  ierr = VecExp(temp1);CHKERRQ(ierr);
  ierr = VecScale(temp2,-1.0);CHKERRQ(ierr);
  ierr = VecExp(temp2);CHKERRQ(ierr);

  ierr = VecAXPY(temp2,1,temp1);CHKERRQ(ierr); // temp2 = temp1 + temp2 = 2*sinh[tau./(s_NORM.*a)]
  ierr = VecPointwiseDivide(temp1,D.psi,D.a);CHKERRQ(ierr); // temp1 = psi./a
  ierr = VecScale(temp1,-1.0);CHKERRQ(ierr); // temp1 = -psi./a
  ierr = VecExp(temp1); // temp1 = exp(-psi./a)

  ierr = VecPointwiseMult(D.V,temp1,temp2); // temp1 = temp1.*temp2
  ierr = VecScale(D.V,D.v0);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending initSlipVel in timeStepping.c\n");CHKERRQ(ierr);
#endif
  return 0;
}

PetscErrorCode computeSlipVel(UserContext& D)
{
  PetscErrorCode ierr = 0;
  Vec            left,right,out;
  PetscScalar    outVal,leftVal,rightVal;
  PetscInt       Ii,Istart,Iend,its;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting computeSlipVel in timeStepping.c\n");CHKERRQ(ierr);
#endif

  ierr = VecDuplicate(D.tau,&right);CHKERRQ(ierr);
  ierr = VecCopy(D.tau,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,D.eta);CHKERRQ(ierr);
  ierr = VecAbs(right);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecCopy(right,left);CHKERRQ(ierr);
  ierr = VecScale(left,-1.0);CHKERRQ(ierr);
  //~ierr = VecSet(left,0.0);CHKERRQ(ierr);

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);
  PetscErrorCode (*frictionLaw)(const PetscInt,const PetscScalar,PetscScalar *, void *) = &stressMstrength;

  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);
    if (leftVal==rightVal) { outVal = leftVal; }
    else {
      ierr = bisect((*frictionLaw),Ii,leftVal,rightVal,&outVal,&its,D.rootTol,1e5,&D);CHKERRQ(ierr);
      D.rootIts += its;
    }
    ierr = VecSetValue(D.V,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(D.V);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.V);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending computeSlipVel in timeStepping.c\n");CHKERRQ(ierr);
#endif

  double endTime = MPI_Wtime();
  D.computeVelTime = D.computeVelTime + (endTime-startTime);

  return ierr;
}

PetscErrorCode rhsFunc(const PetscReal time,const int lenVar,Vec* var,Vec* dvar,void* userContext)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  UserContext    *D = (UserContext*) userContext;
  PetscScalar    val,psiVal;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting rhsFunc in timeStepping.c, at time t=%g\n",time);
  CHKERRQ(ierr);
#endif

  // update boundaries, state
  ierr = VecCopy(var[0],D->gF);CHKERRQ(ierr);
  ierr = VecScale(D->gF,0.5);CHKERRQ(ierr);
  ierr = VecCopy(var[1],D->tempPsi);CHKERRQ(ierr);
  ierr = VecSet(D->gR,D->vp*time/2.0);CHKERRQ(ierr);


  // solve for displacement
  ierr = ComputeRHS(*D);
  double startKspTime = MPI_Wtime();
    ierr = KSPSolve(D->ksp,D->rhs,D->uhat);CHKERRQ(ierr);
  double endKspTime = MPI_Wtime();
  D->kspTime = D->kspTime + (endKspTime-startKspTime);

  ierr = computeTau(*D);CHKERRQ(ierr);
  ierr = computeSlipVel(*D);CHKERRQ(ierr);

  // compute dvar
  ierr = VecGetOwnershipRange(D->V,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(D->V,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(dvar[0],Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(var[1],1,&Ii,&psiVal);
    ierr = agingLaw(Ii,psiVal,&val,D);CHKERRQ(ierr);
    ierr = VecSetValue(dvar[1],Ii,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(dvar[0]);CHKERRQ(ierr); ierr = VecAssemblyBegin(dvar[1]);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(dvar[0]);CHKERRQ(ierr);   ierr = VecAssemblyEnd(dvar[1]);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending rhsFunc in timeStepping.c, at time t=%g\n",time);CHKERRQ(ierr);
#endif

  double endTime = MPI_Wtime();
  D->rhsTime = D->rhsTime + (endTime-startTime);

  return 0;
}


PetscErrorCode timeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec* var, const int lenVar, void* userContext)
{
  PetscErrorCode ierr = 0;
  UserContext*    D = (UserContext*) userContext;

  if ( stepCount % D->strideLength == 0) {
    D->count++;
    D->currTime = time;
    ierr = D->writeCurrentStep();CHKERRQ(ierr);
  }

  return ierr;
}


//~ PetscErrorCode writeData(const char outFileRoot[],const char name[],PetscInt step,Vec w)
//~ {
  //~ PetscErrorCode ierr;
  //~ PetscViewer outviewer;
  //~ char * outFileLoc = malloc(1+strlen(outFileRoot)+strlen(name)+sizeof(PetscInt));
  //~ char *strInt = malloc(1+sizeof(PetscInt));
//~
  //~ strcpy(outFileLoc,outFileRoot);
  //~ strcat(outFileLoc,name);
  //~ sprintf(strInt,"%d",step);
  //~ strcat(outFileLoc,strInt);
//~
//~
  //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  //~ ierr = VecView(w,outviewer);CHKERRQ(ierr);
//~
  //~ return ierr;
//~ }
