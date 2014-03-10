#include <petscts.h>
#include "rootFindingScalar.h"

PetscErrorCode func(PetscScalar in, PetscScalar *out)
{
  *out = in*in*in - 30.0*in*in + 2552;
  //*out = exp(in/4) + exp(-in/4) - in;
  return 0;
}

PetscErrorCode funcPrime(PetscScalar in, PetscScalar *out)
{
  *out = 3*in*in - 60*in;
  return 0;
}

PetscErrorCode secantMethod()
{

  PetscErrorCode ierr;
  PetscScalar    xkm1=10,xk=8,xkp1,fkm1,fk,fkp1,atol=1e-8;
  PetscInt       maxNumIts=7,numIts=0;

  ierr = func(xkm1,&fkm1);CHKERRQ(ierr);
  ierr = func(xk,&fk);CHKERRQ(ierr);
  fkp1 = fk;

  while ( (numIts <= maxNumIts) & (sqrt(fkp1*fkp1) >= atol) ) {

    //ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts = %u, xkm1 = %f, fxkm1 = %f, ",numIts,xkm1,fkm1);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"xk = %f, fk = %f, ",xk,fk);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"atol = %g, check = %f, ",atol, sqrt(fkp1*fkp1));CHKERRQ(ierr);

    xkp1 = xk - fk*(xk-xkm1)/( fk - fkm1 );
    ierr = func(xkp1,&fkp1);CHKERRQ(ierr);

    xkm1 = xk; fkm1 = fk;
    xk = xkp1; fk = fkp1;

    //ierr = PetscPrintf(PETSC_COMM_WORLD,"xk+1 = %f, fxk+1 = %f\n",xk,fk);CHKERRQ(ierr);
    numIts++;
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts = %u, x_guess = %f, f_guess = %f\n",numIts,xkp1,fkp1);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode bisect(PetscErrorCode (*pt2Func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscScalar atol,
    PetscInt itMax,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    fLeft,fRight;

#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting bisect in rootFindingScalar.c\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"left = %g, right = %g,ind=%d\n",left,right,ind);CHKERRQ(ierr);
#endif

  /*
     Check inputs.
  */
  if (left >= right) {
    SETERRQ(PETSC_COMM_WORLD,1,"left bound must be less than right bound");
    return 0;
  }
  else if (atol <= 0) {
    SETERRQ(PETSC_COMM_WORLD,1,"atol must be > 0 for convergence");
    return 0;
  }

  ierr = (*pt2Func)(ind,left,&fLeft,ctx);CHKERRQ(ierr);
  ierr = (*pt2Func)(ind,right,&fRight,ctx);CHKERRQ(ierr);
#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",fLeft,fRight);CHKERRQ(ierr);
#endif

  /*
   * Compute root.
   */
  //~ PetscInt maxNumIts = ceil( log2(right-left) - log2(2*atol) );
  PetscInt numIts=0;
  PetscScalar mid,fMid;

  // if step is too big, assign very large value to output
  if ( isnan(fLeft) || isnan(fRight) || isinf(fLeft) || isinf(fRight) ) {
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",fLeft,fRight);CHKERRQ(ierr);
    //~ SETERRQ(PETSC_COMM_WORLD,1,"Function evaluated to nan");
    mid = 1e9;fMid = 66666e-6;
    return 0;
  }
  else if (sqrt(fLeft*fLeft) <= atol) { *out = left; return 0; }
  else if (sqrt(fRight*fRight) <= atol) { *out = right; return 0; }
  //~else if (fLeft*fRight >= 0) {
    //~ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",fLeft,fRight);CHKERRQ(ierr);
    //~SETERRQ(PETSC_COMM_WORLD,1,"function must cross 0 between bounds");
    //~return 0;
  //~}

  mid = (left + right)*0.5;
  ierr = func(mid,&fMid);CHKERRQ(ierr);
  if (isnan(fMid)) { SETERRQ(PETSC_COMM_WORLD,1,"fMid evaluated to nan"); return 0; }
  while ( (numIts <= itMax) & (sqrt(fMid*fMid) >= atol) ) {
    mid = (left + right)*0.5;
    ierr = (*pt2Func)(ind,mid,&fMid,ctx);CHKERRQ(ierr);

    if (isnan(fMid)) { SETERRQ(PETSC_COMM_WORLD,1,"fMid evaluated to nan"); return 0; }

    //ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts = %u, left = %f, fLeft = %f, ",numIts,left,fLeft);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"right = %f, fRight = %f, mid = %f, fMid = %f, ",right,fRight,mid,fMid);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"atol = %f, bool = %f\n",atol, sqrt(fMid*fMid) );CHKERRQ(ierr);

    if (fLeft*fMid <= 0) {
      right = mid;
      fRight = fMid;
    }
    else {
      left = mid;
      fLeft = fMid;
    }

    numIts++;
  }

#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts/maxNumIts = %u/%u, final mid = %g, fMid = %g\n",numIts,maxNumIts,mid,fMid);CHKERRQ(ierr);
#endif

  *out = mid;
  if (sqrt(fMid*fMid) > atol) {SETERRQ(PETSC_COMM_WORLD,1,"rootFinder did not converge");}

  return ierr;
}

PetscErrorCode rootFinderScalar()
{

  return 0;
}
