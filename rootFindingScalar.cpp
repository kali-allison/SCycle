#include <petscts.h>
#include "rootFindingScalar.h"

//~PetscErrorCode exFunc(PetscScalar in, PetscScalar *out,void *ctx)
PetscErrorCode exFunc(const PetscInt ind, PetscScalar in,PetscScalar *out, void * ctx)
{
  *out = in*in*in - 30.0*in*in + 2552;
  return 0;
}

PetscErrorCode secantMethod(PetscErrorCode (*func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx)
{

  PetscErrorCode ierr = 0;
  PetscScalar    xkm1=10,xk=8,xkp1,fkm1,fk,fkp1;
  PetscInt       numIts=0;

  ierr = func(ind,xkm1,&fkm1,ctx);CHKERRQ(ierr);
  ierr = func(ind,xk,&fk,ctx);CHKERRQ(ierr);
  fkp1 = fk;

  while ( (numIts <= itMax) & (sqrt(fkp1*fkp1) >= atol) ) {

    xkp1 = xk - fk*(xk-xkm1)/( fk - fkm1 );
    ierr = func(ind,xkp1,&fkp1,ctx);CHKERRQ(ierr);

    xkm1 = xk; fkm1 = fk;
    xk = xkp1; fk = fkp1;

    numIts++;
  }
  *its = numIts;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"SECANT: numIts = %u, x = %f, f(x) = %f\n",numIts,xkp1,fkp1);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode bisect(PetscErrorCode (*func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx)
{
  PetscErrorCode ierr = 0;
  PetscScalar    fLeft,fRight;

#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting bisect in rootFindingScalar.c\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"left = %g, right = %g,ind=%d\n",left,right,ind);CHKERRQ(ierr);
#endif

  /*
   * Check inputs.
   */
  if (left >= right) {
    SETERRQ(PETSC_COMM_WORLD,1,"left bound must be less than right bound");
    return 0;
  }
  else if (atol <= 0) {
    SETERRQ(PETSC_COMM_WORLD,1,"atol must be > 0 for convergence");
    return 0;
  }

  ierr = func(ind,left,&fLeft,ctx);CHKERRQ(ierr);
  ierr = func(ind,right,&fRight,ctx);CHKERRQ(ierr);
#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",fLeft,fRight);CHKERRQ(ierr);
#endif

  PetscInt numIts=0;
  PetscScalar mid,fMid;

  if ( isnan(fLeft) || isnan(fRight) || isinf(fLeft) || isinf(fRight) ) {
     SETERRQ(PETSC_COMM_WORLD,1,"Function evaluated to nan");
    return 0;
  }
  else if (sqrt(fLeft*fLeft) <= atol) { *out = left; return 0; }
  else if (sqrt(fRight*fRight) <= atol) { *out = right; return 0; }

  mid = (left + right)*0.5;
  ierr = func(ind,mid,&fMid,ctx);CHKERRQ(ierr);
  if (isnan(fMid)) { SETERRQ(PETSC_COMM_WORLD,1,"fMid evaluated to nan"); return 0; }
  while ( (numIts <= itMax) & (sqrt(fMid*fMid) >= atol) ) {
    mid = (left + right)*0.5;
    ierr = func(ind,mid,&fMid,ctx);CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts/maxIts = %u/%u, final mid = %g, fMid = %g\n",numIts,itMax,mid,fMid);CHKERRQ(ierr);
#endif

  *out = mid;
  *its = numIts;
  if (sqrt(fMid*fMid) > atol) {
    SETERRQ(PETSC_COMM_WORLD,1,"rootFinder did not converge");
    return 0;
  }

  return ierr;
}

PetscErrorCode safeSecant(PetscErrorCode (*func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx)
{
  PetscErrorCode ierr = 0;
  PetscScalar    fLeft,fRight,mid,fMid,xkm2,fkm2,xkm1,fkm1,xk,fk,d=1e-3;
  PetscInt       numIts=0;

  // Check inputs.
  if (left >= right) {
    SETERRQ(PETSC_COMM_WORLD,1,"left bound must be less than right bound");
    return 0;
  }
  else if (atol <= 0) {
    SETERRQ(PETSC_COMM_WORLD,1,"atol must be > 0 for convergence");
    return 0;
  }

  ierr = func(ind,left,&fLeft,ctx);CHKERRQ(ierr);
  ierr = func(ind,right,&fRight,ctx);CHKERRQ(ierr);

  // if step is too big, assign very large value to output
  if ( isnan(fLeft) || isnan(fRight) || isinf(fLeft) || isinf(fRight) ) {
    SETERRQ(PETSC_COMM_WORLD,1,"at least one bound evaluated to inf or nan");
    return 0;
  }
  else if (sqrt(fLeft*fLeft) <= atol) { *out = left; return 0; }
  else if (sqrt(fRight*fRight) <= atol) { *out = right; return 0; }

  mid = (left + right)*0.5; ierr = func(ind,mid,&fMid,ctx);CHKERRQ(ierr); // bisect guess
  if (isnan(fMid)) { SETERRQ(PETSC_COMM_WORLD,1,"fMid evaluated to nan"); return 0; }

  // secant guess
  xkm2 = mid - d*(right-left);            ierr = func(ind,xkm2,&fkm2,ctx);CHKERRQ(ierr);
  xkm1 = mid + d*(right-left);            ierr = func(ind,xkm1,&fkm1,ctx);CHKERRQ(ierr);
  xk = xkm1-fkm1*(xkm1-xkm2)/(fkm1-fkm2); ierr = func(ind,xk,&fk,ctx);CHKERRQ(ierr);
  xkm2 = xkm1; fkm2 = fkm1; // save old steps
  xkm1 = xk; fkm1 = fk;

  while ( (numIts <= itMax) & (sqrt(fMid*fMid) >= atol) ) {

    // bisect guess
    mid = (left + right)*0.5; ierr = func(ind,mid,&fMid,ctx);CHKERRQ(ierr);

    // secant guess
    xk = xkm1-fkm1*(xkm1-xkm2)/(fkm1-fkm2); ierr = func(ind,xk,&fk,ctx);CHKERRQ(ierr);
    xkm2 = xkm1; fkm2 = fkm1; // save old steps
    xkm1 = xk; fkm1 = fk;

    // update bracket
    if (fLeft*fMid <= 0) { right = mid; fRight = fMid; }
    else { left = mid; fLeft = fMid; }

    // if xk is inside bracket and decreases bracket size more than mid, use it!!
    if (left < xk && xk < right) { mid = xk; fMid = fk; }

    numIts++;
  }

  *out = mid;
  *its = numIts;
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"SAFE-SECANT:  numIts = %i, x = %f, f(x) = %f\n",numIts,*out,fMid);CHKERRQ(ierr);
  if (sqrt(fMid*fMid) > atol) {SETERRQ(PETSC_COMM_WORLD,1,"rootFinder did not converge");}

  return ierr;
}
