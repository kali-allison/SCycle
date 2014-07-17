#include <petscts.h>
#include <string>
#include <assert.h>
#include <limits>
#include "userContext.h"
#include "rateAndState.h"

/*
 * Computes rate and state friction as a function of input velocity
 * and relevant rate and state parameters.
 * ind = index corresponding to velocity input (used in a, psi, and sigma_N).
 * out = a.*asinh( (V ./ (2.*v0)) .* exp((psi)./p.a) )
 */
PetscErrorCode stressMstrength(const PetscInt ind,const PetscScalar vel,PetscScalar *out, void * ctx)
{
  PetscErrorCode ierr;
  UserContext    *D = (UserContext*) ctx;
  PetscScalar    psi,a,sigma_n,eta,tau;
  PetscInt       Istart,Iend;

#if VERBOSE > 1
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting stressMstrength in rateAndState.c\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(D->psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(D->tempPsi,1,&ind,&psi);CHKERRQ(ierr);
    ierr = VecGetValues(D->a,1,&ind,&a);CHKERRQ(ierr);
    ierr = VecGetValues(D->sigma_N,1,&ind,&sigma_n);CHKERRQ(ierr);
    ierr = VecGetValues(D->eta,1,&ind,&eta);CHKERRQ(ierr);
    ierr = VecGetValues(D->tau,1,&ind,&tau);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in stressMstrength\n");
  }

   *out = (PetscScalar) a*sigma_n*asinh( (double) (vel/2/D->v0)*exp(psi/a) ) + eta*vel - tau;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/D->v0)=%.9e\n",vel/2/D->v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"eta*vel=%.9e\n",eta*vel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 1
  //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending stressMstrength in rateAndState.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}

/*
 * Computes dPsi, the change in the state evolution parameter
 */
PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi, void *ctx)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,vel;
  UserContext    *D = (UserContext *) ctx;

  double startTime = MPI_Wtime();

  ierr = VecGetOwnershipRange(D->psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(D->b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(D->vel,1,&ind,&vel);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in agingLaw\n");
  }

  //~if (b==0) { *dPsi = 0; }
  if ( isinf(exp(1/b)) ) { *dPsi = 0; }
  else {
    *dPsi = (PetscScalar) (b*D->v0/D->D_c)*( exp((double) ( (D->f0-psi)/b) ) - (vel/D->v0) );
  }

  //~if (ind==0) {
    //~ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,b=%g,f0=%g,D_c=%g,v0=%g,vel=%g\n",psi,b,D->f0,D->D_c,D->v0,vel);
    //~CHKERRQ(ierr);
  //~}

  if (isnan(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,D->f0,D->D_c,D->v0,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*D->v0/D->D_c));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (D->f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/D->v0));
    CHKERRQ(ierr);
  }
  else if (isinf(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,D->f0,D->D_c,D->v0,vel);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*D->v0/D->D_c));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (D->f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/D->v0));
  }

  assert(!isnan(*dPsi));
  assert(!isinf(*dPsi));

  double endTime = MPI_Wtime();
  D->agingLawTime = D->agingLawTime + (endTime-startTime);

  return ierr;
}


PetscErrorCode setRateAndState(UserContext &D)
{
  PetscErrorCode ierr;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v,y,z;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting rateAndStateConstParams in rateAndState.c\n");CHKERRQ(ierr);
#endif

  // Set normal stress along fault
  PetscScalar sigma_NVal = 50.0;
  //~ierr = VecDuplicate(D.a,&D.sigma_N);CHKERRQ(ierr);
  ierr = VecSet(D.sigma_N,sigma_NVal);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(D.sigma_N);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.sigma_N);CHKERRQ(ierr);

  // Set a
  PetscScalar aVal = 0.015;
  ierr = VecSet(D.a,aVal);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(D.a);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.a);CHKERRQ(ierr);

  // Set b
  PetscScalar L1 = D.H;  // Defines depth at which (a-b) begins to increase.
  PetscScalar L2 = 1.5*D.H;  //This is depth at which increase stops and fault is purely velocity strengthening.
  PetscInt    N1 = L1/D.dz;
  PetscInt    N2 = L2/D.dz;
  ierr = VecGetOwnershipRange(D.b,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < N1+1) {
      v=0.02;
      ierr = VecSetValues(D.b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (Ii>N1 && Ii<=N2) {
      v = 0.02/(L1-L2);v = v*Ii*D.dz - v*L2;
      ierr = VecSetValues(D.b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      v = 0.0;
      ierr = VecSetValues(D.b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(D.b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.b);CHKERRQ(ierr);
  //~ierr = VecSet(D.b,0.02);CHKERRQ(ierr); // for spring-slider!!!!!!!!!!!!!!!!

  // set shear modulus
  Vec muVec;
  PetscInt muInds[D.Ny*D.Nz];
  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(muVec,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar r,rbar=0.25*D.W*D.W,rw=1+0.5*D.W/D.D;
  for (Ii=0;Ii<D.Ny*D.Nz;Ii++) {
    z = D.dz*(Ii-D.Nz*(Ii/D.Nz));
    y = D.dy*(Ii/D.Nz);
    r=y*y+(0.25*D.W*D.W/D.D/D.D)*z*z;
    v = 0.5*(D.muOut-D.muIn)*(tanh((double)(r-rbar)/rw)+1) + D.muIn;
    D.muArr[Ii] = v;
    //~D.muArr[Ii] = Ii+2;//!!!!!
    muInds[Ii] = Ii;
  }
  ierr = VecSetValues(muVec,D.Ny*D.Nz,muInds,D.muArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(D.mu,PETSC_DECIDE,PETSC_DECIDE,D.Ny*D.Nz,D.Ny*D.Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D.mu);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D.mu,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D.mu,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D.mu);CHKERRQ(ierr);
  ierr = MatDiagonalSet(D.mu,muVec,INSERT_VALUES);CHKERRQ(ierr);

  // tau, psi, eta
  PetscScalar a,b,eta,tau_inf,sigma_N,grShift;
  ierr = VecGetOwnershipRange(D.tau,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(D.a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(D.b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(D.sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    //~psi_p = D.f0 - b*log(D.vp/D.v0);
    tau_inf = sigma_N*a*asinh( (double) 0.5*D.vp*exp(D.f0/a)/D.v0 );
    z = ((double) Ii)*D.dz;
    if (z < D.D) { eta = 0.5*sqrt(D.rhoIn*D.muArr[Ii]); }
    else { eta = 0.5*sqrt(D.rhoOut*D.muArr[Ii]); }
    //~psi = a*log( 2*D.v0*sinh((double)(tau_inf-eta*D.vp)/(sigma_N*a))/D.vp );
    grShift = tau_inf*D.Ly/D.muArr[Ii];

    ierr = VecSetValue(D.tau,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(D.eta,Ii,eta,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(D.gRShift,Ii,grShift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(D.tau);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(D.eta);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(D.gRShift);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(D.tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.eta);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D.gRShift);CHKERRQ(ierr);

  ierr = VecSet(D.psi,D.f0);
  ierr = VecCopy(D.psi,D.tempPsi);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending rateAndStateConstParams in rateAndState.c\n");CHKERRQ(ierr);
#endif
  return 0;
}


