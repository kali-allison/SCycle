#ifndef USERCONTEXT_H_INCLUDED
#define USERCONTEXT_H_INCLUDED

#define STRINGIFY(name) #name

class UserContext {

  public:
//~
  //~ // debugging folder tree
  //~ std::string debugFolder;
//~
  // output data tree
  std::string outFileRoot;

  // SBP operator order
  PetscInt       order;

  // domain geometry
  PetscInt       Ny,Nz,N;
  PetscScalar    Ly,Lz,H,dy,dz;

  // elastic coefficients and frictional parameters
  PetscScalar    cs,G,rho,f0,v0,vp,D_c,tau_inf;
  Vec            eta,s_NORM,a,b,psi,tau;
  Vec            V;

  // penalty terms for BCs
  PetscScalar    alphaF,alphaR,alphaS,alphaD,beta;

  // boundary conditions
  Vec            gF,gS,gR,gD;

  // linear system
  Mat            A,D2y,D2z;
  Mat            Dy_Iz;
  Mat            IyHinvz_Iye0z,IyHinvz_IyeNz,Hinvy_Iz_e0y_Iz,Hinvy_Iz_BySy_Iz_e0y_Iz;
  Mat            Hinvy_Iz_eNy_Iz,Hinvy_Iz_BySy_Iz_eNy_Iz;
  Mat            Hinvy,Hinvz;

  Vec            rhs,uhat;

  KSP            ksp;
  PC             pc;

  // tolerances for linear and nonlinear (for V) solve
  PetscScalar    kspTol,rootTol;

  // time stepping data
  TS             ts;
  Vec            w,faultDisp,*var;
  PetscInt       strideLength; // stride
  PetscInt       maxStepCount; // largest number of time steps
  PetscReal      initTime,currTime,maxTime,minDeltaT,maxDeltaT;
  int            count;
  PetscScalar    atol;
  PetscScalar    initDeltaT;

  // viewers
  PetscViewer    timeViewer,wViewer,uhatViewer,velViewer,tauViewer,psiViewer;

  // run time info
  double computeTauTime,computeVelTime,kspTime,computeRhsTime,agingLawTime,rhsTime;


  UserContext(PetscInt ord, PetscInt y, PetscInt z, std::string outFile);
  ~UserContext();

  /*
   * Set values for model dimensions.
   */
  friend PetscErrorCode setParameters(UserContext * ctx);
  PetscErrorCode writeParameters();

  /*
   * Output A, Dy_Iz.
   */
  PetscErrorCode writeOperators();

  PetscErrorCode writeInitialStep();

  /*
   * Output displacement, shear stress, and psi on fault and surface.
   * Add current time to timeFile.
   */
  PetscErrorCode writeCurrentStep();

  /*
   * Output operators, time, and all needed vectors so that problem solution
   * can be continued from this point.
   */
  PetscErrorCode saveCurrentPlace();

  PetscErrorCode loadCurrentPlace();

};

#endif
