#ifndef USERCONTEXT_H_INCLUDED
#define USERCONTEXT_H_INCLUDED

#define STRINGIFY(name) #name

class UserContext {

  public:

   // debugging folder tree
   std::string debugFolder;

  // output data tree
  std::string outFileRoot;

  // SBP operator order
  PetscInt       order;

  // domain geometry
  PetscInt       Ny,Nz,N;
  PetscScalar    Ly,Lz,H,dy,dz;

  // elastic coefficients and frictional parameters
  //~PetscScalar    cs,G,rho,f0,v0,vp,D_c,tau_inf;
  PetscScalar    cs,f0,v0,vp,D_c,tau_inf;
  PetscScalar    muIn,muOut,D,W,rhoIn,rhoOut,*muArr;// for basin
  Vec            eta,s_NORM,a,b,psi,tau,gRShift;
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

  Mat            Hinvy_Izxe0y_Iz,Hinvy_IzxeNy_Iz,Iy_HinvzxIy_e0z;
  Mat            Iy_HinvzxIy_eNz,Hinvy_IzxBySy_IzTxe0y_Iz,Hinvy_IzxBySy_IzTxeNy_Iz;

  Mat            mu;

  Vec            rhs,uhat;

  KSP            ksp;
  PC             pc;

  // tolerances for linear and nonlinear (for V) solve
  PetscScalar    kspTol,rootTol;
  PetscInt       rootIts; // total number of its used

  // time stepping data
  TS             ts;
  Vec            faultDisp,dpsi,*var,surfDisp;
  PetscInt       strideLength; // stride
  PetscInt       maxStepCount; // largest number of time steps
  PetscReal      initTime,currTime,maxTime,minDeltaT,maxDeltaT;
  int            count;
  PetscScalar    atol;
  PetscScalar    initDeltaT;

  // lousy temporary solution
  Vec tempPsi;

  // viewers
  PetscViewer    timeViewer,surfDispViewer,faultDispViewer,velViewer,tauViewer,psiViewer;

  // run time info
  double computeTauTime,computeVelTime,kspTime,computeRhsTime,agingLawTime,rhsTime;
  double fullLinOps,arrLinOps;


  UserContext(const PetscInt ord,const PetscInt y,const  PetscInt z,const std::string outFile);
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

  /*
   * Output displacement, shear stress, and psi on fault and surface.
   * Add current time to timeFile.
   */
  PetscErrorCode writeInitialStep();

  PetscErrorCode writeCurrentStep();

  PetscErrorCode printTiming();

  /*
   * Output operators, time, and all needed vectors so that problem solution
   * can be continued from this point.
   */
  PetscErrorCode saveCurrentPlace();

  PetscErrorCode loadCurrentPlace();

  private:
    // disable default copy constructor and assignment operator
    UserContext(const UserContext & that);
    UserContext& operator=( const UserContext& rhs );

};

#endif
