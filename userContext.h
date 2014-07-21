#ifndef USERCONTEXT_H_INCLUDED
#define USERCONTEXT_H_INCLUDED

#define STRINGIFY(name) #name

class UserContext {

  public:

   // debugging folder tree
   std::string debugFolder;

  // output data
  std::string outFileRoot;
  PetscViewer    timeViewer,surfDispViewer,faultDispViewer,velViewer,tauViewer,psiViewer;
  PetscViewer  viewer;

  // SBP operator order
  PetscInt       order;

  // domain geometry
  PetscInt       Ny,Nz,N;
  PetscScalar    Ly,Lz,H,dy,dz;

  // elastic coefficients and frictional parameters
  PetscScalar    cs,f0,v0,vp,D_c,tau_inf;
  PetscScalar    muIn,muOut,D,W,rhoIn,rhoOut,*muArr;// for basin
  Vec            vel, eta,sigma_N,a,b,psi,tempPsi,tau,gRShift;

  // penalty terms for BCs
  PetscScalar    alphaF,alphaR,alphaS,alphaD,beta;

  // boundary conditions
  Vec            gF,gS,gR,gD;

  // linear system
  Mat            A,Dy_Iz;
  Mat            Hinvy_Izxe0y_Iz,Hinvy_IzxeNy_Iz,Iy_HinvzxIy_e0z;
  Mat            Iy_HinvzxIy_eNz,Hinvy_IzxBySy_IzTxe0y_Iz,Hinvy_IzxBySy_IzTxeNy_Iz;

  Mat            mu;

  Vec            rhs,uhat;

  KSP            ksp;
  PC             pc;

  // tolerances for linear and nonlinear (for V) solve
  PetscScalar    kspTol,rootTol;
  PetscInt       rootIts; // total number of iterations

  // time stepping data
  Vec            faultDisp,dpsi,*var,surfDisp;
  PetscInt       strideLength; // stride
  PetscInt       maxStepCount; // largest number of time steps
  PetscReal      initTime,currTime,maxTime,minDeltaT,maxDeltaT;
  int            count;
  PetscScalar    atol;
  PetscScalar    initDeltaT;


  // run time info
  double computeTauTime,computeVelTime,kspTime,computeRhsTime,agingLawTime,rhsTime;
  double fullLinOps,arrLinOps,writeTime;


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
  PetscErrorCode writeStep();
  PetscErrorCode writeInitialStep();
  PetscErrorCode writeRateAndState();

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
