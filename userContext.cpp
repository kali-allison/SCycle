#include <petscts.h>
#include <string>
#include <sstream>
#include <fstream>
#include "odeSolver.h"
#include "userContext.h"

using namespace std;

// constructor
UserContext::UserContext(const PetscInt ord,const PetscInt y,const  PetscInt z,const std::string outFile)
:outFileRoot(outFile),
 order(ord),Ny(y),Nz(z),N(y*z),
 Ly(0),Lz(0),H(0),dy(0),dz(0),
 //~cs(0),G(0),rho(0),f0(0),v0(0),vp(0),D_c(0),
 cs(0),f0(0),v0(0),vp(0),D_c(0),
 muIn(0),muOut(0),D(0),W(0),rhoIn(0),rhoOut(0),
 kspTol(0),rootTol(0),rootIts(0),
 writeStride(1),checkpointStride(1),maxStepCount(1),initTime(0),currTime(0),maxTime(0),
 minDeltaT(1e-14),maxDeltaT(1e-14),
 count(0),atol(1e-14),initDeltaT(1e-14),
 computeTauTime(0),computeVelTime(0),kspTime(0),computeRhsTime(0),agingLawTime(0),rhsTime(0),
 fullLinOps(0),arrLinOps(0)
{

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in userContext.cpp.\n");
#endif

  inFileRoot = outFile.c_str();

  std::ostringstream convert;
  convert << "./matlabAnswers/order" << order << "Ny" << y << "Nz" << z << "/";
  debugFolder = convert.str();

  // Initialize elastic coefficients and frictional parameters
  VecCreate(PETSC_COMM_WORLD,&eta);
  VecSetSizes(eta,PETSC_DECIDE,Nz);
  VecSetFromOptions(eta);
  VecDuplicate(eta,&s_NORM);
  VecDuplicate(eta,&a);
  VecDuplicate(eta,&b);
  VecDuplicate(eta,&psi);
  VecDuplicate(eta,&tau);
  VecDuplicate(eta,&gRShift);

  muArr = new PetscScalar[Ny*Nz];

  // Initialize boundary conditions
  VecDuplicate(eta,&gF);
  VecDuplicate(gF,&gR);
  VecCreate(PETSC_COMM_WORLD,&gS);
  VecSetSizes(gS,PETSC_DECIDE,Ny);
  VecSetFromOptions(gS);
  VecDuplicate(gS,&gD);

  // Initialize the linear system

  MatCreate(PETSC_COMM_WORLD,&A);
  MatCreate(PETSC_COMM_WORLD,&D2y);
  MatCreate(PETSC_COMM_WORLD,&D2z);
  MatCreate(PETSC_COMM_WORLD,&Dy_Iz);

  MatCreate(PETSC_COMM_WORLD,&Hinvy);
  MatCreate(PETSC_COMM_WORLD,&Hinvz);

  MatCreate(PETSC_COMM_WORLD,&Hinvy_Izxe0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxeNy_Iz);
  MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_e0z);
  MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_eNz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxe0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxeNy_Iz);

  MatCreate(PETSC_COMM_WORLD,&mu);

  VecCreate(PETSC_COMM_WORLD,&rhs);
  VecSetSizes(rhs,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(rhs);
  VecDuplicate(rhs,&uhat);

  KSPCreate(PETSC_COMM_WORLD,&ksp);

  // initialize time stepping data
  VecDuplicate(eta,&V);
  VecDuplicate(eta,&faultDisp);
  VecCreate(PETSC_COMM_WORLD,&surfDisp);
  VecSetSizes(surfDisp,PETSC_DECIDE,Ny);
  VecSetFromOptions(surfDisp);
  VecDuplicate(eta,&dpsi);
  var = new Vec[2];
  var[0] = faultDisp; var[1] = psi;

  // lousy temporary solution to fact that psi changes in time stepping but can't be handed around as a variable
  VecDuplicate(eta,&tempPsi);

  // viewers (bc PetSc appears to have an error in the way it handles them)
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outFileRoot+"time.txt").c_str(),&timeViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"surfDisp").c_str(),FILE_MODE_WRITE,&surfDispViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"faultDisp").c_str(),FILE_MODE_WRITE,&faultDispViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"vel").c_str(),FILE_MODE_WRITE,&velViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"tau").c_str(),FILE_MODE_WRITE,&tauViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"psi").c_str(),FILE_MODE_WRITE,&psiViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in userContext.cpp.\n");
#endif


}

// destructor
UserContext::~UserContext()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in userContext.cpp.\n");
#endif

  // Elastic Coefficients and Frictional Parameters
  VecDestroy(&eta);
  VecDestroy(&s_NORM);
  VecDestroy(&a);
  VecDestroy(&b);
  VecDestroy(&psi);
  VecDestroy(&tau);
  VecDestroy(&gRShift);

  // boundary conditions
  VecDestroy(&gF);
  VecDestroy(&gS);
  VecDestroy(&gR);
  VecDestroy(&gD);

  // linear system
  VecDestroy(&rhs);

  MatDestroy(&A);
  MatDestroy(&D2y);
  MatDestroy(&D2z);
  MatDestroy(&Dy_Iz);
  MatDestroy(&Hinvy);
  MatDestroy(&Hinvz);

  MatDestroy(&Hinvy_Izxe0y_Iz);
  MatDestroy(&Hinvy_IzxeNy_Iz);
  MatDestroy(&Iy_HinvzxIy_e0z);
  MatDestroy(&Iy_HinvzxIy_eNz);
  MatDestroy(&Hinvy_IzxBySy_IzTxe0y_Iz);
  MatDestroy(&Hinvy_IzxBySy_IzTxeNy_Iz);

  MatDestroy(&mu);
  delete[] muArr;

  KSPDestroy(&ksp);
  VecDestroy(&rhs);
  VecDestroy(&uhat);

  // time stepping system
  VecDestroy(&faultDisp);
  VecDestroy(&surfDisp);
  VecDestroy(&V);
  delete[] var;

  VecDestroy(&tempPsi);

  // viewers
  PetscViewerDestroy(&timeViewer);
  PetscViewerDestroy(&faultDispViewer);
  PetscViewerDestroy(&surfDispViewer);
  PetscViewerDestroy(&velViewer);
  PetscViewerDestroy(&tauViewer);
  PetscViewerDestroy(&psiViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in userContext.cpp.\n");
#endif
}


/***********************************************************************
 * write state
 **********************************************************************/

PetscErrorCode UserContext::writeParameters()
{

  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  std::string str = outFileRoot + "parameters.txt";
  const char * outFileLoc = str.c_str();
  PetscViewer    outviewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &outviewer);
  PetscViewerSetType(outviewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(outviewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(outviewer, outFileLoc);

  ierr = PetscViewerASCIIPrintf(outviewer,"order = %i\n",order);CHKERRQ(ierr);

  // domain geometry
  ierr = PetscViewerASCIIPrintf(outviewer,"Ny = %i\n",Ny);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Nz = %i\n",Nz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Ly = %.15e\n",Ly);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Lz = %.15e\n",Lz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"H = %.15e\n",H);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"dy = %.15e\n",dy);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"dz = %.15e\n",dz);CHKERRQ(ierr);

  //  frictional parameters
  ierr = PetscViewerASCIIPrintf(outviewer,"f0 = %.15e\n",f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"v0 = %.15e\n",v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"vp = %.15e\n",vp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"D_c = %.15e\n",D_c);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"muIn = %.15e\n",muIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"muOut = %.15e\n",muOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rhoIn = %.15e\n",rhoIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rhoOut = %.15e\n",rhoOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"D = %.15e\n",D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"W = %.15e\n",W);CHKERRQ(ierr);

  // tolerances for linear and nonlinear (for vel) solve
  ierr = PetscViewerASCIIPrintf(outviewer,"kspTol = %.15e\n",kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rootTol = %.15e\n",rootTol);CHKERRQ(ierr);

  //  constitutive parameters
  ierr = PetscViewerASCIIPrintf(outviewer,"cs = %.15e\n",cs);CHKERRQ(ierr);
  //~ierr = PetscViewerASCIIPrintf(outviewer,"G = %f\n",G);CHKERRQ(ierr);
  //~ierr = PetscViewerASCIIPrintf(outviewer,"rho = %f\n",rho);CHKERRQ(ierr);

  // time monitering
  ierr = PetscViewerASCIIPrintf(outviewer,"writeStride = %i\n",writeStride);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"checkpointStride = %i\n",checkpointStride);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxStepCount = %i\n",maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"initTime = %.15e\n",initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"currTime = %.15e\n",currTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxTime = %.15e\n",maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"atol = %.15e\n",atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"initDeltaT = %.15e\n",initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"minDeltaT = %.15e\n",minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxDeltaT = %.15e\n",maxDeltaT);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  return 0;
}

PetscErrorCode UserContext::writeOperators()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeOperators in userContext.cpp\n");CHKERRQ(ierr);
#endif

  PetscViewer    outviewer;
  const char     *outFileLoc;

  std::string str =  outFileRoot + "matA";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(A,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Dy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Dy_Iz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Hinvy_Izxe0y_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_Izxe0y_Iz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Hinvy_IzxBySy_IzTxe0y_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxe0y_Iz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Hinvy_IzxeNy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxeNy_Iz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Hinvy_IzxBySy_IzTxeNy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxeNy_Iz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Iy_HinvzxIy_e0z";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_e0z,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

  str = outFileRoot + "Iy_HinvzxIy_eNz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_eNz,outviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&outviewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOperators in userContext.cpp\n");CHKERRQ(ierr);
#endif

  return ierr;
};

PetscErrorCode UserContext::writeRateAndState()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeRateAndState in rateAndState.c\n");CHKERRQ(ierr);
#endif

  PetscViewer    viewer;
  const char * outFileLoc;

  std::string str = outFileRoot + "a"; outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(a,viewer);CHKERRQ(ierr);

  str = outFileRoot + "b"; outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(b,viewer);CHKERRQ(ierr);

  str = outFileRoot + "eta"; outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(eta,viewer);CHKERRQ(ierr);

  str = outFileRoot + "sigma_N"; outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(s_NORM,viewer);CHKERRQ(ierr);

  str = outFileRoot + "mu"; outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(mu,viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeRateAndState in rateAndState.c\n");CHKERRQ(ierr);
#endif

  return ierr;
}

PetscErrorCode UserContext::writeInitialStep()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeInitialStep in userContext.c\n");CHKERRQ(ierr);
#endif

  double startTime = MPI_Wtime();

  PetscViewerASCIIPrintf(timeViewer, "%f\n", currTime);
  ierr = VecView(faultDisp,faultDispViewer);CHKERRQ(ierr);
  ierr = VecView(surfDisp,surfDispViewer);CHKERRQ(ierr);
  ierr = VecView(V,velViewer);CHKERRQ(ierr);
  ierr = VecView(tau,tauViewer);CHKERRQ(ierr);
  ierr = VecView(psi,psiViewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&faultDispViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&surfDispViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&velViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&tauViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&psiViewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"faultDisp").c_str(),
                               FILE_MODE_APPEND,&faultDispViewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"surfDisp").c_str(),
                               FILE_MODE_APPEND,&surfDispViewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"vel").c_str(),
                               FILE_MODE_APPEND,&velViewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"tau").c_str(),
                               FILE_MODE_APPEND,&tauViewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"psi").c_str(),
                               FILE_MODE_APPEND,&psiViewer);CHKERRQ(ierr);

  writeTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeInitialStep in userContext.c\n");CHKERRQ(ierr);
#endif

  return ierr;
}

PetscErrorCode UserContext::writeCurrentStep()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting writeCurrentStep in userContext.cpp.\n");
#endif

  double startTime = MPI_Wtime();

  PetscViewerASCIIPrintf(timeViewer, "%f\n", currTime);
  ierr = VecView(surfDisp,surfDispViewer);CHKERRQ(ierr);
  ierr = VecView(faultDisp,faultDispViewer);CHKERRQ(ierr);
  ierr = VecView(V,velViewer);CHKERRQ(ierr);
  ierr = VecView(tau,tauViewer);CHKERRQ(ierr);
  ierr = VecView(psi,psiViewer);CHKERRQ(ierr);

  writeTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending writeCurrentStep in userContext.cpp.\n");
#endif

  return 0;
}

PetscErrorCode UserContext::writeCurrentState()
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting writeCurrentState in userContext.cpp.\n");
#endif

  double startTime = MPI_Wtime();

  PetscViewer    viewer;

  ierr = writeParameters();CHKERRQ(ierr);
  ierr = writeRateAndState();CHKERRQ(ierr);
  ierr = writeOperators();CHKERRQ(ierr);
  ierr = solver.saveOdeSolver();CHKERRQ(ierr);

  std::string str = outFileRoot + "currPsi";
  const char* outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(psi,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "currTau";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(tau,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "currVel";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(V,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "currgF";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(gF,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "currgR";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(gR,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  writeTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending writeCurrentState in userContext.cpp.\n");
#endif

  return 0;
}

PetscErrorCode UserContext::printTiming()
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting printTiming in userContext.cpp.\n");
#endif

  ierr = PetscPrintf(PETSC_COMM_WORLD,"fullLinOps = %g\n",fullLinOps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"arrLinOps = %g\n",arrLinOps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeTauTime = %g\n",computeTauTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeVelTime = %g\n",computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"kspTime = %g\n",kspTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"computeRhsTime = %g\n",computeRhsTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"agingLawTime = %g\n",agingLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"rhsTime = %g\n",rhsTime);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"rootIts = %i\n",rootIts);CHKERRQ(ierr);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending printTiming in userContext.cpp.\n");
#endif


  return ierr;
}

/***********************************************************************
 * load state
 **********************************************************************/

PetscErrorCode UserContext::loadCurrentState()
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting loadCurrentState in userContext.cpp.\n");
#endif

  ierr = loadParameters();CHKERRQ(ierr);
  ierr = loadOperators();CHKERRQ(ierr);
  ierr = loadRateAndState();CHKERRQ(ierr);


  const char* outFileLoc = (inFileRoot + "currPsi").c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(psi,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecCopy(psi,tempPsi);CHKERRQ(ierr);

  outFileLoc = (inFileRoot + "currTau").c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(tau,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  outFileLoc = (inFileRoot + "currgF").c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(gF,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecCopy(gF,faultDisp);CHKERRQ(ierr);
  ierr = VecScale(faultDisp,2.0);CHKERRQ(ierr);

  outFileLoc = (inFileRoot + "currgR").c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(gR,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  outFileLoc = (inFileRoot + "currVel").c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(V,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending loadCurrentState in userContext.cpp.\n");
#endif

  return ierr;
}

PetscErrorCode UserContext::loadParameters()
{
  PetscErrorCode ierr = 0;

  ifstream infile( (inFileRoot+"parameters.txt").c_str() );
  string line,var;
  string delim = " = ";
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(delim); // find position of delimiter
    var = line.substr(0,pos);

    if (var=="order") { order = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="Ny") { Ny = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="Nz") { Nz = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    N = Ny*Nz;

    if (var=="Ly") { Ly = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="Lz") { Lz = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="H") { H = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="dy") { dy = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="dz") { dz = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }

    if (var=="f0") { f0 = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="v0") { v0 = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="vp") { vp = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="D_c") { D_c = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="muIn") { muIn = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="muOut") { muOut = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="rhoIn") { rhoIn = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="rhoOut") { rhoOut = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="D") { D = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="W") { W = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="cs") { cs = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }

    if (var=="kspTol") { kspTol = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="rootTol") { rootTol = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }

    if (var=="writeStride") { writeStride = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="checkpointStride") { checkpointStride = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="maxStepCount") { maxStepCount = atoi( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="initTime") { initTime = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="currTime") { currTime = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="maxTime") { maxTime = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="atol") { atol = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="initDeltaT") { initDeltaT = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="minDeltaT") { minDeltaT = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
    if (var=="maxDeltaT") { maxDeltaT = atof( (line.substr(pos+delim.length(),line.npos)).c_str() ); }
  }

  return ierr;
}

PetscErrorCode UserContext::loadOperators()
{
  PetscErrorCode  ierr;
  PetscViewer     fd;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function loadOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  //~int size;
  //~MatType matType;
  //~MPI_Comm_size (MPI_COMM_WORLD, &size);
  //~if (size > 1) {matType = MATMPIAIJ;}
  //~else {matType = MATSEQAIJ;}

  std::string str =  inFileRoot + "matA";
  const char* inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  //~ierr = MatSetType(A,matType);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Dy_Iz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Dy_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(Dy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Dy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Hinvy_Izxe0y_Iz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(Hinvy_Izxe0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Hinvy_Izxe0y_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Hinvy_IzxBySy_IzTxe0y_Iz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(Hinvy_IzxBySy_IzTxe0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Hinvy_IzxBySy_IzTxe0y_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Hinvy_IzxeNy_Iz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(Hinvy_IzxeNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Hinvy_IzxeNy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Hinvy_IzxBySy_IzTxeNy_Iz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(Hinvy_IzxBySy_IzTxeNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Hinvy_IzxBySy_IzTxeNy_Iz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Iy_HinvzxIy_e0z";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  //~ierr = MatSetType(Iy_HinvzxIy_e0z,matType);CHKERRQ(ierr);
  ierr = MatLoad(Iy_HinvzxIy_e0z,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  str =  inFileRoot + "Iy_HinvzxIy_eNz";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  //~ierr = MatSetType(Iy_HinvzxIy_eNz,matType);CHKERRQ(ierr);
  ierr = MatLoad(Iy_HinvzxIy_eNz,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function loadOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

    return ierr;
}

PetscErrorCode UserContext::loadRateAndState()
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function loadOperators in linearSysFuncs.c.\n");CHKERRQ(ierr);
#endif

  std::string str =  inFileRoot + "a";
  const char* inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str =  inFileRoot + "b";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str =  inFileRoot + "eta";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(eta,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str =  inFileRoot + "sigma_N";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(s_NORM,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str =  inFileRoot + "mu";
  inFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inFileLoc,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(mu,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  return ierr;
}
