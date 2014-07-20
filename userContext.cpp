#include <petscts.h>
#include <petscviewerhdf5.h>
//~#include <iostream>
#include <sstream>
#include <string>
#include "userContext.h"

// constructor
UserContext::UserContext(const PetscInt ord,const PetscInt y,const  PetscInt z,const std::string outFile)
:outFileRoot(outFile),
 order(ord),Ny(y),Nz(z),N(y*z),
 Ly(0),Lz(0),H(0),dy(0),dz(0),
 cs(0),f0(0),v0(0),vp(0),D_c(0),
 muIn(0),muOut(0),D(0),W(0),rhoIn(0),rhoOut(0),
 kspTol(0),rootTol(0),rootIts(0),
 strideLength(1),maxStepCount(1),initTime(0),currTime(0),maxTime(0),
 minDeltaT(1e-14),maxDeltaT(1e-14),
 count(0),atol(1e-14),initDeltaT(1e-14),
 computeTauTime(0),computeVelTime(0),kspTime(0),computeRhsTime(0),agingLawTime(0),rhsTime(0),
 fullLinOps(0),arrLinOps(0)
{

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in userContext.cpp.\n");
#endif

  std::ostringstream convert;
  convert << "./matlabAnswers/order" << order << "Ny" << y << "Nz" << z << "/";
  debugFolder = convert.str();

  //~PetscViewerHDF5Open(PETSC_COMM_WORLD, "data.h5", FILE_MODE_WRITE, &viewer);
  //~PetscViewerHDF5PushGroup(viewer, "/");
  //~PetscViewerHDF5PushGroup(viewer, "/timeSeries");
  //~PetscViewerHDF5SetTimestep(viewer, 0);


  // elastic frictional parameters
  VecCreate(PETSC_COMM_WORLD,&eta);
  VecSetSizes(eta,PETSC_DECIDE,Nz);
  VecSetFromOptions(eta);     PetscObjectSetName((PetscObject) eta, "eta");
  VecDuplicate(eta,&sigma_N); PetscObjectSetName((PetscObject) sigma_N, "sigma_N");
  VecDuplicate(eta,&a);       PetscObjectSetName((PetscObject) a, "frica");
  VecDuplicate(eta,&b);       PetscObjectSetName((PetscObject) b, "fricb");
  VecDuplicate(eta,&psi);     PetscObjectSetName((PetscObject) psi, "psi");
  VecDuplicate(eta,&tempPsi);
  VecDuplicate(eta,&tau);     PetscObjectSetName((PetscObject) tau, "tau_qs");
  VecDuplicate(eta,&gRShift);
  MatCreate(PETSC_COMM_WORLD,&mu);     PetscObjectSetName((PetscObject) mu, "mu");
  muArr = new PetscScalar[Ny*Nz];

  // boundary conditions
  VecDuplicate(eta,&gF);
  VecDuplicate(gF,&gR);
  VecCreate(PETSC_COMM_WORLD,&gS);
  VecSetSizes(gS,PETSC_DECIDE,Ny);
  VecSetFromOptions(gS);
  VecDuplicate(gS,&gD);

  // linear system
  MatCreate(PETSC_COMM_WORLD,&A);
  MatCreate(PETSC_COMM_WORLD,&Dy_Iz);

  MatCreate(PETSC_COMM_WORLD,&Hinvy_Izxe0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxeNy_Iz);
  MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_e0z);
  MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_eNz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxe0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxeNy_Iz);

  VecCreate(PETSC_COMM_WORLD,&rhs);
  VecSetSizes(rhs,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(rhs);
  VecDuplicate(rhs,&uhat);

  KSPCreate(PETSC_COMM_WORLD,&ksp);

  // initialize time stepping fields
  VecDuplicate(eta,&vel);                  PetscObjectSetName((PetscObject) vel, "vel");
  VecDuplicate(eta,&faultDisp);            PetscObjectSetName((PetscObject) faultDisp, "faultDisp");
  VecCreate(PETSC_COMM_WORLD,&surfDisp);
  VecSetSizes(surfDisp,PETSC_DECIDE,Ny);
  VecSetFromOptions(surfDisp);             PetscObjectSetName((PetscObject) surfDisp, "surfDisp");
  VecDuplicate(eta,&dpsi);
  var = new Vec[2];
  var[0] = faultDisp; var[1] = psi;

  // viewers
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

  // elastic coefficients and frictional parameters
  VecDestroy(&eta);
  VecDestroy(&sigma_N);
  VecDestroy(&a);
  VecDestroy(&b);
  VecDestroy(&psi);
  VecDestroy(&tempPsi);
  VecDestroy(&dpsi);
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
  MatDestroy(&Dy_Iz);

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
  VecDestroy(&vel);
  delete[] var;

  // viewers
  //~PetscViewerDestroy(&viewer);
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

PetscErrorCode UserContext::writeParameters()
{

  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  std::string str = outFileRoot + "parameters.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"order = %i\n",order);CHKERRQ(ierr);

  // domain geometry
  ierr = PetscViewerASCIIPrintf(viewer,"Ny = %i\n",Ny);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Nz = %i\n",Nz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Ly = %f\n",Ly);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Lz = %f\n",Lz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"H = %f\n",H);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"dy = %f\n",dy);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"dz = %f\n",dz);CHKERRQ(ierr);

  // parameters
  ierr = PetscViewerASCIIPrintf(viewer,"f0 = %f\n",f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %f\n",v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"vp = %f\n",vp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D_c = %f\n",D_c);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"muIn = %f\n",muIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"muOut = %f\n",muOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rhoIn = %f\n",rhoIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rhoOut = %f\n",rhoOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"D = %f\n",D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"W = %f\n",W);CHKERRQ(ierr);

  // tolerances for linear and nonlinear (for vel) solve
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %f\n",kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"rootTol = %f\n",rootTol);CHKERRQ(ierr);

  // time monitering
  ierr = PetscViewerASCIIPrintf(viewer,"strideLength = %i\n",strideLength);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e\n",initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e\n",maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"atol = %f\n",atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initDeltaT = %.15e\n",initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"minDeltaT = %.15e\n",minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxDeltaT = %.15e\n",maxDeltaT);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  return 0;
}

PetscErrorCode UserContext::writeOperators()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeOperators in userContext.c\n");CHKERRQ(ierr);
#endif

  PetscViewer    viewer;

  std::string str =  "A";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Dy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Dy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Hinvy_Izxe0y_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_Izxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Hinvy_IzxBySy_IzTxe0y_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Hinvy_IzxeNy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Hinvy_IzxBySy_IzTxeNy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Iy_HinvzxIy_e0z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_e0z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = "Iy_HinvzxIy_eNz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_eNz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  return ierr;
};

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
  ierr = VecView(vel,velViewer);CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  return ierr;
}


PetscErrorCode UserContext::writeStep()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting writeCurrentStep in userContext.cpp.\n");
#endif

  double startTime = MPI_Wtime();

  //~ierr = PetscViewerHDF5PushGroup(viewer, "/timeSeries");CHKERRQ(ierr);
//~
  //~ierr = VecView(vel, viewer);CHKERRQ(ierr);
  //~ierr = VecView(psi, viewer);CHKERRQ(ierr);
  //~ierr = VecView(tau, viewer);CHKERRQ(ierr);
  //~ierr = VecView(surfDisp, viewer);CHKERRQ(ierr);
  //~ierr = VecView(faultDisp, viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);


  PetscViewerASCIIPrintf(timeViewer, "%f\n", currTime);

  ierr = VecView(faultDisp,faultDispViewer);CHKERRQ(ierr);
  ierr = VecView(surfDisp,surfDispViewer);CHKERRQ(ierr);
  ierr = VecView(vel,velViewer);CHKERRQ(ierr);
  ierr = VecView(tau,tauViewer);CHKERRQ(ierr);
  ierr = VecView(psi,psiViewer);CHKERRQ(ierr);


  writeTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending writeCurrentStep in userContext.cpp.\n");
#endif

  return 0;
}


PetscErrorCode UserContext::writeRateAndState()
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeRateAndState in rateAndState.c\n");CHKERRQ(ierr);
#endif


  //~ierr = PetscViewerHDF5PushGroup(viewer, "/frictionContext");CHKERRQ(ierr);
//~
  //~ierr = VecView(a, viewer);CHKERRQ(ierr);
  //~ierr = VecView(b, viewer);CHKERRQ(ierr);
  //~ierr = VecView(eta, viewer);CHKERRQ(ierr);
  //~ierr = VecView(sigma_N, viewer);CHKERRQ(ierr);
  //~ierr = MatView(mu, viewer);CHKERRQ(ierr);
//~
//~
  //~ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  PetscViewer    viewer;

  std::string str = outFileRoot + "a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "eta";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(eta,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(sigma_N,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outFileRoot + "mu";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(mu,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeRateAndState in rateAndState.c\n");CHKERRQ(ierr);
#endif

  return ierr;
}


PetscErrorCode UserContext::printTiming()
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting printTiming in userContext.cpp.\n");
#endif

  ierr = PetscPrintf(PETSC_COMM_WORLD,"fullLinOps = %g\n",fullLinOps);CHKERRQ(ierr);
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


PetscErrorCode UserContext::loadCurrentPlace()
{
  PetscErrorCode ierr = 0;
  PetscViewer    fd;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"dataMatlab/psi",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(psi,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"dataMatlab/faultDisp",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D->A);CHKERRQ(ierr);
  //~ierr = MatSetType(D->A,matType);CHKERRQ(ierr);
  ierr = VecLoad(faultDisp,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  currTime=4.376188933923020e+05;



  return ierr;
}
