#include <petscts.h>
//~#include <iostream>
#include <sstream>
#include <string>
#include "userContext.h"

// constructor
UserContext::UserContext(const PetscInt ord,const PetscInt y,const  PetscInt z,const std::string outFile)
:outFileRoot(outFile),
 order(ord),Ny(y),Nz(z),N(y*z),
 Ly(0),Lz(0),H(0),dy(0),dz(0),
 //~cs(0),G(0),rho(0),f0(0),v0(0),vp(0),D_c(0),
 cs(0),f0(0),v0(0),vp(0),D_c(0),
 muIn(0),muOut(0),D(0),W(0),rhoIn(0),rhoOut(0),
 kspTol(0),rootTol(0),rootIts(0),
 strideLength(1),maxStepCount(1),initTime(0),currTime(0),maxTime(0),
 minDeltaT(1e-14),maxDeltaT(1e-14),
 count(0),atol(1e-14),initDeltaT(1e-14),
 computeTauTime(0),computeVelTime(0),kspTime(0),computeRhsTime(0),agingLawTime(0),rhsTime(0)
{

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in userContext.cpp.\n");
#endif

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

  MatCreate(PETSC_COMM_WORLD,&IyHinvz_Iye0z);
  MatCreate(PETSC_COMM_WORLD,&IyHinvz_IyeNz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz_e0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz_BySy_Iz_eNy_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz_BySy_Iz_e0y_Iz);
  MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz_eNy_Iz);
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
  VecDuplicate(eta,&dpsi);
  VecCreate(PETSC_COMM_WORLD,&w);
  VecSetSizes(w,PETSC_DECIDE,2*Nz);
  VecSetFromOptions(w);
  TSCreate(PETSC_COMM_WORLD,&ts);
  var = new Vec[2];
  var[0] = faultDisp; var[1] = psi;

  // lousy temporary solution to fact that psi changes in time stepping but can't be handed around as a variable
  VecDuplicate(eta,&tempPsi);

  // viewers (bc PetSc appears to have an error in the way it handles them)
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outFileRoot+"time.txt").c_str(),&timeViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"w").c_str(),FILE_MODE_WRITE,&wViewer);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outFileRoot+"uhat").c_str(),FILE_MODE_WRITE,&uhatViewer);
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
  MatDestroy(&IyHinvz_Iye0z);
  MatDestroy(&IyHinvz_IyeNz);
  MatDestroy(&Hinvy_Iz_e0y_Iz);
  MatDestroy(&Hinvy_Iz_BySy_Iz_e0y_Iz);
  MatDestroy(&Hinvy_Iz_eNy_Iz);
  MatDestroy(&Hinvy_Iz_BySy_Iz_eNy_Iz);
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
  VecDestroy(&w);
  VecDestroy(&faultDisp);
  VecDestroy(&V);
  delete[] var;

  VecDestroy(&tempPsi);

  // viewers
  PetscViewerDestroy(&timeViewer);
  PetscViewerDestroy(&wViewer);
  PetscViewerDestroy(&uhatViewer);
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
  const char * outFileLoc = str.c_str();
  PetscViewer    outviewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &outviewer);
  PetscViewerSetType(outviewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(outviewer, FILE_MODE_APPEND);
  PetscViewerFileSetName(outviewer, outFileLoc);

  ierr = PetscViewerASCIIPrintf(outviewer,"order = %i\n",order);CHKERRQ(ierr);

  // domain geometry
  ierr = PetscViewerASCIIPrintf(outviewer,"Ny = %i\n",Ny);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Nz = %i\n",Nz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Ly = %f\n",Ly);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"Lz = %f\n",Lz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"H = %f\n",H);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"dy = %f\n",dy);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"dz = %f\n",dz);CHKERRQ(ierr);

  //  frictional parameters
  ierr = PetscViewerASCIIPrintf(outviewer,"f0 = %f\n",f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"v0 = %f\n",v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"vp = %f\n",vp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"D_c = %f\n",D_c);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"muIn = %f\n",muIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"muOut = %f\n",muOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rhoIn = %f\n",rhoIn);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rhoOut = %f\n",rhoOut);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"D = %f\n",D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"W = %f\n",W);CHKERRQ(ierr);

  // tolerances for linear and nonlinear (for vel) solve
  ierr = PetscViewerASCIIPrintf(outviewer,"kspTol = %f\n",kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"rootTol = %f\n",rootTol);CHKERRQ(ierr);

  //  constitutive parameters
  //~ierr = PetscViewerASCIIPrintf(outviewer,"cs = %f\n",cs);CHKERRQ(ierr);
  //~ierr = PetscViewerASCIIPrintf(outviewer,"G = %f\n",G);CHKERRQ(ierr);
  //~ierr = PetscViewerASCIIPrintf(outviewer,"rho = %f\n",rho);CHKERRQ(ierr);

  // time monitering
  ierr = PetscViewerASCIIPrintf(outviewer,"strideLength = %i\n",strideLength);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxStepCount = %i\n",maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"initTime = %f\n",initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxTime = %f\n",maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"atol = %f\n",atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"initDeltaT = %f\n",initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"minDeltaT = %f\n",minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(outviewer,"maxDeltaT = %f\n",maxDeltaT);CHKERRQ(ierr);

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

  PetscViewer    outviewer;
  const char     *outFileLoc;

  std::string str =  "A";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(A,outviewer);CHKERRQ(ierr);

  str = "Dy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Dy_Iz,outviewer);CHKERRQ(ierr);

  str = "Hinvy_Izxe0y_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_Izxe0y_Iz,outviewer);CHKERRQ(ierr);

  str = "Hinvy_IzxBySy_IzTxe0y_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxe0y_Iz,outviewer);CHKERRQ(ierr);

  str = "Hinvy_IzxeNy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxeNy_Iz,outviewer);CHKERRQ(ierr);

  str = "Hinvy_IzxBySy_IzTxeNy_Iz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Hinvy_IzxBySy_IzTxeNy_Iz,outviewer);CHKERRQ(ierr);

  str = "Iy_HinvzxIy_e0z";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_e0z,outviewer);CHKERRQ(ierr);

  str = "Iy_HinvzxIy_eNz";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_WRITE,&outviewer);CHKERRQ(ierr);
  ierr = MatView(Iy_HinvzxIy_eNz,outviewer);CHKERRQ(ierr);

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

  PetscViewerASCIIPrintf(timeViewer, "%f\n", currTime);
  ierr = VecView(w,wViewer);CHKERRQ(ierr);
  ierr = VecView(gF,uhatViewer);CHKERRQ(ierr);
  ierr = VecView(V,velViewer);CHKERRQ(ierr);
  ierr = VecView(tau,tauViewer);CHKERRQ(ierr);
  ierr = VecView(psi,psiViewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&uhatViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&velViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&tauViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&psiViewer);CHKERRQ(ierr);

  std::string str = outFileRoot + "w";
  const char * outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_APPEND,&wViewer);CHKERRQ(ierr);

  str = outFileRoot + "uhat";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_APPEND,&uhatViewer);CHKERRQ(ierr);

  str = outFileRoot + "vel";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_APPEND,&velViewer);CHKERRQ(ierr);

  str = outFileRoot + "tau";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_APPEND,&tauViewer);CHKERRQ(ierr);

  str = outFileRoot + "psi";
  outFileLoc = str.c_str();
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outFileLoc,FILE_MODE_APPEND,&psiViewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeParameters in userContext.c\n");CHKERRQ(ierr);
#endif

  return ierr;
}


PetscErrorCode UserContext::writeCurrentStep()
{
  PetscErrorCode ierr;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting writeCurrentStep in userContext.cpp.\n");
#endif

  PetscViewerASCIIPrintf(timeViewer, "%f\n", currTime);
  ierr = VecView(w,wViewer);CHKERRQ(ierr);
  ierr = VecView(faultDisp,uhatViewer);CHKERRQ(ierr);
  ierr = VecView(V,velViewer);CHKERRQ(ierr);
  ierr = VecView(tau,tauViewer);CHKERRQ(ierr);
  ierr = VecView(psi,psiViewer);CHKERRQ(ierr);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending writeCurrentStep in userContext.cpp.\n");
#endif

  return 0;
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
