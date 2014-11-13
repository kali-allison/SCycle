#include "sbpOps.hpp"


//================= constructor and destructor ========================

SbpOps::SbpOps(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(D._muArr),_mu(&D._mu),
  _Hy(_Ny,_Ny),_Hyinv(_Ny,_Ny),_D1y(_Ny,_Ny),_D1yint(_Ny,_Ny),_D2y(_Ny,_Ny),_Sy(_Ny,_Ny),_Iy(_Ny,_Ny),
  _Hz(_Nz,_Nz),_Hzinv(_Nz,_Nz),_D1z(_Nz,_Nz),_D1zint(_Nz,_Nz),_D2z(_Nz,_Nz),_Sz(_Nz,_Nz),_Iz(_Nz,_Nz),
  _alphaF(-13.0/_dy),_alphaR(-13.0/_dy),_alphaS(-1.0),_alphaD(-1.0),_beta(1.0),
  _debugFolder("./matlabAnswers/")
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif


    stringstream ss;
    ss << "order" << _order << "Ny" << _Ny << "Nz" << _Nz << "/";
    _debugFolder += ss.str();


  // create final operator A
  MatCreate(PETSC_COMM_WORLD,&_A); PetscObjectSetName((PetscObject) _A, "_A");

  // map boundary conditions to rhs vector
  MatCreate(PETSC_COMM_WORLD,&_rhsL);
  MatCreate(PETSC_COMM_WORLD,&_rhsR);
  MatCreate(PETSC_COMM_WORLD,&_rhsT);
  MatCreate(PETSC_COMM_WORLD,&_rhsB);

  MatCreate(PETSC_COMM_WORLD,&_AL);
  MatCreate(PETSC_COMM_WORLD,&_AR);
  MatCreate(PETSC_COMM_WORLD,&_AT);
  MatCreate(PETSC_COMM_WORLD,&_AB);


  // Spmats holding 1D SBP operators
  sbpSpmat(_Ny,1/_dy,_Hy,_Hyinv,_D1y,_D1yint,_D2y,_Sy);
  if (_Nz > 1) { sbpSpmat(_Nz,1/_dz,_Hz,_Hzinv,_D1z,_D1zint,_D2z,_Sz); }
  else { _Hz.eye(); }
  _Iy.eye();
  _Iz.eye();

  satBoundaries();
  computeDy_Iz();
  computeA();

  computeH();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in sbpOps.cpp.\n");
#endif
}

SbpOps::~SbpOps()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in sbpOps.cpp.\n");
#endif


  // final operator A
  MatDestroy(&_A);

  // SAT enforcement of boundary conditions
  MatDestroy(&_rhsL);
  MatDestroy(&_rhsR);
  MatDestroy(&_rhsT);
  MatDestroy(&_rhsB);

  MatDestroy(&_AL);
  MatDestroy(&_AR);
  MatDestroy(&_AT);
  MatDestroy(&_AB);

  MatDestroy(&_Dy_Iz);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps.cpp.\n");
#endif
}


//======================== meat ========================================

// enforce SAT boundaries
PetscErrorCode SbpOps::satBoundaries()
{
  PetscErrorCode  ierr = 0;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function satBoundaries in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  // matrices that will be needed multiple times
  // kron(Hyinv,Iz)
  Spmat Hyinv_IzS(_Ny*_Nz,_Ny*_Nz);
  Hyinv_IzS = kron(_Hyinv,_Iz);
  Mat Hyinv_Iz;
  Hyinv_IzS.convert(Hyinv_Iz,1);
  ierr = PetscObjectSetName((PetscObject) Hyinv_Iz, "Hyinv_Iz");CHKERRQ(ierr);


  // mu*kron(BySy,Iz)
  Spmat muxBySy_IzS(_Ny*_Nz,_Ny*_Nz);
  muxBySy_IzS = kron(_Sy,_Iz);
  Mat muxBySy_Iz;
  if (_order==2) { muxBySy_IzS.convert(muxBySy_Iz,3); }
  if (_order==4) { muxBySy_IzS.convert(muxBySy_Iz,5); }
  ierr = MatMatMult(*_mu,muxBySy_Iz,MAT_INITIAL_MATRIX,1.0,&muxBySy_Iz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) muxBySy_Iz, "muxBySy_Iz");CHKERRQ(ierr);

  // kron(E0y,Iz)
  Spmat E0y(_Ny,_Ny);
  E0y(0,0,1.0);
  Spmat E0y_IzS(_Ny*_Nz,_Ny*_Nz);
  E0y_IzS = kron(E0y,_Iz);
  Mat E0y_Iz;
  E0y_IzS.convert(E0y_Iz,1);
  ierr = PetscObjectSetName((PetscObject) E0y_Iz, "E0y_Iz");CHKERRQ(ierr);


  // kron(e0y,Iz)
  Spmat e0y(_Ny,1);
  e0y(0,0,1.0);
  Spmat e0y_IzS(_Ny*_Nz,_Nz);
  e0y_IzS = kron(e0y,_Iz);
  Mat e0y_Iz;
  e0y_IzS.convert(e0y_Iz,1);
  ierr = PetscObjectSetName((PetscObject) e0y_Iz, "e0y_Iz");CHKERRQ(ierr);

  // kron(ENy,Iz)
  Spmat ENy(_Ny,_Ny);
  ENy(_Ny-1,_Ny-1,1.0);
  Spmat ENy_IzS(_Ny*_Nz,_Ny*_Nz);
  ENy_IzS = kron(ENy,_Iz);
  Mat ENy_Iz;
  ENy_IzS.convert(ENy_Iz,1);
  ierr = PetscObjectSetName((PetscObject) ENy_Iz, "ENy_Iz");CHKERRQ(ierr);
  //~MatView(ENy_Iz,PETSC_VIEWER_STDOUT_WORLD);

  // kron(eNy,Iz)
  Spmat eNy(_Ny,1);
  eNy(_Ny-1,0,1.0);
  Spmat eNy_IzS(_Ny*_Nz,_Nz);
  eNy_IzS = kron(eNy,_Iz);
  Mat eNy_Iz;
  eNy_IzS.convert(eNy_Iz,1);
  ierr = PetscObjectSetName((PetscObject) eNy_Iz, "eNy_Iz");CHKERRQ(ierr);


  // enforcement of bcL ================================================
  // map bcL to rhs
  // if bcL = displacement: _alphaF*mu*_Hinvy_Izxe0y_Iz + _beta*_Hinvy_IzxmuxBySy_IzTxe0y_Iz
  ierr = MatMatMatMult(*_mu,Hyinv_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsL);CHKERRQ(ierr);
  ierr = MatScale(_rhsL,_alphaF);CHKERRQ(ierr);
  Mat temp;
  ierr = MatTransposeMatMult(muxBySy_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(Hyinv_Iz,temp,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatAYPX(_rhsL,_beta,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _rhsL, "rhsL");CHKERRQ(ierr);
  //~ierr = MatView(_rhsL,PETSC_VIEWER_STDOUT_WORLD);
  // if bcL = traction-free
  // _rhsL is unneeded bc bcL = 0

  // in computation of A
  // if bcL = displacement: _alphaF*mu*_Hinvy_IzxE0y_Iz + _beta*mu*_Hinvy_IzxBySy_IzTxE0y_Iz
  ierr = MatMatMatMult(*_mu,Hyinv_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_AL);CHKERRQ(ierr);
  ierr = MatScale(_AL,_alphaF);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(muxBySy_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(Hyinv_Iz,temp,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);

  ierr = MatAYPX(_AL,_beta,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _AL, "AL");CHKERRQ(ierr);
  //~ierr = MatView(_AL,PETSC_VIEWER_STDOUT_WORLD);

  // enforcement of bcR ================================================
  // map bcR to rhs
  // if bcR = displacement: _alphaR*mu*_Hinvy_IzxeNy_Iz + _beta*_Hinvy_IzxmuxBySy_IzTxeNy_Iz
  ierr = MatMatMatMult(*_mu,Hyinv_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsR);CHKERRQ(ierr);
  ierr = MatScale(_rhsR,_alphaF);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(muxBySy_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(Hyinv_Iz,temp,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);

  ierr = MatAYPX(_rhsR,_beta,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _rhsR, "rhsR");CHKERRQ(ierr);
  //~ierr = MatView(_rhsR,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  // else if bcR = traction-free
  // _rhsR is unneeded because bcR = 0

  // in computation of A
  // if bcR = displacement: _alphaR*mu*Hinvy_Iz*ENy_Iz + _beta*Hinvy_Iz*(muxBySy_Iz)'*ENy_Iz
  ierr = MatMatMatMult(*_mu,Hyinv_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_AR);CHKERRQ(ierr);
  ierr = MatScale(_AR,_alphaR);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(muxBySy_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatMatMult(Hyinv_Iz,temp,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);

  ierr = MatAYPX(_AR,_beta,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _AR, "AR");CHKERRQ(ierr);
  //~ierr = MatView(_AR,PETSC_VIEWER_STDOUT_WORLD);



  // These matrices are nnz if Nz > 1



  // kron(Iy,Hzinv)
  Spmat Iy_HzinvS(_Ny*_Nz,_Ny*_Nz);
  Iy_HzinvS = kron(_Iy,_Hzinv);
  Mat Iy_Hzinv;
  Iy_HzinvS.convert(Iy_Hzinv,1);
  ierr = PetscObjectSetName((PetscObject) Iy_Hzinv, "Iy_Hzinv");CHKERRQ(ierr);
  //~ierr = MatView(Iy_Hzinv,PETSC_VIEWER_STDOUT_WORLD);

  // mu*kron(Iy,BzSz)
  Spmat muxIy_BzSzS(_Ny*_Nz,_Ny*_Nz);
  muxIy_BzSzS = kron(_Iy,_Sz);
  Mat muxIy_BzSz;
  if (_order==2) { muxIy_BzSzS.convert(muxIy_BzSz,3); }
  if (_order==4) { muxIy_BzSzS.convert(muxIy_BzSz,5); }
  ierr = MatMatMult(*_mu,muxIy_BzSz,MAT_INITIAL_MATRIX,1.0,&muxIy_BzSz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) muxIy_BzSz, "muxIy_BzSz");CHKERRQ(ierr);

  // kron(Iy,E0z)
  Spmat E0z(_Nz,_Nz);
  E0z(0,0,1.0);
  Spmat Iy_E0zS(_Ny*_Nz,_Ny*_Nz);
  Iy_E0zS = kron(_Iy,E0z);
  Mat Iy_E0z;
  Iy_E0zS.convert(Iy_E0z,1);
  ierr = PetscObjectSetName((PetscObject) Iy_E0z, "Iy_E0z");CHKERRQ(ierr);

  // kron(Iy,e0z)
  Spmat e0z(_Nz,1);
  e0z(0,0,1.0);
  Spmat Iy_e0zS(_Ny*_Nz,_Nz);
  Iy_e0zS = kron(_Iy,e0z);
  Mat Iy_e0z;
  Iy_e0zS.convert(Iy_e0z,1);
  ierr = PetscObjectSetName((PetscObject) Iy_e0z, "Iy_e0z");CHKERRQ(ierr);

  // kron(Iy,ENz)
  Spmat ENz(_Nz,_Nz);
  ENz(_Nz-1,_Nz-1,1.0);
  Spmat Iy_ENzS(_Ny*_Nz,_Ny*_Nz);
  Iy_ENzS = kron(_Iy,ENz);
  Mat Iy_ENz;
  Iy_ENzS.convert(Iy_ENz,1);
  ierr = PetscObjectSetName((PetscObject) Iy_ENz, "Iy_ENz");CHKERRQ(ierr);

  // kron(Iy,eNz)
  Spmat eNz(_Nz,1);
  eNz(_Nz-1,0,1.0);
  Spmat Iy_eNzS(_Ny*_Nz,_Nz);
  Iy_eNzS = kron(_Iy,eNz);
  Mat Iy_eNz;
  Iy_eNzS.convert(Iy_eNz,1);
  ierr = PetscObjectSetName((PetscObject) Iy_eNz, "Iy_eNz");CHKERRQ(ierr);


  // enforcement of bcT ================================================
  // map bcS to rhs
  // if bcT = traction:
  ierr = MatMatMult(Iy_Hzinv,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsT);CHKERRQ(ierr);
  ierr = MatScale(_rhsT,_alphaS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _rhsT, "rhsT");CHKERRQ(ierr);
  //~ierr = MatView(_rhsT,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // in computation of A
  // if bcT = traction-free: _alphaS*Iy_Hinvz*Iy_E0z*muxIy_BzSz
  ierr = MatMatMatMult(Iy_Hzinv,Iy_E0z,muxIy_BzSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_AT);CHKERRQ(ierr);
  ierr = MatScale(_AT,_alphaS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _AT, "AT");CHKERRQ(ierr);
  //~ierr = MatView(_AT,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // enforcement of bcB ================================================
  // map bcB to rhs
  // if bcB = traction:
  ierr = MatMatMult(Iy_Hzinv,Iy_eNz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsB);CHKERRQ(ierr);
  ierr = MatScale(_rhsB,_alphaD);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _rhsB, "rhsB");CHKERRQ(ierr);
  //~ierr = MatView(_rhsB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // in computation of A
  // if bcB = traction-free: _alphaD*Iy_Hinvz*Iy_E0z*muxIy_BzSz
  ierr = MatMatMatMult(Iy_Hzinv,Iy_ENz,muxIy_BzSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_AB);CHKERRQ(ierr);
  ierr = MatScale(_AB,_alphaD);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _AB, "AB");CHKERRQ(ierr);
  //~ierr = MatView(_AB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  _runTime += MPI_Wtime() - startTime;

  MatDestroy(&temp);
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function satBoundaries in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}



// compute D2ymu using my class Spmat
PetscErrorCode SbpOps::computeD2ymu(Mat &D2ymu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


  // kron(Dy,Iz) (interior stencil)
  Mat Dy_Iz;
  Spmat Dy_IzS(_Ny*_Nz,_Ny*_Nz);
  Dy_IzS = kron(_D1yint,_Iz);
  if (_order==2) { Dy_IzS.convert(Dy_Iz,2); }
  else if (_order==4) { Dy_IzS.convert(Dy_Iz,5); }
  ierr = PetscObjectSetName((PetscObject) Dy_Iz, "Dyint_Iz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&Dy_Iz,_debugFolder,"Dyint_Iz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


  // mu*kron(BySy,Iz)
  Mat muxBySy_Iz;
  Spmat muxBySy_IzS(_Ny*_Nz,_Ny*_Nz);
  muxBySy_IzS = kron(_Sy,_Iz);
  if (_order==2) { muxBySy_IzS.convert(muxBySy_Iz,3); }
  if (_order==4) { muxBySy_IzS.convert(muxBySy_Iz,5); }
  ierr = MatMatMult(*_mu,muxBySy_Iz,MAT_INITIAL_MATRIX,1.0,&muxBySy_Iz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) muxBySy_Iz, "muxBySy_Iz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = MatView(muxBySy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(muxBySy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


  // mu*kron(Hy,Iz)
  Mat muxHy_Iz;
  Spmat muxHy_IzS(_Ny*_Nz,_Ny*_Nz);
  muxHy_IzS = kron(_Hy,_Iz);
  muxHy_IzS.convert(muxHy_Iz,1);
  ierr = MatMatMult(*_mu,muxHy_Iz,MAT_INITIAL_MATRIX,1.0,&muxHy_Iz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) muxHy_Iz, "muxHy_Iz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&muxHy_Iz,_debugFolder,"muxHy_Iz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(muxHy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

  // kron(Hinvy,Iz)
  Mat Hinvy_Iz;
  Spmat Hyinv_IzS(_Ny*_Nz,_Ny*_Nz);
  Hyinv_IzS = kron(_Hyinv,_Iz);
  Hyinv_IzS.convert(Hinvy_Iz,1);
  ierr = PetscObjectSetName((PetscObject) Hinvy_Iz, "Hinvy_Iz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&Hinvy_Iz,_debugFolder,"Hinvy_Iz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(Hinvy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

  Mat Rymu;
  ierr = computeRymu(Rymu,_order);CHKERRQ(ierr);

  //~Mat D2ymu;
  ierr = MatTransposeMatMult(Dy_Iz,muxHy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = MatMatMult(D2ymu,Dy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = MatScale(D2ymu,-1);CHKERRQ(ierr);
  ierr = MatAXPY(D2ymu,-1,Rymu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D2ymu,1,muxBySy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(Hinvy_Iz,D2ymu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&D2ymu,_debugFolder,"D2ymu");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(D2ymu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

  MatDestroy(&Dy_Iz);
  MatDestroy(&muxBySy_Iz);
  MatDestroy(&muxHy_Iz);
  MatDestroy(&Hinvy_Iz);
  MatDestroy(&Rymu);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SbpOps::computeRzmu(Mat &Rzmu,PetscInt order)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


switch ( order ) {
    case 2:
    {
      Spmat C2z(_Nz,_Nz);
      C2z.eye(); C2z(0,0,0); C2z(_Nz-1,_Nz-1,0);

      // kron(Iy,C2z)
      Mat Iy_C2z;
      Spmat Iy_C2zS(_Ny*_Nz,_Ny*_Nz);
      Iy_C2zS = kron(_Iy,C2z);
      Iy_C2zS.convert(Iy_C2z,1);
      ierr = PetscObjectSetName((PetscObject) Iy_C2z, "Iy_C2zz");CHKERRQ(ierr);
      #if DEBUG > 0
        ierr = checkMatrix(&Iy_C2z,_debugFolder,"Iy_Cz");CHKERRQ(ierr);
      #endif
      #if VERBOSE > 2
        ierr = MatView(Iy_C2z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      #endif

      // kron(Iy,D2z)
      Mat Iy_D2z;
      Spmat Iy_D2zS(_Ny*_Nz,_Ny*_Nz);
      Iy_D2zS = kron(_Iy,_D2z);
      Iy_D2zS.convert(Iy_D2z,5);
      ierr = PetscObjectSetName((PetscObject) Iy_D2z, "Iy_D2z");CHKERRQ(ierr);
      #if DEBUG > 0
        ierr = checkMatrix(&Iy_D2z,_debugFolder,"Iy_D2z");CHKERRQ(ierr);
      #endif
      #if VERBOSE > 2
        ierr = MatView(Iy_D2z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      #endif

      // Rzmu = (Iy_D2z^T x Iy_C2z x mu x Iy_D2z)/4/dz^3;
      ierr = MatTransposeMatMult(Iy_D2z,Iy_C2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatMatMult(Rzmu,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatMatMult(Rzmu,Iy_D2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatScale(Rzmu,0.25*pow(_dz,3));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rzmu, "Rzmu");CHKERRQ(ierr);

      MatDestroy(&Iy_D2z);
      MatDestroy(&Iy_C2z);
      break;
    }

    case 4:
    {
      Spmat D3z(_Nz,_Nz);
      Spmat D4z(_Nz,_Nz);
      Spmat C3z(_Nz,_Nz);
      Spmat C4z(_Nz,_Nz);
      sbpSpmat4(_Nz,1/_dz,D3z,D4z,C3z,C4z);

      Spmat B3(_Ny*_Nz,_Ny*_Nz);
      B3(0,0,0.5*(_muArr[0]+_muArr[1]));
      B3(_Ny*_Nz-1,_Ny*_Nz-1,0.5*(_muArr[_Ny*_Nz-2]+_muArr[_Ny*_Nz-1]));
      for (PetscInt Ii=1;Ii<_Ny*_Nz-1;Ii++)
      {
        B3(Ii,Ii,0.5*(_muArr[Ii]+_muArr[Ii+1]));
      }
      Mat mu3;
      //~Spmat B3(_Ny,_Ny);
      //~B3.eye();
      B3.convert(mu3,1);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nB3:\n");
      //~B3.printPetsc();


      Spmat Iy_D3zS(_Ny*_Nz,_Ny*_Nz);
      Iy_D3zS = kron(_Iy,D3z);
      Mat Iy_D3z;
      Iy_D3zS.convert(Iy_D3z,6);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\n Iy_D3zS:\n");
      //~Iy_D3zS.printPetsc();

      Spmat Iy_D4zS(_Ny*_Nz,_Ny*_Nz);
      Iy_D4zS = kron(_Iy,D4z);
      Mat Iy_D4z;
      Iy_D4zS.convert(Iy_D4z,5);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\n Iy_D4zS:\n");
      //~Iy_D4zS.printPetsc();

      Spmat Iy_C3zS(_Ny*_Nz,_Ny*_Nz);
      Iy_C3zS = kron(_Iy,C3z);
      Mat Iy_C3z;
      Iy_C3zS.convert(Iy_C3z,1);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\n Iy_C3zS:\n");
      //~Iy_C3zS.printPetsc();

      Spmat Iy_C4zS(_Ny*_Nz,_Ny*_Nz);
      Iy_C4zS = kron(_Iy,C4z);
      Mat Iy_C4z;
      Iy_C4zS.convert(Iy_C4z,1);

      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
      Mat temp;
      ierr = MatTransposeMatMult(Iy_D3z,Iy_C3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMult(temp,mu3,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMult(temp,Iy_D3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatScale(temp,1.0/_dz/18);CHKERRQ(ierr);

      ierr = MatTransposeMatMult(Iy_D4z,Iy_C4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatMatMult(Rzmu,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatMatMult(Rzmu,Iy_D4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatScale(Rzmu,1.0/_dz/144);CHKERRQ(ierr);

      ierr = MatAYPX(Rzmu,1.0,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rzmu, "Rzmu");CHKERRQ(ierr);

      MatDestroy(&Iy_D3z);
      MatDestroy(&Iy_D4z);
      MatDestroy(&Iy_C3z);
      MatDestroy(&Iy_C4z);
      MatDestroy(&mu3);
      break;
    }
    default:
      SETERRQ(PETSC_COMM_WORLD,1,"order not understood.");
      break;
  }


#if DEBUG > 0
  ierr = checkMatrix(&Rzmu,_debugFolder,"Rzmu");CHKERRQ(ierr);
#endif
#if VERBOSE > 2
  ierr = MatView(Rzmu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeR2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode SbpOps::computeRymu(Mat &Rymu,PetscInt order)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


switch ( order ) {
    case 2:
    {
      // kron(D2y,Iz)
      Mat D2y_Iz;
      Spmat D2y_IzS(_Ny*_Nz,_Ny*_Nz);
      D2y_IzS = kron(_D2y,_Iz);
      D2y_IzS.convert(D2y_Iz,5);
      ierr = PetscObjectSetName((PetscObject) D2y_Iz, "D2y_Iz");CHKERRQ(ierr);
      #if DEBUG > 0
        ierr = checkMatrix(&D2y_Iz,_debugFolder,"D2y_Iz");CHKERRQ(ierr);
      #endif
      #if VERBOSE > 2
        ierr = MatView(D2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      #endif

      Spmat C2y(_Ny,_Ny);
      C2y.eye(); C2y(0,0,0); C2y(_Ny-1,_Ny-1,0);

      // kron(C2y,Iz)
      Mat C2y_Iz;
      Spmat C2y_IzS(_Ny*_Nz,_Ny*_Nz);
      C2y_IzS = kron(C2y,_Iz);
      C2y_IzS.convert(C2y_Iz,5);
      ierr = PetscObjectSetName((PetscObject) C2y_Iz, "C2y_Iz");CHKERRQ(ierr);
      #if DEBUG > 0
        ierr = MatView(C2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      #endif
      #if VERBOSE > 2
        ierr = MatView(C2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      #endif

      // Rymu = (D2y_Iz^T x C2y_Iz x mu x D2y_Iz)/4/dy^3;
      ierr = MatTransposeMatMult(D2y_Iz,C2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatMatMult(Rymu,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatMatMult(Rymu,D2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatScale(Rymu,0.25*pow(_dy,3));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rymu, "Rymu");CHKERRQ(ierr);

      MatDestroy(&D2y_Iz);
      MatDestroy(&C2y_Iz);

    break;
  }

    case 4:
    {
      Spmat D3y(_Ny,_Ny);
      Spmat D4y(_Ny,_Ny);
      Spmat C3y(_Ny,_Ny);
      Spmat C4y(_Ny,_Ny);
      sbpSpmat4(_Ny,1/_dy,D3y,D4y,C3y,C4y);

      Spmat B3(_Ny*_Nz,_Ny*_Nz);
      B3(0,0,0.5*(_muArr[0]+_muArr[1]));
      B3(_Ny*_Nz-1,_Ny*_Nz-1,0.5*(_muArr[_Ny*_Nz-2]+_muArr[_Ny*_Nz-1]));
      for (PetscInt Ii=1;Ii<_Ny*_Nz-1;Ii++)
      {
        B3(Ii,Ii,0.5*(_muArr[Ii]+_muArr[Ii+1]));
      }
      Mat mu3;
      //~Spmat B3(_Ny,_Ny);
      //~B3.eye();
      B3.convert(mu3,1);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nB3:\n");
      //~B3.printPetsc();


      Spmat D3y_IzS(_Ny*_Nz,_Ny*_Nz);
      D3y_IzS = kron(D3y,_Iz);
      //~Spmat D3y_IzS(_Ny,_Ny);
      //~D3y_IzS = D3y;
      Mat D3y_Iz;
      D3y_IzS.convert(D3y_Iz,6);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nD3y_IzS:\n");
      //~D3y_IzS.printPetsc();

      Spmat D4y_IzS(_Ny*_Nz,_Ny*_Nz);
      D4y_IzS = kron(D4y,_Iz);
      //~Spmat D4y_IzS(_Ny,_Ny);
      //~D4y_IzS = D4y;
      Mat D4y_Iz;
      D4y_IzS.convert(D4y_Iz,5);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nD4y_IzS:\n");
      //~D4y_IzS.printPetsc();

      Spmat C3y_IzS(_Ny*_Nz,_Ny*_Nz);
      C3y_IzS = kron(C3y,_Iz);
      //~Spmat C3y_IzS(_Ny,_Ny);
      //~C3y_IzS = C3y;
      Mat C3y_Iz;
      C3y_IzS.convert(C3y_Iz,1);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nC3y_IzS:\n");
      //~C3y_IzS.printPetsc();

      Spmat C4y_IzS(_Ny*_Nz,_Ny*_Nz);
      C4y_IzS = kron(C4y,_Iz);
      //~Spmat C4y_IzS(_Ny,_Ny);
      //~C4y_IzS = C4y;
      Mat C4y_Iz;
      C4y_IzS.convert(C4y_Iz,1);
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nC4y_IzS:\n");
      //~C4y_IzS.printPetsc();

      // Rymu = (D3y_Iz^T x C3y_Iz x mu3 x D3y_Iz)/18/dy
      //      + (D4y_Iz^T x C4y_Iz x mu x D4y_Iz)/144/dy
      Mat temp;
      ierr = MatTransposeMatMult(D3y_Iz,C3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMult(temp,mu3,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMult(temp,D3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatScale(temp,1.0/_dy/18.0);CHKERRQ(ierr);

      ierr = MatTransposeMatMult(D4y_Iz,C4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatMatMult(Rymu,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      //~ierr = MatMatMult(Rymu,mu3,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatMatMult(Rymu,D4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatScale(Rymu,1.0/_dy/144.0);CHKERRQ(ierr);

      ierr = MatAYPX(Rymu,1.0,temp,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rymu, "Rymu");CHKERRQ(ierr);

      MatDestroy(&D3y_Iz);
      MatDestroy(&C3y_Iz);
      MatDestroy(&D4y_Iz);
      MatDestroy(&C4y_Iz);
      MatDestroy(&mu3);
      break;
    }
    default:
      SETERRQ(PETSC_COMM_WORLD,1,"order not understood.");
      break;
  }
  #if DEBUG > 0
    ierr = checkMatrix(&Rymu,_debugFolder,"Rymu");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(Rymu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeR2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// compute D2zmu using my class Spmat
PetscErrorCode SbpOps::computeD2zmu(Mat &D2zmu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


// kron(Iy,Dz)
  Mat Iy_Dz;
  Spmat Iy_DzS(_Ny*_Nz,_Ny*_Nz);
  Iy_DzS = kron(_Iy,_D1zint);
  if (_order==2) { Iy_DzS.convert(Iy_Dz,2); }
  else if (_order==4) { Iy_DzS.convert(Iy_Dz,5); }
  ierr = PetscObjectSetName((PetscObject) Iy_Dz, "Iy_Dz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&Iy_Dz,_debugFolder,"Iy_Dz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


  // mu*kron(Iy,BzSz)
  Mat muxIy_BzSz;
  Spmat muxIy_BzSzS(_Ny*_Nz,_Ny*_Nz);
  muxIy_BzSzS = kron(_Iy,_Sz);
  if (_order==2) { muxIy_BzSzS.convert(muxIy_BzSz,3); }
  else if (_order==4) { muxIy_BzSzS.convert(muxIy_BzSz,5); }
  ierr = PetscObjectSetName((PetscObject) muxIy_BzSz, "muxIy_BzSz");CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,muxIy_BzSz,MAT_INITIAL_MATRIX,1.0,&muxIy_BzSz);CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&muxIy_BzSz,_debugFolder,"muxIy_BzSz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(muxIy_BzSz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

  // mu*kron(Iy,Hz)
  Mat muxIy_Hz;
  Spmat muxIy_HzS(_Ny*_Nz,_Ny*_Nz);
  muxIy_HzS = kron(_Iy,_Hz);
  muxIy_HzS.convert(muxIy_Hz,1);
  ierr = PetscObjectSetName((PetscObject) muxIy_Hz, "muxIy_Hz");CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,muxIy_Hz,MAT_INITIAL_MATRIX,1.0,&muxIy_Hz);CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&muxIy_Hz,_debugFolder,"muxIy_Hz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(muxIy_Hz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif

  // kron(Iy,Hinvz)
  Mat Iy_Hinvz;
  Spmat Iy_HinvzS(_Ny*_Nz,_Ny*_Nz);
  Iy_HinvzS = kron(_Iy,_Hzinv);
  Iy_HinvzS.convert(Iy_Hinvz,1);
  ierr = PetscObjectSetName((PetscObject) Iy_Hinvz, "Iy_Hinvz");CHKERRQ(ierr);
  #if DEBUG > 0
    ierr = checkMatrix(&Iy_Hinvz,_debugFolder,"Iy_Hinvz");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(Iy_Hinvz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


  Mat Rzmu;
  ierr = computeRzmu(Rzmu,_order);

  //~Mat D2zmu;
  ierr = MatTransposeMatMult(Iy_Dz,muxIy_Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = MatMatMult(D2zmu,Iy_Dz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = MatScale(D2zmu,-1);CHKERRQ(ierr);
  ierr = MatAXPY(D2zmu,-1,Rzmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D2zmu,1,muxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(Iy_Hinvz,D2zmu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);
#if DEBUG > 0
ierr = checkMatrix(&D2zmu,_debugFolder,"D2zmu");CHKERRQ(ierr);
#endif
#if VERBOSE > 2
ierr = MatView(D2zmu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  MatDestroy(&Iy_Dz);
  MatDestroy(&muxIy_BzSz);
  MatDestroy(&muxIy_Hz);
  MatDestroy(&Iy_Hinvz);
  MatDestroy(&Rzmu);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}



// compute matrix for shear stress (scaled by mu!!): kron(Dy,Iz)
PetscErrorCode SbpOps::computeDy_Iz()
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeDy_Iz in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  Spmat Sy_Iz(_Ny*_Nz,_Ny*_Nz);
  Sy_Iz = kron(_D1y,_Iz);
  Sy_Iz.convert(_Dy_Iz,5);

  ierr = MatMatMult(*_mu,_Dy_Iz,MAT_INITIAL_MATRIX,1.0,&_Dy_Iz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _Dy_Iz, "_Dy_Iz");CHKERRQ(ierr);

#if DEBUG > 0
ierr = checkMatrix(&_Dy_Iz,_debugFolder,"Dy_Iz");CHKERRQ(ierr);
#endif
#if VERBOSE > 2
  ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeDy_Iz in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// compute matrix relating displacement to vector b containing boundary conditions
PetscErrorCode SbpOps::computeA()
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeA in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  Mat D2ymu;
  ierr = MatCreate(PETSC_COMM_WORLD,&D2ymu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);
  ierr = computeD2ymu(D2ymu);CHKERRQ(ierr);

  Mat D2zmu;
  ierr = MatCreate(PETSC_COMM_WORLD,&D2zmu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);
  ierr = computeD2zmu(D2zmu);CHKERRQ(ierr);

  Mat D2mu;
  ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&D2mu);CHKERRQ(ierr);
  ierr = MatAYPX(D2mu,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // compute A
  ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
  ierr = MatAYPX(_A,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // use new Mats _AL etc
  ierr = MatAXPY(_A,1.0,_AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,_AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,_AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,_AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"matA");CHKERRQ(ierr);
  //~//ierr = MatView(_A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
//~
  // clean up
  ierr = MatDestroy(&D2ymu);CHKERRQ(ierr);
  ierr = MatDestroy(&D2zmu);CHKERRQ(ierr);


  _runTime = MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeA in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return 0;
}

// Hinv = kron(Hy,Hz)
PetscErrorCode SbpOps::computeH()
{
  PetscErrorCode ierr = 0;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeH in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  // kron(Hyinv, Hzinv)
  //~Spmat Hyinv_Hzinv(_Ny*_Nz,_Ny*_Nz);
  //~Hyinv_Hzinv = kron(_Hyinv,_Hzinv);
  //~Hyinv_Hzinv.convert(_H,1);

  // kron(Hy,Hz)
  Spmat Hy_Hz(_Ny*_Nz,_Ny*_Nz);
  Hy_Hz = kron(_Hy,_Hz);
  Hy_Hz.convert(_H,1);

  ierr = PetscObjectSetName((PetscObject) _H, "H");CHKERRQ(ierr);
  //~ierr = MatView(_H,PETSC_VIEWER_STDOUT_WORLD);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeH in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return ierr;
}

PetscErrorCode SbpOps::sbpSpmat4(const PetscInt N,const PetscScalar scale,
                Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpSpmat4 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  if (N < 2) { return ierr; }

  PetscInt Ii = 0;

  D3(0,0,-1);D3(0,1,3);D3(0,2,-3);D3(0,3,1); // 1st row
  D3(1,0,-1);D3(1,1,3);D3(1,2,-3);D3(1,3,1); // 2nd row
  D3(2,0,-185893.0/301051.0); // 3rd row
  D3(2,1,79000249461.0/54642863857.0);
  D3(2,2,-33235054191.0/54642863857.0);
  D3(2,3,-36887526683.0/54642863857.0);
  D3(2,4,26183621850.0/54642863857.0);
  D3(2,5,-4386.0/181507.0);
  for (Ii=3;Ii<N-4;Ii++)
  {
    D3(Ii,Ii-1,-1.0);
    D3(Ii,Ii,3);
    D3(Ii,Ii+1,-3);
    D3(Ii,Ii+2,1.0);
  }
  D3(N-3,N-1,-D3(2,0));// third to last row
  D3(N-3,N-2,-D3(2,1));
  D3(N-3,N-3,-D3(2,2));
  D3(N-3,N-4,-D3(2,3));
  D3(N-3,N-5,-D3(2,4));
  D3(N-3,N-6,-D3(2,5));
  D3(N-2,N-4,-1);D3(N-2,N-3,3);D3(N-2,N-2,-3);D3(N-2,N-1,1); // 2nd to last row
  D3(N-1,N-4,-1);D3(N-1,N-3,3);D3(N-1,N-2,-3);D3(N-1,N-1,1); // last row
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD3:\n");CHKERRQ(ierr);
  //~D3.printPetsc();


  D4(0,0,1); D4(0,1,-4); D4(0,2,6); D4(0,3,-4); D4(0,4,1); // 1st row
  D4(1,0,1); D4(1,1,-4); D4(1,2,6); D4(1,3,-4); D4(1,4,1); // 1st row
  for (Ii=2;Ii<N-2;Ii++)
  {
    D4(Ii,Ii-2,1);
    D4(Ii,Ii-1,-4);
    D4(Ii,Ii,6);
    D4(Ii,Ii+1,-4);
    D4(Ii,Ii+2,1);
  }
  D4(N-2,N-5,1); D4(N-2,N-4,-4); D4(N-2,N-3,6); D4(N-2,N-2,-4); D4(N-2,N-1,1); // 2nd to last row
  D4(N-1,N-5,1); D4(N-1,N-4,-4); D4(N-1,N-3,6); D4(N-1,N-2,-4); D4(N-1,N-1,1); // last row
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD4:\n");CHKERRQ(ierr);
  //~D4.printPetsc();


  C3.eye();
  C3(0,0,0);
  C3(1,1,0);
  C3(2,2,163928591571.0/53268010936.0);
  C3(3,3,189284.0/185893.0);
  C3(N-5,N-5,C3(3,3));
  C3(N-4,N-4,0);
  C3(N-3,N-3,C3(2,2));
  C3(N-2,N-2,0);
  C3(N-1,N-1,0);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nC3:\n");CHKERRQ(ierr);
  //~C3.printPetsc();

  C4.eye();
  C4(0,0,0);
  C4(1,1,0);
  C4(2,2,1644330.0/301051.0);
  C4(3,3,156114.0/181507.0);
  C4(N-4,N-4,C4(3,3));
  C4(N-3,N-3,C4(2,2));
  C4(N-2,N-2,0);
  C4(N-1,N-1,0);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nC4:\n");CHKERRQ(ierr);
  //~C4.printPetsc();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbpSpmat4 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SbpOps::sbpSpmat(const PetscInt N,const PetscScalar scale,Spmat& H,Spmat& Hinv,Spmat& D1,
                 Spmat& D1int, Spmat& D2, Spmat& S)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpSpmat in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

PetscInt Ii=0;

switch ( _order ) {
    case 2:
    {
      H.eye(); H(0,0,0.5); H(N-1,N-1,0.5); H.scale(1/scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        //~Hinv.printPetsc();
      #endif


      S(0,0,1.5*scale);     S(0,1,-2.0*scale);      S(0,2,0.5*scale); // -1* p666 of Mattsson 2010
      S(N-1,N-3,0.5*scale); S(N-1,N-2,-2.0*scale);  S(N-1,N-1,1.5*scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nS:\n");CHKERRQ(ierr);
        S.printPetsc();
      #endif

      D1int(0,0,-1.0*scale);D1int(0,1,scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1int(Ii,Ii-1,-0.5*scale);
        D1int(Ii,Ii+1,0.5*scale);
      }
      D1int(N-1,N-1,scale);D1int(N-1,N-2,-1*scale); // last row
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1int:\n");CHKERRQ(ierr);
        D1int.printPetsc();
      #endif

      //~D1 = D1int; // copy D1int's interior
      //~D1(N-1,N-3,S(N-1,N-3)); // last row
      //~D1(N-1,N-2,S(N-1,N-2));
      //~D1(N-1,N-1,S(N-1,N-1));
      //~D1 = S;
      // only want shear stress on fault, not interior
      D1(0,0,-S(0,0)); D1(0,1,-S(0,1)); D1(0,2,-S(0,2)); // first row
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1:\n");CHKERRQ(ierr);
        D1.printPetsc();
      #endif

      D2(0,0,scale*scale); D2(0,1,-2.0*scale*scale); D2(0,2,scale*scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D2(Ii,Ii-1,scale*scale);
        D2(Ii,Ii,-2.0*scale*scale);
        D2(Ii,Ii+1,scale*scale);
      }
      D2(N-1,N-3,scale*scale);D2(N-1,N-2,-2.0*scale*scale);D2(N-1,N-1,scale*scale); // last row
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD2:\n");CHKERRQ(ierr);
        D2.printPetsc();
      #endif

      break;
    }
    case 4:
    {
      assert(N>8); // N must be >8 for 4th order SBP
      //~if (N<8) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      H.eye();
      H(0,0,17.0/48.0);
      H(1,1,59.0/48.0);
      H(2,2,43.0/48.0);
      H(3,3,49.0/48.0);
      H(N-1,N-1,17.0/48.0);
      H(N-2,N-2,59.0/48.0);
      H(N-3,N-3,43.0/48.0);
      H(N-4,N-4,49.0/48.0);
      H.scale(1/scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
      #endif

      // row 1: -1* p666 of Mattsson 2010
      S(0,0,11.0/6.0); S(0,1,-3.0); S(0,2,1.5); S(0,3,-1.0/3.0);
      S(N-1,N-1,11.0/6.0); S(N-1,N-2,-3.0); S(N-1,N-3,1.5); S(N-1,N-4,-1.0/3.0);
      S.scale(scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nS:\n");CHKERRQ(ierr);
        S.printPetsc();
      #endif

      // interior stencil for 1st derivative, scaled by multiplication with Hinv's values
      for (Ii=4;Ii<N-4;Ii++)
      {
        D1int(Ii,Ii-2,1.0/12.0*Hinv(Ii,Ii));
        D1int(Ii,Ii-1,-2.0/3.0*Hinv(Ii,Ii));
        D1int(Ii,Ii+1,2.0/3.0*Hinv(Ii,Ii));
        D1int(Ii,Ii+2,-1.0/12.0*Hinv(Ii,Ii));
      }

      // closures
      D1int(0,0,-1.0/2.0*Hinv(0,0)); // row 0
      D1int(0,1,59.0/96.0*Hinv(0,0));
      D1int(0,2,-1.0/12.0*Hinv(0,0));
      D1int(0,3,-1.0/32.0*Hinv(0,0));
      D1int(1,0,-59.0/96.0*Hinv(1,1)); // row 1
      D1int(1,2,59.0/96.0*Hinv(1,1));
      D1int(2,0,1.0/12.0*Hinv(2,2)); // row 2
      D1int(2,1,-59.0/96.0*Hinv(2,2));
      D1int(2,3,59.0/96.0*Hinv(2,2));
      D1int(2,4,-1.0/12.0*Hinv(2,2));
      D1int(3,0,1.0/32.0*Hinv(3,3)); // row 3
      D1int(3,2,-59.0/96.0*Hinv(3,3));
      D1int(3,4,2.0/3.0*Hinv(3,3));
      D1int(3,5,-1.0/12.0*Hinv(3,3));

      D1int(N-1,N-1,1.0/2.0*Hinv(N-1,N-1)); // row N-1
      D1int(N-1,N-2,-59.0/96.0*Hinv(N-1,N-1));
      D1int(N-1,N-3,1.0/12.0*Hinv(N-1,N-1));
      D1int(N-1,N-4,1.0/32.0*Hinv(N-1,N-1));
      D1int(N-2,N-1,59.0/96.0*Hinv(N-2,N-2)); // row N-2
      D1int(N-2,N-3,-59.0/96.0*Hinv(N-2,N-2));
      D1int(N-3,N-1,-1.0/12.0*Hinv(N-3,N-3)); // row N-3
      D1int(N-3,N-2,59.0/96.0*Hinv(N-3,N-3));
      D1int(N-3,N-4,-59.0/96.0*Hinv(N-3,N-3));
      D1int(N-3,N-5,1.0/12.0*Hinv(N-3,N-3));
      D1int(N-4,N-1,-1.0/32.0*Hinv(N-4,N-4)); // row N-4
      D1int(N-4,N-3,59.0/96.0*Hinv(N-4,N-4));
      D1int(N-4,N-5,-2.0/3.0*Hinv(N-4,N-4));
      D1int(N-4,N-6,1.0/12.0*Hinv(N-4,N-4));
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1int:\n");CHKERRQ(ierr);
        D1int.printPetsc();
      #endif


      // 1st derivative, same interior stencil but transition at boundaries
      //~D1 = D1int;
      //~D1(N-1,N-1,S(N-1,N-1)); D1(N-1,N-2,S(N-1,N-2)); D1(N-1,N-3,S(N-1,N-3)); D1(N-1,N-4,S(N-1,N-4)); // last row
      D1 = S; // only need 1st derivative on boundaries, not interior
      D1(0,0,-S(0,0)); D1(0,1,-S(0,1)); D1(0,2,-S(0,2)); D1(0,3,-S(0,3));
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1:\n");CHKERRQ(ierr);
        D1.printPetsc();
      #endif

      break;
    }

    default:
      SETERRQ(PETSC_COMM_WORLD,1,"order not understood.");
      break;
  }


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbpSpmat in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}




//======================== public member functions =====================

// map the boundary condition vectors to rhs
PetscErrorCode SbpOps::setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcS,Vec &_bcD)
{
  PetscErrorCode ierr = 0;

  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function setRhs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  // rhs =  _alphaF*mu*_Hinvy_Izxe0y_Iz*_bcF +...
  // + _beta*_Hinvy_IzxBySy_IzTxe0y_Iz*_bcF + ...
  // + _alphaR*mu*_Hinvy_IzxeNy_Iz*_bcR + ...
  // + _beta*_Hinvy_IzxBySy_IzTxeNy_Iz*_bcR + ...
  // - _alphaS*M.Iy_HinvzxIy_e0z*_bcS + ...
  // + _alphaD*M.Iy_HinvzxIy_eNz*_bcD

  // using new naming conventions
  ierr = VecSet(rhs,0.0);
  ierr = MatMult(_rhsL,_bcF,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcF
  ierr = MatMultAdd(_rhsR,_bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
  ierr = MatMultAdd(_rhsT,_bcS,rhs,rhs);
  ierr = MatMultAdd(_rhsB,_bcD,rhs,rhs);


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function setRhs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  _runTime += MPI_Wtime() - startTime;

  return ierr;
}



//======================= I/O functions ================================

PetscErrorCode SbpOps::loadOps(const std::string inputDir)
{
  PetscErrorCode  ierr = 0;
  PetscViewer     viewer;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function loadOps in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  double startTime = MPI_Wtime();

  int size;
  MatType matType;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  if (size > 1) {matType = MATMPIAIJ;}
  else {matType = MATSEQAIJ;}

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"A",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_A);CHKERRQ(ierr);
  ierr = MatSetType(_A,matType);CHKERRQ(ierr);
  ierr = MatLoad(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Dy_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Dy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(_Dy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Dy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Izxe0y_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(_Hinvy_Izxe0y_Iz,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Hinvy_Izxe0y_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxBySy_IzTxe0y_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(_Hinvy_IzxBySy_IzTxe0y_Iz,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Hinvy_IzxBySy_IzTxe0y_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxeNy_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(_Hinvy_IzxeNy_Iz,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Hinvy_IzxeNy_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxBySy_IzTxeNy_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  //~ierr = MatSetType(_Hinvy_IzxBySy_IzTxeNy_Iz,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Hinvy_IzxBySy_IzTxeNy_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Iy_HinvzxIy_e0z",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  //~ierr = MatSetType(_Iy_HinvzxIy_e0z,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Iy_HinvzxIy_e0z,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Iy_HinvzxIy_eNz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  //~ierr = MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  //~ierr = MatSetType(_Iy_HinvzxIy_eNz,matType);CHKERRQ(ierr);
  //~ierr = MatLoad(_Iy_HinvzxIy_eNz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function loadOps in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
    return ierr;
}


PetscErrorCode SbpOps::writeOps(const std::string outputDir)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeOps in sbpOps.c\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();
  PetscViewer    viewer;

  std::string str =  outputDir + "matA";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Dy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Dy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // matrices to map SAT boundaries to rhs
  str = outputDir + "rhsL";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsL,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "rhsR";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsR,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "rhsT";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsT,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "rhsB";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsB,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // matrices to map SAT boundaries to A
  str = outputDir + "AL";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_AL,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "AR";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsR,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "AT";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_rhsT,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "AB";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_AB,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);



  //~str = outputDir + "Hinvy_Izxe0y_Iz";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Hinvy_Izxe0y_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "Hinvy_IzxBySy_IzTxe0y_Iz";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Hinvy_IzxBySy_IzTxe0y_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "Hinvy_IzxeNy_Iz";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Hinvy_IzxeNy_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "Hinvy_IzxBySy_IzTxeNy_Iz";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Hinvy_IzxBySy_IzTxeNy_Iz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "Iy_HinvzxIy_e0z";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Iy_HinvzxIy_e0z,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "Iy_HinvzxIy_eNz";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_Iy_HinvzxIy_eNz,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOps in sbpOps.cpp\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
};




