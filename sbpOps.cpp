#include "sbpOps.hpp"


//================= constructor and destructor ========================

SbpOps::SbpOps(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(D._muArr),_mu(&D._mu),
  _Sylen(0),_Szlen(0),
  _Hy(_Ny,_Ny),_HyinvS(_Ny,_Ny),_D1yS(_Ny,_Ny),_D1yintS(_Ny,_Ny),_D2yS(_Ny,_Ny),_SyS(_Ny,_Ny),_Iy(_Ny,_Ny),
  _Hz(_Nz,_Nz),_HzinvS(_Nz,_Nz),_D1zS(_Nz,_Nz),_D1zintS(_Nz,_Nz),_D2zS(_Nz,_Nz),_SzS(_Nz,_Nz),_Iz(_Nz,_Nz),
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
  MatCreate(PETSC_COMM_WORLD,&_Hinvy_Izxe0y_Iz);
    PetscObjectSetName((PetscObject) _Hinvy_Izxe0y_Iz, "_Hinvy_Izxe0y_Iz");
  MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxeNy_Iz);
    PetscObjectSetName((PetscObject) _Hinvy_IzxeNy_Iz, "_Hinvy_IzxeNy_Iz");
  MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_e0z);
    PetscObjectSetName((PetscObject) _Iy_HinvzxIy_e0z, "_Iy_HinvzxIy_e0z");
  MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_eNz);
    PetscObjectSetName((PetscObject) _Iy_HinvzxIy_eNz, "_Iy_HinvzxIy_eNz");
  MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxe0y_Iz);
    PetscObjectSetName((PetscObject) _Hinvy_IzxBySy_IzTxe0y_Iz, "_Hinvy_IzxBySy_IzTxe0y_Iz");
  MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxeNy_Iz);
    PetscObjectSetName((PetscObject) _Hinvy_IzxBySy_IzTxeNy_Iz, "_Hinvy_IzxBySy_IzTxeNy_Iz");


  // arrays holding 1D SBP operators
  PetscMalloc(_Ny*sizeof(PetscScalar),&_HinvyArr);
  PetscMalloc(2*_Ny*sizeof(PetscScalar),&_SyArr);
  PetscMalloc(_Ny*_Ny*sizeof(PetscScalar),&_D1y);
  PetscMalloc(_Ny*_Ny*sizeof(PetscScalar),&_D1yint);
  PetscMalloc(_Ny*_Ny*sizeof(PetscScalar),&_D2y);
  sbpArrays(_Ny,1/_dy,_HinvyArr,_D1y,_D1yint,_D2y,_SyArr,&_Sylen);

  PetscMalloc(_Nz*sizeof(PetscScalar),&_HinvzArr);
  PetscMalloc(2*_Nz*sizeof(PetscScalar),&_SzArr);
  PetscMalloc(_Nz*_Nz*sizeof(PetscScalar),&_D1z);
  PetscMalloc(_Nz*_Nz*sizeof(PetscScalar),&_D1zint);
  PetscMalloc(_Nz*_Nz*sizeof(PetscScalar),&_D2z);
  sbpArrays(_Nz,1/_dz,_HinvzArr,_D1z,_D1zint,_D2z,_SzArr,&_Szlen);

  // Spmats holding 1D SBP operators
  sbpSpmat(_Ny,1/_dy,_Hy,_HyinvS,_D1yS,_D1yintS,_D2yS,_SyS);
  sbpSpmat(_Nz,1/_dz,_Hz,_HzinvS,_D1zS,_D1zintS,_D2zS,_SzS);
  _Iy.eye();
  _Iz.eye();


  MatCreate(PETSC_COMM_WORLD,&_Dy_Iz);
  PetscObjectSetName((PetscObject) _Dy_Iz, "Dy_Iz");

  computeDy_Iz();
  computeA();
  computeRhsFactors();

  PetscErrorCode computeH();


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

  // map boundary conditions to rhs vector
  MatDestroy(&_Hinvy_Izxe0y_Iz);
  MatDestroy(&_Hinvy_IzxeNy_Iz);
  MatDestroy(&_Iy_HinvzxIy_e0z);
  MatDestroy(&_Iy_HinvzxIy_eNz);
  MatDestroy(&_Hinvy_IzxBySy_IzTxe0y_Iz);
  MatDestroy(&_Hinvy_IzxBySy_IzTxeNy_Iz);

  MatDestroy(&_Dy_Iz);

  PetscFree(_HinvyArr);
  PetscFree(_D1y);
  PetscFree(_D1yint);
  PetscFree(_D2y);
  PetscFree(_SyArr);

  PetscFree(_HinvzArr);
  PetscFree(_D1z);
  PetscFree(_D1zint);
  PetscFree(_D2z);
  PetscFree(_SzArr);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps.cpp.\n");
#endif
}


//======================== meat ========================================

PetscErrorCode SbpOps::computeRhsFactors()
{
  PetscErrorCode  ierr = 0;
  PetscScalar     v,*vals;
  PetscInt        Ii,J,Istart,Iend,indx,*cols;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeRhsFactors in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  PetscMalloc(_Nz*sizeof(PetscInt),&cols);
  PetscMalloc(_Nz*sizeof(PetscScalar),&vals);

  // Hinvy_Izxe0y_Iz = kron(Hinvy,Iz)*kron(e0y,Iz)
  ierr = MatSetSizes(_Hinvy_Izxe0y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Hinvy_Izxe0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Hinvy_Izxe0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Hinvy_Izxe0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<_Nz;Ii++) {
    ierr = MatSetValues(_Hinvy_Izxe0y_Iz,1,&Ii,1,&Ii,&(_HinvyArr[0]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_Hinvy_Izxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Hinvy_Izxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,_Hinvy_Izxe0y_Iz,MAT_INITIAL_MATRIX,1.0,&_Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_Hinvy_Izxe0y_Iz,_debugFolder,"Hinvy_Izxe0y_Iz");CHKERRQ(ierr);
#endif

  // Hinvy_IzxeNy_Iz = kron(Hinvy,Iz)*kron(eNy,Iz)
  ierr = MatSetSizes(_Hinvy_IzxeNy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Hinvy_IzxeNy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Hinvy_IzxeNy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Hinvy_IzxeNy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  J = _Ny*_Nz - _Nz;
  for (Ii=Iend-1;Ii>=_Ny*_Nz-_Nz;Ii--) {
    indx = Ii-J;
    ierr = MatSetValues(_Hinvy_IzxeNy_Iz,1,&Ii,1,&indx,&(_HinvyArr[_Ny-1]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_Hinvy_IzxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Hinvy_IzxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,_Hinvy_IzxeNy_Iz,MAT_INITIAL_MATRIX,1.0,&_Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_Hinvy_IzxeNy_Iz,_debugFolder,"Hinvy_IzxeNy_Iz");CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_e0z = kron(Iz,Hinvz)*kron(Iy,e0z)
  ierr = MatSetSizes(_Iy_HinvzxIy_e0z,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Iy_HinvzxIy_e0z,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Iy_HinvzxIy_e0z,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Iy_HinvzxIy_e0z,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/_Nz;J=Ii-indx*_Nz;
    if (J==0) {
      ierr = MatSetValues(_Iy_HinvzxIy_e0z,1,&Ii,1,&indx,&_HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(_Iy_HinvzxIy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Iy_HinvzxIy_e0z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_Iy_HinvzxIy_e0z,_debugFolder,"Iy_HinvzxIy_e0z");CHKERRQ(ierr);
#endif

  // Iy_HinvzxIy_eNz = kron(Iy,Hinvz)*kron(Iy,eNz)
  ierr = MatSetSizes(_Iy_HinvzxIy_eNz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Iy_HinvzxIy_eNz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Iy_HinvzxIy_eNz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Iy_HinvzxIy_eNz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/_Nz;J=Ii+1-J*_Nz;
    if (J==0) {
      indx = Ii/_Nz;
      ierr = MatSetValues(_Iy_HinvzxIy_eNz,1,&Ii,1,&indx,&_HinvzArr[J],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(_Iy_HinvzxIy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Iy_HinvzxIy_eNz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_Iy_HinvzxIy_eNz,_debugFolder,"Iy_HinvzxIy_eNz");CHKERRQ(ierr);
#endif

  // Hinvy_IzxBySy_IzTxe0y_Iz = kron(Hinvy,Iz)*kron(BySy,Iz)^T*kron(e0y,Iz)
  ierr = MatSetSizes(_Hinvy_IzxBySy_IzTxe0y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Hinvy_IzxBySy_IzTxe0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Hinvy_IzxBySy_IzTxe0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Hinvy_IzxBySy_IzTxe0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<_Nz*_Sylen;Ii++) {
    indx = Ii-(Ii/_Nz)*_Nz;
    v = _muArr[indx]*_HinvyArr[Ii/_Nz]*_SyArr[Ii/_Nz];
    ierr = MatSetValues(_Hinvy_IzxBySy_IzTxe0y_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_Hinvy_IzxBySy_IzTxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Hinvy_IzxBySy_IzTxe0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_Hinvy_IzxBySy_IzTxe0y_Iz,_debugFolder,"Hinvy_IzxBySy_IzTxe0y_Iz");CHKERRQ(ierr);
#endif


  // Hinvy_IzxBySy_IzTxeNy_Iz = kron(Hinvy,Iz)*[mu*kron(BySy,Iz)]^T*kron(eNy,Iz)
  ierr = MatSetSizes(_Hinvy_IzxBySy_IzTxeNy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Hinvy_IzxBySy_IzTxeNy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Hinvy_IzxBySy_IzTxeNy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Hinvy_IzxBySy_IzTxeNy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>=_Ny*_Nz-_Nz*_Sylen) {
      indx = Ii-(Ii/_Nz)*_Nz;
      v = _muArr[_Ny*_Nz-(_Nz-indx)]*_HinvyArr[Ii/_Nz]*_SyArr[2*_Sylen-1-(_Ny*_Nz-1-Ii)/_Nz];
      ierr = MatSetValues(_Hinvy_IzxBySy_IzTxeNy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(_Hinvy_IzxBySy_IzTxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Hinvy_IzxBySy_IzTxeNy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&_Hinvy_IzxBySy_IzTxeNy_Iz,_debugFolder,"Hinvy_IzxBySy_IzTxeNy_Iz");CHKERRQ(ierr);
#endif

  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  _runTime += MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function createRhsFactors in sbpOps.cpp.\n");CHKERRQ(ierr);
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
  Dy_IzS = kron(_D1yintS,_Iz);
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
  muxBySy_IzS = kron(_SyS,_Iz);
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
  Hyinv_IzS = kron(_HyinvS,_Iz);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
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
      Iy_D2zS = kron(_Iy,_D2zS);
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
      //~ierr = MatMatMatMult(Iy_D2zT,Iy_C2z,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeR2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode SbpOps::computeRymu(Mat &Rymu,PetscInt order)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


switch ( order ) {
    case 2:
    {
      // kron(D2y,Iz)
      Mat D2y_Iz;
      Spmat D2y_IzS(_Ny*_Nz,_Ny*_Nz);
      D2y_IzS = kron(_D2yS,_Iz);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeR2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// compute D2zmu using my class Spmat
PetscErrorCode SbpOps::computeD2zmu(Mat &D2zmu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


// kron(Iy,Dz)
  Mat Iy_Dz;
  Spmat Iy_DzS(_Ny*_Nz,_Ny*_Nz);
  Iy_DzS = kron(_Iy,_D1zintS);
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
  muxIy_BzSzS = kron(_Iy,_SzS);
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
  Iy_HinvzS = kron(_Iy,_HzinvS);
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
  //~PetscScalar     v=0;
  //~PetscInt        Istart,Iend; // range of matrix rows stored locally
  //~PetscInt        Ii, Jj; // local matrix row and col indices
  //~PetscInt        Irow,Icol; // array indices

  //~ierr = MatSetSizes(_Dy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  //~ierr = PetscObjectSetName((PetscObject) _Dy_Iz, "Dy_Iz");CHKERRQ(ierr);
  //~ierr = MatSetFromOptions(_Dy_Iz);CHKERRQ(ierr);
  //~ierr = MatMPIAIJSetPreallocation(_Dy_Iz,_Sylen,NULL,_Sylen,NULL);CHKERRQ(ierr);
  //~ierr = MatSeqAIJSetPreallocation(_Dy_Iz,_Sylen,NULL);CHKERRQ(ierr);
  //~ierr = MatSetUp(_Dy_Iz);CHKERRQ(ierr);
  //~ierr = MatGetOwnershipRange(_Dy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  //~for (Ii=Istart;Ii<Iend;Ii++)
  //~{
    //~Irow = Ii/_Nz;
    //~for (Icol=0;Icol<_Ny;Icol++) // iterate over columns in Dy
    //~{
      //~assert(Irow*_Ny+Icol<_Ny*_Ny);
      //~v = _muArr[Ii]*_D1y[Irow*_Ny+Icol];
      //~Jj = Ii%_Nz + Icol*_Nz; // map cols in Dy to cols in _Dy_Iz
      //~assert(Jj<_Ny*_Nz);
      //~if (v!=0) {
        //~ierr = MatSetValues(_Dy_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES);
        //~}
    //~}
  //~}
  //~ierr = MatAssemblyBegin(_Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatAssemblyEnd(_Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  Spmat Sy_Iz(_Ny*_Nz,_Ny*_Nz);
  Sy_Iz = kron(_D1yS,_Iz);
  Sy_Iz.convert(_Dy_Iz,5);


  ierr = MatMatMult(*_mu,_Dy_Iz,MAT_INITIAL_MATRIX,1.0,&_Dy_Iz);CHKERRQ(ierr);

#if DEBUG > 0
//~ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
ierr = checkMatrix(&_Dy_Iz,_debugFolder,"Dy_Iz");CHKERRQ(ierr);
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
  PetscScalar     v = 0;
  //~PetscInt        Ii,J,Istart,Iend,ncols,indx,*cols,Jj;
  PetscInt        Ii,J,Istart,Iend,indx,*cols;
  PetscScalar *vals;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeA in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  PetscMalloc(_Nz*sizeof(PetscInt),&cols);
  PetscMalloc(_Nz*sizeof(PetscScalar),&vals);



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


  // Hinvy_IzxE0y_Iz = kron(Hinvy,Iz)*kron(E0y,Iz)
  Mat Hinvy_IzxE0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxE0y_Iz);
  PetscObjectSetName((PetscObject) Hinvy_IzxE0y_Iz, "Hinvy_IzxE0y_Iz");
  ierr = MatSetSizes(Hinvy_IzxE0y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxE0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxE0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxE0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<_Nz;Ii++) {
    ierr = MatSetValues(Hinvy_IzxE0y_Iz,1,&Ii,1,&Ii,&(_HinvyArr[0]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,Hinvy_IzxE0y_Iz,MAT_INITIAL_MATRIX,1.0,&Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxE0y_Iz,_debugFolder,"Hinvy_IzxUE0y_Iz");CHKERRQ(ierr);
  //~ierr = MatView(Hinvy_IzxE0y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  // Hinvy_IzxENy_Iz = kron(Hinvy,Iz)*kron(ENy,Iz)
  Mat Hinvy_IzxENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxENy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  J = _Ny*_Nz - _Nz;
  for (Ii=Iend-1;Ii>=_Ny*_Nz-_Nz;Ii--) {
    ierr = MatSetValues(Hinvy_IzxENy_Iz,1,&Ii,1,&Ii,&(_HinvyArr[_Ny-1]),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,Hinvy_IzxENy_Iz,MAT_INITIAL_MATRIX,1.0,&Hinvy_IzxENy_Iz);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxENy_Iz,_debugFolder,"Hinvy_IzxUENy_Iz");CHKERRQ(ierr);
#endif


  // Hinvy_IzxmuxBySy_IzTxE0y_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(E0y,Iz)
  Mat Hinvy_IzxmuxBySy_IzTxE0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxmuxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxmuxBySy_IzTxE0y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxmuxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxmuxBySy_IzTxE0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxmuxBySy_IzTxE0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxmuxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxmuxBySy_IzTxE0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<_Nz*_Sylen;Ii++) {
    indx = Ii-(Ii/_Nz)*_Nz;
    v = _muArr[indx]*_HinvyArr[Ii/_Nz]*_SyArr[Ii/_Nz];
    ierr = MatSetValues(Hinvy_IzxmuxBySy_IzTxE0y_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxmuxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxmuxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxmuxBySy_IzTxE0y_Iz,_debugFolder,"Hinvy_IzxmuxBySy_IzTxUE0y_Iz");CHKERRQ(ierr);
#endif


  // Hinvy_IzxmuxBySy_IzTxENy_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(ENy,Iz)
  Mat Hinvy_IzxmuxBySy_IzTxENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxmuxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxmuxBySy_IzTxENy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxmuxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxmuxBySy_IzTxENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxmuxBySy_IzTxENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxmuxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxmuxBySy_IzTxENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>=_Ny*_Nz-_Nz*_Sylen) {
      indx = (_Ny-1)*_Nz + Ii-(Ii/_Nz)*_Nz;
      v = _muArr[indx]*_HinvyArr[Ii/_Nz]*_SyArr[2*_Sylen-1-(_Ny*_Nz-1-Ii)/_Nz];
      ierr = MatSetValues(Hinvy_IzxmuxBySy_IzTxENy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Hinvy_IzxmuxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxmuxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxmuxBySy_IzTxENy_Iz,_debugFolder,"Hinvy_IzxmuxBySy_IzTxUENy_Iz");CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_E0zxmuxmuxIy_BzSz = kron(Iy,_Hinvz)*kron(Iy,E0z)*mu*kron(Iy,BzSz)
  Mat Iy_HinvzxIy_E0zxmuxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_E0zxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_E0zxmuxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_E0zxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_E0zxmuxIy_BzSz,_Szlen,NULL,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_E0zxmuxIy_BzSz,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_E0zxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_E0zxmuxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/_Nz;J=Ii-indx*_Nz;
    if (J==0) {
      for (indx=0;indx<_Szlen;indx++) {
        cols[indx]=Ii+indx;
        vals[indx]=_muArr[Ii]*_HinvzArr[J]*_SzArr[indx];
      }
      ierr = MatSetValues(Iy_HinvzxIy_E0zxmuxIy_BzSz,1,&Ii,_Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_E0zxmuxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_E0zxmuxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Iy_HinvzxIy_E0zxmuxIy_BzSz,_debugFolder,"Iy_HinvzxIy_E0zxmuxIy_BzSz");CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_ENzxmuxIy_BzSz = kron(Iy,_Hinvz)*kron(Iy,ENz)*mu*kron(Iy,BzSz)
  Mat Iy_HinvzxIy_ENzxmuxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_ENzxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_ENzxmuxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_ENzxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_ENzxmuxIy_BzSz,_Szlen,NULL,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_ENzxmuxIy_BzSz,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_ENzxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_ENzxmuxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/_Nz;J=Ii+1-J*_Nz;
    if (J==0) {
      for (indx=0;indx<_Szlen;indx++) {
        cols[indx]=Ii-_Szlen+1+indx;
        vals[indx]=_muArr[Ii]*_HinvzArr[J]*_SzArr[_Sylen+indx];
      }
      ierr = MatSetValues(Iy_HinvzxIy_ENzxmuxIy_BzSz,1,&Ii,_Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_ENzxmuxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_ENzxmuxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvzxIy_ENzxmuxIy_BzSz,_debugFolder,"Iy_HinvzxIy_ENzxmuxIy_BzSz");CHKERRQ(ierr);
#endif

  // compute A
  ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
  ierr = MatAYPX(_A,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  //~ierr = MatView(_A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage1");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_alphaF,Hinvy_IzxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage2");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_beta,Hinvy_IzxmuxBySy_IzTxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage3");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_alphaR,Hinvy_IzxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage4");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_beta,Hinvy_IzxmuxBySy_IzTxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,_alphaS,Iy_HinvzxIy_E0zxmuxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,_alphaD,Iy_HinvzxIy_ENzxmuxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"matA");CHKERRQ(ierr);
#endif

  // clean up
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  ierr = MatDestroy(&D2ymu);CHKERRQ(ierr);
  ierr = MatDestroy(&D2zmu);CHKERRQ(ierr);
  ierr = MatDestroy(&Hinvy_IzxE0y_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Hinvy_IzxENy_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Hinvy_IzxmuxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Hinvy_IzxmuxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Iy_HinvzxIy_E0zxmuxIy_BzSz);CHKERRQ(ierr);
  ierr = MatDestroy(&Iy_HinvzxIy_ENzxmuxIy_BzSz);CHKERRQ(ierr);

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

   PetscInt Istart,Iend,Ii,i,j;
   PetscScalar v;

  ierr = MatCreate(PETSC_COMM_WORLD,&_H);CHKERRQ(ierr);
  ierr = MatSetSizes(_H,PETSC_DECIDE,PETSC_DECIDE,_Nz*_Ny,_Nz*_Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_H);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_H,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_H,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_H);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_H,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    i = Ii/_Nz; j = Ii-i*_Nz;
    v=1.0/(_HinvyArr[i]*_HinvzArr[j]);
    ierr = MatSetValues(_H,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

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
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
      //~Hinv.printPetsc();

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
      //~Hinv.printPetsc();


      S(0,0,1.5*scale);     S(0,1,-2.0*scale);      S(0,2,0.5*scale); // -1* p666 of Mattsson 2010
      S(N-1,N-3,0.5*scale); S(N-1,N-2,-2.0*scale);  S(N-1,N-1,1.5*scale);
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\nS:\n");CHKERRQ(ierr);
      //~S.printPetsc();

      D1int(0,0,-1.0*scale);D1int(0,1,scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1int(Ii,Ii-1,-0.5*scale);
        D1int(Ii,Ii+1,0.5*scale);
      }
      D1int(N-1,N-1,scale);D1int(N-1,N-2,-1*scale); // last row
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1int:\n");CHKERRQ(ierr);
      //~D1int.printPetsc();

      //~D1 = D1int; // copy D1int's interior
      //~D1(N-1,N-3,S(N-1,N-3)); // last row
      //~D1(N-1,N-2,S(N-1,N-2));
      //~D1(N-1,N-1,S(N-1,N-1));
      D1 = S; // actually only want shear stress on fault, not interior
      D1(0,0,-S(0,0)); D1(0,1,-S(0,1)); D1(0,2,-S(0,2)); // first row


      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1:\n");CHKERRQ(ierr);
      //~D1.printPetsc();

      D2(0,0,scale*scale); D2(0,1,-2.0*scale*scale); D2(0,2,scale*scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D2(Ii,Ii-1,scale*scale);
        D2(Ii,Ii,-2.0*scale*scale);
        D2(Ii,Ii+1,scale*scale);
      }
      D2(N-1,N-3,scale*scale);D2(N-1,N-2,-2.0*scale*scale);D2(N-1,N-1,scale*scale); // last row
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD2:\n");CHKERRQ(ierr);
      //~D2.printPetsc();

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
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
      //~Hinv.printPetsc();

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
      //~Hinv.printPetsc();

      // row 1: -1* p666 of Mattsson 2010
      S(0,0,11.0/6.0); S(0,1,-3.0); S(0,2,1.5); S(0,3,-1.0/3.0);
      S(N-1,N-1,11.0/6.0); S(N-1,N-2,-3.0); S(N-1,N-3,1.5); S(N-1,N-4,-1.0/3.0);
      S.scale(scale);
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nS:\n");CHKERRQ(ierr);
      //~S.printPetsc();

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
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1int:\n");CHKERRQ(ierr);
      //~D1int.printPetsc();


      // 1st derivative, same interior stencil but transition at boundaries
      //~D1 = D1int;
      //~D1(N-1,N-1,S(N-1,N-1)); D1(N-1,N-2,S(N-1,N-2)); D1(N-1,N-3,S(N-1,N-3)); D1(N-1,N-4,S(N-1,N-4)); // last row
      D1 = S; // only need 1st derivative on boundaries, not interior
      D1(0,0,-S(0,0)); D1(0,1,-S(0,1)); D1(0,2,-S(0,2)); D1(0,3,-S(0,3));

      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1:\n");CHKERRQ(ierr);
      //~D1.printPetsc();
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




// creates array versions of the 1D SBP factors
// currently, D1 and D2 are only valid in 1D
// (also, D1 might need to have it's first and last rows modified??)
PetscErrorCode SbpOps::sbpArrays(const PetscInt N,const PetscScalar scale,PetscScalar *Hinv,
                             PetscScalar *D1,PetscScalar *D1int,PetscScalar *D2,
                             PetscScalar *S,PetscInt *Slen)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpArrays in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

PetscInt Ii=0;

switch ( _order ) {
    case 2:
      std::fill_n(Hinv, N, scale);
      Hinv[0] = 2.0*scale;
      Hinv[N-1] = 2.0*scale;

      S[0]=1.5*scale;S[1]=-2*scale;S[2]=0.5*scale;
      S[3]=0.5*scale;S[4]=-2*scale;S[5]=1.5*scale;
      *Slen = 3;

      std::fill_n(D1int, N*N, 0.0);
      D1int[0] = -1*scale; D1int[1] = 1*scale; // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1int[Ii*N+Ii -1] = -0.5*scale;
        D1int[Ii*N+Ii +1] = 0.5*scale;
      }
      D1int[N*N-1] = 1*scale; D1int[N*N-2] = -1*scale; // last row
      //~ierr = printMy2DArray(D1int,N,N);CHKERRQ(ierr);

      std::fill_n(D1, N*N, 0.0);
      D1[0] = -S[0]; D1[1] = -S[1];  D1[2] = -S[2];// first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1[Ii*N+Ii -1] = -0.5*scale;
        D1[Ii*N+Ii +1] = 0.5*scale;
      }
      D1[N*N-3] = S[3]; D1[N*N-2] = S[4]; D1[N*N-1] = S[5];// last row
      //~ierr = printMy2DArray(D1,N,N);CHKERRQ(ierr);

      std::fill_n(D2, N*N, 0.0);
      D2[0] = 1*scale*scale; D2[1] = -2*scale*scale; D2[2] = 1*scale*scale; // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D2[Ii*N+Ii -1] = 1*scale*scale;
        D2[Ii*N+Ii]   = -2*scale*scale;
        D2[Ii*N+Ii +1] = 1*scale*scale;
      }
      D2[N*N-3] = 1*scale*scale; D2[N*N-2] = -2*scale*scale; D2[N*N-1] = 1*scale*scale; // last row
      //~ierr = printMy2DArray(D2,N,N);CHKERRQ(ierr);


      break;

    case 4:
      if (N<8) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      std::fill_n(Hinv,N,scale);
      Hinv[0] = scale*48.0/17.0;
      Hinv[1] = scale*48.0/59.0;
      Hinv[2] = scale*48.0/43.0;
      Hinv[3] = scale*48.0/49.0;
      Hinv[N-4] = scale*48.0/49.0;
      Hinv[N-3] = scale*48.0/43.0;
      Hinv[N-2] = scale*48.0/59.0;
      Hinv[N-1] = scale*48.0/17.0;

      S[0]=11.0/6.0*scale;S[1]=-3.0*scale;S[2]=1.5*scale;S[3]=-scale/3.0;
      S[4]=-scale/3.0;S[5]=1.5*scale;S[6]=-3.0*scale;S[7]=11.0/6.0*scale;
      *Slen = 4;


      // these are actually only 2nd order accurate for now
      std::fill_n(D1, N*N, 0.0);
      D1[0] = -1*scale; D1[1] = 1*scale; // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1[Ii*N+Ii -1] = -0.5*scale;
        D1[Ii*N+Ii +1] = 0.5*scale;
      }
      D1[N*N-1] = 1*scale; D1[N*N-2] = -1*scale; // last row
      //~ierr = printMy2DArray(D1,N,N);CHKERRQ(ierr);


      std::fill_n(D2, N*N, 0.0);
      D2[0] = 1*scale*scale; D2[1] = -2*scale*scale; D2[2] = 1*scale*scale; // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D2[Ii*N+Ii -1] = 1*scale*scale;
        D2[Ii*N+Ii]   = -2*scale*scale;
        D2[Ii*N+Ii +1] = 1*scale*scale;
      }
      D2[N*N-3] = 1*scale*scale; D2[N*N-2] = -2*scale*scale; D2[N*N-1] = 1*scale*scale; // last row
      //~ierr = printMy2DArray(D2,N,N);CHKERRQ(ierr);

      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,1,"order not understood.");
      break;
  }


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbpArrays in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


//PetscErrorCode  SbpOps::sbpOpsMats(PetscInt N, Mat &D, Mat &D2)
//{
  //PetscErrorCode ierr;
  //PetscScalar    v,*stencil;
  //PetscInt       Ii,J,Istart,Iend,*cols,ncols;
  //Mat            Hinv,S,Q;

  //PetscInt const *constCols;
  //PetscScalar const *constVals;

//#if VERBOSE >1
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpOpsMats in sbpOpsMats.cpp.\n");CHKERRQ(ierr);
//#endif

  ///* Create matrix operators: PinvMat, Q, D2, S */
  //ierr = MatCreate(PETSC_COMM_WORLD,&Hinv);CHKERRQ(ierr);
  //ierr = MatSetSizes(Hinv,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Hinv);CHKERRQ(ierr);
  //ierr = MatMPIAIJSetPreallocation(Hinv,1,NULL,0,NULL);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Hinv,1,NULL);CHKERRQ(ierr);
  //ierr = MatSetUp(Hinv);CHKERRQ(ierr);

  //ierr = MatCreate(PETSC_COMM_WORLD,&Q);CHKERRQ(ierr);
  //ierr = MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Q);CHKERRQ(ierr);
  //ierr = MatMPIAIJSetPreallocation(Q,5,NULL,5,NULL);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Q,5,NULL);CHKERRQ(ierr);
  //ierr = MatSetUp(Q);CHKERRQ(ierr);

  //ierr = MatCreate(PETSC_COMM_WORLD,&D2);CHKERRQ(ierr);
  //ierr = MatSetSizes(D2,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  //ierr = MatSetFromOptions(D2);CHKERRQ(ierr);
  //ierr = MatMPIAIJSetPreallocation(D2,5,NULL,5,NULL);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(D2,5,NULL);CHKERRQ(ierr);
  //ierr = MatSetUp(D2);CHKERRQ(ierr);

  //ierr = MatCreate(PETSC_COMM_WORLD,&S);CHKERRQ(ierr);
  //ierr = MatSetSizes(S,PETSC_DECIDE,PETSC_DECIDE,N+1,N+1);CHKERRQ(ierr);
  //ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  //ierr = MatMPIAIJSetPreallocation(S,5,NULL,5,NULL);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(S,5,NULL);CHKERRQ(ierr);
  //ierr = MatSetUp(S);CHKERRQ(ierr);

  //PetscMalloc(16*sizeof(PetscInt),&cols);
  //PetscMalloc(16*sizeof(PetscScalar),&stencil);

  //switch ( _order ) {
    //case 2:

      //ierr = MatGetOwnershipRange(Q,&Istart,&Iend); CHKERRQ(ierr);
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //if (Ii<N) { v=0.5;J=Ii+1; ierr=MatSetValues(Q,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
        //if (Ii>0) { v=-0.5;J=Ii-1; ierr=MatSetValues(Q,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr); }
      //}
      //if (Istart==0) { v=-0.5; ierr=MatSetValues(Q,1,&Istart,1,&Istart,&v,INSERT_VALUES);CHKERRQ(ierr); }
      //if (Iend==N+1) { v=0.5;Ii=N; ierr=MatSetValues(Q,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }

      //ierr = MatGetOwnershipRange(Hinv,&Istart,&Iend); CHKERRQ(ierr);
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //v=1.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Istart==0) { v=2.0; ierr=MatSetValues(Hinv,1,&Istart,1,&Istart,&v,INSERT_VALUES);CHKERRQ(ierr); }
      //if (Iend==N+1) { v=2.0;Ii=Iend-1; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr); }

      //ierr = MatGetOwnershipRange(D2,&Istart,&Iend);CHKERRQ(ierr);
      //stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //if (Ii>0 && Ii<N) {
          //cols[0] = Ii-1;cols[1]=Ii;cols[2]=Ii+1;
          //v=1.0;ierr=MatSetValues(D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        //}
      //}
      //if (Istart==0) {
        //cols[0]=Istart;cols[1]=Istart+1;cols[2]=Istart+2;
        //ierr=MatSetValues(D2,1,&Istart,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=N;
        //cols[2]=Ii;cols[1]=Ii-1;cols[0]=Ii-2;
        //ierr=MatSetValues(D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}

      //ierr = MatGetOwnershipRange(S,&Istart,&Iend);CHKERRQ(ierr);
      //if (Istart==0) {
        //cols[0]=0;cols[1]=1;cols[2]=2;
        //stencil[0]=1.5;stencil[1]=-2;stencil[2]=0.5; // -1 * row from paper
        //ierr=MatSetValues(S,1,&Istart,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=Iend-1;
        //cols[2]=N;cols[1]=N-1;cols[0]=N-2;
        //stencil[0]=0.5;stencil[1]=-2;stencil[2]=1.5;
        //ierr=MatSetValues(S,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}

      //break;

    //case 4:
      //if (N<8) { SETERRQ(PETSC_COMM_WORLD,1,"N too small, must be >8 for order 4 SBP."); }

      //ierr = MatGetOwnershipRange(Q,&Istart,&Iend); CHKERRQ(ierr);
      //if (Istart==0) {
        //cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;
        //stencil[0]=-1.0/2.0;stencil[1]=59.0/96.0;stencil[2]=-1.0/12.0;stencil[3]=-1.0/32.0;
        //ierr=MatSetValues(Q,1,&Istart,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=1;
        //cols[0]=0;cols[1]=2;stencil[0]=-59.0/96.0;stencil[1]=59.0/96.0;
        //ierr=MatSetValues(Q,1,&Ii,2,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=2;
        //cols[0]=0;cols[1]=1;cols[2]=3;cols[3]=4;
        //stencil[0]=1.0/12.0;stencil[1]=-59.0/96.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/12.0;
        //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=3;
        //cols[0]=0;cols[1]=2;cols[2]=4;cols[3]=5;
        //stencil[0]=1.0/32.0;stencil[1]=-59.0/96.0;stencil[2]=2.0/3.0;stencil[3]=-1.0/12.0;
        //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=N;
        //cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        //stencil[0]=1.0/32.0;stencil[1]=1.0/12.0;stencil[2]=-59.0/96.0;stencil[3]=1.0/2.0;
        //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N-1;
        //cols[0]=N-2;cols[1]=N;
        //stencil[0]=-59.0/96.0;stencil[1]=59.0/96.0;
        //ierr=MatSetValues(Q,1,&Ii,2,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N-2;
        //cols[0]=N-4;cols[1]=N-3;cols[2]=N-1;cols[3]=N;
        //stencil[0]=1.0/12.0;stencil[1]=-59.0/96.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/12.0;
        //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N-3;
        //cols[0]=N-5;cols[1]=N-4;cols[2]=N-2;cols[3]=N;
        //stencil[0]=1.0/12.0;stencil[1]=-2.0/3.0;stencil[2]=59.0/96.0;stencil[3]=-1.0/32.0;
        //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //if (Ii>3 && Ii<N-3) {
          //cols[0]=Ii-2;cols[1]=Ii-1;cols[2]=Ii+1;cols[3]=Ii+2;
          //stencil[0]=1.0/12.0;stencil[1]=-2.0/3.0;stencil[2]=2.0/3.0;stencil[3]=-1.0/12.0;
          //ierr=MatSetValues(Q,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        //}
      //}

      //ierr = MatGetOwnershipRange(Hinv,&Istart,&Iend); CHKERRQ(ierr);
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //v=1.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Istart==0) {
        //Ii=Istart;v=48.0/17.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=Istart+1;v=48.0/59.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=Istart+2;v=48.0/43.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=Istart+3;v=48.0/49.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=N;v=48.0/17.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=N-1;v=48.0/59.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=N-2;v=48.0/43.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
        //Ii=N-3;v=48.0/49.0; ierr=MatSetValues(Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      //}


      //ierr = MatGetOwnershipRange(D2,&Istart,&Iend);CHKERRQ(ierr);
      //for (Ii=Istart;Ii<Iend;Ii++) {
        //if (Ii>3 && Ii<N-3) {
          //cols[0]=Ii-2;cols[1]=Ii-1;cols[2]=Ii;cols[3]=Ii+1;cols[4]=Ii+2;
          //stencil[0]=-1.0/12.0;stencil[1]=4.0/3.0;stencil[2]=-5.0/2.0;stencil[3]=4.0/3.0;stencil[4]=-1.0/12.0;
          //ierr=MatSetValues(D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
        //}
      //}
      //if (Istart==0) {
        //Ii=Istart;
        //cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        //stencil[0]=2.0;stencil[1]=-5.0;stencil[2]=4.0;stencil[3]=-1.0;
        //ierr=MatSetValues(D2,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=Istart+1;
        //cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        //stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
        //ierr=MatSetValues(D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=Istart+2;
        //cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;cols[4]=4;
        //stencil[0]=-4.0/43.0;stencil[1]=59.0/43.0;stencil[2]=-110.0/43.0;stencil[3]=59.0/43;stencil[4]=-4.0/43.0;
        //ierr=MatSetValues(D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=Istart+3;
        //cols[0]=0;cols[1]=2;cols[2]=3;cols[3]=4;cols[4]=5;
        //stencil[0]=-1.0/49.0;stencil[1]=59.0/49.0;stencil[2]=-118.0/49.0;stencil[3]=64.0/49.0;stencil[4]=-4.0/49.0;
        //ierr=MatSetValues(D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=N-3;
        //cols[0]=N-5;cols[1]=N-4;cols[2]=N-3;cols[3]=N-2;cols[4]=N;
        //stencil[0]=-4.0/49.;stencil[1]=64.0/49.0;stencil[2]=-118.0/49.0;stencil[3]=59.0/49.0;stencil[4]=-1.0/49.0;
        //ierr=MatSetValues(D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N-2;
        //cols[0]=N-4;cols[1]=N-3;cols[2]=N-2;cols[3]=N-1;cols[4]=N;
        //stencil[0]=-4.0/43.0;stencil[1]=59.0/43.0;stencil[2]=-110.0/43.0;stencil[3]=59.0/43.0;stencil[4]=-4.0/43.0;
        //ierr=MatSetValues(D2,1,&Ii,5,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N-1;
        //cols[0]=N-2;cols[1]=N-1;cols[2]=N;
        //stencil[0]=1.0;stencil[1]=-2.0;stencil[2]=1.0;
        //ierr=MatSetValues(D2,1,&Ii,3,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);

        //Ii=N;
        //cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        //stencil[0]=-1.0;stencil[1]=4.0;stencil[2]=-5.0;stencil[3]=2.0;
        //ierr=MatSetValues(D2,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}

      //ierr = MatGetOwnershipRange(S,&Istart,&Iend);CHKERRQ(ierr);
      //if (Istart==0) {
        //cols[0]=0;cols[1]=1;cols[2]=2;cols[3]=3;
        //stencil[0]=11.0/6.0;stencil[1]=-3.0;stencil[2]=3.0/2.0;stencil[3]=-1.0/3.0; // -1 * row from paper
        //ierr=MatSetValues(S,1,&Istart,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}
      //if (Iend==N+1) {
        //Ii=Iend-1;
        //cols[0]=N-3;cols[1]=N-2;cols[2]=N-1;cols[3]=N;
        //stencil[0]=-1.0/3.0;stencil[1]=3.0/2.0;stencil[2]=-3.0;stencil[3]=11.0/6.0;
        //ierr=MatSetValues(S,1,&Ii,4,cols,stencil,INSERT_VALUES);CHKERRQ(ierr);
      //}

      //break;


    //default:
      //SETERRQ(PETSC_COMM_WORLD,1,"SBP order not understood.");
      //break;
  //}

  //ierr = MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyBegin(Hinv,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyBegin(D2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyEnd(Hinv,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyEnd(D2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatMatMult(Hinv,Q,MAT_INITIAL_MATRIX,1.0,&D);CHKERRQ(ierr);
  //ierr = MatSetOption(D,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);
  //ierr = MatSetFromOptions(D);CHKERRQ(ierr);

  //// D(1,:) = S(1,:); D(end,:) = S(end,:)
  //ierr = MatGetOwnershipRange(S,&Istart,&Iend);CHKERRQ(ierr);
  //if (Istart == 0) {
    //ierr = MatGetRow(S,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    //for (J=0;J<ncols;J++) { stencil[J]=-1.0*constVals[J];}
    //ierr = MatSetValues(D,1,&Istart,ncols,constCols,stencil,INSERT_VALUES);CHKERRQ(ierr);
    //ierr = MatRestoreRow(S,Istart,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  //}
  //Ii=N;
  //if (Ii>=Istart && Ii<Iend) {
    //ierr = MatGetRow(S,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
    //ierr = MatSetValues(D,1,&Ii,ncols,constCols,constVals,INSERT_VALUES);CHKERRQ(ierr);
    //ierr = MatRestoreRow(S,Ii,&ncols,&constCols,&constVals);CHKERRQ(ierr);
  //}
  //ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = PetscFree(cols);CHKERRQ(ierr);
  //ierr = PetscFree(stencil);CHKERRQ(ierr);
  //ierr = MatDestroy(&Hinv);CHKERRQ(ierr);
  //ierr = MatDestroy(&S);CHKERRQ(ierr);
  //ierr = MatDestroy(&Q);CHKERRQ(ierr);

//#if VERBOSE > 1
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbpOpsMats in sbpOpsMats.cpp.\n");CHKERRQ(ierr);
//#endif

  //return 0;
//}


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
  ierr = MatMult(_Hinvy_Izxe0y_Iz,_bcF,rhs);CHKERRQ(ierr);
  ierr = VecScale(rhs,_alphaF);CHKERRQ(ierr);

  // + _beta*mu*_Hinvy_IzxBySy_IzTxe0y_Iz*_bcF + ...
  Vec temp;
  ierr = VecDuplicate(rhs,&temp);CHKERRQ(ierr);
  ierr = MatMult(_Hinvy_IzxBySy_IzTxe0y_Iz,_bcF,temp);CHKERRQ(ierr);
  ierr = VecAXPY(rhs,_beta,temp);CHKERRQ(ierr);

  // + _alphaR*mu*_Hinvy_IzxeNy_Iz*_bcR + ...
  ierr = MatMult(_Hinvy_IzxeNy_Iz,_bcR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(rhs,_alphaR,temp);CHKERRQ(ierr);

  // + _beta*mu*_Hinvy_IzxBySy_IzTxeNy_Iz*_bcR + ...
  ierr = MatMult(_Hinvy_IzxBySy_IzTxeNy_Iz,_bcR,temp);CHKERRQ(ierr);
  ierr = VecAXPY(rhs,_beta,temp);CHKERRQ(ierr);

  //~ // - _alphaS*M.Iy_HinvzxIy_e0z*_bcS + ...
  //~ ierr = MatMult(_IyHinvz_Iye0z,_bcS,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(_rhs,_alphaS,temp);CHKERRQ(ierr);
//~
  //~ // + _alphaD*M.Iy_HinvzxIy_eNz*_bcD
  //~ ierr = MatMult(_IyHinvz_IyeNz,_bcD,temp);CHKERRQ(ierr);
  //~ ierr = VecAXPY(_rhs,_alphaD,temp);CHKERRQ(ierr);

  ierr = VecDestroy(&temp);CHKERRQ(ierr);

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

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_Izxe0y_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_Izxe0y_Iz);CHKERRQ(ierr);
  ierr = MatSetType(_Hinvy_Izxe0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Hinvy_Izxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxBySy_IzTxe0y_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxe0y_Iz);CHKERRQ(ierr);
  ierr = MatSetType(_Hinvy_IzxBySy_IzTxe0y_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Hinvy_IzxBySy_IzTxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxeNy_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxeNy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(_Hinvy_IzxeNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Hinvy_IzxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Hinvy_IzxBySy_IzTxeNy_Iz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Hinvy_IzxBySy_IzTxeNy_Iz);CHKERRQ(ierr);
  ierr = MatSetType(_Hinvy_IzxBySy_IzTxeNy_Iz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Hinvy_IzxBySy_IzTxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Iy_HinvzxIy_e0z",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_e0z);CHKERRQ(ierr);
  ierr = MatSetType(_Iy_HinvzxIy_e0z,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Iy_HinvzxIy_e0z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Iy_HinvzxIy_eNz",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&_Iy_HinvzxIy_eNz);CHKERRQ(ierr);
  ierr = MatSetType(_Iy_HinvzxIy_eNz,matType);CHKERRQ(ierr);
  ierr = MatLoad(_Iy_HinvzxIy_eNz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

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

  str = outputDir + "Hinvy_Izxe0y_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Hinvy_Izxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Hinvy_IzxBySy_IzTxe0y_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Hinvy_IzxBySy_IzTxe0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Hinvy_IzxeNy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Hinvy_IzxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Hinvy_IzxBySy_IzTxeNy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Hinvy_IzxBySy_IzTxeNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Iy_HinvzxIy_e0z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_HinvzxIy_e0z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Iy_HinvzxIy_eNz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_HinvzxIy_eNz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOps in sbpOps.cpp\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
};



