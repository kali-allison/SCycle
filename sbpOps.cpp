#include "sbpOps.hpp"


//================= constructor and destructor ========================

SbpOps::SbpOps(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(D._muArr),_mu(&D._mu),
  _Sylen(0),_Szlen(0),
  _alphaF(-13.0/_dy),_alphaR(-13.0/_dy),_alphaS(-1.0),_alphaD(-1.0),_beta(1.0),
  _debugFolder("./matlabAnswers/")
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif

    stringstream ss;
    ss << "order" << _order << "Ny" << _Ny << "Nz" << _Nz << "/";
    _debugFolder += ss.str();

  //~VecCreate(PETSC_COMM_WORLD,&_rhs);
  //~VecSetSizes(_rhs,PETSC_DECIDE,D.Nz*D.Nz);
  //~VecSetFromOptions(_rhs);     PetscObjectSetName((PetscObject) _rhs, "_rhs");

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
  PetscMalloc(_Ny*_Ny*sizeof(PetscScalar),&_D2y);
  sbpArrays(_Ny,1/_dy,_HinvyArr,_D1y,_D2y,_SyArr,&_Sylen);

  PetscMalloc(_Nz*sizeof(PetscScalar),&_HinvzArr);
  PetscMalloc(2*_Nz*sizeof(PetscScalar),&_SzArr);
  PetscMalloc(_Nz*_Nz*sizeof(PetscScalar),&_D1z);
  PetscMalloc(_Nz*_Nz*sizeof(PetscScalar),&_D2z);
  sbpArrays(_Nz,1/_dz,_HinvzArr,_D1z,_D2z,_SzArr,&_Szlen);


  MatCreate(PETSC_COMM_WORLD,&_Dy_Iz); PetscObjectSetName((PetscObject) _Dy_Iz, "_Dy_Iz");

  computeDy_Iz();
  computeA();
  computeRhsFactors();

  MatMatMult(*_mu,_Dy_Iz,MAT_INITIAL_MATRIX,1.0,&_Dy_Iz); // scale for shear stress computations


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
  PetscFree(_D2y);
  PetscFree(_SyArr);

  PetscFree(_HinvzArr);
  PetscFree(_D1z);
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
  //~checkMatrix(&_Iy_HinvzxIy_e0z,_debugFolder,"Iy_HinvzxIy_e0z",&D);CHKERRQ(ierr);
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

// this is only valid for 2nd order
// to do 4th order, I would need to change my computation of Rmuy
PetscErrorCode SbpOps::computeD2ymu(Mat &D2ymu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  PetscScalar     v=0;
  PetscInt        Istart,Iend; // range of matrix rows stored locally
  PetscInt        Ii,Jj; // local matrix indices
  PetscInt        Irow,Icol; // array indices


  // kron muxBySy_Iz
  Mat muxBySy_Iz;
  MatCreate(PETSC_COMM_WORLD,&muxBySy_Iz);
  ierr = PetscObjectSetName((PetscObject) muxBySy_Iz, "muxBySy_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(muxBySy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(muxBySy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(muxBySy_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(muxBySy_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(muxBySy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(muxBySy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    Irow = Ii/_Nz; // corresponding row in By*Sy
    if (Irow == 0) {
      for (Icol=0;Icol<_Sylen;Icol++) // iterate over columns in By*Sy
      {
        v = _SyArr[Icol];
        Jj = Ii%_Nz + Icol*_Nz; // map cols in Dy to cols in Dy_Iz
        if (v!=0) { ierr = MatSetValues(muxBySy_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
      }
    }
    if (Ii > _Ny*_Nz -_Nz-1) {
      for (Icol=0;Icol<_Sylen;Icol++) // iterate over columns in By*Sy
      {
        v = _SyArr[_Sylen + Icol];
        Jj = Ii%_Nz + (_Ny-_Sylen + Icol)*_Nz; // map cols in SyArr to cols in BySy_Iz
        ierr = MatSetValues(muxBySy_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES);
      }
    }
  }
  ierr = MatAssemblyBegin(muxBySy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(muxBySy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,muxBySy_Iz,MAT_INITIAL_MATRIX,1.0,&muxBySy_Iz);CHKERRQ(ierr);
  //~ierr = MatView(muxBySy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // mu*kron(Hy,Iz)
  Mat muxHy_Iz;
  MatCreate(PETSC_COMM_WORLD,&muxHy_Iz);
  ierr = PetscObjectSetName((PetscObject) muxHy_Iz, "muxHy_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(muxHy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(muxHy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(muxHy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(muxHy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(muxHy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(muxHy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii/_Nz; // corresponding row in Hy
    v = _muArr[Irow]/_HinvyArr[Irow];
    ierr = MatSetValues(muxHy_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(muxHy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(muxHy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(muxHy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(Hinvy,Iz)
  Mat Hinvy_Iz;
  MatCreate(PETSC_COMM_WORLD,&Hinvy_Iz);
  ierr = PetscObjectSetName((PetscObject) Hinvy_Iz, "Hinvy_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii/_Nz; // corresponding row in Hiny
    v = _HinvyArr[Irow];
    ierr = MatSetValues(Hinvy_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(Hinvy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Hinvy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(D2y,Iz)
  Mat D2y_Iz;
  MatCreate(PETSC_COMM_WORLD,&D2y_Iz);
  ierr = PetscObjectSetName((PetscObject) D2y_Iz, "D2y_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(D2y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D2y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D2y_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D2y_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D2y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii/_Nz; // corresponding row in D2y
    for (Icol=0;Icol<_Ny;Icol++)
    { // iterate over columns in Dy
      v = _D2y[Irow*_Ny+Icol];
      Jj = Ii%_Nz + Icol*_Nz; // map cols in D2y to cols in D2y_Iz
      if (v!=0) { ierr = MatSetValues(D2y_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D2y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(D2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(D2y',Iz)
  Mat D2yT_Iz;
  MatCreate(PETSC_COMM_WORLD,&D2yT_Iz);
  ierr = PetscObjectSetName((PetscObject) D2yT_Iz, "D2yT_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(D2yT_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D2yT_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(D2yT_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(D2yT_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(D2yT_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(D2yT_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii/_Nz; // corresponding row in D2y
    for (Icol=0;Icol<_Ny;Icol++) // iterate over columns in D2y
    {
      v = _D2y[Icol*_Ny+Irow]; // swapped Irow and Icol rel to kron(D2y,Iz) eq to access D2y'
      Jj = Ii%_Nz + Icol*_Nz; // map cols in D2y' to cols in D2y_Iz
      if (v!=0) { ierr = MatSetValues(D2yT_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(D2yT_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D2yT_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(D2yT_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  // kron(C,Iz)
  Mat C_Iz;
  MatCreate(PETSC_COMM_WORLD,&C_Iz);
  ierr = PetscObjectSetName((PetscObject) C_Iz, "C_Iz");CHKERRQ(ierr);
  ierr = MatSetSizes(C_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C_Iz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C_Iz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(C_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    if  (Ii>_Nz-1 && Ii<(_Ny*_Nz-_Nz) ) {
      v = 1;
      ierr = MatSetValues(C_Iz,1,&Ii,1,&Ii,&v,INSERT_VALUES);
    }
  }
  ierr = MatAssemblyBegin(C_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(C_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  // Rymu = (D2yT x C_Iz x mu x D2y_Iz)/4/dy^3;
  Mat Rymu;
  ierr = MatMatMatMult(D2yT_Iz,C_Iz,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
  ierr = MatMatMult(Rymu,D2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
  ierr = MatScale(Rymu,0.25*_dy*_dy*_dy);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"_dy=%f",_dy);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) Rymu, "Rymu");CHKERRQ(ierr);
  //~ierr = MatView(Rymu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //~Mat D2ymu;
  ierr = MatTransposeMatMult(_Dy_Iz,muxHy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = MatMatMult(D2ymu,_Dy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = MatScale(D2ymu,-1);CHKERRQ(ierr);
  ierr = MatAXPY(D2ymu,-1,Rymu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D2ymu,1,muxBySy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(Hinvy_Iz,D2ymu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);
  //~ierr = MatView(D2ymu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  MatDestroy(&muxBySy_Iz);
  MatDestroy(&muxHy_Iz);
  MatDestroy(&Hinvy_Iz);
  MatDestroy(&D2y_Iz);
  MatDestroy(&D2yT_Iz);
  MatDestroy(&C_Iz);
  MatDestroy(&Rymu);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}



// this is only valid for 2nd order
// to do 4th order, I would need to change my computation of Rmuz
PetscErrorCode SbpOps::computeD2zmu(Mat &D2zmu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  PetscScalar     v=0;
  PetscInt        Istart,Iend; // range of matrix rows stored locally
  PetscInt        Ii,Jj; // local matrix indices
  PetscInt        Irow,Icol; // array indices

  // kron(Iy,Dz)
  Mat Iy_Dz;
  MatCreate(PETSC_COMM_WORLD,&Iy_Dz);
  ierr = PetscObjectSetName((PetscObject) Iy_Dz, "Iy_Dz");CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_Dz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_Dz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_Dz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_Dz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_Dz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_Dz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii%_Nz; // corresponding row in D1z
    for (Icol=0;Icol<_Nz;Icol++)
    { // iterate over columns in Dy
      v = _D1z[Irow*_Nz+Icol];
      Jj = Ii + Icol-Irow; // map cols in Dy to cols in Iy_Dz
      if (v!=0) { ierr = MatSetValues(Iy_Dz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(Iy_Dz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_Dz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // mux kron(Iy,BzSz)
  Mat muxIy_BzSz;
  MatCreate(PETSC_COMM_WORLD,&muxIy_BzSz);
  ierr = PetscObjectSetName((PetscObject) muxIy_BzSz, "muxIy_BzSz");CHKERRQ(ierr);
  ierr = MatSetSizes(muxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(muxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(muxIy_BzSz,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(muxIy_BzSz,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(muxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(muxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    //~Irow = Ii%_Nz; // corresponding row in By*Sy
    if (Ii%_Nz == 0) {
      for (Icol=0;Icol<_Szlen;Icol++) // iterate over columns in Bz*Sz
      {
        v = _SzArr[Icol];
        Jj = Ii + Icol; // map cols in Dy to cols in Dy_Iz
        ierr = MatSetValues(muxIy_BzSz,1,&Ii,1,&Jj,&v,INSERT_VALUES);
      }
    }
    if (Ii%_Nz ==_Ny ) {
      for (Icol=0;Icol<_Szlen;Icol++) // iterate over columns in By*Sy
      {
        v = _SzArr[_Szlen + Icol];
        Jj = Ii + Icol-_Szlen+1; // map cols in SyArr to cols in BySy_Iz
        if (v!=0) { ierr = MatSetValues(muxIy_BzSz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
      }
    }
  }
  ierr = MatAssemblyBegin(muxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(muxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMatMult(*_mu,muxIy_BzSz,MAT_INITIAL_MATRIX,1.0,&muxIy_BzSz);CHKERRQ(ierr);
  //~ierr = MatView(muxIy_BzSz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // mu*kron(Iy,Hz)
  Mat muxIy_Hz;
  MatCreate(PETSC_COMM_WORLD,&muxIy_Hz);
  ierr = PetscObjectSetName((PetscObject) muxIy_Hz, "muxIy_Hz");CHKERRQ(ierr);
  ierr = MatSetSizes(muxIy_Hz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(muxIy_Hz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(muxIy_Hz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(muxIy_Hz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(muxIy_Hz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(muxIy_Hz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii%_Nz; // corresponding row in Hz
    v = _muArr[Irow]/_HinvzArr[Irow];
    ierr = MatSetValues(muxIy_Hz,1,&Ii,1,&Ii,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(muxIy_Hz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(muxIy_Hz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(muxIy_Hz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(Iy,Hinvz)
  Mat Iy_Hinvz;
  MatCreate(PETSC_COMM_WORLD,&Iy_Hinvz);
  ierr = PetscObjectSetName((PetscObject) Iy_Hinvz, "Iy_Hinvz");CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_Hinvz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_Hinvz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_Hinvz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_Hinvz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_Hinvz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii%_Nz; // corresponding row in Hinvz
    v = _HinvzArr[Irow];
    ierr = MatSetValues(Iy_Hinvz,1,&Ii,1,&Ii,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_Hinvz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Iy_Hinvz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(Iy,D2z)
  Mat Iy_D2z;
  MatCreate(PETSC_COMM_WORLD,&Iy_D2z);
  ierr = PetscObjectSetName((PetscObject) Iy_D2z, "Iy_D2z");CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_D2z,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_D2z);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_D2z,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_D2z,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_D2z);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_D2z,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii%_Nz; // corresponding row in D2z
    for (Icol=0;Icol<_Nz;Icol++)
    { // iterate over columns in Dy
      v = _D2z[Irow*_Nz+Icol];
      Jj = Ii + Icol - Irow; // map cols in D2z to cols in Iy_D2z
      if (v!=0) { ierr = MatSetValues(Iy_D2z,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_D2z,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Iy_D2z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // kron(Iy,D2z')
  Mat Iy_D2zT;
  MatCreate(PETSC_COMM_WORLD,&Iy_D2zT);
  ierr = PetscObjectSetName((PetscObject) Iy_D2zT, "Iy_D2zT");CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_D2zT,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_D2zT);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_D2zT,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_D2zT,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_D2zT);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_D2zT,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii%_Nz; // corresponding row in D2y
    for (Icol=0;Icol<_Nz;Icol++) // iterate over columns in D2y
    {
      v = _D2z[Icol*_Nz+Irow]; // swapped Irow and Icol rel to kron(Iy,D2z) eq to access D2z'
      Jj = Ii + Icol - Irow; // map cols in D2y' to cols in D2y_Iz
      //~Jj = Ii - Icol + Irow; // map cols in D2z' to cols in Iy_D2zT
      if (v!=0) { ierr = MatSetValues(Iy_D2zT,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(Iy_D2zT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_D2zT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Iy_D2zT,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  // kron(Iy,C)
  Mat Iy_C;
  MatCreate(PETSC_COMM_WORLD,&Iy_C);
  ierr = PetscObjectSetName((PetscObject) Iy_C, "Iy_C");CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_C,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_C);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_C,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_C,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_C,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    if  (Ii%_Nz!=0 && Ii%_Nz!=_Ny ) {
      v = 1;
      ierr = MatSetValues(Iy_C,1,&Ii,1,&Ii,&v,INSERT_VALUES);
    }
  }
  ierr = MatAssemblyBegin(Iy_C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(Iy_C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // Rzmu = (Iy_D2zT x Iy_C x mu x Iy_D2z)/4/dz^3;
  Mat Rzmu;
  ierr = MatMatMatMult(Iy_D2zT,Iy_C,*_mu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
  ierr = MatMatMult(Rzmu,Iy_D2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
  ierr = MatScale(Rzmu,0.25*_dz*_dz*_dz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Rzmu, "Rzmu");CHKERRQ(ierr);
  //~ierr = MatView(Rzmu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //~Mat D2zmu;
  ierr = MatTransposeMatMult(Iy_Dz,muxIy_Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = MatMatMult(D2zmu,Iy_Dz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = MatScale(D2zmu,-1);CHKERRQ(ierr);
  ierr = MatAXPY(D2zmu,-1,Rzmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(D2zmu,1,muxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(Iy_Hinvz,D2zmu,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);
  //~ierr = MatView(D2zmu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  MatDestroy(&Iy_Dz);
  MatDestroy(&muxIy_BzSz);
  MatDestroy(&muxIy_Hz);
  MatDestroy(&Iy_Hinvz);
  MatDestroy(&Iy_D2z);
  MatDestroy(&Iy_D2zT);
  MatDestroy(&Iy_C);
  MatDestroy(&Rzmu);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// compute matrix for shear stress (not scaled by mu!!): kron(Dy,Iz)
PetscErrorCode SbpOps::computeDy_Iz()
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeDy_Iz in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  PetscScalar     v=0;
  PetscInt        Istart,Iend; // range of matrix rows stored locally
  PetscInt        Ii, Jj; // local matrix row and col indices
  PetscInt        Irow,Icol; // array indices

  ierr = MatSetSizes(_Dy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Dy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Dy_Iz,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Dy_Iz,2,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Dy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Dy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    Irow = Ii/_Nz; // corresponding row in Dy
    for (Icol=0;Icol<_Ny;Icol++)
    { // iterate over columns in Dy
      assert(Irow*_Ny+Icol<_Ny*_Nz);
      v = _D1y[Irow*_Ny+Icol];
      //~v = 0;
      Jj = Ii%_Nz + Icol*_Nz; // map cols in Dy to cols in _Dy_Iz
      assert(Jj<_Ny*_Nz);
      if (v!=0) { ierr = MatSetValues(_Dy_Iz,1,&Ii,1,&Jj,&v,INSERT_VALUES); }
    }
  }
  ierr = MatAssemblyBegin(_Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Dy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //~ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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


  // Hinvy_IzxBySy_IzTxE0y_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(E0z,Iz)
  Mat Hinvy_IzxBySy_IzTxE0y_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxBySy_IzTxE0y_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxBySy_IzTxE0y_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxBySy_IzTxE0y_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxBySy_IzTxE0y_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<_Nz*_Sylen;Ii++) {
    indx = Ii-(Ii/_Nz)*_Nz;
    v = _muArr[indx]*_HinvyArr[Ii/_Nz]*_SyArr[Ii/_Nz];
    ierr = MatSetValues(Hinvy_IzxBySy_IzTxE0y_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Hinvy_IzxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxBySy_IzTxE0y_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxBySy_IzTxE0y_Iz,_debugFolder,"Hinvy_IzxBySy_IzTxUE0y_Iz");CHKERRQ(ierr);
#endif


  // Hinvy_IzxBySy_IzTxENy_Iz = kron(Hinvy,Iz)*mu*kron(BySy,Iz)^T*kron(ENz,Iz)
  Mat Hinvy_IzxBySy_IzTxENy_Iz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatSetSizes(Hinvy_IzxBySy_IzTxENy_Iz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Hinvy_IzxBySy_IzTxENy_Iz,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Hinvy_IzxBySy_IzTxENy_Iz,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Hinvy_IzxBySy_IzTxENy_Iz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>=_Ny*_Nz-_Nz*_Sylen) {
      indx = (_Ny-1)*_Nz + Ii-(Ii/_Nz)*_Nz;
      v = _muArr[indx]*_HinvyArr[Ii/_Nz]*_SyArr[2*_Sylen-1-(_Ny*_Nz-1-Ii)/_Nz];
      ierr = MatSetValues(Hinvy_IzxBySy_IzTxENy_Iz,1,&Ii,1,&indx,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Hinvy_IzxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hinvy_IzxBySy_IzTxENy_Iz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Hinvy_IzxBySy_IzTxENy_Iz,_debugFolder,"Hinvy_IzxBySy_IzTxUENy_Iz");CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_E0zxIy_BzSz = kron(Iy,_Hinvz)*kron(Iy,E0z)*mu*kron(Iy,BzSz)
  Mat Iy_HinvzxIy_E0zxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_E0zxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_E0zxIy_BzSz,_Szlen,NULL,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_E0zxIy_BzSz,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_E0zxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    indx=Ii/_Nz;J=Ii-indx*_Nz;
    if (J==0) {
      for (indx=0;indx<_Szlen;indx++) {
        cols[indx]=Ii+indx;
        vals[indx]=_muArr[Ii]*_HinvzArr[J]*_SzArr[indx];
      }
      ierr = MatSetValues(Iy_HinvzxIy_E0zxIy_BzSz,1,&Ii,_Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_E0zxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_E0zxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  ierr = checkMatrix(&Iy_HinvzxIy_E0zxIy_BzSz,_debugFolder,"Iy_HinvzxIy_E0zxIy_BzSz");CHKERRQ(ierr);
#endif


  // Iy_HinvzxIy_ENzxIy_BzSz = kron(Iy,_Hinvz)*kron(Iy,ENz)*mu*kron(Iy,BzSz)
  Mat Iy_HinvzxIy_ENzxIy_BzSz;
  ierr = MatCreate(PETSC_COMM_WORLD,&Iy_HinvzxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatSetSizes(Iy_HinvzxIy_ENzxIy_BzSz,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Iy_HinvzxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Iy_HinvzxIy_ENzxIy_BzSz,_Szlen,NULL,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Iy_HinvzxIy_ENzxIy_BzSz,_Szlen,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(Iy_HinvzxIy_ENzxIy_BzSz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Iy_HinvzxIy_ENzxIy_BzSz,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    J=(Ii+1)/_Nz;J=Ii+1-J*_Nz;
    if (J==0) {
      for (indx=0;indx<_Szlen;indx++) {
        cols[indx]=Ii-_Szlen+1+indx;
        vals[indx]=_muArr[Ii]*_HinvzArr[J]*_SzArr[_Sylen+indx];
      }
      ierr = MatSetValues(Iy_HinvzxIy_ENzxIy_BzSz,1,&Ii,_Szlen,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Iy_HinvzxIy_ENzxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Iy_HinvzxIy_ENzxIy_BzSz,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&Iy_HinvzxIy_ENzxIy_BzSz,_debugFolder,"Iy_HinvzxIy_ENzxIy_BzSz");CHKERRQ(ierr);
#endif

  PetscViewer viewer;

  // compute A
  //~ierr = MatMatMult(*_mu,D2yplusD2z,MAT_INITIAL_MATRIX,1.0,&_A);CHKERRQ(ierr);
  ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
  ierr = MatAYPX(_A,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  //~ierr = MatView(_A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage1",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage1");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_alphaF,Hinvy_IzxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage2",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage2");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_beta,Hinvy_IzxBySy_IzTxE0y_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage3",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage3");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_alphaR,Hinvy_IzxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage4",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"Astage4");CHKERRQ(ierr);
#endif
  ierr = MatAXPY(_A,_beta,Hinvy_IzxBySy_IzTxENy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatAXPY(_A,_alphaS,Iy_HinvzxIy_E0zxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/Astage6",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatAXPY(_A,_alphaD,Iy_HinvzxIy_ENzxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/Users/kallison/dataFE/matA",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
  ierr = MatDestroy(&Hinvy_IzxBySy_IzTxE0y_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Hinvy_IzxBySy_IzTxENy_Iz);CHKERRQ(ierr);
  ierr = MatDestroy(&Iy_HinvzxIy_E0zxIy_BzSz);CHKERRQ(ierr);
  ierr = MatDestroy(&Iy_HinvzxIy_ENzxIy_BzSz);CHKERRQ(ierr);

  _runTime = MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeA in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return 0;
}

// Hinv = kron(Hy,Hz)
PetscErrorCode SbpOps::computeHinv()
{
  PetscErrorCode ierr = 0;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeH in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

   PetscInt Istart,Iend,Ii,i,j;
   PetscScalar v;

  ierr = MatCreate(PETSC_COMM_WORLD,&_Hinv);CHKERRQ(ierr);
  ierr = MatSetSizes(_Hinv,PETSC_DECIDE,PETSC_DECIDE,_Nz*_Ny,_Nz*_Ny);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Hinv);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_Hinv,1,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_Hinv,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_Hinv);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(_Hinv,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    i = Ii/_Nz; j = Ii-i*_Nz;
    v=1.0/(_HinvyArr[i]*_HinvzArr[j]);
    ierr = MatSetValues(_Hinv,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_Hinv,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_Hinv,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeH in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return ierr;
}



// creates array versions of the 1D SBP factors
// currently, D1 and D2 are only valid in 1D
// (also, D1 might need to have it's first and last rows modified??)
PetscErrorCode SbpOps::sbpArrays(const PetscInt N,const PetscScalar scale,PetscScalar *Hinv,
                             PetscScalar *D1,PetscScalar *D2,PetscScalar *S,PetscInt *Slen)
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
