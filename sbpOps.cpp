#include "sbpOps.hpp"




//================= constructor and destructor ========================
/* SAT params _alphaF,_alphaR set to values that work for both 2nd and
 * 4th order but are not ideal for 4th.
 */
SbpOps::SbpOps(Domain&D,PetscScalar& muArr,Mat& mu)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(&muArr),_mu(&mu),
  _rhsL(NULL),_rhsR(NULL),_rhsT(NULL),_rhsB(NULL),
  _alphaF(-4.0/_dy),_alphaR(-4.0/_dy),_alphaS(-1.0),_alphaD(-1.0),_beta(1.0),
  _debugFolder("./matlabAnswers/"),_A(NULL),_Dy_Iz(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif

  if(_muArr) { // ensure that _muArr is not NULL
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

    MatCreate(PETSC_COMM_WORLD,&_Dy_Iz);

    {
      /* NOT a member of this class, contains stuff to be deleted before
       * end of constructor to save on memory usage.
       */
      TempMats tempFactors(_order,_Ny,_dy,_Nz,_dz,_mu);

      // reset SAT params
      if (_order==4) {
        _alphaF = tempFactors._Hy(0,0);
        _alphaR = tempFactors._Hy(0,0);
      }


      computeDy_Iz(tempFactors);
      satBoundaries(tempFactors);
      computeA(tempFactors);
    }
}

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

  MatDestroy(&_Dy_Iz);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps.cpp.\n");
#endif
}


//======================== meat ========================================

/* Enforce boundary conditions using SAT penalty terms, computing both
 * matrices used to build the rhs vector from the boundary vectors and
 * the matrices that are added to the spatial derivative matrices to
 * form the A matrix.
 */
PetscErrorCode SbpOps::satBoundaries(TempMats& tempMats)
{
  PetscErrorCode  ierr = 0;

  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function satBoundaries in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  // temporary matrices needed throughout function
  Mat temp1,temp2,temp3;

// matrices in y-direction (left and right boundaries)
{ // enclose in brackets so that destructors are called at end
  // kron(E0y,Iz)
  Mat E0y_Iz;
  {
    Spmat E0y(_Ny,_Ny);
    E0y(0,0,1.0);
    kronConvert(E0y,tempMats._Iz,E0y_Iz,1,1);
    ierr = PetscObjectSetName((PetscObject) E0y_Iz, "E0y_Iz");CHKERRQ(ierr);
  }


  // kron(e0y,Iz)
  Mat e0y_Iz;
  {
    Spmat e0y(_Ny,1);
    e0y(0,0,1.0);
    kronConvert(e0y,tempMats._Iz,e0y_Iz,1,1);
    ierr = PetscObjectSetName((PetscObject) e0y_Iz, "e0y_Iz");CHKERRQ(ierr);
  }

  // kron(ENy,Iz)
  Mat ENy_Iz;
  {
    Spmat ENy(_Ny,_Ny);
    ENy(_Ny-1,_Ny-1,1.0);
    kronConvert(ENy,tempMats._Iz,ENy_Iz,1,1);
    ierr = PetscObjectSetName((PetscObject) ENy_Iz, "ENy_Iz");CHKERRQ(ierr);
  }

  // kron(eNy,Iz)
  Mat eNy_Iz;
  {
    Spmat eNy(_Ny,1);
    eNy(_Ny-1,0,1.0);
    kronConvert(eNy,tempMats._Iz,eNy_Iz,1,1);
    ierr = PetscObjectSetName((PetscObject) eNy_Iz, "eNy_Iz");CHKERRQ(ierr);
  }


  // enforcement of left boundary _bcL =================================
  // map bcL to rhs
  // if bcL = displacement: _alphaF*mu*_Hinvy_Izxe0y_Iz + _beta*_Hinvy_IzxmuxBySy_IzTxe0y_Iz
  MatDestroy(&_rhsL);
  ierr = MatMatMatMult(*_mu,tempMats._Hyinv_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsL);CHKERRQ(ierr);
  ierr = MatScale(_rhsL,_alphaF);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(tempMats._muxBySy_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
  ierr = MatAYPX(_rhsL,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._H,_rhsL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
  ierr = MatCopy(temp3,_rhsL,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
  MatDestroy(&temp3); //!!!
  ierr = PetscObjectSetName((PetscObject) _rhsL, "rhsL");CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&temp2);
  //ierr = MatView(_rhsL,PETSC_VIEWER_STDOUT_WORLD);
  // if bcL = traction-free
  // _rhsL is unneeded bc bcL = 0

  // in computation of A
  // if bcL = displacement: _alphaF*mu*_Hinvy_IzxE0y_Iz + _beta*mu*_Hinvy_IzxBySy_IzTxE0y_Iz
  ierr = MatMatMatMult(*_mu,tempMats._Hyinv_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AL);CHKERRQ(ierr);
  ierr = MatScale(tempMats._AL,_alphaF);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(tempMats._muxBySy_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

  ierr = MatAYPX(tempMats._AL,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tempMats._AL, "AL");CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&temp2);

  // enforcement of right boundary bcR =================================
  // map bcR to rhs
  // if bcR = displacement: _alphaR*mu*_Hinvy_IzxeNy_Iz + _beta*_Hinvy_IzxmuxBySy_IzTxeNy_Iz
  MatDestroy(&_rhsR);
  ierr = MatMatMatMult(*_mu,tempMats._Hyinv_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsR);CHKERRQ(ierr);
  ierr = MatScale(_rhsR,_alphaF);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(tempMats._muxBySy_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

  ierr = MatAYPX(_rhsR,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatMatMult(tempMats._H,_rhsR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
  ierr = MatCopy(temp3,_rhsR,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
  MatDestroy(&temp3); //!!!
  ierr = PetscObjectSetName((PetscObject) _rhsR, "rhsR");CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&temp2);
  //~ierr = MatView(_rhsR,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  // else if bcR = traction-free
  // _rhsR is unneeded because bcR = 0

  // in computation of A
  // if bcR = displacement: _alphaR*mu*Hinvy_Iz*ENy_Iz + _beta*Hinvy_Iz*(muxBySy_Iz)'*ENy_Iz
  ierr = MatMatMatMult(*_mu,tempMats._Hyinv_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AR);CHKERRQ(ierr);
  ierr = MatScale(tempMats._AR,_alphaR);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(tempMats._muxBySy_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

  ierr = MatAYPX(tempMats._AR,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tempMats._AR, "AR");CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&temp2);


   MatDestroy(&e0y_Iz);
   MatDestroy(&eNy_Iz);
   MatDestroy(&E0y_Iz);
   MatDestroy(&ENy_Iz);

}


// matrices in z-direction (top and bottom boundaries)
  // These matrices are nnz if Nz > 1
{
  // kron(Iy,E0z)
  Mat Iy_E0z;
  {
    Spmat E0z(_Nz,_Nz);
    E0z(0,0,1.0);
    kronConvert(tempMats._Iy,E0z,Iy_E0z,1,1);
    ierr = PetscObjectSetName((PetscObject) Iy_E0z, "Iy_E0z");CHKERRQ(ierr);
  }

  // kron(Iy,e0z)
  Mat Iy_e0z;
  {
    Spmat e0z(_Nz,1);
    e0z(0,0,1.0);
    kronConvert(tempMats._Iy,e0z,Iy_e0z,1,1);
    ierr = PetscObjectSetName((PetscObject) Iy_e0z, "Iy_e0z");CHKERRQ(ierr);
  }

  // kron(Iy,ENz)
  Mat Iy_ENz;
  {
    Spmat ENz(_Nz,_Nz);
    ENz(_Nz-1,_Nz-1,1.0);
    kronConvert(tempMats._Iy,ENz,Iy_ENz,1,1);
    ierr = PetscObjectSetName((PetscObject) Iy_ENz, "Iy_ENz");CHKERRQ(ierr);
  }

  // kron(Iy,eNz)
  Mat Iy_eNz;
  {
    Spmat eNz(_Nz,1);
    eNz(_Nz-1,0,1.0);
    kronConvert(tempMats._Iy,eNz,Iy_eNz,1,1);
    ierr = PetscObjectSetName((PetscObject) Iy_eNz, "Iy_eNz");CHKERRQ(ierr);
  }


  // enforcement of top boundary bcT ===================================
  // map bcS to rhs
  // if bcT = traction:
  MatDestroy(&_rhsT);
  ierr = MatMatMult(tempMats._Iy_Hzinv,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsT);CHKERRQ(ierr);
  ierr = MatScale(_rhsT,_alphaS);CHKERRQ(ierr);
  ierr = MatMatMult(tempMats._H,_rhsT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
  ierr = MatCopy(temp3,_rhsT,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
  MatDestroy(&temp3); //!!!
  ierr = PetscObjectSetName((PetscObject) _rhsT, "rhsT");CHKERRQ(ierr);
  //~ierr = MatView(_rhsT,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // in computation of A
  // if bcT = traction-free: _alphaS*Iy_Hinvz*Iy_E0z*muxIy_BzSz
  ierr = MatMatMatMult(tempMats._Iy_Hzinv,Iy_E0z,tempMats._muxIy_BzSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AT);CHKERRQ(ierr);
  ierr = MatScale(tempMats._AT,_alphaS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tempMats._AT, "AT");CHKERRQ(ierr);
  //~ierr = MatView(_AT,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // enforcement of bottom boundary bcB ================================
  // map bcB to rhs
  // if bcB = traction:
  MatDestroy(&_rhsB);
  ierr = MatMatMult(tempMats._Iy_Hzinv,Iy_eNz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsB);CHKERRQ(ierr);
  ierr = MatScale(_rhsB,_alphaD);CHKERRQ(ierr);
    ierr = MatMatMult(tempMats._H,_rhsB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
  ierr = MatCopy(temp3,_rhsB,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
  MatDestroy(&temp3); //!!!
  ierr = PetscObjectSetName((PetscObject) _rhsB, "rhsB");CHKERRQ(ierr);
  //~ierr = MatView(_rhsB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // in computation of A
  // if bcB = traction-free: _alphaD*Iy_Hinvz*Iy_E0z*muxIy_BzSz
  ierr = MatMatMatMult(tempMats._Iy_Hzinv,Iy_ENz,tempMats._muxIy_BzSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AB);CHKERRQ(ierr);
  ierr = MatScale(tempMats._AB,_alphaD);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) tempMats._AB, "AB");CHKERRQ(ierr);


   MatDestroy(&Iy_e0z);
   MatDestroy(&Iy_eNz);
   MatDestroy(&Iy_E0z);
   MatDestroy(&Iy_ENz);

}
  _runTime += MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function satBoundaries in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}



// compute D2ymu using my class Spmat
PetscErrorCode SbpOps::computeD2ymu(const TempMats& tempMats, Mat &D2ymu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


  // kron(Dy,Iz) (interior stencil)
  Mat Dy_Iz;
  {
    if (_order==2) { kronConvert(tempMats._D1yint,tempMats._Iz,Dy_Iz,2,2); }
    else if (_order==4) { kronConvert(tempMats._D1yint,tempMats._Iz,Dy_Iz,5,5); }
    ierr = PetscObjectSetName((PetscObject) Dy_Iz, "Dyint_Iz");CHKERRQ(ierr);
    #if DEBUG > 0
      ierr = checkMatrix(&Dy_Iz,_debugFolder,"Dyint_Iz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
  }


  // mu*kron(Hy,Iz)
  Mat muxHy_Iz;
  {
    Mat temp;
    kronConvert(tempMats._Hy,tempMats._Iz,temp,1,0);
    ierr = MatMatMult(*_mu,temp,MAT_INITIAL_MATRIX,1.0,&muxHy_Iz);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) muxHy_Iz, "muxHy_Iz");CHKERRQ(ierr);
    #if DEBUG > 0
      ierr = checkMatrix(&muxHy_Iz,_debugFolder,"muxHy_Iz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(muxHy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
    MatDestroy(&temp);
  }

  Mat temp1,temp2;
  ierr = MatTransposeMatMult(Dy_Iz,muxHy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(temp1,Dy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
  ierr = MatScale(temp2,-1);CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&Dy_Iz);
  MatDestroy(&muxHy_Iz);


  Mat Rymu;
  ierr = computeRymu(tempMats,Rymu);CHKERRQ(ierr);
  ierr = MatAXPY(temp2,-1,Rymu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  MatDestroy(&Rymu);

  ierr = MatAXPY(temp2,1,tempMats._muxBySy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatMatMult(tempMats._Hyinv_Iz,temp2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2ymu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);
  MatDestroy(&temp2);
  #if DEBUG > 0
    ierr = checkMatrix(&D2ymu,_debugFolder,"D2ymu");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(D2ymu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SbpOps::computeRzmu(const TempMats& tempMats,Mat &Rzmu)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


switch ( _order ) {
    case 2:
    {
      Spmat D2z(_Nz,_Nz);
      Spmat C2z(_Nz,_Nz);
      if (_Nz > 1) { sbpSpmat2(_Nz,1.0/_dz,D2z,C2z); }


      // kron(Iy,C2z)
      Mat Iy_C2z;
      {
        kronConvert(tempMats._Iy,C2z,Iy_C2z,1,0);
        ierr = PetscObjectSetName((PetscObject) Iy_C2z, "Iy_C2zz");CHKERRQ(ierr);
        #if DEBUG > 0
          ierr = checkMatrix(&Iy_C2z,_debugFolder,"Iy_Cz");CHKERRQ(ierr);
        #endif
        #if VERBOSE > 2
          ierr = MatView(Iy_C2z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        #endif
      }

      // kron(Iy,D2z)
      Mat Iy_D2z;
      {
        kronConvert(tempMats._Iy,D2z,Iy_D2z,5,5);
        ierr = PetscObjectSetName((PetscObject) Iy_D2z, "Iy_D2z");CHKERRQ(ierr);
        #if DEBUG > 0
          ierr = checkMatrix(&Iy_D2z,_debugFolder,"Iy_D2z");CHKERRQ(ierr);
        #endif
        #if VERBOSE > 2
          ierr = MatView(Iy_D2z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        #endif
      }

      // Rzmu = (Iy_D2z^T x Iy_C2z x mu x Iy_D2z)/4/dz^3;
      Mat temp;
      ierr = MatTransposeMatMult(Iy_D2z,Iy_C2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp,*_mu,Iy_D2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
      ierr = MatScale(Rzmu,0.25*pow(_dz,3));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rzmu, "Rzmu");CHKERRQ(ierr);

      MatDestroy(&temp);
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
      if (_Nz > 1) { sbpSpmat4(_Nz,1/_dz,D3z,D4z,C3z,C4z); }

      Mat mu3;
      {
        Spmat B3(_Ny*_Nz,_Ny*_Nz);
        B3(0,0,0.5*(_muArr[0]+_muArr[1]));
        B3(_Ny*_Nz-1,_Ny*_Nz-1,0.5*(_muArr[_Ny*_Nz-2]+_muArr[_Ny*_Nz-1]));
        for (PetscInt Ii=1;Ii<_Ny*_Nz-1;Ii++)
        {
          B3(Ii,Ii,0.5*(_muArr[Ii]+_muArr[Ii+1]));
        }
        //~Spmat B3(_Ny,_Ny);
        //~B3.eye();
        B3.convert(mu3,1);
      }

      Mat Iy_D3z;
      kronConvert(tempMats._Iy,D3z,Iy_D3z,6,6);

      Mat Iy_C3z;
      kronConvert(tempMats._Iy,C3z,Iy_C3z,1,0);

      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
      Mat temp1,temp2;
      ierr = MatTransposeMatMult(Iy_D3z,Iy_C3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp1,mu3,Iy_D3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
      ierr = MatScale(temp2,1.0/_dz/18);CHKERRQ(ierr);
      MatDestroy(&temp1);
      MatDestroy(&Iy_D3z);
      MatDestroy(&Iy_C3z);
      MatDestroy(&mu3);


      Mat Iy_D4z;
        kronConvert(tempMats._Iy,D4z,Iy_D4z,5,5);

      Mat Iy_C4z;
      kronConvert(tempMats._Iy,C4z,Iy_C4z,1,1);


      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
      ierr = MatTransposeMatMult(Iy_D4z,Iy_C4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp1,*_mu,Iy_D4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);
      ierr = MatScale(Rzmu,1.0/_dz/144);CHKERRQ(ierr);

      ierr = MatAYPX(Rzmu,1.0,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rzmu, "Rzmu");CHKERRQ(ierr);

      MatDestroy(&temp1);
      MatDestroy(&temp2);
      MatDestroy(&Iy_D4z);
      MatDestroy(&Iy_C4z);

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



PetscErrorCode SbpOps::computeRymu(const TempMats& tempMats,Mat &Rymu)
{
  PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeR2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


switch ( _order ) {
    case 2:
    {
      Spmat D2y(_Ny,_Ny);
      Spmat C2y(_Ny,_Ny);
      sbpSpmat2(_Ny,1/_dy,D2y,C2y);


      // kron(D2y,Iz)
      Mat D2y_Iz;
      {
        kronConvert(D2y,tempMats._Iz,D2y_Iz,5,5);
        ierr = PetscObjectSetName((PetscObject) D2y_Iz, "D2y_Iz");CHKERRQ(ierr);
        #if DEBUG > 0
          ierr = checkMatrix(&D2y_Iz,_debugFolder,"D2y_Iz");CHKERRQ(ierr);
        #endif
        #if VERBOSE > 2
          ierr = MatView(D2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        #endif
      }

      // kron(C2y,Iz)
      Mat C2y_Iz;
      {
        kronConvert(C2y,tempMats._Iz,C2y_Iz,5,5);
        ierr = PetscObjectSetName((PetscObject) C2y_Iz, "C2y_Iz");CHKERRQ(ierr);
        #if DEBUG > 0
          ierr = MatView(C2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        #endif
        #if VERBOSE > 2
          ierr = MatView(C2y_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        #endif
      }

      // Rymu = (D2y_Iz^T x C2y_Iz x mu x D2y_Iz)/4/dy^3;
      Mat temp;
      ierr = MatTransposeMatMult(D2y_Iz,C2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp,*_mu,D2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatScale(Rymu,0.25*pow(_dy,3));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rymu, "Rymu");CHKERRQ(ierr);

      MatDestroy(&temp);
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

      Mat mu3;
      {
        Spmat B3(_Ny*_Nz,_Ny*_Nz);
        B3(0,0,0.5*(_muArr[0]+_muArr[1]));
        B3(_Ny*_Nz-1,_Ny*_Nz-1,0.5*(_muArr[_Ny*_Nz-2]+_muArr[_Ny*_Nz-1]));
        for (PetscInt Ii=1;Ii<_Ny*_Nz-1;Ii++)
        {
          B3(Ii,Ii,0.5*(_muArr[Ii]+_muArr[Ii+1]));
        }
        B3.convert(mu3,1);
        //~PetscPrintf(PETSC_COMM_WORLD,"\n\nB3:\n");
        //~B3.printPetsc();
      }

      Mat D3y_Iz;
      kronConvert(D3y,tempMats._Iz,D3y_Iz,6,6);

      Mat C3y_Iz;
      kronConvert(C3y,tempMats._Iz,C3y_Iz,1,1);


      // Rymu = (D3y_Iz^T x C3y_Iz x mu3 x D3y_Iz)/18/dy
      //      + (D4y_Iz^T x C4y_Iz x mu x D4y_Iz)/144/dy
      Mat temp1,temp2;
      ierr = MatTransposeMatMult(D3y_Iz,C3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp1,mu3,D3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
      ierr = MatScale(temp2,1.0/_dy/18.0);CHKERRQ(ierr);
      MatDestroy(&temp1);
      MatDestroy(&D3y_Iz);
      MatDestroy(&C3y_Iz);
      MatDestroy(&mu3);


      Mat D4y_Iz;
      kronConvert(D4y,tempMats._Iz,D4y_Iz,5,5);

      Mat C4y_Iz;
      kronConvert(C4y,tempMats._Iz,C4y_Iz,1,0);

      ierr = MatTransposeMatMult(D4y_Iz,C4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      ierr = MatMatMatMult(temp1,*_mu,D4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
      ierr = MatScale(Rymu,1.0/_dy/144.0);CHKERRQ(ierr);

      ierr = MatAYPX(Rymu,1.0,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) Rymu, "Rymu");CHKERRQ(ierr);

      MatDestroy(&temp1);
      MatDestroy(&temp2);
      MatDestroy(&D4y_Iz);
      MatDestroy(&C4y_Iz);

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
PetscErrorCode SbpOps::computeD2zmu(const TempMats& tempMats,Mat &D2zmu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


  // kron(Iy,Dz)
  Mat Iy_Dz;
  {
    if (_order==2) { kronConvert(tempMats._Iy,tempMats._D1zint,Iy_Dz,2,2); }
    else if (_order==4) { kronConvert(tempMats._Iy,tempMats._D1zint,Iy_Dz,5,5); }
    ierr = PetscObjectSetName((PetscObject) Iy_Dz, "Iy_Dz");CHKERRQ(ierr);
    #if DEBUG > 0
      ierr = checkMatrix(&Iy_Dz,_debugFolder,"Iy_Dz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
  }


  // mu*kron(Iy,Hz)
  Mat muxIy_Hz;
  {
    Spmat muxIy_HzS(_Ny*_Nz,_Ny*_Nz);
    muxIy_HzS = kron(tempMats._Iy,tempMats._Hz);
    muxIy_HzS.convert(muxIy_Hz,1);
    ierr = PetscObjectSetName((PetscObject) muxIy_Hz, "muxIy_Hz");CHKERRQ(ierr);
    ierr = MatMatMult(*_mu,muxIy_Hz,MAT_INITIAL_MATRIX,1.0,&muxIy_Hz);CHKERRQ(ierr);
    #if DEBUG > 0
      ierr = checkMatrix(&muxIy_Hz,_debugFolder,"muxIy_Hz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(muxIy_Hz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
  }

  Mat temp1,temp2;
  ierr = MatTransposeMatMult(Iy_Dz,muxIy_Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  ierr = MatMatMult(temp1,Iy_Dz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
  ierr = MatScale(temp2,-1);CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&Iy_Dz);
  MatDestroy(&muxIy_Hz);

  Mat Rzmu;
  ierr = computeRzmu(tempMats,Rzmu);
  ierr = MatAXPY(temp2,-1,Rzmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  MatDestroy(&Rzmu);

  ierr = MatAXPY(temp2,1,tempMats._muxIy_BzSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatMatMult(tempMats._Iy_Hzinv,temp2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D2zmu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);
  MatDestroy(&temp2);
  #if DEBUG > 0
    ierr = checkMatrix(&D2zmu,_debugFolder,"D2zmu");CHKERRQ(ierr);
  #endif
  #if VERBOSE > 2
    ierr = MatView(D2zmu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  #endif


  #if VERBOSE >1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}



// compute matrix for shear stress (scaled by mu!!): kron(Dy,Iz)
PetscErrorCode SbpOps::computeDy_Iz(const TempMats& tempMats)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeDy_Iz in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  Mat temp;
  kronConvert(tempMats._D1y,tempMats._Iz,temp,5,5);

  MatDestroy(&_Dy_Iz);
  ierr = MatMatMult(*_mu,temp,MAT_INITIAL_MATRIX,1.0,&_Dy_Iz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _Dy_Iz, "_Dy_Iz");CHKERRQ(ierr);

  MatDestroy(&temp);

#if DEBUG > 0
ierr = checkMatrix(&_Dy_Iz,_debugFolder,"Dy_Iz");CHKERRQ(ierr);
#endif
#if VERBOSE > 2
  ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeDy_Iz in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
}


// compute matrix relating displacement to vector b containing boundary conditions
PetscErrorCode SbpOps::computeA(const TempMats& tempMats)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function computeA in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  Mat D2ymu;
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D2ymu);CHKERRQ(ierr);
  ierr = computeD2ymu(tempMats,D2ymu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);

  Mat D2zmu;
  //~ierr = MatCreate(PETSC_COMM_WORLD,&D2zmu);CHKERRQ(ierr);
  ierr = computeD2zmu(tempMats,D2zmu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);

  // compute A
  MatDestroy(&_A);
  ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
  ierr = MatAYPX(_A,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // use new Mats _AL etc
  ierr = MatAXPY(_A,1.0,tempMats._AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,tempMats._AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,tempMats._AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(_A,1.0,tempMats._AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"matA");CHKERRQ(ierr);
  //ierr = MatView(_A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  // clean up
  ierr = MatDestroy(&D2ymu);CHKERRQ(ierr);
  ierr = MatDestroy(&D2zmu);CHKERRQ(ierr);

  // if using H A uhat = H rhs
  Mat temp;
  ierr = MatMatMult(tempMats._H,_A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatCopy(temp,_A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&temp);CHKERRQ(ierr);
  ierr = MatSetOption(_A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) _A, "_A");CHKERRQ(ierr);

  _runTime = MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function computeA in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return 0;
}






PetscErrorCode sbpSpmat2(const PetscInt N,const PetscScalar scale, Spmat& D2, Spmat& C2)
{
PetscErrorCode ierr = 0;
//~double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpSpmat2 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  assert(N > 2);

  PetscInt Ii=0;

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


  C2.eye();
  C2(0,0,0);
  C2(N-1,N-1,0);
  #if VERBOSE > 2
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nC2:\n");CHKERRQ(ierr);
    C2.printPetsc();
  #endif


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbpSpmat2 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  //~_runTime = MPI_Wtime() - startTime;
  return ierr;
}






PetscErrorCode sbpSpmat4(const PetscInt N,const PetscScalar scale,
                Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpSpmat4 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  assert(N > 8);

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


PetscErrorCode sbpSpmat(const PetscInt order, const PetscInt N,const PetscScalar scale,Spmat& H,Spmat& Hinv,Spmat& D1,
                 Spmat& D1int, Spmat& S)
{
PetscErrorCode ierr = 0;
//~double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbpSpmat in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

PetscInt Ii=0;

switch ( order ) {
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

      D1 = D1int; // copy D1int's interior
      D1(N-1,N-3,S(N-1,N-3)); // last row
      D1(N-1,N-2,S(N-1,N-2));
      D1(N-1,N-1,S(N-1,N-1));
      // only want shear stress on fault, not interior
      D1(0,0,-S(0,0)); D1(0,1,-S(0,1)); D1(0,2,-S(0,2)); // first row
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1:\n");CHKERRQ(ierr);
        D1.printPetsc();
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
  //~_runTime = MPI_Wtime() - startTime;
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


  //~// matrices to map SAT boundaries to A
  //~str = outputDir + "AL";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_AL,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "AR";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_rhsR,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "AT";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_rhsT,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
//~
  //~str = outputDir + "AB";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = MatView(_AB,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOps in sbpOps.cpp\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
};


//=================== functions for struct =============================

TempMats::TempMats(const PetscInt order,const PetscInt Ny,const PetscScalar dy,const PetscInt Nz,const PetscScalar dz, Mat*mu)
: _order(order),_Ny(Ny),_Nz(Nz),_dy(dy),_dz(dz),_mu(mu),
  _Hy(Ny,Ny),_D1y(Ny,Ny),_D1yint(Ny,Ny),_Iy(Ny,Ny),
  _Hz(Nz,Nz),_D1zint(Nz,Nz),_Iz(Nz,Nz)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats::TempMats in sbpOps.cpp.\n");
#endif

  // so the destructor can run happily
  MatCreate(PETSC_COMM_WORLD,&_muxBySy_Iz);
  MatCreate(PETSC_COMM_WORLD,&_Hyinv_Iz);

  MatCreate(PETSC_COMM_WORLD,&_muxIy_BzSz);
  MatCreate(PETSC_COMM_WORLD,&_Iy_Hzinv);

  MatCreate(PETSC_COMM_WORLD,&_AL);
  MatCreate(PETSC_COMM_WORLD,&_AR);
  MatCreate(PETSC_COMM_WORLD,&_AT);
  MatCreate(PETSC_COMM_WORLD,&_AB);

  MatCreate(PETSC_COMM_WORLD,&_H);

  _Iy.eye(); // matrix size is set during colon initialization
  _Iz.eye();


  // going from 1D in y to 2D:
  {
    Spmat Hyinv(_Ny,_Ny),Sy(_Ny,_Ny);
    sbpSpmat(order,Ny,1.0/dy,_Hy,Hyinv,_D1y,_D1yint,Sy);

    // kron(Hyinv,Iz)
    {
      Spmat Hyinv_Iz(_Ny*_Nz,_Ny*_Nz);
      Hyinv_Iz = kron(Hyinv,_Iz);
      Hyinv_Iz.convert(_Hyinv_Iz,1);
      PetscObjectSetName((PetscObject) _Hyinv_Iz, "Hyinv_Iz");
    }


    // mu*kron(BySy,Iz)
    {
      Mat temp;
      if (_order==2) { kronConvert(Sy,_Iz,temp,3,3); }
      if (_order==4) { kronConvert(Sy,_Iz,temp,5,5); }
      MatMatMult(*_mu,temp,MAT_INITIAL_MATRIX,1.0,&_muxBySy_Iz);
      PetscObjectSetName((PetscObject) _muxBySy_Iz, "muxBySy_Iz");
      MatDestroy(&temp);
    }
  }



  // going from 1D in z to 2D:
  {
    Spmat Hzinv(_Nz,_Nz),D1z(_Nz,_Nz),Sz(_Nz,_Nz);
    if (Nz > 1) { sbpSpmat(order,Nz,1/dz,_Hz,Hzinv,D1z,_D1zint,Sz); }
    else { _Hz.eye(); }

    // kron(Iy,Hzinv)
    {
      kronConvert(_Iy,Hzinv,_Iy_Hzinv,1,0);
      PetscObjectSetName((PetscObject) _Iy_Hzinv, "Iy_Hzinv");
    }

    // mu*kron(Iy,BzSz)
    {
      Mat temp;
      if (_order==2) { kronConvert(_Iy,Sz,temp,3,3); }
      if (_order==4) { kronConvert(_Iy,Sz,temp,5,5); }
      MatMatMult(*_mu,temp,MAT_INITIAL_MATRIX,1.0,&_muxIy_BzSz);
      PetscObjectSetName((PetscObject) _muxIy_BzSz, "muxIy_BzSz");
      MatDestroy(&temp);
    }
  }


  // H matrix
  {
    // kron(Hy,Hz)
    kronConvert(_Hy,_Hz,_H,1,0);
  }
  PetscObjectSetName((PetscObject) _H, "H");

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats::TempMats in sbpOps.cpp.\n");
#endif
};

TempMats::~TempMats()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats::~TempMats in sbpOps.cpp.\n");
#endif

  MatDestroy(&_muxBySy_Iz);
  MatDestroy(&_Hyinv_Iz);

  MatDestroy(&_muxIy_BzSz);
  MatDestroy(&_Iy_Hzinv);

  MatDestroy(&_AL);
  MatDestroy(&_AR);
  MatDestroy(&_AT);
  MatDestroy(&_AB);

  MatDestroy(&_H);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats::~TempMats in sbpOps.cpp.\n");
#endif
};


