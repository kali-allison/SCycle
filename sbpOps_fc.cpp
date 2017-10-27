#include "sbpOps_fc.hpp"




//================= constructor and destructor ========================
SbpOps_fc::SbpOps_fc(Domain&D,PetscInt Ny, PetscInt Nz,Vec& muVec,
  string bcT,string bcR,string bcB, string bcL, string type)
: _order(D._order),_Ny(Ny),_Nz(Nz),_dy(D._Ly/(D._Ny-1.)),_dz(D._Lz/(D._Nz-1.)),
  _muVec(&muVec),_mu(NULL),
  _type(type),_bcTType(bcT),_bcRType(bcR),_bcBType(bcB),_bcLType(bcL),
  _rhsL(NULL),_rhsR(NULL),_rhsT(NULL),_rhsB(NULL),
  _Hyinv_Iz(NULL),_Iy_Hzinv(NULL),_e0y_Iz(NULL),_eNy_Iz(NULL),
  _E0y_Iz(NULL),_ENy_Iz(NULL),_Iy_E0z(NULL),_Iy_ENz(NULL),
  _alphaT(-1.0),_alphaDy(-4.0/_dy),_alphaDz(-4.0/_dz),_beta(1.0),
  _debugFolder("./matlabAnswers/"),_H(NULL),_Hinv(NULL),_A(NULL),
  _Dy_Iz(NULL),_Iy_Dz(NULL),
  _Ry(NULL),_Rz(NULL),_By_Iz(NULL),_Iy_Bz(NULL),_Iy_e0z(NULL),_Iy_eNz(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif

  //~ if (_Ny == 1) { return; }

  stringstream ss;
  ss << "order" << _order << "Ny" << _Ny << "Nz" << _Nz << "/";
  _debugFolder += ss.str();


  // construct matrix mu
  MatCreate(PETSC_COMM_WORLD,&_mu);
  MatSetSizes(_mu,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);
  MatSetFromOptions(_mu);
  MatMPIAIJSetPreallocation(_mu,1,NULL,1,NULL);
  MatSeqAIJSetPreallocation(_mu,1,NULL);
  MatSetUp(_mu);
  MatDiagonalSet(_mu,*_muVec,INSERT_VALUES);

  // NOT a member of this class, contains stuff to be deleted before
  // end of constructor to save on memory usage.
  TempMats_fc tempFactors(_order,_Ny,_dy,_Nz,_dz,_mu);

  // reset SAT params
  if (_order==4) {
    _alphaDy = 2.0*-48.0/17.0 /_dy;
    _alphaDz = 2.0*-48.0/17.0 /_dz;
  }

  constructH(tempFactors);
  constructHinv(tempFactors);
  construct1stDerivs(tempFactors);
  satBoundaries(tempFactors);

  constructA(tempFactors);

  MatDuplicate(tempFactors._Hyinv_Iz,MAT_COPY_VALUES,&_Hyinv_Iz);
  MatDuplicate(tempFactors._Iy_Hzinv,MAT_COPY_VALUES,&_Iy_Hzinv);

  Spmat e0y(_Ny,1); e0y(0,0,1.0);
  kronConvert(e0y,tempFactors._Iz,_e0y_Iz,1,1);

  Spmat eNy(_Ny,1); eNy(_Ny-1,0,1.0);
  kronConvert(eNy,tempFactors._Iz,_eNy_Iz,1,1);

  Spmat E0y(_Ny,_Ny); E0y(0,0,1.0);
  kronConvert(E0y,tempFactors._Iz,_E0y_Iz,1,1);

  Spmat ENy(_Ny,_Ny); ENy(_Ny-1,_Ny-1,1.0);
  kronConvert(ENy,tempFactors._Iz,_ENy_Iz,1,1);

  Spmat E0z(_Nz,_Nz); E0z(0,0,1.0);
  kronConvert(tempFactors._Iy,E0z,_Iy_E0z,1,1);

  Spmat ENz(_Nz,_Nz); ENz(_Nz-1,_Nz-1,1.0);
  kronConvert(tempFactors._Iy,ENz,_Iy_ENz,1,1);

  Spmat e0z(_Nz,1); e0z(0,0,1.0);
  kronConvert(tempFactors._Iy,e0z,_Iy_e0z,1,1);

  Spmat eNz(_Nz,1); eNz(_Nz-1,0,1.0);
  kronConvert(tempFactors._Iy,eNz,_Iy_eNz,1,1);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in sbpOps.cpp.\n");
#endif
}

SbpOps_fc::~SbpOps_fc()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in sbpOps_fc.cpp.\n");
#endif

  MatDestroy(&_H);
  MatDestroy(&_Hinv);

  // final operator A
  MatDestroy(&_A);

  // SAT enforcement of boundary conditions
  MatDestroy(&_rhsL);
  MatDestroy(&_rhsR);
  MatDestroy(&_rhsT);
  MatDestroy(&_rhsB);

  MatDestroy(&_Dy_Iz);
  MatDestroy(&_Iy_Dz);

  MatDestroy(&_mu);

  MatDestroy(&_Hyinv_Iz);
  MatDestroy(&_Iy_Hzinv);
  MatDestroy(&_e0y_Iz);
  MatDestroy(&_eNy_Iz);
  MatDestroy(&_E0y_Iz);
  MatDestroy(&_ENy_Iz);
  MatDestroy(&_Iy_E0z);
  MatDestroy(&_Iy_ENz);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps_fc.cpp.\n");
#endif
}

PetscErrorCode SbpOps_fc::getA(Mat &mat)
{
  #if VERBOSE > 1
    string funcName = "SbpOps_fc::getA";
    string fileName = "SbpOps_fc.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // return shallow copy of A:
  mat = _A;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
  return 0;
}

PetscErrorCode SbpOps_fc::getAlphay(PetscScalar &alphaDy)
{
  #if VERBOSE > 2
    string funcName = "SbpOps_fc::getAlphay";
    string fileName = "SbpOps_fc.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // return copy of alphaDy:
  alphaDy = _alphaDy;
  #if VERBOSE > 2
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
  return 0;
}

PetscErrorCode SbpOps_fc::getAlphaz(PetscScalar &alphaDz)
{
  #if VERBOSE > 2
    string funcName = "SbpOps_fc::getAlphaz";
    string fileName = "SbpOps_fc.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // return copy of alphaDz:
  alphaDz = _alphaDz;

  #if VERBOSE > 2
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
  return 0;
}

PetscErrorCode SbpOps_fc::getH(Mat &mat)
{
  #if VERBOSE > 1
    string funcName = "SbpOps_fc::getH";
    string fileName = "SbpOps_fc.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // return shallow copy of H:
  mat = _H;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
  return 0;
}


// for energy balance
PetscErrorCode SbpOps_fc::getDs(Mat &Dy,Mat &Dz) { Dy = _Dy_Iz; Dz = _Iy_Dz; return 0; }
PetscErrorCode SbpOps_fc::getMus(Mat &mu,Mat &muqy,Mat &murz) { mu = _mu; muqy = _mu; murz = _mu; return 0; }
PetscErrorCode SbpOps_fc::getR(Mat& Ry, Mat& Rz) { Ry=_Ry; Rz=_Rz; return 0; }
PetscErrorCode SbpOps_fc::getEs(Mat& E0y_Iz,Mat& ENy_Iz,Mat& Iy_E0z,Mat& Iy_ENz)
{
  E0y_Iz = _E0y_Iz;
  ENy_Iz = _ENy_Iz;
  Iy_E0z = _Iy_E0z;
  Iy_ENz = _Iy_ENz;
  return 0;
}
PetscErrorCode SbpOps_fc::getes(Mat& e0y_Iz,Mat& eNy_Iz,Mat& Iy_e0z,Mat& Iy_eNz)
{
  e0y_Iz = _e0y_Iz;
  eNy_Iz = _eNy_Iz;
  Iy_e0z = _Iy_e0z;
  Iy_eNz = _Iy_eNz;
  return 0;
}
PetscErrorCode SbpOps_fc::getBs(Mat& By_Iz,Mat& Iy_Bz)
{
  By_Iz = _By_Iz;
  Iy_Bz = _Iy_Bz;
  return 0;
}
PetscErrorCode SbpOps_fc::getHs(Mat& Hy_Iz,Mat& Iy_Hz)
{
  Hy_Iz = _Hy_Iz;
  Iy_Hz = _Iy_Hz;
  return 0;
}
PetscErrorCode SbpOps_fc::getHinvs(Mat& Hyinv_Iz,Mat& Iy_Hzinv)
{
  Hyinv_Iz = _Hyinv_Iz;
  Iy_Hzinv = _Iy_Hzinv;
  return 0;
}
PetscErrorCode SbpOps_fc::getCoordTrans(Mat& qy,Mat& rz, Mat& yq, Mat& zr) { assert(0); return 0; }


//======================================================================

/* Enforce boundary conditions using SAT penalty terms, computing both
 * matrices used to build the rhs vector from the boundary vectors and
 * the matrices that are added to the spatial derivative matrices to
 * form the A matrix.
 */
PetscErrorCode SbpOps_fc::satBoundaries(TempMats_fc& tempMats)
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

  // if bcL = displacement: _alphaD*mu*_Hinvy_Iz*e0y_Iz + _beta*_Hinvy_Iz*e0y_Iz*muxBSy_IzT
  if (!_bcLType.compare("Dirichlet")) {
    MatDestroy(&_rhsL);
    ierr = MatMatMatMult(_mu,tempMats._Hyinv_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsL);CHKERRQ(ierr);
    ierr = MatScale(_rhsL,_alphaDy);CHKERRQ(ierr);
    //~ ierr = MatTransposeMatMult(tempMats._muxBSy_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    Mat muxBSy_IzT;
    MatTranspose(tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,&muxBSy_IzT);
    MatMatMult(muxBSy_IzT,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxBSy_IzT);
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
    // if bcL = displacement: _alphaD*mu*_Hinvy_Iz*E0y_Iz + _beta*_Hinvy_Iz*muxBSy_IzT*E0y_Iz
    ierr = MatMatMatMult(_mu,tempMats._Hyinv_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AL);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AL,_alphaDy);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxBSy_Iz,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    MatTranspose(tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,&muxBSy_IzT);
    MatMatMult(muxBSy_IzT,E0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxBSy_IzT);
    ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(tempMats._AL,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AL, "AL");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);
    }
  else {
    // For rhsL: if bcL = traction: alphaT * Hinvy_Iz * e0y_Iz
    MatDestroy(&_rhsL);
    ierr = MatMatMult(tempMats._Hyinv_Iz,e0y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsL);CHKERRQ(ierr);
    ierr = MatScale(_rhsL,-_alphaT);CHKERRQ(ierr);
    ierr = MatMatMult(tempMats._H,_rhsL,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
    ierr = MatCopy(temp3,_rhsL,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3); //!!!
    ierr = PetscObjectSetName((PetscObject) _rhsL, "rhsL");CHKERRQ(ierr);

    // For A:if bcL = traction: alphaT * Hinvy_Iz * E0y_Iz * muxBySy_Iz
    ierr = MatMatMatMult(tempMats._Hyinv_Iz,E0y_Iz,tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AL);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AL,_alphaT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AL, "AL");CHKERRQ(ierr);
  }

  // enforcement of right boundary bcR =================================
  // map bcR to rhs
  if (!_bcRType.compare("Dirichlet")) {
    // if bcR = displacement: _alphaD*mu*_Hinvy_Iz*eNy_Iz + _beta*_Hinvy_Iz*muxBSy_IzT*eNy_Iz
    MatDestroy(&_rhsR);
    ierr = MatMatMatMult(_mu,tempMats._Hyinv_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsR);CHKERRQ(ierr);
    ierr = MatScale(_rhsR,_alphaDy);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxBSy_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    Mat muxBSy_IzT;
    MatTranspose(tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,&muxBSy_IzT);
    MatMatMult(muxBSy_IzT,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxBSy_IzT);
    ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(_rhsR,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatMatMult(tempMats._H,_rhsR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
    ierr = MatCopy(temp3,_rhsR,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3); //!!!
    ierr = PetscObjectSetName((PetscObject) _rhsR, "rhsR");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);


    // in computation of A
    // if bcR = displacement: _alphaD*mu*Hinvy_Iz*ENy_Iz + _beta*Hinvy_Iz*muxBSy_IzT*ENy_Iz
    ierr = MatMatMatMult(_mu,tempMats._Hyinv_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AR);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AR,_alphaDy);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxBSy_Iz,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    MatTranspose(tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,&muxBSy_IzT);
    MatMatMult(muxBSy_IzT,ENy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxBSy_IzT);
    ierr = MatMatMult(tempMats._Hyinv_Iz,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(tempMats._AR,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AR, "AR");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);
  }
  else {
    // For rhsR: if bcR = traction: alphaT * Hinvy_Iz * eNy_Iz
    MatDestroy(&_rhsR);
    ierr = MatMatMult(tempMats._Hyinv_Iz,eNy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsR);CHKERRQ(ierr);
    ierr = MatScale(_rhsR,_alphaT);CHKERRQ(ierr);
    ierr = MatMatMult(tempMats._H,_rhsR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
    ierr = MatCopy(temp3,_rhsR,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3); //!!!
    ierr = PetscObjectSetName((PetscObject) _rhsR, "rhsR");CHKERRQ(ierr);


    // For A:if bcR = traction: alphaT * Hinvy_Iz * ENy_Iz * muxBySy_Iz
    ierr = MatMatMatMult(tempMats._Hyinv_Iz,ENy_Iz,tempMats._muxBSy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AR);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AR,_alphaT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AR, "AR");CHKERRQ(ierr);
    //~ierr = MatView(_AR,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }


   MatDestroy(&e0y_Iz);
   MatDestroy(&eNy_Iz);
   MatDestroy(&E0y_Iz);
   MatDestroy(&ENy_Iz);

}


// matrices in z-direction (top and bottom boundaries)
  // These matrices are nonzero if Nz > 1
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
  if (!_bcTType.compare("Dirichlet")) {
    // if bcR = displacement: _alphaD*mu*Iy_Hzinv*Iy_e0z + _beta*Iy_Hzinv*_muxIy_BSzT*Iy_e0z
    MatDestroy(&_rhsT);
    ierr = MatMatMatMult(_mu,tempMats._Iy_Hzinv,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsT);CHKERRQ(ierr);
    ierr = MatScale(_rhsT,_alphaDz);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxIy_BSz,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    Mat muxIy_BSzT;
    MatTranspose(tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,&muxIy_BSzT);
    MatMatMult(muxIy_BSzT,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxIy_BSzT);
    ierr = MatMatMult(tempMats._Iy_Hzinv,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(_rhsT,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatMatMult(tempMats._H,_rhsT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr);
    ierr = MatCopy(temp3,_rhsT,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    MatDestroy(&temp3);
    ierr = PetscObjectSetName((PetscObject) _rhsT, "rhsT");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);

    // in computation of A
    // if bcR = displacement: _alphaD*mu*Iy_Hzinv*Iy_E0z + _beta*Iy_Hzinv*_muxIy_BSzT*Iy_E0z
    ierr = MatMatMatMult(_mu,tempMats._Iy_Hzinv,Iy_E0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AT);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AT,_alphaDz);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxIy_BSz,Iy_E0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    MatTranspose(tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,&muxIy_BSzT);
    MatMatMult(muxIy_BSzT,Iy_E0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
    MatDestroy(&muxIy_BSzT);
    ierr = MatMatMult(tempMats._Iy_Hzinv,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(tempMats._AT,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AT, "AT");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);
  }
  else {
    // if bcT = traction: alphaT*Iy_Hzinv*Iy_e0z
    MatDestroy(&_rhsT);
    ierr = MatMatMult(tempMats._Iy_Hzinv,Iy_e0z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsT);CHKERRQ(ierr);
    ierr = MatScale(_rhsT,-_alphaT);CHKERRQ(ierr);
    ierr = MatMatMult(tempMats._H,_rhsT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
    ierr = MatCopy(temp3,_rhsT,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3);
    ierr = PetscObjectSetName((PetscObject) _rhsT, "rhsT");CHKERRQ(ierr);

    // in computation of A
    // if bcT = traction-free: _alphaT*Iy_Hinvz*Iy_E0z*muxIy_BzSz
    ierr = MatMatMatMult(tempMats._Iy_Hzinv,Iy_E0z,tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AT);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AT,_alphaT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AT, "AT");CHKERRQ(ierr);
  }

  // enforcement of bottom boundary bcB ================================
  // map bcB to rhs
  if (!_bcBType.compare("Dirichlet")) {
    // if bcR = displacement: _alphaD*mu*Iy_Hzinv*Iy_eNz + _beta*Iy_Hzinv*_muxIy_BSzT*Iy_eNz
    MatDestroy(&_rhsB);
    ierr = MatMatMatMult(_mu,tempMats._Iy_Hzinv,Iy_eNz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsB);CHKERRQ(ierr);
    ierr = MatScale(_rhsB,_alphaDz);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxIy_BSz,Iy_eNz,MAT_INITIAL_MATRIX,1.0,&temp1);CHKERRQ(ierr);
    Mat muxIy_BSzT;
    MatTranspose(tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,&muxIy_BSzT);
    MatMatMult(muxIy_BSzT,Iy_eNz,MAT_INITIAL_MATRIX,1.0,&temp1);
    MatDestroy(&muxIy_BSzT);
    ierr = MatMatMult(tempMats._Iy_Hzinv,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
    ierr = MatAYPX(_rhsB,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    ierr = MatMatMult(tempMats._H,_rhsB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr);
    ierr = MatCopy(temp3,_rhsB,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3); //!!!
    ierr = PetscObjectSetName((PetscObject) _rhsB, "rhsB");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);


    // in computation of A
    // if bcR = displacement: _alphaD*mu*Iy_Hzinv*Iy_ENz + _beta*Iy_Hzinv*(_muxIy_BSz)'*Iy_ENz
    ierr = MatMatMatMult(_mu,tempMats._Iy_Hzinv,Iy_ENz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AB);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AB,_alphaDz);CHKERRQ(ierr);

    //~ ierr = MatTransposeMatMult(tempMats._muxIy_BSz,Iy_ENz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
    MatTranspose(tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,&muxIy_BSzT);
    MatMatMult(muxIy_BSzT,Iy_ENz,MAT_INITIAL_MATRIX,1.0,&temp1);
    MatDestroy(&muxIy_BSzT);
    ierr = MatMatMult(tempMats._Iy_Hzinv,temp1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);

    ierr = MatAYPX(tempMats._AB,_beta,temp2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AB, "AB");CHKERRQ(ierr);
    MatDestroy(&temp1);
    MatDestroy(&temp2);
  }
  else {
    // if bcB = traction: alphaT*Iy_Hinvz*Iy_eNz
    MatDestroy(&_rhsB);
    ierr = MatMatMult(tempMats._Iy_Hzinv,Iy_eNz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_rhsB);CHKERRQ(ierr);
    ierr = MatScale(_rhsB,_alphaT);CHKERRQ(ierr);
      ierr = MatMatMult(tempMats._H,_rhsB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp3);CHKERRQ(ierr); //!!!
    ierr = MatCopy(temp3,_rhsB,SAME_NONZERO_PATTERN);CHKERRQ(ierr); //!!!
    MatDestroy(&temp3);
    ierr = PetscObjectSetName((PetscObject) _rhsB, "rhsB");CHKERRQ(ierr);
    //~ierr = MatView(_rhsB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // in computation of A
    // if bcB = traction-free: _alphaT*Iy_Hinvz*Iy_ENz*muxIy_BzSz
    ierr = MatMatMatMult(tempMats._Iy_Hzinv,Iy_ENz,tempMats._muxIy_BSz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMats._AB);CHKERRQ(ierr);
    ierr = MatScale(tempMats._AB,_alphaT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tempMats._AB, "AB");CHKERRQ(ierr);
  }


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
PetscErrorCode SbpOps_fc::constructD2ymu(const TempMats_fc& tempMats, Mat &D2ymu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


  // kron(Dy,Iz) (interior stencil)
  Mat Dy_Iz;
  {
    if (_order==2) { kronConvert(tempMats._D1yint,tempMats._Iz,Dy_Iz,2,2); }
    else  { kronConvert(tempMats._D1yint,tempMats._Iz,Dy_Iz,5,5); }
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
    ierr = MatMatMult(_mu,temp,MAT_INITIAL_MATRIX,1.0,&muxHy_Iz);CHKERRQ(ierr);
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
  //~ ierr = MatTransposeMatMult(Dy_Iz,muxHy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  Mat Dy_IzT;
  MatTranspose(Dy_Iz,MAT_INITIAL_MATRIX,&Dy_IzT);
  MatMatMult(Dy_IzT,muxHy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
  MatDestroy(&Dy_IzT);
  ierr = MatMatMult(temp1,Dy_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
  ierr = MatScale(temp2,-1);CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&Dy_Iz);
  MatDestroy(&muxHy_Iz);


  Mat Rymu;
  ierr = constructRymu(tempMats,Rymu);CHKERRQ(ierr);
  ierr = MatAXPY(temp2,-1,Rymu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  #if CALCULATE_ENERGY == 1
    MatDuplicate(Rymu,MAT_COPY_VALUES,&_Ry);
  #endif
  MatDestroy(&Rymu);

  ierr = MatAXPY(temp2,1,tempMats._muxBSy_Iz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructD2ymu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SbpOps_fc::constructRzmu(const TempMats_fc& tempMats,Mat &Rzmu)
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
      sbp_fc_Spmat2(_Nz,1.0/_dz,D2z,C2z);
      //~ if (_Nz > 1) { sbp_fc_Spmat2(_Nz,1.0/_dz,D2z,C2z); }

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
      //~ ierr = MatTransposeMatMult(Iy_D2z,Iy_C2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      Mat Iy_D2zT;
      MatTranspose(Iy_D2z,MAT_INITIAL_MATRIX,&Iy_D2zT);
      MatMatMult(Iy_D2zT,Iy_C2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);
      MatDestroy(&Iy_D2zT);
      ierr = MatMatMatMult(temp,_mu,Iy_D2z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);CHKERRQ(ierr);
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
      sbp_fc_Spmat4(_Nz,1/_dz,D3z,D4z,C3z,C4z);
      //~ if (_Nz > 1) { sbp_fc_Spmat4(_Nz,1/_dz,D3z,D4z,C3z,C4z); }

      Mat mu3;
      {
        PetscScalar mu=0;
        MatDuplicate(_mu,MAT_COPY_VALUES,&mu3);
        PetscInt Ii,Jj,Istart,Iend=0;
        VecGetOwnershipRange(*_muVec,&Istart,&Iend);
        if (Istart==0) {
          Jj = Istart + 1;
          VecGetValues(*_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Istart,1,&Istart,&mu,ADD_VALUES);
        }
        if (Iend==_Ny*_Nz) {
          Jj = Iend - 2;
          Ii = Iend - 1;
          VecGetValues(*_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Ii,1,&Ii,&mu,ADD_VALUES);
        }
        for (Ii=Istart+1;Ii<Iend-1;Ii++) {
          VecGetValues(*_muVec,1,&Ii,&mu);
          Jj = Ii - 1;
          MatSetValues(mu3,1,&Jj,1,&Jj,&mu,ADD_VALUES);
        }
        MatAssemblyBegin(mu3,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mu3,MAT_FINAL_ASSEMBLY);
        MatScale(mu3,0.5);
      }

      Mat Iy_D3z;
      kronConvert(tempMats._Iy,D3z,Iy_D3z,6,6);

      Mat Iy_C3z;
      kronConvert(tempMats._Iy,C3z,Iy_C3z,1,0);

      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
      Mat temp1,temp2;
      //~ ierr = MatTransposeMatMult(Iy_D3z,Iy_C3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      Mat Iy_D3zT;
      MatTranspose(Iy_D3z,MAT_INITIAL_MATRIX,&Iy_D3zT);
      MatMatMult(Iy_D3zT,Iy_C3z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
      MatDestroy(&Iy_D3zT);
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
      //~ ierr = MatTransposeMatMult(Iy_D4z,Iy_C4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      Mat Iy_D4zT;
      MatTranspose(Iy_D4z,MAT_INITIAL_MATRIX,&Iy_D4zT);
      MatMatMult(Iy_D4zT,Iy_C4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
      MatDestroy(&Iy_D4zT);
      ierr = MatMatMatMult(temp1,_mu,Iy_D4z,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rzmu);
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



PetscErrorCode SbpOps_fc::constructRymu(const TempMats_fc& tempMats,Mat &Rymu)
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
      sbp_fc_Spmat2(_Ny,1/_dy,D2y,C2y);


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

      //~ // Rymu = (D2y_Iz^T x C2y_Iz x mu x D2y_Iz)/4/dy^3;
      Mat temp;
      //~ ierr = MatTransposeMatMult(D2y_Iz,C2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
      Mat D2y_IzT;
      MatTranspose(D2y_Iz,MAT_INITIAL_MATRIX,&D2y_IzT);
      MatMatMult(D2y_IzT,C2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);
      MatDestroy(&D2y_IzT);
      ierr = MatMatMatMult(temp,_mu,D2y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
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
      sbp_fc_Spmat4(_Ny,1/_dy,D3y,D4y,C3y,C4y);

      Mat mu3;
      {
        PetscScalar mu=0;
        MatDuplicate(_mu,MAT_COPY_VALUES,&mu3);
        PetscInt Ii,Jj,Istart,Iend=0;
        VecGetOwnershipRange(*_muVec,&Istart,&Iend);
        if (Iend==_Ny*_Nz) {
          Jj = Iend - 2;
          Ii = Iend - 1;
          VecGetValues(*_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Ii,1,&Ii,&mu,ADD_VALUES);
        }
        for (Ii=Istart+1;Ii<Iend;Ii++) {
          VecGetValues(*_muVec,1,&Ii,&mu);
          Jj = Ii - 1;
          MatSetValues(mu3,1,&Jj,1,&Jj,&mu,ADD_VALUES);
        }
        MatAssemblyBegin(mu3,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mu3,MAT_FINAL_ASSEMBLY);
        MatScale(mu3,0.5);
      }

      Mat D3y_Iz;
      kronConvert(D3y,tempMats._Iz,D3y_Iz,6,6);

      Mat C3y_Iz;
      kronConvert(C3y,tempMats._Iz,C3y_Iz,1,1);


      // Rymu = (D3y_Iz^T x C3y_Iz x mu3 x D3y_Iz)/18/dy
      //      + (D4y_Iz^T x C4y_Iz x mu x D4y_Iz)/144/dy
      Mat temp1,temp2;
      //~ ierr = MatTransposeMatMult(D3y_Iz,C3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      Mat D3y_IzT;
      MatTranspose(D3y_Iz,MAT_INITIAL_MATRIX,&D3y_IzT);
      MatMatMult(D3y_IzT,C3y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
      MatDestroy(&D3y_IzT);
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

      //~ ierr = MatTransposeMatMult(D4y_Iz,C4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
      Mat D4y_IzT;
      MatTranspose(D4y_Iz,MAT_INITIAL_MATRIX,&D4y_IzT);
      MatMatMult(D4y_IzT,C4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
      MatDestroy(&D4y_IzT);
      ierr = MatMatMatMult(temp1,_mu,D4y_Iz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Rymu);CHKERRQ(ierr);
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
PetscErrorCode SbpOps_fc::constructD2zmu(const TempMats_fc& tempMats,Mat &D2zmu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif


  // kron(Iy,Dz)
  Mat Iy_Dz = NULL;
  {
    if (_order==2) { kronConvert(tempMats._Iy,tempMats._D1zint,Iy_Dz,2,2); }
    else  { kronConvert(tempMats._Iy,tempMats._D1zint,Iy_Dz,5,5); }
    ierr = PetscObjectSetName((PetscObject) Iy_Dz, "Iy_Dz");CHKERRQ(ierr);
    #if DEBUG > 0
      ierr = checkMatrix(&Iy_Dz,_debugFolder,"Iy_Dz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
  }


  // mu*kron(Iy,Hz)
  Mat muxIy_Hz = NULL;
  {
    Mat Iy_Hz = NULL;
    kronConvert(tempMats._Iy,tempMats._Hz,Iy_Hz,1,1);
    ierr = MatMatMult(_mu,Iy_Hz,MAT_INITIAL_MATRIX,1.0,&muxIy_Hz);CHKERRQ(ierr);
     ierr = PetscObjectSetName((PetscObject) muxIy_Hz, "muxIy_Hz");CHKERRQ(ierr);
    MatDestroy(&Iy_Hz);
    #if DEBUG > 0
      ierr = checkMatrix(&muxIy_Hz,_debugFolder,"muxIy_Hz");CHKERRQ(ierr);
    #endif
    #if VERBOSE > 2
      ierr = MatView(muxIy_Hz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    #endif
  }

  Mat temp1,temp2 = NULL;
  //~ ierr = MatTransposeMatMult(Iy_Dz,muxIy_Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);CHKERRQ(ierr);
  Mat Iy_DzT;
  MatTranspose(Iy_Dz,MAT_INITIAL_MATRIX,&Iy_DzT);
  MatMatMult(Iy_DzT,muxIy_Hz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp1);
  MatDestroy(&Iy_DzT);
  ierr = MatMatMult(temp1,Iy_Dz,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp2);CHKERRQ(ierr);
  ierr = MatScale(temp2,-1);CHKERRQ(ierr);
  MatDestroy(&temp1);
  MatDestroy(&Iy_Dz);
  MatDestroy(&muxIy_Hz);

  Mat Rzmu = NULL;
  ierr = constructRzmu(tempMats,Rzmu);
  ierr = MatAXPY(temp2,-1,Rzmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  #if CALCULATE_ENERGY == 1
    MatDuplicate(Rzmu,MAT_COPY_VALUES,&_Rz);
  #endif
  MatDestroy(&Rzmu);

  ierr = MatAXPY(temp2,1,tempMats._muxIy_BSz,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructD2zmu in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute H matrix (Hy kron Hz)
PetscErrorCode SbpOps_fc::constructH(const TempMats_fc& tempMats)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructH in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  MatDuplicate(tempMats._H,MAT_COPY_VALUES,&_H);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructH in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif
_runTime = MPI_Wtime() - startTime;
  return ierr;
}

// compute Hinv matrix (Hyinv kron Hzinv)
PetscErrorCode SbpOps_fc::constructHinv(const TempMats_fc& tempMats)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructHinv in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  if (_Ny > 1 && _Nz > 1) {
    MatMatMult(tempMats._Hyinv_Iz,tempMats._Iy_Hzinv,MAT_INITIAL_MATRIX,1.5,&_Hinv);
  }
  else if (_Nz == 1) { MatDuplicate(tempMats._Hyinv_Iz,MAT_COPY_VALUES,&_Hinv); }
  else if (_Ny == 1) { MatDuplicate(tempMats._Iy_Hzinv,MAT_COPY_VALUES,&_Hinv); }
  PetscObjectSetName((PetscObject) _Hinv, "Hinv");

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructHinv in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif
_runTime = MPI_Wtime() - startTime;
  return ierr;
}

// compute matrices for 1st derivatives
PetscErrorCode SbpOps_fc::construct1stDerivs(const TempMats_fc& tempMats)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function construct1stDerivs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  //~ PetscPrintf(PETSC_COMM_WORLD,"D1y:\n");
  //~ tempMats._D1y.printPetsc();
  //~ PetscPrintf(PETSC_COMM_WORLD,"\n");
  //~ PetscPrintf(PETSC_COMM_WORLD,"D1z:\n");
  //~ tempMats._D1y.printPetsc();
  //~ PetscPrintf(PETSC_COMM_WORLD,"\n");

  kronConvert(tempMats._D1y,tempMats._Iz,_Dy_Iz,5,5);
  ierr = PetscObjectSetName((PetscObject) _Dy_Iz, "_Dy_Iz");CHKERRQ(ierr);

  // create _Iy_Dz
  kronConvert(tempMats._Iy,tempMats._D1z,_Iy_Dz,5,5);
  ierr = PetscObjectSetName((PetscObject) _Iy_Dz, "_Iy_Dz");CHKERRQ(ierr);

  //~ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //~ierr = MatView(_Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


#if VERBOSE > 2
  ierr = MatView(_Dy_Iz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(_Iy_Dz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function construct1stDerivs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
}

PetscErrorCode SbpOps_fc::updateVarCoeff(const Vec& coeff)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function updateVarCoeff in sbpOps.cpp.\n");
    CHKERRQ(ierr);
  #endif

  ierr = MatDiagonalSet(_mu,coeff,INSERT_VALUES); CHKERRQ(ierr);
  TempMats_fc tempFactors(_order,_Ny,_dy,_Nz,_dz,_mu);

  // this approach fundamentally reallocates all the matrices
  ierr = satBoundaries(tempFactors); CHKERRQ(ierr);
  ierr = constructA(tempFactors); CHKERRQ(ierr);


  _runTime = MPI_Wtime() - startTime;
  #if VERBOSE >1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function updateVarCoeff in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute matrix relating displacement to vector b containing boundary conditions
PetscErrorCode SbpOps_fc::constructA(const TempMats_fc& tempMats)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructA in sbpOps.cpp.\n");
  CHKERRQ(ierr);
#endif

  if (_type.compare("yz")==0) {
    Mat D2ymu = NULL;
    ierr = constructD2ymu(tempMats,D2ymu);CHKERRQ(ierr);

    Mat D2zmu = NULL;
    ierr = constructD2zmu(tempMats,D2zmu);CHKERRQ(ierr);

    // compute A
    ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
    ierr = MatAYPX(_A,1.0,D2zmu,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    // clean up
    ierr = MatDestroy(&D2ymu);CHKERRQ(ierr);
    ierr = MatDestroy(&D2zmu);CHKERRQ(ierr);

    // use new Mats _AL etc
    ierr = MatAXPY(_A,1.0,tempMats._AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,tempMats._AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,tempMats._AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,tempMats._AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_type.compare("y")==0) {
    Mat D2ymu;
    ierr = constructD2ymu(tempMats,D2ymu);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) D2ymu, "D2ymu");CHKERRQ(ierr);

    // compute A
    MatDestroy(&_A);
    ierr = MatDuplicate(D2ymu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);

    ierr = MatDestroy(&D2ymu);CHKERRQ(ierr); // clean up

    // use new Mats _AL etc
    ierr = MatAXPY(_A,1.0,tempMats._AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,tempMats._AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_type.compare("z")==0) {
    Mat D2zmu;
    ierr = constructD2zmu(tempMats,D2zmu);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) D2zmu, "D2zmu");CHKERRQ(ierr);

    // compute A
    MatDestroy(&_A);
    ierr = MatDuplicate(D2zmu,MAT_COPY_VALUES,&_A);CHKERRQ(ierr);
    ierr = MatDestroy(&D2zmu);CHKERRQ(ierr); // clean up

    // add matrices for boundary conditions
    ierr = MatAXPY(_A,1.0,tempMats._AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,tempMats._AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: sbp member 'type' not understood. Choices: 'yz', 'y', 'z'.\n");
    assert(0);
  }

#if DEBUG > 0
  checkMatrix(&_A,_debugFolder,"matA");CHKERRQ(ierr);
#endif




  // if using H A uhat = H rhs
  Mat temp;
  ierr = MatMatMult(tempMats._H,_A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&temp);CHKERRQ(ierr);
  ierr = MatCopy(temp,_A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&temp);CHKERRQ(ierr);
  ierr = MatSetOption(_A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) _A, "_A");CHKERRQ(ierr);


  //~MatView(_A,PETSC_VIEWER_STDOUT_WORLD);

#if VERBOSE > 2
  MatView(_A,PETSC_VIEWER_STDOUT_WORLD);
#endif

  _runTime = MPI_Wtime() - startTime;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructA in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  return 0;
}






PetscErrorCode sbp_fc_Spmat2(const PetscInt N,const PetscScalar scale, Spmat& D2, Spmat& C2)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbp_fc_Spmat2 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  assert(N > 2 || N == 1);

  if (N == 1) {
    D2.eye();
    C2.eye();
    return ierr;
  }

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbp_fc_Spmat2 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  //~_runTime = MPI_Wtime() - startTime;
  return ierr;
}






PetscErrorCode sbp_fc_Spmat4(const PetscInt N,const PetscScalar scale,
                Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbp_fc_Spmat4 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  assert(N > 8 || N == 1);

  if (N==1) {
    D3.eye();
    D4.eye();
    C3.eye();
    C4.eye();
    return ierr;
  }


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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbp_fc_Spmat4 in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode sbp_fc_Spmat(const PetscInt order, const PetscInt N,const PetscScalar scale,Spmat& H,Spmat& Hinv,Spmat& D1, Spmat& D1int, Spmat& BS)
{
PetscErrorCode ierr = 0;
//~double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function sbp_fc_Spmat in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  if (N == 1) {
    H.eye();
    Hinv.eye();
    D1.eye();
    D1int.eye();
    BS.eye();
    return ierr;
  }

PetscInt Ii=0;

switch ( order ) {
    case 2:
    {
      H.eye(); H(0,0,0.5); H(N-1,N-1,0.5); H.scale(1/scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
        H.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
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

      // not fully compatible
      //~BS(0,0,1.5*scale);     BS(0,1,-2.0*scale);     BS(0,2,0.5*scale); // -1* p666 of Mattsson 2010
      //~BS(N-1,N-3,0.5*scale); BS(N-1,N-2,-2.0*scale); BS(N-1,N-1,1.5*scale);

      // fully compatible
      BS(0,0,-D1int(0,0)); BS(0,1,-D1int(0,1));
      BS(N-1,N-2,D1int(N-1,N-2)); BS(N-1,N-1,D1int(N-1,N-1));
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBS:\n");CHKERRQ(ierr);
        BS.printPetsc();
      #endif

      D1 = D1int; // copy D1int's interior
      // last row
      D1(N-1,N-2,BS(N-1,N-2));
      D1(N-1,N-1,BS(N-1,N-1));
      D1(0,0,-BS(0,0)); D1(0,1,-BS(0,1)); // first row
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
        H.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1.0/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
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

      // not fully compatible
      // row 1: -1* p666 of Mattsson 2010
      //~BS(0,0,11.0/6.0*scale); BS(0,1,-3.0*scale); BS(0,2,1.5*scale); BS(0,3,-1.0/3.0*scale);
      //~BS(N-1,N-1,11.0/6.0); BS(N-1,N-2,-3.0); BS(N-1,N-3,1.5); BS(N-1,N-4,-1.0/3.0);

      // fully compatible
      BS(0,0,24.0/17.0*scale); BS(0,1,-59.0/34.0*scale);
      BS(0,2,4.0/17.0*scale); BS(0,3,3.0/34.0*scale);
      BS(N-1,N-1,24.0/17.0*scale); BS(N-1,N-2,-59.0/34.0*scale);
      BS(N-1,N-3,4.0/17.0*scale); BS(N-1,N-4,3.0/34.0*scale);

      #if VERBOBSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nBS:\n");CHKERRQ(ierr);
        BS.printPetsc();
      #endif

     // for simulations with viscoelasticity, need
     // 1st deriv on interior as well
     D1 = D1int;

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function sbp_fc_Spmat in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  //~_runTime = MPI_Wtime() - startTime;
  return ierr;
}




//======================== public member functions =====================

// map the boundary condition vectors to rhs
PetscErrorCode SbpOps_fc::setRhs(Vec&rhs,Vec &bcL,Vec &bcR,Vec &bcT,Vec &bcB)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function setRhs in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  if (_type.compare("yz")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsL,bcL,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcL
    ierr = MatMultAdd(_rhsR,bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
    ierr = MatMultAdd(_rhsT,bcT,rhs,rhs);
    ierr = MatMultAdd(_rhsB,bcB,rhs,rhs);
  }
  else if (_type.compare("y")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsL,bcL,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcL
    ierr = MatMultAdd(_rhsR,bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
  }
  else if (_type.compare("z")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsT,bcT,rhs);CHKERRQ(ierr);
    ierr = MatMultAdd(_rhsB,bcB,rhs,rhs);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: sbp member 'type' not understood. Choices: 'yz', 'y', 'z'.\n");
    assert(0);
  }


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function setRhs in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  _runTime += MPI_Wtime() - startTime;
  return ierr;
}



//======================= I/O functions ================================

PetscErrorCode SbpOps_fc::loadOps(const std::string inputDir)
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

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function loadOps in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
    return ierr;
}


PetscErrorCode SbpOps_fc::writeOps(const std::string outputDir)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeOps in sbpOps_fc.cpp\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();
  PetscViewer    viewer;

  if (_Ny == 1) { return 0;}

  std::string str =  outputDir + "A";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // first derivative operators
  str = outputDir + "Dy_Iz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Dy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Iy_Dz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_Dz,viewer);CHKERRQ(ierr);
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


  // H and Hinv
  str = outputDir + "H";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_H,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Hyinv";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Hyinv_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "Hzinv";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_Hzinv,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "E0y";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_E0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  str = outputDir + "ee0y";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_e0y_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "ENy";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_ENy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  str = outputDir + "eeNy";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_eNy_Iz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "E0z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_E0z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  //~ MatView(_Iy_e0z,PETSC_VIEWER_STDOUT_WORLD);
  //~ assert(0);
  str = outputDir + "ee0z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_e0z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "ENz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_ENz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  str = outputDir + "eeNz";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_Iy_eNz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOps in sbpOps.cpp\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
};




// out = Dy * in
PetscErrorCode SbpOps_fc::Dy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dy";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = MatMult(_Dy_Iz,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};

 // out = mu * Dy * in
PetscErrorCode SbpOps_fc::muxDy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_Dy_Iz,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};

// out = Dy * mu * in
PetscErrorCode SbpOps_fc::Dyxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dyxmu";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_Dy_Iz,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};


// out = Dz * in
PetscErrorCode SbpOps_fc::Dz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dz";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = MatMult(_Iy_Dz,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};


// out = mu * Dz * in
PetscErrorCode SbpOps_fc::muxDz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Dz,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Dz * mu * in
PetscErrorCode SbpOps_fc::Dzxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dzxmu";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Dz,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = H * in
PetscErrorCode SbpOps_fc::H(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "H";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = MatMult(_H,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hinv * in
PetscErrorCode SbpOps_fc::Hinv(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hinv";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = MatMult(_Hinv,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * e0y * in
PetscErrorCode SbpOps_fc::Hyinvxe0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hyinvxe0y";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_e0y_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~ ierr = VecScale(temp1,_alphaDy); CHKERRQ(ierr);
  //~ ierr = VecCopy(temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_fc::HyinvxeNy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxeNy";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_eNy_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~ ierr = VecScale(temp1,_alphaDy); CHKERRQ(ierr);
  //~ ierr = VecCopy(temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * E0y * in
PetscErrorCode SbpOps_fc::HyinvxE0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxE0y";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_E0y_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~ ierr = VecScale(temp1,_alphaDy); CHKERRQ(ierr);
  //~ ierr = VecCopy(temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_fc::HyinvxENy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxENy";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_ENy_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~ ierr = VecScale(temp1,_alphaDy); CHKERRQ(ierr);
  //~ ierr = VecCopy(temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * e0z * in
PetscErrorCode SbpOps_fc::HzinvxE0z(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxE0z";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_E0z,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * eNz * in
PetscErrorCode SbpOps_fc::HzinvxENz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxENz";
  string fileName = "sbpOps_fc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~PetscInt ENzM,ENzN,HzinvM,HzinvN,vecN;
  //~VecGetSize(in,&vecN);
  //~MatGetSize(_Iy_ENz,&ENzM,&ENzN);
  //~MatGetSize(_Iy_Hzinv,&HzinvM,&HzinvN);
  //~PetscPrintf(PETSC_COMM_WORLD,"vecN = %i\n",vecN);
  //~PetscPrintf(PETSC_COMM_WORLD,"ENzM = %i, ENzN = %i\n",ENzM,ENzN);
  //~PetscPrintf(PETSC_COMM_WORLD,"HzinvM = %i, HzinvN = %i\n",HzinvM,HzinvN);
  //~assert(0>1);

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_ENz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}




//=================== functions for struct =============================

TempMats_fc::TempMats_fc(const PetscInt order,const PetscInt Ny,
      const PetscScalar dy,const PetscInt Nz,const PetscScalar dz,Mat& mu)
: _order(order),_Ny(Ny),_Nz(Nz),_dy(dy),_dz(dz),_mu(NULL),
  _Hy(Ny,Ny),_D1y(Ny,Ny),_D1yint(Ny,Ny),_Iy(Ny,Ny),
  _Hz(Nz,Nz),_D1z(Nz,Nz),_D1zint(Nz,Nz),_Iz(Nz,Nz),
  _muxBSy_Iz(NULL),_Hyinv_Iz(NULL),_muxIy_BSz(NULL),_Iy_Hzinv(NULL),
  _AL(NULL),_AR(NULL),_AT(NULL),_AB(NULL),_H(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats_fc::TempMats_fc in sbpOps.cpp.\n");
#endif

  _mu = mu; // shallow copy

  _Iy.eye(); // matrix size is set during colon initialization
  _Iz.eye();


  // going from 1D in y to 2D:
  {
    Spmat Hyinv(_Ny,_Ny),BSy(_Ny,_Ny);
    sbp_fc_Spmat(order,Ny,1.0/dy,_Hy,Hyinv,_D1y,_D1yint,BSy);
    //~ if (Ny > 1) { sbp_fc_Spmat(order,Ny,1.0/dy,_Hy,Hyinv,_D1y,_D1yint,BSy); }
    //~ else { _Hy.eye(); }

    // kron(Hyinv,Iz)
    {
      Spmat Hyinv_Iz(_Ny*_Nz,_Ny*_Nz);
      //~ Hyinv_Iz = kron(Hyinv,_Iz);
      //~ Hyinv_Iz.convert(_Hyinv_Iz,1);
      kronConvert(Hyinv,_Iz,_Hyinv_Iz,1,0);
      PetscObjectSetName((PetscObject) _Hyinv_Iz, "Hyinv_Iz");
    }


    // mu*kron(BSy,Iz)
    {
      Mat temp;
      if (_order==2) { kronConvert(BSy,_Iz,temp,3,3); }
      if (_order==4) { kronConvert(BSy,_Iz,temp,5,5); }
      MatMatMult(_mu,temp,MAT_INITIAL_MATRIX,1.0,&_muxBSy_Iz);
      PetscObjectSetName((PetscObject) _muxBSy_Iz, "muxBSy_Iz");
      MatDestroy(&temp);
    }
  }



  // going from 1D in z to 2D:
  {
    Spmat Hzinv(_Nz,_Nz),BSz(_Nz,_Nz);
    sbp_fc_Spmat(order,Nz,1/dz,_Hz,Hzinv,_D1z,_D1zint,BSz);
    //~ if (Nz > 1) { sbp_fc_Spmat(order,Nz,1/dz,_Hz,Hzinv,_D1z,_D1zint,BSz); }
    //~ else { _Hz.eye(); }

    // kron(Iy,Hzinv)
    {
      kronConvert(_Iy,Hzinv,_Iy_Hzinv,1,0);
      PetscObjectSetName((PetscObject) _Iy_Hzinv, "Iy_Hzinv");
    }

    // mu*kron(Iy,BSz)
    {
      Mat temp;
      if (_order==2) { kronConvert(_Iy,BSz,temp,3,3); }
      if (_order==4) { kronConvert(_Iy,BSz,temp,5,5); }
      MatMatMult(_mu,temp,MAT_INITIAL_MATRIX,1.0,&_muxIy_BSz);
      PetscObjectSetName((PetscObject) _muxIy_BSz, "muxIy_BzSz");
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats_fc::TempMats_fc in sbpOps.cpp.\n");
#endif
}

TempMats_fc::~TempMats_fc()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats_fc::~TempMats_fc in sbpOps.cpp.\n");
#endif

  MatDestroy(&_muxBSy_Iz);
  MatDestroy(&_Hyinv_Iz);

  MatDestroy(&_muxIy_BSz);
  MatDestroy(&_Iy_Hzinv);

  MatDestroy(&_AL);
  MatDestroy(&_AR);
  MatDestroy(&_AT);
  MatDestroy(&_AB);

  MatDestroy(&_H);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats_fc::~TempMats_fc in sbpOps.cpp.\n");
#endif
}


