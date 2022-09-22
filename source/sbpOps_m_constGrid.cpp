#include "sbpOps_m_constGrid.hpp"

#define FILENAME "sbpOps_m_constGrid.cpp"


//======================================================================
// constructor and destructor
//======================================================================
SbpOps_m_constGrid::SbpOps_m_constGrid(const int order,const PetscInt Ny,const PetscInt Nz,const PetscScalar Ly,const PetscScalar Lz,Vec& muVec)
: _order(order),_Ny(Ny),_Nz(Nz),_dy(Ly/(Ny-1.)),_dz(Lz/(Nz-1.)),
  _bcRType("unspecified"),_bcTType("unspecified"),
  _bcLType("unspecified"),_bcBType("unspecified"),
  _runTime(0),_compatibilityType("fullyCompatible"),_D2type("yz"),
  _multByH(0),_deleteMats(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in SbpOps_m_constGrid.cpp.\n");
#endif

  // ensure this is in an acceptable state
  setMatsToNull();
  assert(order == 2 || order == 4);
  assert(Ny > 0); assert(Nz > 0);
  assert(Ly > 0); assert(Lz > 0);
  if (Ny == 1) { _dy = Ly; }
  if (Nz == 1) { _dz = Lz; }
  assert(muVec != NULL);
  VecDuplicate(muVec, &_muVec);
  VecCopy(muVec, _muVec);

  // penalty weights
  _alphaT = -1.0; // von Neumann
  _beta= 1.0; // 1 part of Dirichlet
  if (_order == 2) {
    _alphaDy = -4.0/_dy;
    _alphaDz = -4.0/_dz;
    _h11y = 0.5 * _dy;
    _h11z = 0.5 * _dz;
  }
  else if (_order == 4) {
    _alphaDy = 2.0*-48.0/17.0 /_dy;
    _alphaDz = 2.0*-48.0/17.0 /_dz;
    _h11y = 17.0/48.0 * _dy;
    _h11z = 17.0/48.0 * _dz;
  }

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in SbpOps_m_constGrid.cpp.\n");
#endif
}


SbpOps_m_constGrid::~SbpOps_m_constGrid()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in sbpOps_fc.cpp.\n");
  #endif

  VecDestroy(&_muVec);
  MatDestroy(&_mu);

  MatDestroy(&_AR_N); MatDestroy(&_AT_N); MatDestroy(&_AL_N); MatDestroy(&_AB_N);
  MatDestroy(&_rhsL_N); MatDestroy(&_rhsR_N); MatDestroy(&_rhsT_N); MatDestroy(&_rhsB_N);
  MatDestroy(&_AR_D); MatDestroy(&_AT_D); MatDestroy(&_AL_D); MatDestroy(&_AB_D);
  MatDestroy(&_rhsL_D); MatDestroy(&_rhsR_D); MatDestroy(&_rhsT_D); MatDestroy(&_rhsB_D);

  MatDestroy(&_A);
  MatDestroy(&_D2);
  MatDestroy(&_Dy_Iz);
  MatDestroy(&_Iy_Dz);
  MatDestroy(&_Hinv); MatDestroy(&_H);
  MatDestroy(&_Hyinv_Iz); MatDestroy(&_Iy_Hzinv);
  MatDestroy(&_Hy_Iz); MatDestroy(&_Iy_Hz);
  MatDestroy(&_e0y_Iz); MatDestroy(&_eNy_Iz); MatDestroy(&_Iy_e0z); MatDestroy(&_Iy_eNz);
  MatDestroy(&_E0y_Iz); MatDestroy(&_ENy_Iz); MatDestroy(&_Iy_E0z); MatDestroy(&_Iy_ENz);
  MatDestroy(&_muxBySy_IzT); MatDestroy(&_Iy_muxBzSzT);
  MatDestroy(&_BSy_Iz); MatDestroy(&_Iy_BSz);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps_fc.cpp.\n");
  #endif
}


//======================================================================
// functions for setting options for class
//======================================================================

PetscErrorCode SbpOps_m_constGrid::setMatsToNull()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::setMatsToNull";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _mu = NULL;

  _AR = NULL; _AT = NULL; _AL = NULL; _AB = NULL;
  _rhsL = NULL; _rhsR = NULL; _rhsT = NULL; _rhsB = NULL;
  _AR_N = NULL; _AT_N = NULL; _AL_N = NULL; _AB_N = NULL;
  _rhsL_N = NULL; _rhsR_N = NULL; _rhsT_N = NULL; _rhsB_N = NULL;
  _AR_D = NULL; _AT_D = NULL; _AL_D = NULL; _AB_D = NULL;
  _rhsL_D = NULL; _rhsR_D = NULL; _rhsT_D = NULL; _rhsB_D = NULL;

  _A = NULL;
  _Dy_Iz = NULL; _Iy_Dz = NULL;
  _D2 = NULL;
  _Hinv = NULL; _H = NULL; _Hyinv_Iz = NULL; _Iy_Hzinv = NULL; _Hy_Iz = NULL; _Iy_Hz = NULL;
  _e0y_Iz = NULL; _eNy_Iz = NULL; _Iy_e0z = NULL; _Iy_eNz = NULL;
  _E0y_Iz = NULL; _ENy_Iz = NULL; _Iy_E0z = NULL; _Iy_ENz = NULL;
  _BSy_Iz = NULL; _Iy_BSz = NULL;
  _muxBySy_IzT = NULL; _Iy_muxBzSzT = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode SbpOps_m_constGrid::setBCTypes(std::string bcR, std::string bcT, std::string bcL, std::string bcB)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::setBCTypes";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // check that each string is a valid option
  assert(bcR.compare("Dirichlet") == 0 || bcR.compare("Neumann") == 0 );
  assert(bcT.compare("Dirichlet") == 0 || bcT.compare("Neumann") == 0 );
  assert(bcL.compare("Dirichlet") == 0 || bcL.compare("Neumann") == 0 );
  assert(bcB.compare("Dirichlet") == 0 || bcB.compare("Neumann") == 0 );

  _bcRType = bcR;
  _bcTType = bcT;
  _bcLType = bcL;
  _bcBType = bcB;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// not used for this type of SBP operator
PetscErrorCode SbpOps_m_constGrid::setGrid(Vec* y, Vec* z) { return 0; }

PetscErrorCode SbpOps_m_constGrid::setMultiplyByH(const int multByH)
{
  assert( multByH == 1 || multByH == 0 );
  _multByH = multByH;
  return 0;
}

PetscErrorCode SbpOps_m_constGrid::setLaplaceType(const std::string type)
{
  assert(_D2type.compare("yz") == 0 || _D2type.compare("y") == 0 || _D2type.compare("z") == 0 );
  _D2type = type;
  return 0;
}

PetscErrorCode SbpOps_m_constGrid::setCompatibilityType(const string type)
{
  _compatibilityType = type;
  assert(_compatibilityType.compare("fullyCompatible") == 0 || _compatibilityType.compare("compatible") == 0 );
  return 0;
}

PetscErrorCode SbpOps_m_constGrid::changeBCTypes(std::string bcR, std::string bcT, std::string bcL, std::string bcB)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::setBCTypes";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // check that each string is a valid option
  assert(bcR.compare("Dirichlet") == 0 || bcR.compare("Neumann") == 0 );
  assert(bcT.compare("Dirichlet") == 0 || bcT.compare("Neumann") == 0 );
  assert(bcL.compare("Dirichlet") == 0 || bcL.compare("Neumann") == 0 );
  assert(bcB.compare("Dirichlet") == 0 || bcB.compare("Neumann") == 0 );

  _bcRType = bcR;
  _bcTType = bcT;
  _bcLType = bcL;
  _bcBType = bcB;

  constructBCMats();
  updateA_BCs();

  if (_deleteMats) { deleteIntermediateFields(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode SbpOps_m_constGrid::setDeleteIntermediateFields(const int deleteMats)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::setDeleteIntermediateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(deleteMats == 0 || deleteMats == 1);
  _deleteMats = deleteMats;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::deleteIntermediateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::deleteIntermediateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if ( _bcRType.compare("Dirichlet") == 0 ) { MatDestroy(&_AR_N); MatDestroy(&_rhsR_N); }
  else if ( _bcRType.compare("Neumann") == 0 ) { MatDestroy(&_AR_D); MatDestroy(&_rhsR_D); }

  if ( _bcTType.compare("Dirichlet") == 0 ) { MatDestroy(&_AT_N); MatDestroy(&_rhsT_N); }
  else if ( _bcTType.compare("Neumann") == 0 ) { MatDestroy(&_AT_D); MatDestroy(&_rhsT_D); }

  if ( _bcLType.compare("Dirichlet") == 0 ) { MatDestroy(&_AL_N); MatDestroy(&_rhsL_N); }
  else if ( _bcLType.compare("Neumann") == 0 ) { MatDestroy(&_AL_D); MatDestroy(&_rhsL_D); }

  if ( _bcBType.compare("Dirichlet") == 0 ) { MatDestroy(&_AB_N); MatDestroy(&_rhsB_N); }
  else if ( _bcBType.compare("Neumann") == 0 ) { MatDestroy(&_AB_D); MatDestroy(&_rhsB_D); }

  MatDestroy(&_D2);

  //~ MatDestroy(&_Hinv); MatDestroy(&_H);
  //~ MatDestroy(&_Hyinv_Iz); MatDestroy(&_Iy_Hzinv);
  MatDestroy(&_Hy_Iz); MatDestroy(&_Iy_Hz);
  //~ MatDestroy(&_e0y_Iz); MatDestroy(&_eNy_Iz); MatDestroy(&_Iy_e0z); MatDestroy(&_Iy_eNz);
  //~ MatDestroy(&_E0y_Iz); MatDestroy(&_ENy_Iz); MatDestroy(&_Iy_E0z); MatDestroy(&_Iy_ENz);
  //~ MatDestroy(&_muxBySy_IzT); MatDestroy(&_Iy_muxBzSzT);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// functions for computing matrices
//======================================================================

// matrices not constructed until now
PetscErrorCode SbpOps_m_constGrid::computeMatrices()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::computeMatrices";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  TempMats_m_constGrid tempMats(_order,_Ny,_dy,_Nz,_dz,_compatibilityType);

  constructMu(_muVec);
  constructEs(tempMats); // ok
  constructes(tempMats); // ok
  constructHs(tempMats);//ok
  constructBs(tempMats);

  construct1stDerivs(tempMats);
  constructBCMats();
  constructA(tempMats);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::constructMu(Vec& muVec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructMu";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // construct matrix mu
  MatCreate(PETSC_COMM_WORLD,&_mu);
  MatSetSizes(_mu,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);
  MatSetFromOptions(_mu);
  MatMPIAIJSetPreallocation(_mu,1,NULL,1,NULL);
  MatSeqAIJSetPreallocation(_mu,1,NULL);
  MatSetUp(_mu);
  MatDiagonalSet(_mu,muVec,INSERT_VALUES);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode SbpOps_m_constGrid::constructEs(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructEs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Spmat E0y(_Ny,_Ny);
  if (_Ny > 1) { E0y(0,0,1.0); }
  kronConvert(E0y,tempMats._Iz,_E0y_Iz,1,1);

  Spmat ENy(_Ny,_Ny);
  if (_Ny > 1) { ENy(_Ny-1,_Ny-1,1.0); }
  kronConvert(ENy,tempMats._Iz,_ENy_Iz,1,1);

  Spmat E0z(_Nz,_Nz);
  if (_Nz > 1) { E0z(0,0,1.0); }
  kronConvert(tempMats._Iy,E0z,_Iy_E0z,1,1);

  Spmat ENz(_Nz,_Nz);
  if (_Nz > 1) { ENz(_Nz-1,_Nz-1,1.0); }
  kronConvert(tempMats._Iy,ENz,_Iy_ENz,1,1);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::constructes(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructes";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Spmat e0y(_Ny,1);
  if (_Ny > 1) { e0y(0,0,1.0); }
  kronConvert(e0y,tempMats._Iz,_e0y_Iz,1,1);

  Spmat eNy(_Ny,1);
  if (_Ny > 1) { eNy(_Ny-1,0,1.0); }
  kronConvert(eNy,tempMats._Iz,_eNy_Iz,1,1);

  Spmat e0z(_Nz,1);
  if (_Nz > 1) { e0z(0,0,1.0); }
  kronConvert(tempMats._Iy,e0z,_Iy_e0z,1,1);

  Spmat eNz(_Nz,1);
  if (_Nz > 1) { eNz(_Nz-1,0,1.0); }
  kronConvert(tempMats._Iy,eNz,_Iy_eNz,1,1);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::constructBs(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_order==2 && _BSy_Iz == NULL) { kronConvert(tempMats._BSy,tempMats._Iz,_BSy_Iz,3,0); }
  if (_order==4 && _BSy_Iz == NULL) { kronConvert(tempMats._BSy,tempMats._Iz,_BSy_Iz,5,0); }
  if (_muxBySy_IzT == NULL) { MatTransposeMatMult(_BSy_Iz,_mu,MAT_INITIAL_MATRIX,1.,&_muxBySy_IzT); }
  else{ MatTransposeMatMult(_BSy_Iz,_mu,MAT_REUSE_MATRIX,1.,&_muxBySy_IzT); }

  if (_order==2 && _Iy_BSz == NULL) { kronConvert(tempMats._Iy,tempMats._BSz,_Iy_BSz,3,0); }
  if (_order==4 && _Iy_BSz == NULL) { kronConvert(tempMats._Iy,tempMats._BSz,_Iy_BSz,5,0); }
  if (_Iy_muxBzSzT == NULL) { MatTransposeMatMult(_Iy_BSz,_mu,MAT_INITIAL_MATRIX,1.,&_Iy_muxBzSzT); }
  else{ MatTransposeMatMult(_Iy_BSz,_mu,MAT_REUSE_MATRIX,1.,&_Iy_muxBzSzT); }

  if (_deleteMats) {
    MatDestroy(&_BSy_Iz);
    MatDestroy(&_Iy_BSz);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::constructHs(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructHs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // H, Hy, and Hz
  kronConvert(tempMats._Hy,tempMats._Iz,_Hy_Iz,1,0);
  PetscObjectSetName((PetscObject) _Hy_Iz, "Hy_Iz");
  kronConvert(tempMats._Iy,tempMats._Hz,_Iy_Hz,1,0);
  PetscObjectSetName((PetscObject) _Iy_Hz, "Iy_Hz");
  ierr = MatMatMult(_Hy_Iz,_Iy_Hz,MAT_INITIAL_MATRIX,1.,&_H);
  PetscObjectSetName((PetscObject) _H, "H");

  // Hinv, and Hinvy and Hinvz
  kronConvert(tempMats._Hyinv,tempMats._Iz,_Hyinv_Iz,1,0);
  PetscObjectSetName((PetscObject) _Hyinv_Iz, "Hyinv_Iz");
  kronConvert(tempMats._Iy,tempMats._Hzinv,_Iy_Hzinv,1,0);
  PetscObjectSetName((PetscObject) _Iy_Hzinv, "Iy_Hzinv");
  ierr = MatMatMult(_Hyinv_Iz,_Iy_Hzinv,MAT_INITIAL_MATRIX,1.,&_Hinv);
  PetscObjectSetName((PetscObject) _Hinv, "Hinv");

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// computes SAT term for von Neumann BC to be added to u
// out = alphaT * Bfact * Hinv* E * mu * D1
// out = alphaT * Bfact * H * Hinv* E * mu * D1
// scall = MAT_INITIAL_MATRIX, or MAT_REUSE_MATRIX
// Bfact = -1 or 1, instead of passing in matrix B
PetscErrorCode SbpOps_m_constGrid::constructBC_Neumann(Mat& out, Mat& Hinv, PetscScalar Bfact, Mat& E, Mat& mu, Mat& D1,MatReuse scall)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBC_Neumann(Mat& out, Mat& Hinv, Mat& B, Mat& E, Mat& mu, Mat& D1)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // temporary matrix using only diagonal matrices, should be efficient to generate and destroy
  Mat HinvxExmu;
  ierr = MatMatMatMult(Hinv,E,mu,MAT_INITIAL_MATRIX,1.,&HinvxExmu); CHKERRQ(ierr);

  if (!_multByH) { // if do not multiply by H
    ierr = MatMatMult(HinvxExmu,D1,scall,PETSC_DECIDE,&out); CHKERRQ(ierr);
  }
  else {
    // if do multiply by H
    ierr = MatMatMatMult(_H,HinvxExmu,D1,scall,PETSC_DECIDE,&out); CHKERRQ(ierr);
  }
  MatDestroy(&HinvxExmu);

  PetscScalar a = Bfact * _alphaT;
  MatScale(out,a);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// computes SAT term for von Neumann BC for construction of rhs
// out = alphaT * Bfact * Hinv* e
// out = alphaT * Bfact * H * Hinv* e
// scall = MAT_INITIAL_MATRIX, or MAT_REUSE_MATRIX
// Bfact = -1 or 1, instead of passing in matrix B
PetscErrorCode SbpOps_m_constGrid::constructBC_Neumann(Mat& out, Mat& Hinv, PetscScalar Bfact, Mat& e, MatReuse scall)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBC_Neumann(Mat& out, Mat& Hinv, Mat& B, Mat& E, Mat& mu, Mat& D1)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (!_multByH) { // if do not multiply by H
    ierr = MatMatMult(Hinv,e,scall,1.,&out); CHKERRQ(ierr);
  }
  else {
    // if do multiply by H
    ierr = MatMatMatMult(_H,Hinv,e,scall,1.,&out); CHKERRQ(ierr);
  }

  PetscScalar a = Bfact * _alphaT;
  MatScale(out,a);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// computes SAT term for Dirichlet BC
// out =     Hinv * (alphaD * mu + BD1T) * E
// out = H * Hinv * (alphaD * mu + BD1T) * E
// scall = MAT_INITIAL_MATRIX, or MAT_REUSE_MATRIX
// Bfact = -1 or 1, instead of passing in matrix B
PetscErrorCode SbpOps_m_constGrid::constructBC_Dirichlet(Mat& out,PetscScalar alphaD,Mat& mu,Mat& Hinv,Mat& BD1T,Mat& E,MatReuse scall)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBC_Dirichlet";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Mat HxHinv;
  if (!_multByH) { // if do not multiply by H
    HxHinv = Hinv;
  }
  else {
    ierr = MatMatMult(_H,Hinv,MAT_INITIAL_MATRIX,1.,&HxHinv); CHKERRQ(ierr);
  }

  Mat HinvxmuxE;
  ierr = MatMatMatMult(HxHinv,mu,E,MAT_INITIAL_MATRIX,1.,&HinvxmuxE); CHKERRQ(ierr);

  ierr = MatMatMatMult(HxHinv,BD1T,E,scall,PETSC_DECIDE,&out); CHKERRQ(ierr);
  ierr = MatAXPY(out,alphaD,HinvxmuxE,SUBSET_NONZERO_PATTERN);


  if (_multByH) { MatDestroy(&HxHinv); }

  MatDestroy(&HinvxmuxE);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::constructBCMats()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBCMats";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_bcRType.compare("Dirichlet")==0) {
    if (_AR_D == NULL) { constructBC_Dirichlet(_AR_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_ENy_Iz,MAT_INITIAL_MATRIX); }
    if (_rhsR_D == NULL) { constructBC_Dirichlet(_rhsR_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_eNy_Iz,MAT_INITIAL_MATRIX); }
    _AR = _AR_D;
    _rhsR = _rhsR_D;
  }
  else if (_bcRType.compare("Neumann")==0) {
    if (_AR_N == NULL) { constructBC_Neumann(_AR_N,_Hyinv_Iz, 1.,_ENy_Iz,_mu,_Dy_Iz,MAT_INITIAL_MATRIX); }
    if (_rhsR_N == NULL) { constructBC_Neumann(_rhsR_N,_Hyinv_Iz, 1.,_eNy_Iz,MAT_INITIAL_MATRIX); }
    _AR = _AR_N;
    _rhsR = _rhsR_N;
  }

  if (_bcTType.compare("Dirichlet")==0) {
    if (_AT_D == NULL) { constructBC_Dirichlet(_AT_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_E0z,MAT_INITIAL_MATRIX); }
    if (_rhsT_D == NULL) { constructBC_Dirichlet(_rhsT_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_e0z,MAT_INITIAL_MATRIX); }
    _AT = _AT_D;
    _rhsT = _rhsT_D;
  }
  else if (_bcTType.compare("Neumann")==0) {
    if (_AT_N == NULL) { constructBC_Neumann(_AT_N,_Iy_Hzinv, -1.,_Iy_E0z,_mu,_Iy_Dz,MAT_INITIAL_MATRIX); }
    if (_rhsT_N == NULL) { constructBC_Neumann(_rhsT_N,_Iy_Hzinv, -1.,_Iy_e0z,MAT_INITIAL_MATRIX); }
    _AT = _AT_N;
    _rhsT = _rhsT_N;
  }


  if (_bcLType.compare("Dirichlet")==0) {
    if (_AL_D == NULL) { constructBC_Dirichlet(_AL_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_E0y_Iz,MAT_INITIAL_MATRIX); }
    if (_rhsL_D == NULL) { constructBC_Dirichlet(_rhsL_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_e0y_Iz,MAT_INITIAL_MATRIX); }
    _AL = _AL_D;
    _rhsL = _rhsL_D;
  }
  else if (_bcLType.compare("Neumann")==0) {
    if (_AL_N == NULL) { constructBC_Neumann(_AL_N, _Hyinv_Iz, -1., _E0y_Iz, _mu, _Dy_Iz,MAT_INITIAL_MATRIX); }
    if (_rhsL_N == NULL) { constructBC_Neumann(_rhsL_N, _Hyinv_Iz, -1., _e0y_Iz,MAT_INITIAL_MATRIX); }
    _AL = _AL_N;
    _rhsL = _rhsL_N;
  }


  if (_bcBType.compare("Dirichlet")==0) {
    if (_AB_D == NULL) { constructBC_Dirichlet(_AB_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_ENz,MAT_INITIAL_MATRIX); }
    if (_rhsB_D == NULL) { constructBC_Dirichlet(_rhsB_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_eNz,MAT_INITIAL_MATRIX); }
    _AB = _AB_D;
    _rhsB = _rhsB_D;
  }
  else if (_bcBType.compare("Neumann")==0) {
    if (_AB_N == NULL) { constructBC_Neumann(_AB_N,_Iy_Hzinv, 1.,_Iy_ENz,_mu,_Iy_Dz,MAT_INITIAL_MATRIX); }
    if (_rhsB_N == NULL) { constructBC_Neumann(_rhsB_N,_Iy_Hzinv, 1.,_Iy_eNz,MAT_INITIAL_MATRIX); }
    _AB = _AB_N;
    _rhsB = _rhsB_N;
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// construct // D2 = d/dy(mu d/dy) + d/dz(mu d/dz)
PetscErrorCode SbpOps_m_constGrid::constructD2(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructA";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
    #if VERBOSE > 2
    MatView(_A,PETSC_VIEWER_STDOUT_WORLD);
  #endif

  Mat Dyymu = NULL;
  Mat Dzzmu = NULL;

  if (_D2type.compare("yz")==0) { // D2 = d/dy(mu d/dy) + d/dz(mu d/dz)
    ierr = constructDyymu(tempMats,Dyymu); CHKERRQ(ierr);
    ierr = constructDzzmu(tempMats,Dzzmu); CHKERRQ(ierr);
    ierr = MatDuplicate(Dyymu,MAT_COPY_VALUES,&_D2); CHKERRQ(ierr);
    ierr = MatAYPX(_D2,1.0,Dzzmu,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  }
  else if (_D2type.compare("y")==0) { // D2 = d/dy(mu d/dy)
    ierr = constructDyymu(tempMats,Dyymu); CHKERRQ(ierr);
    ierr = MatDuplicate(Dyymu,MAT_COPY_VALUES,&_D2); CHKERRQ(ierr);
  }
  else if (_D2type.compare("z")==0) { // D2 = d/dz(mu d/dz)
    ierr = constructDzzmu(tempMats,Dzzmu); CHKERRQ(ierr);
    ierr = MatDuplicate(Dzzmu,MAT_COPY_VALUES,&_D2); CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: sbp member 'type' not understood. Choices: 'yz', 'y', 'z'.\n");
    assert(0);
  }

  // clean up
  ierr = MatDestroy(&Dyymu);CHKERRQ(ierr);
  ierr = MatDestroy(&Dzzmu);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  _runTime = MPI_Wtime() - startTime;
  return 0;
}

// assumes A has not been computed before
PetscErrorCode SbpOps_m_constGrid::constructA(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructA";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_D2 == NULL) { constructD2(tempMats); }
  MatDuplicate(_D2,MAT_COPY_VALUES,&_A);

  if (_deleteMats) { MatDestroy(&_D2); }

  // add SAT boundary condition terms
  constructBCMats();

  if (_D2type.compare("yz")==0) {
    // use new Mats _AL etc
    ierr = MatAXPY(_A,1.0,_AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("y")==0) {
    ierr = MatAXPY(_A,1.0,_AL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("z")==0) {

    ierr = MatAXPY(_A,1.0,_AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);


  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning in SbpOps: D2type of %s not understood. Choices: 'yz', 'y', 'z'.\n",_D2type.c_str());
    assert(0);
  }
  ierr = PetscObjectSetName((PetscObject) _A, "_A");CHKERRQ(ierr);

  #if VERBOSE > 2
    MatView(_A,PETSC_VIEWER_STDOUT_WORLD);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  _runTime = MPI_Wtime() - startTime;
  return 0;
}

// update A based on new BCs
PetscErrorCode SbpOps_m_constGrid::updateA_BCs()
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
  #if VERBOSE > 1
    string funcName = "SbpOps_m_varGridSpacing::updateA_BCs";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_D2 == NULL) {
    TempMats_m_constGrid tempMats(_order,_Ny,_dy,_Nz,_dz,_compatibilityType);
    constructD2(tempMats);
  }

  MatZeroEntries(_A);
  MatCopy(_D2,_A,SAME_NONZERO_PATTERN);

  if (_deleteMats) { MatDestroy(&_D2); }

  // add SAT boundary condition terms
  constructBCMats();

  if (_D2type.compare("yz")==0) {
    // use new Mats _AL etc
    ierr = MatAXPY(_A,1.0,_AL,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AT,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("y")==0) {
    ierr = MatAXPY(_A,1.0,_AL,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("z")==0) {
    ierr = MatAXPY(_A,1.0,_AT,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning in SbpOps: D2type of %s not understood. Choices: 'yz', 'y', 'z'.\n",_D2type.c_str());
    assert(0);
  }

  #if VERBOSE > 2
    MatView(_A,PETSC_VIEWER_STDOUT_WORLD);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  _runTime = MPI_Wtime() - startTime;
  return 0;
}

// update A based on new BCs
PetscErrorCode SbpOps_m_constGrid::updateA_BCs(TempMats_m_constGrid& tempMats)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::updateA_BCs(TempMats_m_constGrid& tempMats)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update D2 component of A
  if (_D2 == NULL) { constructD2(tempMats); }
  MatZeroEntries(_A);
  MatCopy(_D2,_A,SAME_NONZERO_PATTERN);
  if (_deleteMats) { MatDestroy(&_D2); }

  // add SAT boundary condition terms
  constructBCMats();
  if (_D2type.compare("yz")==0) {
    ierr = MatAXPY(_A,1.0,_AL,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AT,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("y")==0) {
    ierr = MatAXPY(_A,1.0,_AL,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AR,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else if (_D2type.compare("z")==0) {
    ierr = MatAXPY(_A,1.0,_AT,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(_A,1.0,_AB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning in SbpOps: D2type of %s not understood. Choices: 'yz', 'y', 'z'.\n",_D2type.c_str());
    assert(0);
  }

  #if VERBOSE > 2
    MatView(_A,PETSC_VIEWER_STDOUT_WORLD);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  _runTime = MPI_Wtime() - startTime;
  return 0;
}

// update SAT matrices for boundary conditions if the variable coefficient has changed
PetscErrorCode SbpOps_m_constGrid::updateBCMats()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SbpOps_m_constGrid::constructBCMats";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_bcRType.compare("Dirichlet")==0) {
    constructBC_Dirichlet(_AR_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_ENy_Iz,MAT_REUSE_MATRIX);
    constructBC_Dirichlet(_rhsR_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_eNy_Iz,MAT_REUSE_MATRIX);
    _AR = _AR_D; _rhsR = _rhsR_D;
    MatDestroy(&_AR_N); MatDestroy(&_rhsR_N);
  }
  else if (_bcRType.compare("Neumann")==0) {
    constructBC_Neumann(_AR_N,_Hyinv_Iz, 1.,_ENy_Iz,_mu,_Dy_Iz,MAT_REUSE_MATRIX);
    constructBC_Neumann(_rhsR_N,_Hyinv_Iz, 1.,_eNy_Iz,MAT_REUSE_MATRIX);
    _AR = _AR_N; _rhsR = _rhsR_N;
    MatDestroy(&_AR_D); MatDestroy(&_rhsR_D);
  }

  if (_bcTType.compare("Dirichlet")==0) {
    constructBC_Dirichlet(_AT_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_E0z,MAT_REUSE_MATRIX);
    constructBC_Dirichlet(_rhsT_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_e0z,MAT_REUSE_MATRIX);
    _AT = _AT_D; _rhsT = _rhsT_D;
    MatDestroy(&_AR_D); MatDestroy(&_rhsR_D);
  }
  else if (_bcTType.compare("Neumann")==0) {
    constructBC_Neumann(_AT_N,_Iy_Hzinv, -1.,_Iy_E0z,_mu,_Iy_Dz,MAT_REUSE_MATRIX);
    constructBC_Neumann(_rhsT_N,_Iy_Hzinv, -1.,_Iy_e0z,MAT_REUSE_MATRIX);
    _AT = _AT_N; _rhsT = _rhsT_N;
    MatDestroy(&_AT_D); MatDestroy(&_rhsT_D);
  }


  if (_bcLType.compare("Dirichlet")==0) {
    constructBC_Dirichlet(_AL_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_E0y_Iz,MAT_REUSE_MATRIX);
    constructBC_Dirichlet(_rhsL_D,_alphaDy,_mu,_Hyinv_Iz,_muxBySy_IzT,_e0y_Iz,MAT_REUSE_MATRIX);
    _AL = _AL_D; _rhsL = _rhsL_D;
    MatDestroy(&_AL_N); MatDestroy(&_rhsL_N);
  }
  else if (_bcLType.compare("Neumann")==0) {
    constructBC_Neumann(_AL_N, _Hyinv_Iz, -1., _E0y_Iz, _mu, _Dy_Iz,MAT_REUSE_MATRIX);
    constructBC_Neumann(_rhsL_N, _Hyinv_Iz, -1., _e0y_Iz,MAT_REUSE_MATRIX);
    _AL = _AL_N; _rhsL = _rhsL_N;
    MatDestroy(&_AL_D); MatDestroy(&_rhsL_D);
  }


  if (_bcBType.compare("Dirichlet")==0) {
    constructBC_Dirichlet(_AB_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_ENz,MAT_REUSE_MATRIX);
    constructBC_Dirichlet(_rhsB_D,_alphaDz,_mu,_Iy_Hzinv,_Iy_muxBzSzT,_Iy_eNz,MAT_REUSE_MATRIX);
    _AB = _AB_D; _rhsB = _rhsB_D;
    MatDestroy(&_AB_N); MatDestroy(&_rhsB_N);
  }
  else if (_bcBType.compare("Neumann")==0) {
    constructBC_Neumann(_AB_N,_Iy_Hzinv, 1.,_Iy_ENz,_mu,_Iy_Dz,MAT_REUSE_MATRIX);
    constructBC_Neumann(_rhsB_N,_Iy_Hzinv, 1.,_Iy_eNz,MAT_REUSE_MATRIX);
    _AB = _AB_N; _rhsB = _rhsB_N;
    MatDestroy(&_AB_D); MatDestroy(&_rhsB_D);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// functions to allow user access to various matrices
//======================================================================

// map the boundary condition vectors to rhs
PetscErrorCode SbpOps_m_constGrid::setRhs(Vec&rhs,Vec &bcL,Vec &bcR,Vec &bcT,Vec &bcB)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function setRhs in %s.\n", FILENAME);CHKERRQ(ierr);
#endif

  if (_D2type.compare("yz")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsL,bcL,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcL
    ierr = MatMultAdd(_rhsR,bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
    ierr = MatMultAdd(_rhsT,bcT,rhs,rhs);
    ierr = MatMultAdd(_rhsB,bcB,rhs,rhs);
  }
  else if (_D2type.compare("y")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsL,bcL,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcL
    ierr = MatMultAdd(_rhsR,bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
  }
  else if (_D2type.compare("z")==0) {
    ierr = VecSet(rhs,0.0);
    ierr = MatMult(_rhsT,bcT,rhs);CHKERRQ(ierr);
    ierr = MatMultAdd(_rhsB,bcB,rhs,rhs);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning in SbpOps: D2type of %s not understood. Choices: 'yz', 'y', 'z'.\n",_D2type.c_str());
    assert(0);
  }

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function setRhs in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  _runTime += MPI_Wtime() - startTime;
  return ierr;
}

PetscErrorCode SbpOps_m_constGrid::geth11(PetscScalar &h11y, PetscScalar &h11z) { h11y = _h11y; h11z = _h11z; return 0; }

PetscErrorCode SbpOps_m_constGrid::getA(Mat &mat) { mat = _A; return 0; }

PetscErrorCode SbpOps_m_constGrid::getH(Mat &mat) { mat = _H; return 0; }

PetscErrorCode SbpOps_m_constGrid::getDs(Mat &Dy,Mat &Dz) { Dy = _Dy_Iz; Dz = _Iy_Dz; return 0; }

PetscErrorCode SbpOps_m_constGrid::getMus(Mat &mu,Mat &muqy,Mat &murz) { mu = _mu; muqy = _mu; murz = _mu; return 0; }

PetscErrorCode SbpOps_m_constGrid::getEs(Mat& E0y_Iz,Mat& ENy_Iz,Mat& Iy_E0z,Mat& Iy_ENz)
{
  E0y_Iz = _E0y_Iz;
  ENy_Iz = _ENy_Iz;
  Iy_E0z = _Iy_E0z;
  Iy_ENz = _Iy_ENz;
  return 0;
}

PetscErrorCode SbpOps_m_constGrid::getes(Mat& e0y_Iz,Mat& eNy_Iz,Mat& Iy_e0z,Mat& Iy_eNz)
{
  e0y_Iz = _e0y_Iz;
  eNy_Iz = _eNy_Iz;
  Iy_e0z = _Iy_e0z;
  Iy_eNz = _Iy_eNz;
  return 0;
}

PetscErrorCode SbpOps_m_constGrid::getHs(Mat& Hy_Iz,Mat& Iy_Hz)
{
  Hy_Iz = _Hy_Iz;
  Iy_Hz = _Iy_Hz;
  return 0;
}
PetscErrorCode SbpOps_m_constGrid::getHinvs(Mat& Hyinv_Iz,Mat& Iy_Hzinv)
{
  Hyinv_Iz = _Hyinv_Iz;
  Iy_Hzinv = _Iy_Hzinv;
  return 0;
}
PetscErrorCode SbpOps_m_constGrid::getCoordTrans(Mat&J, Mat& Jinv,Mat& qy,Mat& rz, Mat& yq, Mat& zr) { assert(0); return 0; }

PetscErrorCode SbpOps_m_constGrid::getBCTypes(std::string& bcRType,std::string& bcTType,std::string& bcLType,std::string& bcBType)
{
  bcRType = _bcRType;
  bcTType = _bcTType;
  bcLType = _bcLType;
  bcBType = _bcBType;
  return 0;
}

//======================================================================


// compute Dyymu using my class Spmat
PetscErrorCode SbpOps_m_constGrid::constructDyymu(const TempMats_m_constGrid& tempMats, Mat &Dyymu)
{
  PetscErrorCode  ierr = 0;
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructDyymu in sbpOps_fc.cpp.\n");CHKERRQ(ierr);
#endif

  Mat Rymu,HinvxRymu;
  ierr = constructRymu(tempMats,Rymu); CHKERRQ(ierr);
  ierr = MatMatMult(_Hyinv_Iz,Rymu,MAT_INITIAL_MATRIX,1.,&HinvxRymu); CHKERRQ(ierr);
  ierr = MatMatMatMult(_Dy_Iz,_mu,_Dy_Iz,MAT_INITIAL_MATRIX,1.,&Dyymu); CHKERRQ(ierr);
  ierr = MatAXPY(Dyymu,-1.,HinvxRymu,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  MatDestroy(&HinvxRymu);
  MatDestroy(&Rymu);

  if (_multByH) {
    Mat temp;
    MatMatMult(_H,Dyymu,MAT_INITIAL_MATRIX,1.,&temp); CHKERRQ(ierr);
    MatCopy(temp,Dyymu,SAME_NONZERO_PATTERN);
    MatDestroy(&temp);
  }

  //~ writeMat(Dyymu,"/Users/kallison/eqcycle/data/mms_ops_p_Dyymu");

  #if VERBOSE >1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructDyymu in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute Dzzmu using my class Spmat
PetscErrorCode SbpOps_m_constGrid::constructDzzmu(const TempMats_m_constGrid& tempMats,Mat &Dzzmu)
{
  PetscErrorCode  ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function constructDzzmu in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif

  Mat Rzmu,HinvxRzmu;
  ierr = constructRzmu(tempMats,Rzmu); CHKERRQ(ierr);
  ierr = MatMatMult(_Iy_Hzinv,Rzmu,MAT_INITIAL_MATRIX,1.,&HinvxRzmu); CHKERRQ(ierr);
  ierr = MatMatMatMult(_Iy_Dz,_mu,_Iy_Dz,MAT_INITIAL_MATRIX,1.,&Dzzmu); CHKERRQ(ierr);
  ierr = MatAXPY(Dzzmu,-1.,HinvxRzmu,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  MatDestroy(&HinvxRzmu);
  MatDestroy(&Rzmu);

  if (_multByH) {
    Mat temp;
    MatMatMult(_H,Dzzmu,MAT_INITIAL_MATRIX,1.,&temp); CHKERRQ(ierr);
    MatCopy(temp,Dzzmu,SAME_NONZERO_PATTERN);
    MatDestroy(&temp);
  }

  #if VERBOSE >1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function constructDzzmu in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SbpOps_m_constGrid::constructRzmu(const TempMats_m_constGrid& tempMats,Mat &Rzmu)
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
      sbp_Spmat2(_Nz,1.0/_dz,D2z,C2z);

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
        kronConvert(tempMats._Iy,D2z,Iy_D2z,5,0);
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
      sbp_Spmat4(_Nz,1/_dz,D3z,D4z,C3z,C4z);

      Mat mu3;
      {
        PetscScalar mu=0;
        MatDuplicate(_mu,MAT_COPY_VALUES,&mu3);
        PetscInt Ii,Jj,Istart,Iend=0;
        VecGetOwnershipRange(_muVec,&Istart,&Iend);
        if (Istart==0) {
          Jj = Istart + 1;
          VecGetValues(_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Istart,1,&Istart,&mu,ADD_VALUES);
        }
        if (Iend==_Ny*_Nz) {
          Jj = Iend - 2;
          Ii = Iend - 1;
          VecGetValues(_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Ii,1,&Ii,&mu,ADD_VALUES);
        }
        for (Ii=Istart+1;Ii<Iend-1;Ii++) {
          VecGetValues(_muVec,1,&Ii,&mu);
          Jj = Ii - 1;
          MatSetValues(mu3,1,&Jj,1,&Jj,&mu,ADD_VALUES);
        }
        MatAssemblyBegin(mu3,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mu3,MAT_FINAL_ASSEMBLY);
        MatScale(mu3,0.5);
      }

      Mat Iy_D3z; kronConvert(tempMats._Iy,D3z,Iy_D3z,6,0);
      Mat Iy_C3z; kronConvert(tempMats._Iy,C3z,Iy_C3z,1,0);

      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
      Mat temp1,temp2;
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

      Mat Iy_D4z; kronConvert(tempMats._Iy,D4z,Iy_D4z,5,0);
      Mat Iy_C4z; kronConvert(tempMats._Iy,C4z,Iy_C4z,1,0);

      // Rzmu = (Iy_D3z^T x Iy_C3z x mu3 x Iy_D3z)/18/dy
      //      + (Iy_D4z^T x Iy_C4z x mu x Iy_D4z)/144/dy
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


PetscErrorCode SbpOps_m_constGrid::constructRymu(const TempMats_m_constGrid& tempMats,Mat &Rymu)
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
      sbp_Spmat2(_Ny,1/_dy,D2y,C2y);

      // kron(D2y,Iz)
      Mat D2y_Iz;
      {
        kronConvert(D2y,tempMats._Iz,D2y_Iz,5,0);
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
        kronConvert(C2y,tempMats._Iz,C2y_Iz,5,0);
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
      sbp_Spmat4(_Ny,1/_dy,D3y,D4y,C3y,C4y);

      Mat mu3;
      {
        PetscScalar mu=0;
        MatDuplicate(_mu,MAT_COPY_VALUES,&mu3);
        PetscInt Ii,Jj,Istart,Iend=0;
        VecGetOwnershipRange(_muVec,&Istart,&Iend);
        if (Iend==_Ny*_Nz) {
          Jj = Iend - 2;
          Ii = Iend - 1;
          VecGetValues(_muVec,1,&Jj,&mu);
          MatSetValues(mu3,1,&Ii,1,&Ii,&mu,ADD_VALUES);
        }
        for (Ii=Istart+1;Ii<Iend;Ii++) {
          VecGetValues(_muVec,1,&Ii,&mu);
          Jj = Ii - 1;
          MatSetValues(mu3,1,&Jj,1,&Jj,&mu,ADD_VALUES);
        }
        MatAssemblyBegin(mu3,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mu3,MAT_FINAL_ASSEMBLY);
        MatScale(mu3,0.5);
      }

      Mat D3y_Iz; kronConvert(D3y,tempMats._Iz,D3y_Iz,6,0);
      Mat C3y_Iz; kronConvert(C3y,tempMats._Iz,C3y_Iz,1,0);


      // Rymu = (D3y_Iz^T x C3y_Iz x mu3 x D3y_Iz)/18/dy
      //      + (D4y_Iz^T x C4y_Iz x mu x D4y_Iz)/144/dy
      Mat temp1,temp2;
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


      Mat D4y_Iz; kronConvert(D4y,tempMats._Iz,D4y_Iz,5,0);
      Mat C4y_Iz; kronConvert(C4y,tempMats._Iz,C4y_Iz,1,0);

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


// compute matrices for 1st derivatives
PetscErrorCode SbpOps_m_constGrid::construct1stDerivs(const TempMats_m_constGrid& tempMats)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function construct1stDerivs in %s.\n", FILENAME);CHKERRQ(ierr);
#endif

  kronConvert(tempMats._D1y,tempMats._Iz,_Dy_Iz,5,5);
  ierr = PetscObjectSetName((PetscObject) _Dy_Iz, "_Dy_Iz");CHKERRQ(ierr);

  kronConvert(tempMats._Iy,tempMats._D1z,_Iy_Dz,5,5);
  ierr = PetscObjectSetName((PetscObject) _Iy_Dz, "_Iy_Dz");CHKERRQ(ierr);

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

PetscErrorCode SbpOps_m_constGrid::updateVarCoeff(const Vec& coeff)
{
  PetscErrorCode  ierr = 0;
  double startTime = MPI_Wtime();
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function updateVarCoeff in %s.\n", FILENAME);
  CHKERRQ(ierr);
#endif

  MatDestroy(&_D2);

  // update coefficient Vec and Mat
  VecCopy(coeff,_muVec);
  MatDiagonalSet(_mu,coeff,INSERT_VALUES);

  // update Mats
  TempMats_m_constGrid tempMats(_order,_Ny,_dy,_Nz,_dz,_compatibilityType);
  constructBs(tempMats);
  updateBCMats();
  updateA_BCs(tempMats);

  _runTime = MPI_Wtime() - startTime;
  #if VERBOSE >1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function updateVarCoeff in sbpOps.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


//======================== public member functions =====================


//======================= I/O functions ================================

PetscErrorCode SbpOps_m_constGrid::loadOps(const std::string inputDir)
{
  PetscErrorCode  ierr = 0;
  PetscViewer     viewer;

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function loadOps in %s.\n", FILENAME);CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function loadOps in %s.\n", FILENAME);CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
    return ierr;
}


PetscErrorCode SbpOps_m_constGrid::writeOps(const std::string outputDir)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeOps in %s.\n", FILENAME);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  writeMat(_A,outputDir + "A");

  writeMat(_Dy_Iz,outputDir + "Dy_Iz");
  writeMat(_Iy_Dz,outputDir + "Iy_Dz");

  writeMat(_rhsR,outputDir + "rhsR");
  writeMat(_rhsT,outputDir + "rhsT");
  writeMat(_rhsL,outputDir + "rhsL");
  writeMat(_rhsB,outputDir + "rhsB");
  writeMat(_AR,outputDir + "AR");
  writeMat(_AT,outputDir + "AT");
  writeMat(_AL,outputDir + "AL");
  writeMat(_AB,outputDir + "AB");

  writeMat(_H,outputDir + "H");
  writeMat(_Hinv,outputDir + "Hinv");
  writeMat(_Hyinv_Iz,outputDir + "Hyinv");
  writeMat(_Iy_Hzinv,outputDir + "Hzinv");

  writeMat(_E0y_Iz,outputDir + "E0y");
  writeMat(_ENy_Iz,outputDir + "ENy");
  writeMat(_Iy_E0z,outputDir + "E0z");
  writeMat(_Iy_ENz,outputDir + "ENz");
  writeMat(_e0y_Iz,outputDir + "ee0y");
  writeMat(_eNy_Iz,outputDir + "eeNy");
  writeMat(_Iy_e0z,outputDir + "ee0z");
  writeMat(_Iy_eNz,outputDir + "eeNz");

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeOps in sbpOps.cpp\n");CHKERRQ(ierr);
#endif
  _runTime = MPI_Wtime() - startTime;
  return ierr;
};




// out = Dy * in
PetscErrorCode SbpOps_m_constGrid::Dy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dy";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  ierr = MatMult(_Dy_Iz,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
};

 // out = mu * Dy * in
PetscErrorCode SbpOps_m_constGrid::muxDy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_Dy_Iz,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
};

// out = Dy * mu * in
PetscErrorCode SbpOps_m_constGrid::Dyxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dyxmu";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_Dy_Iz,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
};


// out = Dz * in
PetscErrorCode SbpOps_m_constGrid::Dz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dz";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  ierr = MatMult(_Iy_Dz,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
};


// out = mu * Dz * in
PetscErrorCode SbpOps_m_constGrid::muxDz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Dz,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Dz * mu * in
PetscErrorCode SbpOps_m_constGrid::Dzxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dzxmu";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp;
  ierr = VecDuplicate(in,&temp); CHKERRQ(ierr);
  ierr = MatMult(_mu,in,temp); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Dz,temp,out); CHKERRQ(ierr);

  VecDestroy(&temp);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = H * in
PetscErrorCode SbpOps_m_constGrid::H(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "H";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  ierr = MatMult(_H,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hinv * in
PetscErrorCode SbpOps_m_constGrid::Hinv(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hinv";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  ierr = MatMult(_Hinv,in,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * e0y * in
PetscErrorCode SbpOps_m_constGrid::Hyinvxe0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hyinvxe0y";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_e0y_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_m_constGrid::HyinvxeNy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxeNy";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_eNy_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * E0y * in
PetscErrorCode SbpOps_m_constGrid::HyinvxE0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxE0y";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_E0y_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_m_constGrid::HyinvxENy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxENy";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_ENy_Iz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * e0z * in
PetscErrorCode SbpOps_m_constGrid::HzinvxE0z(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxE0z";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_E0z,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * eNz * in
PetscErrorCode SbpOps_m_constGrid::HzinvxENz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxENz";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif

  Vec temp1;
  ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_ENz,in,temp1); CHKERRQ(ierr);
  ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);CHKERRQ(ierr);
#endif
  return ierr;
}


//=================== functions for struct =============================

TempMats_m_constGrid::TempMats_m_constGrid(const PetscInt order,const PetscInt Ny,
      const PetscScalar dy,const PetscInt Nz,const PetscScalar dz,const string type)
: _order(order),_Ny(Ny),_Nz(Nz),_dy(dy),_dz(dz),
  _Hy(Ny,Ny),_Hyinv(Ny,Ny),_D1y(Ny,Ny),_D1yint(Ny,Ny),_BSy(Ny,Ny),_Iy(Ny,Ny),
  _Hz(Nz,Nz),_Hzinv(Nz,Nz),_D1z(Nz,Nz),_D1zint(Nz,Nz),_BSz(Nz,Nz),_Iz(Nz,Nz)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats_m_constGrid::TempMats_m_constGrid in %s.\n", FILENAME);
#endif

  _Iy.eye(); // matrix size is set during colon initialization
  _Iz.eye();

  sbp_Spmat(order,Ny,1./dy,_Hy,_Hyinv,_D1y,_D1yint,_BSy,type);
  sbp_Spmat(order,Nz,1./dz,_Hz,_Hzinv,_D1z,_D1zint,_BSz,type);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats_m_constGrid::TempMats_m_constGrid in %s.\n", FILENAME);
#endif
}

TempMats_m_constGrid::~TempMats_m_constGrid()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting TempMats_m_constGrid::~TempMats_m_constGrid in %s.\n", FILENAME);
#endif

 // do nothing

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending TempMats_m_constGrid::~TempMats_m_constGrid in %s.\n", FILENAME);
#endif
}
