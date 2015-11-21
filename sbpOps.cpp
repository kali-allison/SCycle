#include "sbpOps.hpp"




//================= constructor and destructor ========================
/* SAT params _alphaD,_alphaD set to values that work for both 2nd and
 * 4th order but are not ideal for 4th.
 */
SbpOps::SbpOps(Domain&D,PetscScalar& muArr,Mat& mu)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(&muArr),_mu(&mu),_internalSBP(D,*D._muArrPlus,D._muP),
  _alphaDy(_internalSBP._alphaDy)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif

  if (_Ny == 1) { return;}

  //~_H = _internalSBP._H; // shallow copy to avoid memory cost
  _A = _internalSBP._A;

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in sbpOps.cpp.\n");
#endif
}

SbpOps::~SbpOps()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in sbpOps.cpp.\n");
#endif


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps.cpp.\n");
#endif
}


PetscErrorCode SbpOps::setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcS,Vec &_bcD)
{
  return _internalSBP.setRhs(rhs,_bcF,_bcR,_bcS,_bcD);
}

// read/write commands
PetscErrorCode SbpOps::loadOps(const std::string inputDir)
{
  return _internalSBP.loadOps(inputDir);
}

PetscErrorCode SbpOps::writeOps(const std::string outputDir)
{
  return _internalSBP.writeOps(outputDir);
}

 // out = Dy * in
PetscErrorCode SbpOps::Dy(const Vec &in, Vec &out)
{
  return _internalSBP.Dy(in,out);
}

// out = mu * Dy * in
PetscErrorCode SbpOps::muxDy(const Vec &in, Vec &out)
{
  return _internalSBP.muxDy(in,out);
}

// out = Dy * mu * in
PetscErrorCode SbpOps::Dyxmu(const Vec &in, Vec &out)
{
  return _internalSBP.Dyxmu(in,out);
}


// out = Dz * in
PetscErrorCode SbpOps::Dz(const Vec &in, Vec &out)
{
  return _internalSBP.Dz(in,out);
}


// out = mu * Dz * in
PetscErrorCode SbpOps::muxDz(const Vec &in, Vec &out)
{
  return _internalSBP.muxDz(in,out);
}


// out = Dz * mu * in
PetscErrorCode SbpOps::Dzxmu(const Vec &in, Vec &out)
{
  return _internalSBP.Dzxmu(in,out);
}


// out = H * in
PetscErrorCode SbpOps::H(const Vec &in, Vec &out)
{
  return _internalSBP.H(in,out);
}

// out = H * Iy_Bz * 2 * mu * in
PetscErrorCode SbpOps::HBzx2mu(const Vec &in, Vec &out)
{
  return _internalSBP.HBzx2mu(in,out);
}

// out = By_Iz * in
PetscErrorCode SbpOps::By(const Vec &in, Vec &out)
{
  return _internalSBP.By(in,out);
}

// out = e0y_Iz * in
PetscErrorCode SbpOps::e0y(const Vec &in, Vec &out)
{
  return _internalSBP.e0y(in,out);
}

// out = eNy_Iz * in
PetscErrorCode SbpOps::eNy(const Vec &in, Vec &out)
{
  return _internalSBP.eNy(in,out);
}

