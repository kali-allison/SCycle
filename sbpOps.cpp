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

// out = Hy^-1 * e0y * in
PetscErrorCode SbpOps::Hyinvxe0y(const Vec &in, Vec &out)
{
  return _internalSBP.Hyinvxe0y(in,out);
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps::HyinvxeNy(const Vec &in, Vec &out)
{
  return _internalSBP.HyinvxeNy(in,out);
}

// out = Hy^-1 * E0y * in
PetscErrorCode SbpOps::HyinvxE0y(const Vec &in, Vec &out)
{
  return _internalSBP.HyinvxE0y(in,out);
}

// out = Hy^-1 * ENy * in
PetscErrorCode SbpOps::HyinvxENy(const Vec &in, Vec &out)
{
  return _internalSBP.HyinvxENy(in,out);
}

// out = Hy^-1 * E0z * in
PetscErrorCode SbpOps::HzinvxE0z(const Vec &in, Vec &out)
{
  return _internalSBP.HzinvxE0z(in,out);
}

// out = Hy^-1 * ENz * in
PetscErrorCode SbpOps::HzinvxENz(const Vec &in, Vec &out)
{
  return _internalSBP.HzinvxENz(in,out);
}

