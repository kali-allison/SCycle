#include "SbpOps_sc.hpp"




//================= constructor and destructor ========================
/* SAT params _alphaD,_alphaD set to values that work for both 2nd and
 * 4th order but are not ideal for 4th.
 */
SbpOps_sc::SbpOps_sc(Domain&D,PetscScalar& muArr,Mat& mu)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),_dy(D._dy),_dz(D._dz),
  _muArr(&muArr),_mu(&mu),_muVP(D._muVP),
  _da(D._da),
  _bcTType(D._bcTType),_bcRType(D._bcRType),_bcBType(D._bcBType),_bcLType(D._bcLType),
  _alphaT(-1.0),_alphaDy(-4.0/_dy),_alphaDz(-4.0/_dz),_beta(1.0)

{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in sbpOps.cpp.\n");
#endif

  if (_Ny == 1) { return;}

  if(_muArr) { // ensure that _muArr is not NULL
      /* NOT a member of this class, contains stuff to be deleted before
       * end of constructor to save on memory usage.
       */
      PetscInt zn,yn;
      DMDAGetCorners(_da, &_zS, &_yS, 0, &zn, &yn, 0);
      _zE = _zS + zn;
      _yE = _yS + yn;

      // reset SAT params
      if (_order==4) {
        _alphaDy = -2.0*48.0/17.0 /_dy;
        _alphaDz = -2.0*48.0/17.0 /_dz;

    }
}

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in sbpOps.cpp.\n");
#endif
}

SbpOps_sc::~SbpOps_sc()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in sbpOps.cpp.\n");
#endif


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in sbpOps.cpp.\n");
#endif
}


//======================== public member functions =====================

// map the boundary condition vectors to rhs
PetscErrorCode SbpOps_sc::setRhs(Vec&rhs,Vec &bcL,Vec &bcR,Vec &bcT,Vec &bcB)
{
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function setRhs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  //~ierr = VecSet(rhs,0.0);
  //~ierr = MatMult(_rhsL,bcL,rhs);CHKERRQ(ierr); // rhs = _rhsL * _bcL
  //~ierr = MatMultAdd(_rhsR,bcR,rhs,rhs); // rhs = rhs + _rhsR * _bcR
  //~ierr = MatMultAdd(_rhsT,bcT,rhs,rhs);
  //~ierr = MatMultAdd(_rhsB,bcB,rhs,rhs);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function setRhs in sbpOps.cpp.\n");CHKERRQ(ierr);
#endif

  _runTime += MPI_Wtime() - startTime;
  return ierr;
}


// out = Dy * in
PetscErrorCode SbpOps_sc::Dy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Dy";
    string fileName = "SbpOps_sc.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  for (yI = _yS; yI < _yE; yI++) {
    for (zI = _zS; zI < _zE; zI++) {
      if (yI > 0 && yI < _Ny - 1) { lout[yI][zI] = 0.5*(lin[yI+1][zI] - lin[yI-1][zI]); }
      else if (yI == 0) { lout[yI][zI] = -1.5*lin[0][zI] + 2.0*lin[1][zI] - 0.5*lin[2][zI]; }
      else if (yI == _Ny-1) { lout[yI][zI] = 0.5*lin[_Ny-3][zI] - 2.0*lin[_Ny-2][zI] + 1.5*lin[_Ny-1][zI]; }
      lout[yI][zI] = lout[yI][zI]/_dy;
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif
  return ierr;
}


 // out = mu * Dy * in
PetscErrorCode SbpOps_sc::muxDy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = Dy(in,out); CHKERRQ(ierr);
  ierr = VecPointwiseMult(_muVP,out,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};

// out = Dy * mu * in
PetscErrorCode SbpOps_sc::Dyxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dyxmu";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = VecPointwiseMult(_muVP,in,out); CHKERRQ(ierr);
  ierr = Dy(out,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
};



// out = d/dz * in
PetscErrorCode SbpOps_sc::Dz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dz";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  for (yI = _yS; yI < _yE; yI++) {
    for (zI = _zS; zI < _zE; zI++) {
      if (zI > 0 && zI < _Nz - 1) { lout[yI][zI] = 0.5*(lin[yI][zI+1] - lin[yI][zI-1]); }
      else if (zI == 0) { lout[yI][zI] = -1.5*lin[yI][0] + 2.0*lin[yI][1] - 0.5*lin[yI][2]; }
      else if (zI == _Nz - 1) { lout[yI][zI] = 0.5*lin[yI][_Nz-3] - 2.0*lin[yI][_Nz-2] + 1.5*lin[yI][_Nz-1]; }
      lout[yI][zI] = lout[yI][zI]/_dz;
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = mu * Dz * in
PetscErrorCode SbpOps_sc::muxDz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "muxDy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = Dz(in,out); CHKERRQ(ierr);
  ierr = VecPointwiseMult(_muVP,out,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Dz * mu * in
PetscErrorCode SbpOps_sc::Dzxmu(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Dzxmu";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  ierr = VecPointwiseMult(_muVP,in,out); CHKERRQ(ierr);
  ierr = Dz(out,out); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = H * in
PetscErrorCode SbpOps_sc::H(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "H";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  // apply Hy then Hz
  ierr = Hy(in,out);
  ierr = Hz(out,out);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}


// out = Hy * in
PetscErrorCode SbpOps_sc::Hy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  VecCopy(in,out);

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  if (_order == 2) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (yI == 0 || yI == _Ny-1) { lout[yI][zI] = 0.5*lin[0][zI]; }
      }
    }
  }
  else if (_order == 4) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (yI == 0 || yI == _Ny-1) { lout[yI][zI] = 17.0/48.0*lin[yI][zI]; }
        if (yI == 1 || yI == _Ny-2) { lout[yI][zI] = 59.0/48.0*lin[yI][zI]; }
        if (yI == 2 || yI == _Ny-3) { lout[yI][zI] = 43.0/48.0*lin[yI][zI]; }
        if (yI == 3 || yI == _Ny-4) { lout[yI][zI] = 49.0/48.0*lin[yI][zI]; }
      }
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

  VecScale(out,_dy);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz * in
PetscErrorCode SbpOps_sc::Hz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hz";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  VecCopy(in,out);

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  if (_order == 2) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (zI == 0 || zI == _Ny-1) { lout[yI][zI] = 0.5*lin[0][zI]; }
      }
    }
  }
  else if (_order == 4) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (zI == 0 || zI == _Ny-1) { lout[yI][zI] = 17.0/48.0*lin[yI][zI]; }
        if (zI == 1 || zI == _Ny-2) { lout[yI][zI] = 59.0/48.0*lin[yI][zI]; }
        if (zI == 2 || zI == _Ny-3) { lout[yI][zI] = 43.0/48.0*lin[yI][zI]; }
        if (zI == 3 || zI == _Ny-4) { lout[yI][zI] = 49.0/48.0*lin[yI][zI]; }
      }
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

  VecScale(out,_dz);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}


// out = Hy^-1 * in
PetscErrorCode SbpOps_sc::Hinvy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hinvy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  VecCopy(in,out);

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  if (_order == 2) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (yI == 0 || yI == _Ny-1) { lout[yI][zI] = 2.0*lin[0][zI]; }
      }
    }
  }
  else if (_order == 4) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (yI == 0 || yI == _Ny-1) { lout[yI][zI] = 48.0/17.0*lin[yI][zI]; }
        if (yI == 1 || yI == _Ny-2) { lout[yI][zI] = 48.0/59.0*lin[yI][zI]; }
        if (yI == 2 || yI == _Ny-3) { lout[yI][zI] = 48.0/43.0*lin[yI][zI]; }
        if (yI == 3 || yI == _Ny-4) { lout[yI][zI] = 48.0/49.0*lin[yI][zI]; }
      }
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

  VecScale(out,1.0/_dy);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hinvz * in
PetscErrorCode SbpOps_sc::Hinvz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hinvz";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  VecCopy(in,out);

  Vec loutVec, linVec;
  PetscScalar** lout;
  PetscScalar** lin;
  ierr = DMCreateLocalVector(_da, &loutVec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(_da, &linVec);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(_da, in, INSERT_VALUES, linVec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(_da, linVec, &lin); CHKERRQ(ierr);

  PetscInt yI,zI;
  if (_order == 2) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (zI == 0 || zI == _Ny-1) { lout[yI][zI] = 2.0*lin[0][zI]; }
      }
    }
  }
  else if (_order == 4) {
    for (yI = _yS; yI < _yE; yI++) {
      for (zI = _zS; zI < _zE; zI++) {
        if (zI == 0 || zI == _Ny-1) { lout[yI][zI] = 48.0/17.0*lin[yI][zI]; }
        if (zI == 1 || zI == _Ny-2) { lout[yI][zI] = 48.0/59.0*lin[yI][zI]; }
        if (zI == 2 || zI == _Ny-3) { lout[yI][zI] = 48.0/43.0*lin[yI][zI]; }
        if (zI == 3 || zI == _Ny-4) { lout[yI][zI] = 48.0/49.0*lin[yI][zI]; }
      }
    }
  }

  ierr = DMDAVecRestoreArray(_da, loutVec, &lout);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(_da, linVec, &lin);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(_da, loutVec, INSERT_VALUES, out);CHKERRQ(ierr);

  VecDestroy(&loutVec);
  VecDestroy(&linVec);

  VecScale(out,1.0/_dz);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}


// out = Hy^-1 * e0y * in
PetscErrorCode SbpOps_sc::Hyinvxe0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "Hyinvxe0y";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_e0y_Iz,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_sc::HyinvxeNy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxeNy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(out,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_eNy_Iz,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * E0y * in
PetscErrorCode SbpOps_sc::HyinvxE0y(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxE0y";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(in,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_E0y_Iz,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hy^-1 * eNy * in
PetscErrorCode SbpOps_sc::HyinvxENy(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HyinvxENy";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(in,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_ENy_Iz,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Hyinv_Iz,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * e0z * in
PetscErrorCode SbpOps_sc::HzinvxE0z(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxE0z";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(in,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Iy_E0z,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}

// out = Hz^-1 * eNz * in
PetscErrorCode SbpOps_sc::HzinvxENz(const Vec &in, Vec &out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "HzinvxE0z";
  string fileName = "SbpOps_sc.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  //~Vec temp1;
  //~ierr = VecDuplicate(in,&temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Iy_ENz,in,temp1); CHKERRQ(ierr);
  //~ierr = MatMult(_Iy_Hzinv,temp1,out); CHKERRQ(ierr);

  //~VecDestroy(&temp1);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}


