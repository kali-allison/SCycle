#include "domain.hpp"

using namespace std;

Domain::Domain()
  :  _sbpType("mfc_coordTrans"),_Ny(201),_Nz(1),_Ly(30),_Lz(30),
     _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(-1),_dr(-1),
     _bCoordTrans(5)
{
  if (_Ny > 1) {
    _dq = 1.0 / (_Ny - 1.0);
  }
  else {
    _dq = 1;
  }

  if (_Nz > 1) {
    _dr = 1.0 / (_Nz - 1.0);
  }
  else {
    _dr = 1;
  }
}

// destructor
Domain::~Domain()
{
  // free memory
  VecDestroy(&_q);
  VecDestroy(&_r);
  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&_y0);
  VecDestroy(&_z0);

  // set map iterator, free memory from VecScatter
  map<string,VecScatter>::iterator it;
  for (it = _scatters.begin(); it != _scatters.end(); it++) {
    VecScatterDestroy(&(it->second));
  }
}

// construct coordinate transform, setting vectors q, r, y, z
PetscErrorCode Domain::setFields()
{
  PetscErrorCode ierr = 0;

  // generate vector _y with size _Ny*_Nz
  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y"); CHKERRQ(ierr);

  // duplicate _y into _z, _q, _r
  ierr = VecDuplicate(_y,&_z); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _z, "z"); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_q); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _q, "q"); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_r); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _r, "r"); CHKERRQ(ierr);

  // construct coordinate transform
  PetscInt Ii,Istart,Iend,Jj = 0;
  PetscScalar *y,*z,*q,*r;
  ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);

  // return pointers to local data arrays (the processor's portion of vector data)
  ierr = VecGetArray(_y,&y); CHKERRQ(ierr);
  ierr = VecGetArray(_z,&z); CHKERRQ(ierr);
  ierr = VecGetArray(_q,&q); CHKERRQ(ierr);
  ierr = VecGetArray(_r,&r); CHKERRQ(ierr);

  // set vector entries for q, r (coordinate transform) and y, z (no transform)
  for (Ii=Istart; Ii<Iend; Ii++) {
    q[Jj] = _dq*(Ii/_Nz);
    r[Jj] = _dr*(Ii-_Nz*(Ii/_Nz));

    // matrix-based, fully compatible, allows curvilinear coordinate transformation
    if (_sbpType.compare("mfc_coordTrans") ) {
      y[Jj] = (_dq*_Ly)*(Ii/_Nz);
      z[Jj] = (_dr*_Lz)*(Ii-_Nz*(Ii/_Nz));
    }
    else {
      // hardcoded transformation (not available for z)
      if (_bCoordTrans > 0) {
	y[Jj] = _Ly * sinh(_bCoordTrans * q[Jj]) / sinh(_bCoordTrans);
      }
      // no transformation
      y[Jj] = q[Jj]*_Ly;
      z[Jj] = r[Jj]*_Lz;
    }
    Jj++;
  }

  // restore arrays
  ierr = VecRestoreArray(_y,&y); CHKERRQ(ierr);
  ierr = VecRestoreArray(_z,&z); CHKERRQ(ierr);
  ierr = VecRestoreArray(_q,&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(_r,&r); CHKERRQ(ierr);

  return ierr;
}


// scatters values from one vector to another
PetscErrorCode Domain::setScatters() {
  PetscErrorCode ierr = 0;

  ierr = VecCreate(PETSC_COMM_WORLD,&_y0); CHKERRQ(ierr);
  ierr = VecSetSizes(_y0,PETSC_DECIDE,_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y0); CHKERRQ(ierr);
  ierr = VecSet(_y0,0.0); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&_z0); CHKERRQ(ierr);
  ierr = VecSetSizes(_z0,PETSC_DECIDE,_Ny); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_z0); CHKERRQ(ierr);
  ierr = VecSet(_z0,0.0); CHKERRQ(ierr);

  PetscInt *indices;
  IS is;
  ierr = PetscMalloc1(_Nz,&indices); CHKERRQ(ierr);

  // we want to scatter from index 0 to _Nz - 1, i.e. take the first _Nz components of the vector to scatter from
  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    indices[Ii] = Ii;
  }

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, indices, PETSC_COPY_VALUES, &is); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, is, _y0, is, &_scatters["body2L"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(indices); CHKERRQ(ierr);
  ierr = ISDestroy(&is); CHKERRQ(ierr);

  //===============================================================================
  // set up scatter context to take values for y = Ly from body field and put them on a Vec of size Nz
  PetscInt *fi;
  IS isf;
  ierr = PetscMalloc1(_Nz,&fi); CHKERRQ(ierr);

  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    fi[Ii] = Ii + (_Ny*_Nz-_Nz);
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, fi, PETSC_COPY_VALUES, &isf); CHKERRQ(ierr);

  PetscInt *ti;
  IS ist;
  ierr = PetscMalloc1(_Nz,&ti); CHKERRQ(ierr);
  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    ti[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, ti, PETSC_COPY_VALUES, &ist); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf, _y0, ist, &_scatters["body2R"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(fi); CHKERRQ(ierr);
  ierr = PetscFree(ti); CHKERRQ(ierr);
  ierr = ISDestroy(&isf); CHKERRQ(ierr);
  ierr = ISDestroy(&ist); CHKERRQ(ierr);

  
  //============================================================================== 
  IS isf2;
  ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, 0, _Nz, &isf2); CHKERRQ(ierr);

  PetscInt *ti2;
  IS ist2;
  ierr = PetscMalloc1(_Ny,&ti2); CHKERRQ(ierr);

  for (PetscInt Ii=0; Ii<_Ny; Ii++) {
    ti2[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti2, PETSC_COPY_VALUES, &ist2); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf2, _z0, ist2, &_scatters["body2T"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(ti2); CHKERRQ(ierr);
  ierr = ISDestroy(&isf2); CHKERRQ(ierr);
  ierr = ISDestroy(&ist2); CHKERRQ(ierr);


  //==============================================================================
  IS isf3;
  ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, _Nz - 1, _Nz, &isf3); CHKERRQ(ierr);

  PetscInt *ti3;
  IS ist3;
  ierr = PetscMalloc1(_Ny,&ti3); CHKERRQ(ierr);
  for (PetscInt Ii = 0; Ii<_Ny; Ii++) {
    ti3[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti3, PETSC_COPY_VALUES, &ist3); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf3, _z0, ist3, &_scatters["body2B"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(ti3); CHKERRQ(ierr);
  ierr = ISDestroy(&isf3); CHKERRQ(ierr);
  ierr = ISDestroy(&ist3); CHKERRQ(ierr);

  return ierr;
}
