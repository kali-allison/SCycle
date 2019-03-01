#include "domain.hpp"

#define FILENAME "sbpOps_fc.cpp"

using namespace std;

Domain::Domain(const char *file)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),
  _momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),_operatorType("matrix-based"),
  _sbpCompatibilityType("fullyCompatible"),
  _gridSpacingType("variableGridSpacing"),
  _isMMS(0),_loadICs(0), _inputDir("unspecified_"),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),
  _vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(-1)
{
  // load data from file
  loadData(_file);

  // check domain size and set grid spacing in y direction
  if (_Ny > 1) {
    _dq = 1.0 / (_Ny - 1.0);
  }
  else {
    _dq = 1;
  }

  // set grid spacing in z-direction
  if (_Nz > 1) {
    _dr = 1.0 / (_Nz - 1.0);
  }
  else {
    _dr = 1;
  }

  checkInput(); // perform some basic value checking to prevent NaNs
  setFields();
  setScatters();
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
  for (it = _scatters.begin(); it != _scatters.end(); it++ ) {
    VecScatterDestroy(&it->second);
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
  VecDuplicate(_y,&_z); PetscObjectSetName((PetscObject) _z, "z");
  VecDuplicate(_y,&_q); PetscObjectSetName((PetscObject) _q, "q");
  VecDuplicate(_y,&_r); PetscObjectSetName((PetscObject) _r, "r");

  // construct coordinate transform
  PetscInt Ii,Istart,Iend,Jj = 0;
  PetscScalar *y,*z,*q,*r;
  ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);

  // return pointers to local data arrays (the processor's portion of vector data)
  VecGetArray(_y,&y);
  VecGetArray(_z,&z);
  VecGetArray(_q,&q);
  VecGetArray(_r,&r);

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
  VecRestoreArray(_y,&y);
  VecRestoreArray(_z,&z);
  VecRestoreArray(_q,&q);
  VecRestoreArray(_r,&r);

  return ierr;
}
