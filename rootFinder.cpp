#include "rootFinder.hpp"


//================= RootFinder member functions ========================

RootFinder::RootFinder(const PetscInt maxNumIts,const PetscScalar atol)
: _numIts(0),_maxNumIts(maxNumIts),_atol(atol)
{
  assert(_maxNumIts >= 0);
  assert(_atol >= 0);
}

RootFinder::~RootFinder(){}


PetscInt RootFinder::getNumIts() const
{
  return _numIts;
}


//=============== Bisect member functions ==============================

Bisect::Bisect(const PetscInt maxNumIts,const PetscScalar atol)
: RootFinder(maxNumIts,atol),//_numIts(0),_maxNumIts(maxNumIts),
  _atol(atol),
  _left(0),_fLeft(0),_right(0),_fRight(0),_mid(0),_fMid(2*atol)
{}

Bisect::~Bisect(){}




PetscErrorCode Bisect::findRoot(Fault *obj,const PetscInt ind,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting bisect in rootFinder.cpp\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"..left = %e, right = %g,mid=%g Ii=%i\n",_left,_right,_mid,ind);CHKERRQ(ierr);
#endif

  ierr = obj->getResid(ind,_left,&_fLeft);CHKERRQ(ierr);
  ierr = obj->getResid(ind,_right,&_fRight);CHKERRQ(ierr);
  //~ierr = func(ind,_left,&_fLeft,ctx);CHKERRQ(ierr);
  //~ierr = func(ind,_right,&_fRight,ctx);CHKERRQ(ierr);

  assert(!isnan(_fLeft)); assert(!isnan(_fRight));
  assert(!isinf(_fLeft)); assert(!isinf(_fRight));
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",_fLeft,_fRight);CHKERRQ(ierr);
#endif

  //~if (sqrt(fLeft*fLeft) <= atol) { *out = left; return 0; }
  //~else if (sqrt(fRight*fRight) <= atol) { *out = right; return 0; }
  PetscInt numIts = 0;
  while ( (numIts <= _maxNumIts) & (sqrt(_fMid*_fMid) >= _atol) ) {
    _mid = (_left + _right)*0.5;
    ierr = obj->getResid(ind,_mid,&_fMid);CHKERRQ(ierr);
    //~ierr = func(ind,mid,&fMid,ctx);CHKERRQ(ierr);
#if VERBOSE > 4
    ierr = PetscPrintf(PETSC_COMM_WORLD,"!!%i: %i %.15f %.15f %.15f %.15f\n",
                       ind,numIts,_left,_right,_mid,_fMid);CHKERRQ(ierr);
#endif
    if (_fLeft*_fMid <= 0) {
      _right = _mid;
      _fRight = _fMid;
    }
    else {
      _left = _mid;
      _fLeft = _fMid;
    }
   numIts++;
  }

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts/maxIts = %u/%u, final mid = %g, fMid = %g\n",
                     numIts,_maxNumIts,_mid,_fMid);CHKERRQ(ierr);
#endif

  *out = _mid;
  if (sqrt(_fMid*_fMid) > _atol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"rootFinder did not converge in %i iterations\n",numIts);
    assert(sqrt(_fMid*_fMid) < _atol);
    return 1;
  }

  return ierr;
}


PetscErrorCode Bisect::setBounds(PetscScalar left,PetscScalar right)
{
  assert(left < right);

  _left=left;
  _right=right;

  _mid = (_left + _right)*0.5;

  return 0;
}
