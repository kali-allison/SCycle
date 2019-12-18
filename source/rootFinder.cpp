#include "rootFinder.hpp"

using namespace std;


/*======================================================================
 *
 * Test functions for RootFinder.
 *
 *====================================================================*/

PetscErrorCode TestRoots::getResid(const PetscInt ind,const PetscScalar val,PetscScalar *out)
{
  PetscErrorCode ierr = 0;

  //~double temp = val-2;
  //~*out = temp*temp*temp;

  //~*out = (PetscScalar) a*sigma_N*asinh( (double) (vel/2./_v0)*exp(psi/a) ) + 0.5*zPlus*vel - tauQS;
  PetscScalar tauQS = 0.75*(3.000075e+01 - 9.999920e-06);
  *out = (PetscScalar) 0.015*50*asinh( (double) (val/2./1e-6)*exp(0.6/0.015) ) + 0.5*12*val - tauQS;

  return ierr;
}


//================= RootFinder member functions ========================

RootFinder::RootFinder(const PetscInt maxNumIts,const PetscScalar atol)
: _numIts(0),_maxNumIts(maxNumIts),_atol(atol)
{
  assert(_maxNumIts >= 0);
  assert(_atol >= 0);
}

//~RootFinder::~RootFinder(){};

PetscInt RootFinder::getNumIts() const
{
  return _numIts;
}


//=============== Bisect member functions ==============================

Bisect::Bisect(const PetscInt maxNumIts,const PetscScalar atol)
: RootFinder(maxNumIts,atol),
  _left(0),_fLeft(0),_right(0),_fRight(0),_mid(0),_fMid(2*atol)
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting Bisect::Bisect in rootFinder.cpp.\n");
#endif

#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending Bisect::Bisect in rootFinder.cpp.\n");
#endif
}

Bisect::~Bisect()
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting Bisect::~Bisect in rootFinder.cpp.\n");
#endif


#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending Bisect::~Bisect in rootFinder.cpp.\n");
#endif
};


PetscErrorCode Bisect::findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out)
{
  return findRoot(obj,ind,out);
}

PetscErrorCode Bisect::findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting bisect in rootFinder.cpp\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"..left = %e, right = %g,mid=%g Ii=%i\n",_left,_right,_mid,ind);CHKERRQ(ierr);
#endif

  ierr = obj->getResid(ind,_left,&_fLeft);CHKERRQ(ierr);
  ierr = obj->getResid(ind,_right,&_fRight);CHKERRQ(ierr);

  assert(!std::isnan(_fLeft)); assert(!std::isnan(_fRight));
  assert(!std::isinf(_fLeft)); assert(!std::isinf(_fRight));

  if (sqrt(_fLeft*_fLeft) <= _atol) { *out = _left; return 0; }
  else if (sqrt(_fRight*_fRight) <= _atol) { *out = _right; return 0; }

  PetscInt numIts = 0;
  while ( (numIts <= _maxNumIts) & (sqrt(_fMid*_fMid) >= _atol) ) {
    _mid = (_left + _right)*0.5;
    ierr = obj->getResid(ind,_mid,&_fMid); CHKERRQ(ierr);

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
  // assign left and right bounds, ensuring left < right
  if (left > right) {
    _left = right;
    _right = _left;
  }
  else {
    _left=left;
    _right=right;
  }

  _mid = (_left + _right)*0.5;
  _fMid = 2 * _atol;

  return 0;
}


//=============== BracketedNewton member functions ==============================

BracketedNewton::BracketedNewton(const PetscInt maxNumIts,const PetscScalar atol)
: RootFinder(maxNumIts,atol),
  _left(0),_fLeft(0),_right(0),_fRight(0)
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting BracketedNewton::BracketedNewton in rootFinder.cpp.\n");
#endif

#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending BracketedNewton::BracketedNewton in rootFinder.cpp.\n");
#endif
}

BracketedNewton::~BracketedNewton()
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting BracketedNewton::~BracketedNewton in rootFinder.cpp.\n");
#endif


#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending BracketedNewton::~BracketedNewton in rootFinder.cpp.\n");
#endif
};


PetscErrorCode BracketedNewton::findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out)
{
  return findRoot(obj,ind,0.5*(_left+_right),out);
}


PetscErrorCode BracketedNewton::findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar x0,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting BracketedNewton::findRoot in rootFinder.cpp\n");
#endif

  // check if initial input is the root
  ierr = obj->getResid(ind,_x,&_f,&_fPrime);CHKERRQ(ierr);
  assert(!std::isinf(_f)); assert(!std::isnan(_f));
  assert(!std::isinf(x0)); assert(!std::isnan(x0));
  if (abs(_f) <= _atol) { *out = x0; return 0; }

  // check if endpoints are root
  ierr = obj->getResid(ind,_left,&_fLeft);CHKERRQ(ierr);
  ierr = obj->getResid(ind,_right,&_fRight);CHKERRQ(ierr);
  assert(!std::isnan(_fLeft));
  assert(!std::isnan(_fRight));
  assert(!std::isinf(_fLeft));
  assert(!std::isinf(_fRight));

  if (sqrt(_fLeft*_fLeft) <= _atol) {
    *out = _left;
    return 0;
  }
  else if (sqrt(_fRight*_fRight) <= _atol) {
    *out = _right;
    return 0;
  }
  else if (sqrt(_fRight*_fRight) <= _atol) {
    *out = _right;
    return 0;
  }

  // ensure that _fLeft < 0
  if (_fLeft > 0) {
    PetscReal temp = _left;
    PetscReal fTemp = _fLeft;
    _left = _right; _fLeft = _fRight;
    _right = temp; _fRight = fTemp;
  }

  // proceed with iteration
  PetscInt numIts = 0;
  PetscScalar _x = x0;
  PetscScalar dxOld = fabs(_right - _left);
  PetscScalar dx = dxOld;
  ierr = obj->getResid(ind,_x,&_f,&_fPrime); CHKERRQ(ierr);

  while ( (numIts <= _maxNumIts) && (abs(_f) >= _atol) ) {

    // use bisection if Newton out of range or not converging quickly enough
    if ( ((_x-_right)*_fPrime-_f)*((_x-_left)*_fPrime-_f) > 0.0
  || fabs(2.0*_f) > fabs(dxOld*_fPrime) ) {
      dxOld = dx;
      dx = 0.5*(_right - _left);
      _x = _left + dx;
    }
    else {
      // Newton step is acceptable
      dxOld = dx;
      dx = _f/_fPrime;
      _x -= dx;
    }

    ierr = obj->getResid(ind,_x,&_f,&_fPrime); CHKERRQ(ierr);

    // update bounds
    if (_f < 0.0) { _left = _x; }
    else { _right = _x; }

   numIts++;
  }

  *out = _x;
  if (abs(_f) > _atol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"rootFinder BracketedNewton did not converge in %i iterations\n",numIts);
    PetscPrintf(PETSC_COMM_WORLD,"ind = %i, residual = %g\n",ind,_f);
    assert(abs(_f) < _atol);
    return 1;
  }

  return ierr;
}


PetscErrorCode BracketedNewton::setBounds(PetscScalar left,PetscScalar right)
{
  // it is ok if left < right or vice versa
  _left=left;
  _right=right;
  return 0;
}


//=============== RegulaFalsi member functions ==============================

RegulaFalsi::RegulaFalsi(const PetscInt maxNumIts,const PetscScalar atol)
: RootFinder(maxNumIts,atol),
  _left(0),_fLeft(0),_right(0),_fRight(0),_x(0),_f(2*atol)
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting RegulaFalsi::RegulaFalsi in rootFinder.cpp.\n");
#endif

#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending RegulaFalsi::RegulaFalsi in rootFinder.cpp.\n");
#endif
}

RegulaFalsi::~RegulaFalsi()
{
#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Starting RegulaFalsi::~RegulaFalsi in rootFinder.cpp.\n");
#endif


#if VERBOSE > 3
  PetscPrintf(PETSC_COMM_WORLD,"Ending RegulaFalsi::~RegulaFalsi in rootFinder.cpp.\n");
#endif
};



PetscErrorCode RegulaFalsi::findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out)
{
  return findRoot(obj,ind,0.5*(_left+_right),out);
}

PetscErrorCode RegulaFalsi::findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting RegulaFalsi in rootFinder.cpp\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"..left = %e, right = %g Ii=%i\n",_left,_right,ind);CHKERRQ(ierr);
#endif

  // initial guess
  _x = in;

  ierr = obj->getResid(ind,_left,&_fLeft);CHKERRQ(ierr);
  ierr = obj->getResid(ind,_right,&_fRight);CHKERRQ(ierr);
  ierr = obj->getResid(ind,_x,&_f);CHKERRQ(ierr);

  assert(!std::isnan(_fLeft)); assert(!std::isnan(_fRight));
  assert(!std::isinf(_fLeft)); assert(!std::isinf(_fRight));

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"fLeft = %g, fRight = %g\n",_fLeft,_fRight);CHKERRQ(ierr);
#endif

  if (sqrt(_fLeft*_fLeft) <= _atol) { *out = _left; return 0; }
  else if (sqrt(_fRight*_fRight) <= _atol) { *out = _right; return 0; }

  PetscInt numIts = 0;
  PetscScalar diff = 10*_atol;
  PetscScalar prev = _x;

  while ( (numIts <= _maxNumIts) & (sqrt(_f*_f) >= _atol) & (sqrt(diff*diff) >= _atol)) {
#if VERBOSE > 4
    ierr = PetscPrintf(PETSC_COMM_WORLD,"!!%i: %i %.15f %.15f %.15f %.15f\n",
                       ind,numIts,_left,_right,_mid,_f);CHKERRQ(ierr);
#endif
    prev = _x;
    if (_fLeft*_f > _atol) {
      _left = _x;
      _x = _right - (_right - _left) * (_fRight / (_fRight - _fLeft));
      _fLeft = _f;
    }
    else {
      _right = _x;
      _x = _right - (_right - _left) * (_fRight / (_fRight - _fLeft));
      _fRight = _f;
    }

  diff = (_x-prev)/_x;
  ierr = obj->getResid(ind,_x,&_f);CHKERRQ(ierr);
  numIts++;
  }

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"numIts/maxIts = %u/%u, final mid = %g, fMid = %g\n",
                     numIts,_maxNumIts,_mid,_f);CHKERRQ(ierr);
#endif

  *out = _x;
  if ( (sqrt(_f*_f) > _atol) & (sqrt(diff*diff) > _atol)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"rootFinder did not converge in %i iterations\n",numIts);
    assert(sqrt(_f*_f) < _atol);
    return 1;
  }

  return ierr;
}

PetscErrorCode RegulaFalsi::setBounds(PetscScalar left,PetscScalar right)
{
  assert(left <= right);

  _left=left;
  _right=right;

  return 0;
}
