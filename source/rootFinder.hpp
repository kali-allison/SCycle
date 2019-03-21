#ifndef ROOTFINDER_HPP_INCLUDED
#define ROOTFINDER_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <assert.h>
#include "rootFinderContext.hpp"


class TestRoots : public RootFinderContext
{
  public:
    PetscErrorCode getResid(const PetscInt ind,const PetscScalar val,PetscScalar *out);
};


class RootFinder
{
  protected:
    PetscInt _numIts,_maxNumIts;
    PetscScalar _atol;

    // disable default copy constructor and assignment operator
    RootFinder(const RootFinder & that);
    RootFinder& operator=(const RootFinder& rhs);

  public:

    RootFinder(const PetscInt maxNumIts,const PetscScalar atol);

    virtual PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out) = 0;
    virtual PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out) = 0;
    virtual PetscErrorCode setBounds(PetscScalar left,PetscScalar right) = 0;

    PetscInt getNumIts() const;
};


// bisection method
class Bisect : public RootFinder
{
  private:

    PetscScalar _left,_fLeft;
    PetscScalar _right,_fRight;
    PetscScalar _mid,_fMid;

  public:

    Bisect(const PetscInt maxNumIts,const PetscScalar atol);
    ~Bisect();

    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out);
    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out);
    PetscErrorCode setBounds(PetscScalar left,PetscScalar right);
};


// Bracketed Newton solver
class BracketedNewton : public RootFinder
{
  private:

    PetscScalar _left,_fLeft;
    PetscScalar _right,_fRight;
    PetscScalar _x,_f,_fPrime;

  public:

    BracketedNewton(const PetscInt maxNumIts,const PetscScalar atol);
    ~BracketedNewton();

    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out);
    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out);
    PetscErrorCode setBounds(PetscScalar left,PetscScalar right);
};


class RegulaFalsi : public RootFinder
{
  private:

    PetscScalar _left,_fLeft;
    PetscScalar _right,_fRight;
    PetscScalar _x,_f;

  public:

    RegulaFalsi(const PetscInt maxNumIts,const PetscScalar atol);
    ~RegulaFalsi();

    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,PetscScalar *out);
    PetscErrorCode findRoot(RootFinderContext *obj,const PetscInt ind,const PetscScalar in,PetscScalar *out);
    PetscErrorCode setBounds(PetscScalar left,PetscScalar right);
};


#endif
