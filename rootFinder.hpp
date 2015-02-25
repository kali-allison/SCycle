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
    virtual PetscErrorCode setBounds(PetscScalar left,PetscScalar right) = 0;

    PetscInt getNumIts() const;
};



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
    PetscErrorCode setBounds(PetscScalar left,PetscScalar right);
};




#endif
