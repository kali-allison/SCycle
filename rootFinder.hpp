#ifndef ROOTFINDER_HPP_INCLUDED
#define ROOTFINDER_HPP_INCLUDED


#include <petscts.h>
#include <string>
#include <assert.h>
//~#include "userContext.h"
#include "fault.hpp"


class RootFinder
{
  protected:
    PetscInt _numIts,_maxNumIts;
    PetscScalar _atol;


    // disable default copy constructor and assignment operator
    RootFinder(const RootFinder & that);
    RootFinder& operator=(const RootFinder& rhs);

  public:

    //~typedef fault model;

    RootFinder(const PetscInt maxNumIts,const PetscScalar atol);
    virtual ~RootFinder();

    virtual PetscErrorCode findRoot(Fault *obj,const PetscInt ind,PetscScalar *out) = 0;
    virtual PetscErrorCode setBounds(PetscScalar left,PetscScalar right) = 0;

    PetscInt getNumIts() const;

};



class Bisect : public RootFinder
{
  private:
    PetscScalar _atol;

    PetscScalar _left,_fLeft;
    PetscScalar _right,_fRight;
    PetscScalar _mid,_fMid;

    // disable default copy constructor and assignment operator
    //~Bisect(const Bisect & that);
    //~Bisect& operator=(const Bisect& rhs);

  public:

    //~typedef fault<Bisect> model;

    Bisect(const PetscInt maxNumIts,const PetscScalar atol);
    ~Bisect();

    PetscErrorCode findRoot(Fault *obj,const PetscInt ind,PetscScalar *out);

    PetscErrorCode setBounds(PetscScalar left,PetscScalar right);
};




#endif
