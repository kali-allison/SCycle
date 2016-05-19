#ifndef SBPOPS_H_INCLUDED
#define SBPOPS_H_INCLUDED

#include <petscksp.h>
#include "domain.hpp"


/*
 * This is an abstract that defines an interface for SBP operators.
 *
 * Current supported options:
 * matrix (m)/stencil (s)     order        compatible (c)/fully compatible (fc)
 *         m                   2                 c
 *         m                   4                 c
 *         m                   2                 fc
 *         m                   4                 fc
 *
 * Coming soon: matrix-free versions!
 */

class SbpOps
{

  public:
    SbpOps(){};
    virtual ~SbpOps(){};

    // don't want _A to belong to the abstract class
    virtual PetscErrorCode getA(Mat &mat) = 0;

    // create the vector rhs out of the boundary conditions (_bc*)
    virtual PetscErrorCode setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcT,Vec &_bcB) = 0;

    // functions to compute various derivatives of input vectors
    virtual PetscErrorCode Dy(const Vec &in, Vec &out) = 0; // out = Dy * in
    virtual PetscErrorCode muxDy(const Vec &in, Vec &out) = 0; // out = mu * Dy * in
    virtual PetscErrorCode Dyxmu(const Vec &in, Vec &out) = 0; // out = Dy * mu * in
    virtual PetscErrorCode Dz(const Vec &in, Vec &out) = 0; // out = Dz * in
    virtual PetscErrorCode muxDz(const Vec &in, Vec &out) = 0; // out = mu * Dz * in
    virtual PetscErrorCode Dzxmu(const Vec &in, Vec &out) = 0; // out = Dz * mu * in


    virtual PetscErrorCode H(const Vec &in, Vec &out) = 0; // out = H * in
    virtual PetscErrorCode Hyinvxe0y(const Vec &in, Vec &out) = 0; // out = Hy^-1 * e0y * in
    virtual PetscErrorCode HyinvxeNy(const Vec &in, Vec &out) = 0; // out = Hy^-1 * eNy * in
    virtual PetscErrorCode HyinvxE0y(const Vec &in, Vec &out) = 0; // out = Hy^-1 * E0y * in
    virtual PetscErrorCode HyinvxENy(const Vec &in, Vec &out) = 0; // out = Hy^-1 * ENy * in
    virtual PetscErrorCode HzinvxE0z(const Vec &in, Vec &out) = 0; // out = Hz^-1 * E0z * in
    virtual PetscErrorCode HzinvxENz(const Vec &in, Vec &out) = 0; // out = Hz^-1 * ENz * in
};


#endif
