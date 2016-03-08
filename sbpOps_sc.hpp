#ifndef SBPOPS_SC_H_INCLUDED
#define SBPOPS_SC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <assert.h>
#include "domain.hpp"
#include "debuggingFuncs.hpp"
#include "spmat.hpp"
#include "sbpOps.hpp"

using namespace std;


/*
 * Constructs matrix-free compatible linear SBP operators (NOT fully
 * compatible) with SAT boundary conditions:
 *
 *   D2^(mu) = -H^(-1) [ -D1^T H mu D1 - R + mu BS]
 *
 * where S is a 1st derivative operator only on the boundaries, and has
 * a higher order of accuracy than D1 there.
 *
 */



/*
 * This class contains the matrix-free summation-by-parts (SBP) matrices needed at
 * each time step to (1) compute the displacement in the medium, and (2) the
 * shear stress from the displacement. (1) Is accomplished by first
 * forming the vector rhs, which contains the boundary conditions, using
 * the function setRhs, and then using the matrix _A to compute the
 * displacement vector (uhat) from the linear equation A uhat = rhs.
 */

class SbpOps_sc
{

  private:

    // disable default copy constructor and assignment operator
    SbpOps_sc(const SbpOps_sc & that);
    SbpOps_sc& operator=( const SbpOps_sc& rhs );

  public:

    const PetscInt    _order,_Ny,_Nz;
    const PetscReal   _dy,_dz;
    PetscScalar      *_muArr;
    Mat              *_mu;
    Vec               _muVecP;

    Mat               _A;

    // DMDA dimensions
    DM _da;
    PetscInt _yS,_yE,_zS,_zE; // for for loops below

    double _runTime;

    // map boundary conditions to rhs vector
    string _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction

    // boundary conditions
    PetscScalar _alphaT,_alphaDy,_alphaDz,_beta; // penalty terms for traction and displacement respectively

    // helper functions for SBP operators
    PetscErrorCode Hy(const Vec &in, Vec &out); // out = Hy * in
    PetscErrorCode Hz(const Vec &in, Vec &out); // out = Hz * in
    PetscErrorCode Hinvy(const Vec &in, Vec &out); // out = Hinvy * in
    PetscErrorCode Hinvz(const Vec &in, Vec &out); // out = Hinvz * in
    PetscErrorCode Dmuyy(const Vec &in, Vec &out); // out = Dmuyy * in
    PetscErrorCode Dmuzz(const Vec &in, Vec &out); // out = Dmuzz * in

  public:

    //~SbpOps_sc(Domain&D,PetscScalar& muArr,Mat& mu);
    SbpOps_sc(Domain&D,PetscScalar& muArr,Mat& mu,string bcT,string bcR,string bcB, string bcL, string type);
    ~SbpOps_sc();

    // create the vector rhs out of the boundary conditions (_bc*)
    PetscErrorCode setRhs(Vec&rhs,Vec &bcL,Vec &bcR,Vec &bcT,Vec &bcB);

    PetscErrorCode getA(Mat &mat);

    // mat-based versions
    PetscErrorCode matDy(const Vec &in, Vec &out); // out = Dy * in


    // functions to compute various derivatives of input vectors (this
    // allows the matrix-free version of these operators to present
    // the exact same interface to the as the matrix version).
    PetscErrorCode Dy(const Vec &in, Vec &out); // out = Dy * in
    PetscErrorCode muxDy(const Vec &in, Vec &out); // out = mu * Dy * in
    PetscErrorCode Dyxmu(const Vec &in, Vec &out); // out = Dy * mu * in
    PetscErrorCode Dz(const Vec &in, Vec &out); // out = Dz * in
    PetscErrorCode muxDz(const Vec &in, Vec &out); // out = mu * Dz * in
    PetscErrorCode Dzxmu(const Vec &in, Vec &out); // out = Dz * mu * in

    PetscErrorCode H(const Vec &in, Vec &out); // out = H * in
    PetscErrorCode Hyinvxe0y(const Vec &in, Vec &out); // out = Hy^-1 * e0y * in
    PetscErrorCode HyinvxeNy(const Vec &in, Vec &out); // out = Hy^-1 * eNy * in
    PetscErrorCode HyinvxE0y(const Vec &in, Vec &out); // out = Hy^-1 * E0y * in
    PetscErrorCode HyinvxENy(const Vec &in, Vec &out); // out = Hy^-1 * ENy * in
    PetscErrorCode HzinvxE0z(const Vec &in, Vec &out); // out = Hz^-1 * e0z * in
    PetscErrorCode HzinvxENz(const Vec &in, Vec &out); // out = Hz^-1 * eNz * in

};

#endif
