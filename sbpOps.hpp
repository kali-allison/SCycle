#ifndef SBPOPS_DDD_H_INCLUDED
#define SBPOPS_DDD_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <assert.h>
#include "domain.hpp"
#include "debuggingFuncs.hpp"
#include "spmat.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"

using namespace std;


/*
 * Contains linear SBP operators. Currently supported options:
 *   compatible
 *   fully compatible (untested)
 * Coming soon:
 *   matrix-free compatible operators!
 *
 */


/*
 * Note: PETSc's ability to count matrix creation/destructions is off.
 * For every MATAXPY, the number of destructions increments by 1 more than
 * the number of creations. Thus, after satBoundaries() the number will
 * be off by 4.
 */

class SbpOps
{

  private:
      // disable default copy constructor and assignment operator
    SbpOps(const SbpOps & that);
    SbpOps& operator=( const SbpOps& rhs );

  public:

    const PetscInt    _order,_Ny,_Nz;
    const PetscReal   _dy,_dz;
    PetscScalar      *_muArr;
    Mat              *_mu;

    SbpOps_c   _internalSBP;
    PetscScalar _alphaDy;

    Mat _A;

    SbpOps(Domain&D,PetscScalar& muArr,Mat& mu);
    ~SbpOps();

    // create the vector rhs out of the boundary conditions (_bc*)
    PetscErrorCode setRhs(Vec&rhs,Vec &_bcF,Vec &_bcR,Vec &_bcS,Vec &_bcD);

    // read/write commands
    PetscErrorCode loadOps(const std::string inputDir);
    PetscErrorCode writeOps(const std::string outputDir);


    //~// functions to compute various derivatives of input vectors (this
    //~// will allow the matrix-free version of these operators to present
    //~// the exact same interface to the as the matrix version).
    PetscErrorCode Dy(const Vec &in, Vec &out); // out = Dy * in
    PetscErrorCode muxDy(const Vec &in, Vec &out); // out = mu * Dy * in
    PetscErrorCode Dyxmu(const Vec &in, Vec &out); // out = Dy * mu * in
    PetscErrorCode Dz(const Vec &in, Vec &out); // out = Dz * in
    PetscErrorCode muxDz(const Vec &in, Vec &out); // out = mu * Dz * in
    PetscErrorCode Dzxmu(const Vec &in, Vec &out); // out = Dz * mu * in


    PetscErrorCode H(const Vec &in, Vec &out); // out = H * in
    PetscErrorCode HBzx2mu(const Vec &in, Vec &out); // out = H * Iy_Bz * 2 * mu * in
    PetscErrorCode By(const Vec &in, Vec &out); // out = By_Iz * in
    PetscErrorCode e0y(const Vec &in, Vec &out); // out = eNy_Iz * in
    PetscErrorCode eNy(const Vec &in, Vec &out); // out = eNy_Iz * in

};


#endif
