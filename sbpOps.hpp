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
    // temporarily available for energy balance calculations
    //~ Mat _mu,_H,_Hy_Iz,_Iy_Hz,_e0y_Iz,_eNy_Iz,_E0y_Iz,_ENy_Iz,_Iy_E0z,_Iy_ENz;
    //~ Mat _Ry,_Rz,_By_Iz,_Iy_Bz;

    SbpOps(){};
    virtual ~SbpOps(){};

    // don't want _A to belong to the abstract class
    virtual PetscErrorCode getA(Mat &mat) = 0;
    virtual PetscErrorCode getH(Mat &mat) = 0;

    // temporarily available for energy balance
    virtual PetscErrorCode getMus(Mat &muqy,Mat &murz) = 0;
    virtual PetscErrorCode getR(Mat& Ry, Mat& Rz) = 0;
    virtual PetscErrorCode getEs(Mat& E0y_Iz,Mat& ENy_Iz,Mat& Iy_E0z,Mat& Iy_ENz) = 0;
    virtual PetscErrorCode getes(Mat& e0y_Iz,Mat& eNy_Iz,Mat& Iy_e0z,Mat& Iy_eNz) = 0;
    virtual PetscErrorCode getBs(Mat& By_Iz,Mat& Iy_Bz) = 0;
    virtual PetscErrorCode getHs(Mat& Hy_Iz,Mat& Iy_Hz) = 0;
    virtual PetscErrorCode getCoordTrans(Mat& qy,Mat& rz, Mat& yq, Mat& zr) = 0;

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
    virtual PetscErrorCode Hinv(const Vec &in, Vec &out) = 0; // out = H^-1 * in
    virtual PetscErrorCode Hyinvxe0y(const Vec &in, Vec &out) = 0; // out = Hy^-1 * e0y * in
    virtual PetscErrorCode HyinvxeNy(const Vec &in, Vec &out) = 0; // out = Hy^-1 * eNy * in
    virtual PetscErrorCode HyinvxE0y(const Vec &in, Vec &out) = 0; // out = Hy^-1 * E0y * in
    virtual PetscErrorCode HyinvxENy(const Vec &in, Vec &out) = 0; // out = Hy^-1 * ENy * in
    virtual PetscErrorCode HzinvxE0z(const Vec &in, Vec &out) = 0; // out = Hz^-1 * E0z * in
    virtual PetscErrorCode HzinvxENz(const Vec &in, Vec &out) = 0; // out = Hz^-1 * ENz * in

    // file I/O
    virtual PetscErrorCode writeOps(const std::string outputDir) = 0;
};


#endif
