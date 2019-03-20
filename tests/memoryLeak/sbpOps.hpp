#ifndef SBPOPS_H_INCLUDED
#define SBPOPS_H_INCLUDED

#include <petscksp.h>
#include <string>

#include "genFuncs.hpp"

using namespace std;


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
 *
 * To create a member of this class, several functions need to be called called to set up
 * the context:
 *
 * Example usage:
 *    SbpOps *sbp = new SbpOps_fc(order,Ny,Nz,Ly,Lz,coeff); // Sample constructor call. Note that this does NOT compute any matrices.
 *    setBCTypes("Dirichlet","Neumann","Dirichlet","Neumann"); // sets what types of boundaries will be enforced
 *    computeMatrices(); // this call actually creates the matrices
 *
 * Optional additional functions, which must be called prior to computeMatrices:
 *    setGrid(y,z); // y- and z- coordinate vectors
 *    setLaplaceType("yz"); // (default: "yz") determines whether to
 *        // construct d/dy(coeff * d/dy) + d/dz(coeff * d/dz) or only 1 or the other term
 *    setMultiplyByH(1); // (default: 0) 1 for yes, 0 for no
 *    setDeleteIntermediateFields(1); // (default: 0) removes intermediate matrices and old BC matrices to save on memory usage
 *
 * It is possible to change the type of the boundary conditions:
 * changeBCTypes("Neumann","Neumann","Neumann","Neumann");  // if you want to switch
 *              // which type of BC to enforce (will compute the new matrices for you if necessary)
 *
 */



class SbpOps
{

  public:

    SbpOps(){};
    virtual ~SbpOps(){};


    // set context options
    virtual PetscErrorCode setBCTypes(string bcR, string bcT, string bcL, string bcB) = 0;
    virtual PetscErrorCode setGrid(Vec* y, Vec* z) = 0;
    virtual PetscErrorCode setMultiplyByH(const int multByH) = 0;
    virtual PetscErrorCode setLaplaceType(const string type) = 0; // "y", "z", or "yz"
    virtual PetscErrorCode setDeleteIntermediateFields(const int deleteMats) = 0;
    virtual PetscErrorCode changeBCTypes(string bcR, string bcT, string bcL, string bcB) = 0;
    virtual PetscErrorCode computeMatrices() = 0; // matrices not constructed until now


    // allow variable coefficient to change
    virtual PetscErrorCode updateVarCoeff(const Vec& coeff) = 0;

    // return penalty weight h11 (the first element of the H matrix)
    virtual PetscErrorCode geth11(PetscScalar &h11y, PetscScalar &h11z) = 0;

    // allow access to matrices
    virtual PetscErrorCode getA(Mat &mat) = 0;
    virtual PetscErrorCode getH(Mat &mat) = 0;
    virtual PetscErrorCode getDs(Mat &Dy,Mat &Dz) = 0;
    virtual PetscErrorCode getMus(Mat &mu,Mat &muqy,Mat &murz) = 0;
    virtual PetscErrorCode getEs(Mat& E0y_Iz,Mat& ENy_Iz,Mat& Iy_E0z,Mat& Iy_ENz) = 0;
    virtual PetscErrorCode getes(Mat& e0y_Iz,Mat& eNy_Iz,Mat& Iy_e0z,Mat& Iy_eNz) = 0;
    virtual PetscErrorCode getHs(Mat& Hy_Iz,Mat& Iy_Hz) = 0;
    virtual PetscErrorCode getHinvs(Mat& Hyinv_Iz,Mat& Iy_Hzinv) = 0;

    // various pieces of the Jacobian of a coordinate transformation
    virtual PetscErrorCode getCoordTrans(Mat&J, Mat& Jinv,Mat& qy,Mat& rz, Mat& yq, Mat& zr) = 0;

    // create the vector rhs out of the boundary conditions
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
