#ifndef SBPOPS_M_CONSTGRIDSPACING_H_INCLUDED
#define SBPOPS_M_CONSTGRIDSPACING_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <assert.h>
#include "spmat.hpp"
#include "sbpOps.hpp"

using namespace std;


/*
 * Constructs fully compatible linear SBP operators
 * with SAT boundary conditions:
 *
 *   D2^(mu) = -H^(-1) [ -D1^T H mu D1 - R + mu BD1]
 *
 */

struct TempMats_m_constGrid
{
  const PetscInt    _order,_Ny,_Nz;
  const PetscReal   _dy,_dz;

  Spmat _Hy,_Hyinv,_D1y,_D1yint,_BSy,_Iy;
  Spmat _Hz,_Hzinv,_D1z,_D1zint,_BSz,_Iz;

  TempMats_m_constGrid(const PetscInt order,const PetscInt Ny,const PetscScalar dy,const PetscInt Nz,const PetscScalar dz,const string type);
  ~TempMats_m_constGrid();

private:

  // disable default copy constructor and assignment operator
  TempMats_m_constGrid(const TempMats_m_constGrid & that);
  TempMats_m_constGrid& operator=( const TempMats_m_constGrid& rhs );
};


class SbpOps_m_constGrid : public SbpOps
{
  public:

    const PetscInt      _order,_Ny,_Nz;
    PetscScalar         _dy,_dz;
    Vec                 _muVec; // variable coefficient
    Mat                 _mu; // matrix of coefficient
    std::string         _bcRType,_bcTType,_bcLType,_bcBType; // options: "Dirichlet", "Traction"
    double              _runTime;
    string              _compatibilityType; // "fullyCompatible" (S = D),  or "compatible" (S =/= D)
    string              _D2type; // "yz", "y", or "z"
    int                 _multByH; // (default: 0) 1 if yes, 0 if no
    int                 _deleteMats; // (default: 0) 1 if yes, 0 if no

    // enforce boundary conditions
    Mat    _AR,_AT,_AL,_AB,_rhsL,_rhsR,_rhsT,_rhsB; // pointer to currently used matrices
    Mat    _AR_N,_AT_N,_AL_N,_AB_N,_rhsL_N,_rhsR_N,_rhsT_N,_rhsB_N; // for Neumann conditions
    Mat    _AR_D,_AT_D,_AL_D,_AB_D,_rhsL_D,_rhsR_D,_rhsT_D,_rhsB_D; // for Dirichlet conditions

    // boundary condition penalty weights
    PetscScalar _alphaT,_alphaDy,_alphaDz,_beta;
    PetscScalar _h11y,_h11z;

    // various SBP factors
    Mat _A;
    Mat _Dy_Iz, _Iy_Dz;
    Mat _D2; // Dyy + Dzz w/out BCs
    Mat _Hinv,_H,_Hyinv_Iz,_Iy_Hzinv,_Hy_Iz,_Iy_Hz;
    Mat _e0y_Iz,_eNy_Iz,_Iy_e0z,_Iy_eNz;
    Mat _E0y_Iz,_ENy_Iz,_Iy_E0z,_Iy_ENz;
    Mat _muxBySy_IzT,_Iy_muxBzSzT;
    Mat _BSy_Iz, _Iy_BSz;


    SbpOps_m_constGrid(const int order,const PetscInt Ny,const PetscInt Nz,const PetscScalar Ly, const PetscScalar Lz,Vec& muVec);
    ~SbpOps_m_constGrid();

    PetscErrorCode setBCTypes(std::string bcR, std::string bcT, std::string bcL, std::string bcB);
    PetscErrorCode setGrid(Vec* y, Vec* z);
    PetscErrorCode setMultiplyByH(const int multByH);
    PetscErrorCode setLaplaceType(const string type); // "y", "z", or "yz"
    PetscErrorCode setCompatibilityType(const string type); // "fullyCompatible" or "compatible"
    PetscErrorCode setDeleteIntermediateFields(const int deleteMats);
    PetscErrorCode changeBCTypes(string bcR, string bcT, string bcL, string bcB);
    PetscErrorCode computeMatrices(); // matrices not constructed until now


    // create the vector rhs out of the boundary conditions (_bc*)
    PetscErrorCode setRhs(Vec&rhs,Vec &bcL,Vec &bcR,Vec &bcT,Vec &bcB);

    // read/write commands
    PetscErrorCode loadOps(const std::string inputDir);
    PetscErrorCode writeOps(const std::string outputDir);


    // allow variable coefficient to change
    PetscErrorCode updateVarCoeff(const Vec& coeff);


    // functions to compute various derivatives of input vectors (this
    // will allow the matrix-free version of these operators to present
    // the exact same interface to the as the matrix version).
    PetscErrorCode Dy(const Vec &in, Vec &out); // out = Dy * in
    PetscErrorCode muxDy(const Vec &in, Vec &out); // out = mu * Dy * in
    PetscErrorCode Dyxmu(const Vec &in, Vec &out); // out = Dy * mu * in
    PetscErrorCode Dz(const Vec &in, Vec &out); // out = Dz * in
    PetscErrorCode muxDz(const Vec &in, Vec &out); // out = mu * Dz * in
    PetscErrorCode Dzxmu(const Vec &in, Vec &out); // out = Dz * mu * in

    PetscErrorCode H(const Vec &in, Vec &out); // out = H * in
    PetscErrorCode Hinv(const Vec &in, Vec &out); // out = H * in
    PetscErrorCode Hyinvxe0y(const Vec &in, Vec &out); // out = Hy^-1 * e0y * in
    PetscErrorCode HyinvxeNy(const Vec &in, Vec &out); // out = Hy^-1 * eNy * in
    PetscErrorCode HyinvxE0y(const Vec &in, Vec &out); // out = Hy^-1 * E0y * in
    PetscErrorCode HyinvxENy(const Vec &in, Vec &out); // out = Hy^-1 * ENy * in
    PetscErrorCode HzinvxE0z(const Vec &in, Vec &out); // out = Hz^-1 * e0z * in
    PetscErrorCode HzinvxENz(const Vec &in, Vec &out); // out = Hz^-1 * eNz * in

    // return penalty weight h11 (the first element of the H matrix)
    PetscErrorCode geth11(PetscScalar &h11y, PetscScalar &h11z);

    // allow access to matrices
    PetscErrorCode getCoordTrans(Mat&J, Mat& Jinv,Mat& qy,Mat& rz, Mat& yq, Mat& zr);
    PetscErrorCode getA(Mat &mat);
    PetscErrorCode getH(Mat &mat);
    PetscErrorCode getDs(Mat &Dy,Mat &Dz);
    PetscErrorCode getMus(Mat &mu,Mat &muqy,Mat &murz);
    PetscErrorCode getEs(Mat& E0y_Iz,Mat& ENy_Iz,Mat& Iy_E0z,Mat& Iy_ENz);
    PetscErrorCode getes(Mat& e0y_Iz,Mat& eNy_Iz,Mat& Iy_e0z,Mat& Iy_eNz);
    PetscErrorCode getHs(Mat& Hy_Iz,Mat& Iy_Hz);
    PetscErrorCode getHinvs(Mat& Hyinv_Iz,Mat& Iy_Hzinv);

  private:
    // disable default copy constructor and assignment operator
    SbpOps_m_constGrid(const SbpOps_m_constGrid & that);
    SbpOps_m_constGrid& operator=( const SbpOps_m_constGrid& rhs );

    PetscErrorCode setMatsToNull();

    // functions to construct various matrices
    PetscErrorCode constructMu(Vec& muVec);
    PetscErrorCode constructEs(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructes(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructBs(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructHs(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructH(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructHinv(const TempMats_m_constGrid& tempMats);
    PetscErrorCode construct1stDerivs(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructA(const TempMats_m_constGrid& tempMats);
    PetscErrorCode updateA_BCs();
    PetscErrorCode updateA_BCs(TempMats_m_constGrid& tempMats);
    PetscErrorCode updateBCMats();
    PetscErrorCode constructDyymu(const TempMats_m_constGrid& tempMats, Mat &Dyymu);
    PetscErrorCode constructDzzmu(const TempMats_m_constGrid& tempMats, Mat &Dzzmu);
    PetscErrorCode constructD2(const TempMats_m_constGrid& tempMats);
    PetscErrorCode constructRymu(const TempMats_m_constGrid& tempMats,Mat &Rymu);
    PetscErrorCode constructRzmu(const TempMats_m_constGrid& tempMats,Mat &Rzmu);
    PetscErrorCode deleteIntermediateFields();

    PetscErrorCode constructBC_Dirichlet(Mat& out,PetscScalar alphaD,Mat& mu,Mat& Hinv,Mat& BD1T,Mat& E,MatReuse scall);
    PetscErrorCode constructBC_Neumann(Mat& out, Mat& Hinv, PetscScalar Bfact, Mat& E, Mat& mu, Mat& D1,MatReuse scall); // for A
    PetscErrorCode constructBC_Neumann(Mat& out, Mat& Hinv, PetscScalar Bfact, Mat& e, MatReuse scall); // for rhs
    PetscErrorCode constructBCMats();
};

#endif
