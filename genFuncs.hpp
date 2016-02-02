#ifndef GENFUNCS_HPP_INCLUDED
#define GENFUNCS_HPP_INCLUDED

#include <petscts.h>
#include <petscdmda.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <iostream>

using namespace std;

// Print out a vector with 15 significant figures.
void printVec(Vec vec);

// Print out a vectorfrom a DMDA
void printVec(const Vec vec, const DM da);

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsDiff(Vec vec1,Vec vec2);

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsSum(Vec vec1,Vec vec2);

// Write vec to the file loc in binary format.
PetscErrorCode writeVec(Vec vec,const char * loc);

// Write mat to the file loc in binary format.
PetscErrorCode writeMat(Mat mat,const char * loc);

// Print all entries of 2D DMDA global vector out, including which
// processor each entry lives on, and the corresponding subscripting
// indices.
PetscErrorCode printf_DM_2d(const Vec gvec, const DM dm);

// vector norms
double computeNormDiff_Mat(const Mat& mat,const Vec& vec1,const Vec& vec2);
double computeNormDiff_2(const Vec& vec1,const Vec& vec2);
double computeNorm_Mat(const Mat& mat,const Vec& vec);


PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName);
PetscErrorCode loadVectorFromInputFile(const string& str,vector<double>& vec);


PetscErrorCode printArray(const PetscScalar * arr,const PetscScalar len);


// helper functions for testing derivatives
double MMS_test(double y,double z);


// MMS functions (acting on scalars)
double MMS_uA(double y,double z, double t);
double MMS_uA_y(double y,double z, double t);
double MMS_uA_yy(const double y,const double z,const double t);
double MMS_uA_z(const double y,const double z,const double t);
double MMS_uA_zz(const double y,const double z,const double t);
double MMS_uA_t(const double y,const double z,const double t);

double MMS_mu(const double y,const double z);
double MMS_mu_y(const double y,const double z);
double MMS_mu_z(const double y,const double z);

double MMS_sigmaxy(const double y,const double z,const double t);
double MMS_sigmaxz(const double y,const double z,const double t);

double MMS_visc(const double y,const double z);
double MMS_inVisc(const double y,const double z);
double MMS_invVisc_y(const double y,const double z);
double MMS_invVisc_z(const double y,const double z);

double MMS_gxy(const double y,const double z,const double t);
double MMS_gxy_y(const double y,const double z,const double t);
double MMS_gxy_t(const double y,const double z,const double t);

double MMS_gxz(const double y,const double z,const double t);
double MMS_gxz_z(const double y,const double z,const double t);
double MMS_gxz_t(const double y,const double z,const double t);


double MMS_uSource(const double y,const double z,const double t);
double MMS_gSource(const double y,const double z,const double t);

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const int N, const double dy, const double dz, const double t);

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const int N, const double dy, const double dz);

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const int N, const double dy, const double dz,const DM da);

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const int N, const double dy, const double dz,const double t,const DM da);



#endif
