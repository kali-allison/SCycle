#ifndef GENFUNCS_HPP_INCLUDED
#define GENFUNCS_HPP_INCLUDED

#include <petscts.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <map>
#include <iostream>

/*
 * This file, and its accompanying cpp file, contains an assortment of
 * small functions which may be useful in a variety of contexts. The goal
 * is to avoid implementing these functions multiple times, to limit the
 * introduction of errors.
 *
 */

using namespace std;

typedef vector<Vec>::iterator it_vec;
typedef vector<Vec>::const_iterator const_it_vec;

// detect if file exists
bool doesFileExist(const string fileName);

// clean up a C++ std library vector of PETSc Vecs
void destroyVector(vector<Vec>& vec);

// clean up a C++ std library map of PETSc Vecs
void destroyVector(map<string,Vec>& vec);

// Print out a vector with 15 significant figures.
void printVec(Vec vec);

// Print out a vectorfrom a DMDA
void printVec(const Vec vec, const DM da);

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsDiff(Vec vec1,Vec vec2);

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsSum(Vec vec1,Vec vec2);

// check that a Vec contains no inf or NaN values
double anyIsnan(const Vec& vec, string str);

// if Vec contains no inf or NaN values, change them to newVal
double changeAnyIsnan(Vec& vec, string str, float newVal);

// write a single vector to file in HDF5 format
PetscErrorCode writeVec_hdf5(Vec vec, const string outFileName, const string group, const string objectName);

// Write vec to the file loc in binary format.
PetscErrorCode writeVec(Vec vec, const string filename);

// Write vec to the file loc in binary format.
PetscErrorCode writeVecAppend(Vec vec, const string filename);

// Write mat to the file loc in binary format.
PetscErrorCode writeMat(Mat mat, const string filename);

// initiate a viewer for binary output
PetscViewer initiateViewer(string filename);
PetscErrorCode appendViewer(PetscViewer& vw, const string filename);

// loop over all viewers in the map vwL and switch then all to append mode
PetscErrorCode appendViewers(map<string,PetscViewer> &vwL, const string filename);
PetscErrorCode io_initiateWriteAppend(map<string,pair<PetscViewer,string>> &vwL, const string key, const Vec& vec, const string filename);

/* Print all entries of 2D DMDA global vector out, including which
 * processor each entry lives on, and the corresponding subscripting indices
 */
PetscErrorCode printf_DM_2d(const Vec gvec, const DM dm);

// vector norms
double computeNormDiff_Mat(const Mat& mat,const Vec& vec1,const Vec& vec2);
double computeNormDiff_2(const Vec& vec1,const Vec& vec2);
double computeNormDiff_L2_scaleL2(const Vec& vec1,const Vec& vec2);
double computeMaxDiff_scaleVec1(const Vec& vec1,const Vec& vec2);
double computeNorm_Mat(const Mat& mat,const Vec& vec);

// functions to make computing the energy much easier
double multVecMatsVec(const Vec& vecL, const Mat& A, const Vec& vecR);
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Vec& vecR);
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Mat& C, const Vec& vecR);
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Mat& C, const Mat& D, const Vec& vecR);
PetscErrorCode multMatsVec(Vec& out, const Mat& A, const Mat& B, const Vec& vecR);
PetscErrorCode multMatsVec(const Mat& A, const Mat& B, Vec& vecR);

// log10(out) = a*log10(vec1) + b*log10(vec2)
// out may be not be the same as vec1 or vec2
PetscErrorCode MyVecLog10AXPBY(Vec& out,const double a, const Vec& vec1, const double b, const Vec& vec2);

// load vector from input file
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName);
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName, bool& fileExists);
PetscErrorCode loadVectorFromInputFile(const string& str,vector<double>& vec);
PetscErrorCode loadVectorFromInputFile(const string& str,vector<int>& vec);
PetscErrorCode loadVectorFromInputFile(const string& str,vector<string>& vec);

// convert vector to string
string vector2str(const vector<double> vec);
string vector2str(const vector<int> vec);
string vector2str(const vector<string> vec);
PetscErrorCode printArray(const PetscScalar * arr,const PetscScalar len);

// helper functions for testing derivatives
double MMS_test(double z);
double MMS_test(double y,double z);

// setting vector in z direction
PetscErrorCode setVec(Vec& vec, const Vec& coord, vector<double>& vals,vector<double>& depths);

// map to vector
PetscErrorCode mapToVec(Vec& vec, double(*func)(double),const Vec& yV);
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double), const Vec& yV, const double t);
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double), const Vec& yV,const Vec& zV, const double t);
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double), const Vec& yV, const Vec& zV);

// map to vector for 1D da
PetscErrorCode mapToVec(Vec& vec, double(*func)(double), const int N, const double dz,const DM da);
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double), const int N, const double dy, const double dz,const DM da);
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double), const int N, const double dy, const double dz,const double t,const DM da);

// repmat for vecs (i.e. vec -> [vec vec]
PetscErrorCode repVec(Vec& out, const Vec& in, const PetscInt n);
PetscErrorCode sepVec(Vec& out, const Vec& in, const PetscInt gIstart, const PetscInt gIend);
PetscErrorCode distributeVec(Vec& out, const Vec& in, const PetscInt gIstart, const PetscInt gIend);


// checkpoint functions
PetscErrorCode loadValueFromCheckpoint(const string outputDir, const string filename, PetscScalar &value);
PetscErrorCode loadValueFromCheckpoint(const string outputDir, const string filename, PetscInt &value);

PetscErrorCode writeASCII(const string outputDir, const string filename, PetscInt var,const string format);
PetscErrorCode writeASCII(const string outputDir, const string filename, PetscScalar var,const string format);



PetscErrorCode initiateWriteASCII(const string outputDir, const string filename, const PetscFileMode mode, PetscViewer &viewer, const string format, PetscScalar var);
PetscErrorCode initiateWriteASCII(const string outputDir, const string filename, const PetscFileMode mode, PetscViewer &viewer, const string format, PetscInt var);

PetscErrorCode initiate_writeVec_hdf5(map<string, pair<PetscViewer, string>> &vwL, const string key, const Vec &vec, const string filename, const PetscFileMode mode);

PetscErrorCode initiate_appendVecToOutput(map<string, pair<PetscViewer, string>> &vwL, const string key, const Vec &vec, const string filename, const PetscFileMode mode);

PetscErrorCode io_initiateWriteAppend(map<string, pair<PetscViewer,string>> &vwL, const string key, const Vec& vec, const string filename);

#endif
