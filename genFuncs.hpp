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



#endif
