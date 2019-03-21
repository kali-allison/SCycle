#ifndef SPMAT_H_INCLUDED
#define SPMAT_H_INCLUDED

#include <cstdlib>
#include <map>
#include <vector>
#include <iostream>
#include <assert.h>
#include <petscts.h>
#include <petscdmda.h>
#include <string>

/*
 * Small class for sparse matrices, supporting a very limited set of
 * operations.
 * Based on code from http://www.cplusplus.com/forum/general/8352/
 */

class Spmat
{
public:
  typedef std::map<size_t, std::map<size_t, double> > mat_t;
  typedef mat_t::iterator row_iter;
  typedef mat_t::const_iterator const_row_iter;
  typedef std::map<size_t, double> col_t;
  typedef col_t::iterator col_iter;
  typedef col_t::const_iterator const_col_iter;

  Spmat(size_t rowSize, size_t colSize); // constructor
  Spmat(const Spmat &that); // copy constructor
  void eye(); // initialize matrix as identity matrix

  size_t size(const int dim) const; // (1 = row, 2 = col, as in matlab)
  void scale(double val); // multiply every entry in the matrix by val
  void transpose();
  void print() const;
  void printPetsc() const;

  // convert to PETSc style matrix
  void convert(Mat& petscMat, PetscInt N) const;

  friend Spmat kron(const Spmat& left,const Spmat& right);
  friend void kronConvert(const Spmat& left,const Spmat& right,Mat& mat,PetscInt diag,PetscInt offDiag);
  friend void kronConvert_symbolic(const Spmat& left,const Spmat& right,Mat& mat,PetscInt* d_nnz,PetscInt* o_nnz);


  // inline functions are defined in the header file

  // insert a value into the matrix
  inline void operator()(size_t row, size_t col,double val)
  {
    assert(row<_rowSize && col<_colSize);
    _mat[row][col] = val;
  };

  // return value at (row,col) from the matrix
  inline double operator()(size_t row, size_t col) const
  {
    assert(row<_rowSize && col<_colSize);

    const_row_iter it = _mat.find(row);
    if ( it == _mat.end() ) { return 0.0; }
    const_col_iter temp = it->second.find(col);
    if ( temp == it->second.end() ) { return 0.0; }
    return it->second.find(col)->second;
    //~return _mat[row][col]; // this creates new entry if it doesn't exist
  };

  //~private:
  protected:
    mat_t _mat;
    size_t _rowSize;
    size_t _colSize;

};

// functions to construct 1D sbp operators
PetscErrorCode sbp_Spmat(const PetscInt order,const PetscInt N,const PetscScalar scale,
                        Spmat& H,Spmat& Hinv,Spmat& D1,Spmat& D1int, Spmat& S, const std::string type);
PetscErrorCode sbp_Spmat2(const PetscInt N,const PetscScalar scale,Spmat& D2,Spmat& C2);
PetscErrorCode sbp_Spmat4(const PetscInt N,const PetscScalar scale,
                         Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4);

#endif
