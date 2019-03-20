#include "spmat.hpp"

using namespace std;

// constructor
Spmat::Spmat(size_t rowSize, size_t colSize)
:_rowSize(rowSize),_colSize(colSize)
{}

// copy constructor
Spmat::Spmat(const Spmat& that)
:_mat(that._mat),
 _rowSize(that._rowSize),_colSize(that._colSize)
{}

void Spmat::eye()
{
  _mat.clear(); // ensure matrix is currently empty

  size_t Ii;
  for (Ii=0;Ii<_rowSize;Ii++) // iterate over rows
  {
    _mat[Ii][Ii] = 1.0;
  }
}


void Spmat::print() const
{
  const_row_iter Ii;
  const_col_iter Jj;
  for(Ii=_mat.begin(); Ii!=_mat.end(); Ii++) // iterate over rows
  {
    for( Jj=(Ii->second).begin(); Jj!=(Ii->second).end(); Jj++)
    {
      std::cout << "(" << Ii->first << ",";
      std::cout << Jj->first << "): ";
      std::cout << Jj->second << std::endl;
    }
  }
}

// same as print, but formatted like PETSc
void Spmat::printPetsc() const
{
  const_row_iter Ii;
  const_col_iter Jj;
  for(Ii=_mat.begin(); Ii!=_mat.end(); Ii++) // iterate over rows
  {
    std::cout << "row " << Ii->first << ": ";
    for( Jj=(Ii->second).begin(); Jj!=(Ii->second).end(); Jj++)
    {

      std::cout << "(" << Jj->first << ", ";
      std::cout << Jj->second << ") ";
    }
    std::cout << std::endl;
  }
}


// convert to PETSc style matrix
// assumes matrix has already been created, but not allocated
void Spmat::convert(Mat& petscMat, PetscInt N) const
{
  PetscInt Istart,Iend;

  MatCreate(PETSC_COMM_WORLD,&petscMat);
  MatSetSizes(petscMat,PETSC_DECIDE,PETSC_DECIDE,_rowSize,_colSize);
  MatSetFromOptions(petscMat);
  MatMPIAIJSetPreallocation(petscMat,N,NULL,N,NULL);
  MatSeqAIJSetPreallocation(petscMat,N,NULL);
  MatSetUp(petscMat);
  MatGetOwnershipRange(petscMat,&Istart,&Iend);

  PetscInt row,col;
  double val = 0.0;

  // iterate over nnz entries in this, placing them in PETSc matrix
  const_row_iter Ii;
  const_col_iter Jj;
  for(Ii=_mat.begin(); Ii!=_mat.end(); Ii++) // iterate over rows
  {
    for( Jj=(Ii->second).begin(); Jj!=(Ii->second).end(); Jj++)
    {
      row = Ii->first;
      col = Jj->first;
      val = Jj->second;
      if (row>=Istart && row<Iend) {
        MatSetValues(petscMat,1,&row,1,&col,&val,INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(petscMat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(petscMat,MAT_FINAL_ASSEMBLY);
}

void Spmat::transpose()
{
  Spmat temp(size(1),size(2)); // copy input
  row_iter Ii;
  col_iter Jj;
  for(Ii=_mat.begin(); Ii!=_mat.end(); Ii++) // iterate over rows
  {
    for( Jj=(Ii->second).begin(); Jj!=(Ii->second).end(); Jj++)
    {
      temp(Jj->first,Ii->first,Jj->second);
    }
  }
  *this = temp;
}

// returns the size of the matrix along the specified direction
// (1 = row, 2 = col, as in matlab)
size_t Spmat::size(const int dim) const
{
  if (dim==1) { return _rowSize;}
  else if (dim==2) { return _colSize;}
  else {assert(0<1);} // I really need to learn how to throw exceptions
  return 0;
}

void Spmat::scale(double val)
{
  row_iter Ii;
  col_iter Jj;

  for (Ii=_mat.begin();Ii!=_mat.end();Ii++) // iterate over rows
  {
    for(Jj=Ii->second.begin();Jj!=Ii->second.end();Jj++) // iterate over cols
    {
      Jj->second = Jj->second*val;
    }
  }
}

// performs Kronecker product
Spmat kron(const Spmat& left,const Spmat& right)
{
  size_t leftRowSize = left.size(1);
  size_t leftColSize = left.size(2);
  size_t rightRowSize = right.size(1);
  size_t rightColSize = right.size(2);

  Spmat result(leftRowSize*rightRowSize,leftColSize*rightColSize);

  // iterate over only nnz entries
  Spmat::const_row_iter IiL,IiR;
  Spmat::const_col_iter JjL,JjR;
  double valL=0.0,valR=0.0,val=0.0;
  size_t rowL,colL,rowR,colR,row,col;
  for(IiL=left._mat.begin(); IiL!=left._mat.end(); IiL++) // loop over all values in left
  {
    for( JjL=(IiL->second).begin(); JjL!=(IiL->second).end(); JjL++)
    {
      rowL = IiL->first;
      colL = JjL->first;
      valL = JjL->second;
      if (valL==0) {break;}

      // loop over all values in right
      for(IiR=right._mat.begin(); IiR!=right._mat.end(); IiR++)
      {
        for( JjR=(IiR->second).begin(); JjR!=(IiR->second).end(); JjR++)
        {
          rowR = IiR->first;
          colR = JjR->first;
          valR = JjR->second;

          val = valL*valR;
          row = rowL*rightRowSize + rowR;
          col = colL*rightColSize + colR;
          if (val!=0) { result(row,col,val); }
        }
      }
    }
  }

  return result;
}


// calculate the exact nonzero structure which results from the kronecker outer product of
// left and right
void kronConvert_symbolic(const Spmat& left,const Spmat& right,Mat& mat,PetscInt* d_nnz,PetscInt* o_nnz)
{
  size_t rightRowSize = right.size(1);
  size_t rightColSize = right.size(2);

  PetscInt Istart,Iend; // rows owned by current processor
  PetscInt Jstart,Jend; // cols owned by current processor

  // allocate space for mat
  MatGetOwnershipRange(mat,&Istart,&Iend);
  MatGetOwnershipRangeColumn(mat,&Jstart,&Jend);
  PetscInt m = Iend - Istart;

  for(int ii=0; ii<m; ii++) { d_nnz[ii] = 0; }
  for(int ii=0; ii<m; ii++) { o_nnz[ii] = 0; }


  // iterate over only nnz entries
  Spmat::const_row_iter IiL,IiR;
  Spmat::const_col_iter JjL,JjR;
  double valL=0.0,valR=0.0,val=0.0;
  PetscInt row,col;
  size_t rowL,colL,rowR,colR;
  for(IiL=left._mat.begin(); IiL!=left._mat.end(); IiL++) // loop over all values in left
  {
    for( JjL=(IiL->second).begin(); JjL!=(IiL->second).end(); JjL++)
    {
      rowL = IiL->first;
      colL = JjL->first;
      valL = JjL->second;
      if (valL==0) { continue; }

      // loop over all values in right
      for(IiR=right._mat.begin(); IiR!=right._mat.end(); IiR++)
      {
        for( JjR=(IiR->second).begin(); JjR!=(IiR->second).end(); JjR++)
        {
          rowR = IiR->first;
          colR = JjR->first;
          valR = JjR->second;

          // the new values and coordinates for the product matrix
          val = valL*valR;
          row = rowL*rightRowSize + rowR;
          col = colL*rightColSize + colR;

          PetscInt ii = row - Istart; // array index for d_nnz and o_nnz
          if (val!=0 && row >= Istart && row < Iend && col >= Jstart && col < Jend) { d_nnz[ii]++; }
          if ( (val!=0 && row >= Istart && row < Iend) && (col < Jstart || col >= Jend) ) { o_nnz[ii]++; }
        }
      }
    }
  }
}


// performs Kronecker product and converts to PETSc Mat
void kronConvert(const Spmat& left,const Spmat& right,Mat& mat,PetscInt diag,PetscInt offDiag)
{
  size_t leftRowSize = left.size(1);
  size_t leftColSize = left.size(2);
  size_t rightRowSize = right.size(1);
  size_t rightColSize = right.size(2);

  // create matrix
  MatCreate(PETSC_COMM_WORLD,&mat);
  MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,leftRowSize*rightRowSize,leftColSize*rightColSize);
  MatSetFromOptions(mat);
  MatSetUp(mat);

  // symbolic kronConvert to allocate space for matrix
  PetscInt Istart,Iend; // rows owned by processor
  PetscInt Jstart,Jend; // cols owned by processor
  MatGetOwnershipRange(mat,&Istart,&Iend);
  MatGetOwnershipRangeColumn(mat,&Jstart,&Jend);
  PetscInt m = Iend - Istart;
  PetscInt d_nnz[m], o_nnz[m];
  kronConvert_symbolic(left,right,mat,d_nnz,o_nnz);

  // allocate space for mat
  MatMPIAIJSetPreallocation(mat,diag,d_nnz,offDiag,o_nnz); // arguments diag, offdiag will be ignored
  MatSeqAIJSetPreallocation(mat,diag,d_nnz); // argument diag will be ignored
  MatSetUp(mat);

  // iterate over only nnz entries
  Spmat::const_row_iter IiL,IiR;
  Spmat::const_col_iter JjL,JjR;
  double valL=0.0,valR=0.0,val=0.0;
  PetscInt row,col;
  size_t rowL,colL,rowR,colR;
  for(IiL=left._mat.begin(); IiL!=left._mat.end(); IiL++) // loop over all values in left
  {
    for( JjL=(IiL->second).begin(); JjL!=(IiL->second).end(); JjL++)
    {
      rowL = IiL->first;
      colL = JjL->first;
      valL = JjL->second;
      if (valL==0) { continue; }

      // loop over all values in right
      for(IiR=right._mat.begin(); IiR!=right._mat.end(); IiR++)
      {
        for( JjR=(IiR->second).begin(); JjR!=(IiR->second).end(); JjR++)
        {
          rowR = IiR->first;
          colR = JjR->first;
          valR = JjR->second;

          // the new values and coordinates for the product matrix
          val = valL*valR;
          row = rowL*rightRowSize + rowR;
          col = colL*rightColSize + colR;
          if (val!=0 && row>=Istart && row<Iend) { // if entry is nnz and belongs to processor
            MatSetValues(mat,1,&row,1,&col,&val,INSERT_VALUES);
          }
        }
      }
    }
  }
  MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
}



PetscErrorCode sbp_Spmat(const PetscInt order, const PetscInt N,const PetscScalar scale,
  Spmat& H,Spmat& Hinv,Spmat& D1,Spmat& D1int, Spmat& BS, const std::string type)
{
PetscErrorCode ierr = 0;
//~double startTime = MPI_Wtime();
#if VERBOSE >1
  string funcName = "sbp_Spmat";
  string fileName = "spmat.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif


  if (N == 1) {
    H.eye();
    return ierr;
  }

PetscInt Ii=0;

switch ( order ) {
    case 2:
    {
      H.eye(); H(0,0,0.5); H(N-1,N-1,0.5); H.scale(1/scale);
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
        H.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
      #endif

      D1int(0,0,-1.0*scale);D1int(0,1,scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D1int(Ii,Ii-1,-0.5*scale);
        D1int(Ii,Ii+1,0.5*scale);
      }
      D1int(N-1,N-1,scale);D1int(N-1,N-2,-1*scale); // last row
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1int:\n");CHKERRQ(ierr);
        D1int.printPetsc();
      #endif

      D1 = D1int; // copy D1int's interior

      // fully compatible
      if (type.compare("fullyCompatible")==0 ) {
        BS(0,0,-D1int(0,0)); BS(0,1,-D1int(0,1));
        BS(N-1,N-2,D1int(N-1,N-2)); BS(N-1,N-1,D1int(N-1,N-1));
      }
      else if (type.compare("compatible")==0) {
        BS(0,0,1.5*scale);     BS(0,1,-2.0*scale);     BS(0,2,0.5*scale); // -1* p666 of Mattsson 2010
        BS(N-1,N-3,0.5*scale); BS(N-1,N-2,-2.0*scale); BS(N-1,N-1,1.5*scale);

        D1(0,0,-BS(0,0)); D1(0,1,-BS(0,1)); D1(0,2,-BS(0,2));
        D1(N-1,N-3,BS(N-1,N-3)); D1(N-1,N-2,BS(N-1,N-2)); D1(N-1,N-1,BS(N-1,N-1));
      }
      else { PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n"); assert(0); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBS:\n");CHKERRQ(ierr);
        BS.printPetsc();
      #endif

      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD1:\n");CHKERRQ(ierr);
        D1.printPetsc();
      #endif

      break;
    }
    case 4:
    {
      assert(N>8); // N must be >8 for 4th order SBP

      H.eye();
      H(0,0,17.0/48.0);
      H(1,1,59.0/48.0);
      H(2,2,43.0/48.0);
      H(3,3,49.0/48.0);
      H(N-1,N-1,17.0/48.0);
      H(N-2,N-2,59.0/48.0);
      H(N-3,N-3,43.0/48.0);
      H(N-4,N-4,49.0/48.0);
      H.scale(1/scale);

      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nH:\n");CHKERRQ(ierr);
        H.printPetsc();
      #endif

      for (Ii=0;Ii<N;Ii++) { Hinv(Ii,Ii,1/H(Ii,Ii)); }
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nHinv:\n");CHKERRQ(ierr);
        Hinv.printPetsc();
      #endif

      // interior stencil for 1st derivative, scaled by multiplication with Hinv's values
      for (Ii=4;Ii<N-4;Ii++)
      {
        D1int(Ii,Ii-2,1.0/12.0*Hinv(Ii,Ii));
        D1int(Ii,Ii-1,-2.0/3.0*Hinv(Ii,Ii));
        D1int(Ii,Ii+1,2.0/3.0*Hinv(Ii,Ii));
        D1int(Ii,Ii+2,-1.0/12.0*Hinv(Ii,Ii));
      }

      // closures
      D1int(0,0,-1.0/2.0*Hinv(0,0)); // row 0
      D1int(0,1,59.0/96.0*Hinv(0,0));
      D1int(0,2,-1.0/12.0*Hinv(0,0));
      D1int(0,3,-1.0/32.0*Hinv(0,0));
      D1int(1,0,-59.0/96.0*Hinv(1,1)); // row 1
      D1int(1,2,59.0/96.0*Hinv(1,1));
      D1int(2,0,1.0/12.0*Hinv(2,2)); // row 2
      D1int(2,1,-59.0/96.0*Hinv(2,2));
      D1int(2,3,59.0/96.0*Hinv(2,2));
      D1int(2,4,-1.0/12.0*Hinv(2,2));
      D1int(3,0,1.0/32.0*Hinv(3,3)); // row 3
      D1int(3,2,-59.0/96.0*Hinv(3,3));
      D1int(3,4,2.0/3.0*Hinv(3,3));
      D1int(3,5,-1.0/12.0*Hinv(3,3));

      D1int(N-1,N-1,1.0/2.0*Hinv(N-1,N-1)); // row N-1
      D1int(N-1,N-2,-59.0/96.0*Hinv(N-1,N-1));
      D1int(N-1,N-3,1.0/12.0*Hinv(N-1,N-1));
      D1int(N-1,N-4,1.0/32.0*Hinv(N-1,N-1));
      D1int(N-2,N-1,59.0/96.0*Hinv(N-2,N-2)); // row N-2
      D1int(N-2,N-3,-59.0/96.0*Hinv(N-2,N-2));
      D1int(N-3,N-1,-1.0/12.0*Hinv(N-3,N-3)); // row N-3
      D1int(N-3,N-2,59.0/96.0*Hinv(N-3,N-3));
      D1int(N-3,N-4,-59.0/96.0*Hinv(N-3,N-3));
      D1int(N-3,N-5,1.0/12.0*Hinv(N-3,N-3));
      D1int(N-4,N-1,-1.0/32.0*Hinv(N-4,N-4)); // row N-4
      D1int(N-4,N-3,59.0/96.0*Hinv(N-4,N-4));
      D1int(N-4,N-5,-2.0/3.0*Hinv(N-4,N-4));
      D1int(N-4,N-6,1.0/12.0*Hinv(N-4,N-4));
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1int:\n");CHKERRQ(ierr);
        D1int.printPetsc();
      #endif

      D1 = D1int;

      // fully compatible
      if (type.compare("fullyCompatible")==0 ) {
        BS(0,0,24.0/17.0*scale); BS(0,1,-59.0/34.0*scale);
        BS(0,2,4.0/17.0*scale); BS(0,3,3.0/34.0*scale);
        BS(N-1,N-1,24.0/17.0*scale); BS(N-1,N-2,-59.0/34.0*scale);
        BS(N-1,N-3,4.0/17.0*scale); BS(N-1,N-4,3.0/34.0*scale);
      }
      else if (type.compare("compatible")==0) {
        BS(0,0,11.0/6.0*scale); BS(0,1,-3.0*scale); BS(0,2,1.5*scale); BS(0,3,-1.0/3.0*scale);
        BS(N-1,N-1,11.0/6.0); BS(N-1,N-2,-3.0); BS(N-1,N-3,1.5); BS(N-1,N-4,-1.0/3.0);

        D1(0,0,-BS(0,0)); D1(0,1,-BS(0,1)); D1(0,2,-BS(0,2)); D1(0,3,-BS(0,3));
        D1(N-1,N-4,BS(N-1,N-4)); D1(N-1,N-4,BS(N-1,N-4)); D1(N-1,N-2,BS(N-1,N-2)); D1(N-1,N-1,BS(N-1,N-1));
      }
      #if VERBOBSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nBS:\n");CHKERRQ(ierr);
        BS.printPetsc();
      #endif
      #if VERBOSE > 2
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\nD1:\n");CHKERRQ(ierr);
        D1.printPetsc();
      #endif

      break;
    }

    default:
      SETERRQ(PETSC_COMM_WORLD,1,"order not understood.");
      break;
  }


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  //~_runTime = MPI_Wtime() - startTime;
  return ierr;
}


PetscErrorCode sbp_Spmat2(const PetscInt N,const PetscScalar scale, Spmat& D2, Spmat& C2)
{
PetscErrorCode ierr = 0;
//~double startTime = MPI_Wtime();
#if VERBOSE >1
  string funcName = "sbp_Spmat2";
  string fileName = "spmat.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  assert(N > 2 || N == 1);

  if (N == 1) {
    //~ D2.eye();
    //~ C2.eye();
    return ierr;
  }

  assert(N > 2);

  PetscInt Ii=0;

  D2(0,0,scale*scale); D2(0,1,-2.0*scale*scale); D2(0,2,scale*scale); // first row
      for (Ii=1;Ii<N-1;Ii++) {
        D2(Ii,Ii-1,scale*scale);
        D2(Ii,Ii,-2.0*scale*scale);
        D2(Ii,Ii+1,scale*scale);
      }
  D2(N-1,N-3,scale*scale);D2(N-1,N-2,-2.0*scale*scale);D2(N-1,N-1,scale*scale); // last row

  #if VERBOSE > 2
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nD2:\n");CHKERRQ(ierr);
    D2.printPetsc();
  #endif


  C2.eye();
  C2(0,0,0);
  C2(N-1,N-1,0);
  #if VERBOSE > 2
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nC2:\n");CHKERRQ(ierr);
    C2.printPetsc();
  #endif


#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  //~_runTime = MPI_Wtime() - startTime;
  return ierr;
}


PetscErrorCode sbp_Spmat4(const PetscInt N,const PetscScalar scale,
                Spmat& D3, Spmat& D4, Spmat& C3, Spmat& C4)
{
PetscErrorCode ierr = 0;
#if VERBOSE >1
  string funcName = "sbp_Spmat4";
  string fileName = "spmat.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  assert(N > 8 || N == 1);

  if (N==1) {
    return ierr;
  }

  PetscInt Ii = 0;

  D3(0,0,-1);D3(0,1,3);D3(0,2,-3);D3(0,3,1); // 1st row
  D3(1,0,-1);D3(1,1,3);D3(1,2,-3);D3(1,3,1); // 2nd row
  D3(2,0,-185893.0/301051.0); // 3rd row
  D3(2,1,79000249461.0/54642863857.0);
  D3(2,2,-33235054191.0/54642863857.0);
  D3(2,3,-36887526683.0/54642863857.0);
  D3(2,4,26183621850.0/54642863857.0);
  D3(2,5,-4386.0/181507.0);
  for (Ii=3;Ii<N-4;Ii++)
  {
    D3(Ii,Ii-1,-1.0);
    D3(Ii,Ii,3);
    D3(Ii,Ii+1,-3);
    D3(Ii,Ii+2,1.0);
  }
  D3(N-3,N-1,-D3(2,0));// third to last row
  D3(N-3,N-2,-D3(2,1));
  D3(N-3,N-3,-D3(2,2));
  D3(N-3,N-4,-D3(2,3));
  D3(N-3,N-5,-D3(2,4));
  D3(N-3,N-6,-D3(2,5));
  D3(N-2,N-4,-1);D3(N-2,N-3,3);D3(N-2,N-2,-3);D3(N-2,N-1,1); // 2nd to last row
  D3(N-1,N-4,-1);D3(N-1,N-3,3);D3(N-1,N-2,-3);D3(N-1,N-1,1); // last row

  D4(0,0,1); D4(0,1,-4); D4(0,2,6); D4(0,3,-4); D4(0,4,1); // 1st row
  D4(1,0,1); D4(1,1,-4); D4(1,2,6); D4(1,3,-4); D4(1,4,1); // 1st row
  for (Ii=2;Ii<N-2;Ii++)
  {
    D4(Ii,Ii-2,1);
    D4(Ii,Ii-1,-4);
    D4(Ii,Ii,6);
    D4(Ii,Ii+1,-4);
    D4(Ii,Ii+2,1);
  }
  D4(N-2,N-5,1); D4(N-2,N-4,-4); D4(N-2,N-3,6); D4(N-2,N-2,-4); D4(N-2,N-1,1); // 2nd to last row
  D4(N-1,N-5,1); D4(N-1,N-4,-4); D4(N-1,N-3,6); D4(N-1,N-2,-4); D4(N-1,N-1,1); // last row

  C3.eye();
  C3(0,0,0);
  C3(1,1,0);
  C3(2,2,163928591571.0/53268010936.0);
  C3(3,3,189284.0/185893.0);
  C3(N-5,N-5,C3(3,3));
  C3(N-4,N-4,0);
  C3(N-3,N-3,C3(2,2));
  C3(N-2,N-2,0);
  C3(N-1,N-1,0);

  C4.eye();
  C4(0,0,0);
  C4(1,1,0);
  C4(2,2,1644330.0/301051.0);
  C4(3,3,156114.0/181507.0);
  C4(N-4,N-4,C4(3,3));
  C4(N-3,N-3,C4(2,2));
  C4(N-2,N-2,0);
  C4(N-1,N-1,0);

#if VERBOSE >1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending function %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif
  return ierr;
}
