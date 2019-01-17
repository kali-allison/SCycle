#include "spmat.hpp"

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

  // naive implementation loops over all possible entries ( very slow)
  //~double val = 0.0;
  //~size_t Ii,Jj;
  //~for (Ii=0;Ii<leftRowSize*rightRowSize;Ii++)
  //~{
    //~for (Jj=0;Jj<leftColSize*rightColSize;Jj++)
    //~{
      //~val = left(Ii/rightRowSize,Jj/rightColSize);
      //~val = val * right(Ii%rightRowSize,Jj%rightColSize);
      //~if (val!=0) { result(Ii,Jj,val); }
    //~}
  //~}

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



// performs Kronecker product and converts to PETSc Mat
void kronConvert(const Spmat& left,const Spmat& right,Mat& mat,PetscInt diag,PetscInt offDiag)
{
  size_t leftRowSize = left.size(1);
  size_t leftColSize = left.size(2);
  size_t rightRowSize = right.size(1);
  size_t rightColSize = right.size(2);


  PetscInt Istart,Iend;

  // allocate space for mat
  MatCreate(PETSC_COMM_WORLD,&mat);
  MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,leftRowSize*rightRowSize,leftColSize*rightColSize);
  MatSetFromOptions(mat);
  MatMPIAIJSetPreallocation(mat,diag,NULL,offDiag,NULL);
  MatSeqAIJSetPreallocation(mat,diag+offDiag,NULL);
  MatSetUp(mat);
  MatGetOwnershipRange(mat,&Istart,&Iend);

   // NOTE: This might potentially really slow things down!!
  MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

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
      if (valL==0) {break;}

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




int spmatTests()
{
  PetscErrorCode ierr = 0;

  Spmat mat(3,3);
  mat(0,0,1); // assign 1 to the (0,0) entry
  mat(1,2,3);

  mat.printPetsc();


  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting Copy of mat:\n");CHKERRQ(ierr);
  Spmat copied(mat);
  copied = mat;
  copied.printPetsc();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting transpose of mat:\n");CHKERRQ(ierr);
  mat.transpose();
  mat.printPetsc();
  mat.transpose();
  //~mat.printPetsc();


  double val = mat(0,0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(0,0): val = %f\n",val);CHKERRQ(ierr);
  val = mat(1,1);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(1,1): val = %f\n",val);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Original values for mat:\n");CHKERRQ(ierr);
  mat.print();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Result of scaling mat by 3.0:\n");CHKERRQ(ierr);
  mat.scale(3.0);
  mat.printPetsc();


  ierr = PetscPrintf(PETSC_COMM_WORLD,"Result of mat.spy():\n");CHKERRQ(ierr);
  mat.eye();
  mat.printPetsc();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBuilding new mat:\n");CHKERRQ(ierr);
  size_t Ii,Jj;
  val = 1.0;
  for(Ii=0;Ii<3;Ii++)
  {
    for(Jj=0;Jj<3;Jj++)
    {
      mat(Ii,Jj,val);
      val++;
    }
  }
  mat.printPetsc();

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nkron(mat,eye(1,1)):\n");CHKERRQ(ierr);
  Spmat I(1,1);
  I.eye();
  Spmat result = kron(mat,I);
  result.printPetsc();


  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nkron(mat,eye(2,2)):\n");CHKERRQ(ierr);
  Spmat I2(2,2);
  I2.eye();
  Spmat result2 = kron(mat,I2);
  result2.printPetsc();


  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nkron(eye(2,2),mat):\n");CHKERRQ(ierr);
  Spmat result3 = kron(I2,mat);
  result3.printPetsc();

  Mat petscMat;
  result3.convert(petscMat,6);
  ierr = PetscObjectSetName((PetscObject) petscMat, "petscMat");CHKERRQ(ierr);
  ierr = MatView(petscMat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  return ierr;
}
