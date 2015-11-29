#include "genFuncs.hpp"

// Print out a vector with 15 significant figures.
void printVec(Vec vec)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v;
  VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec,1,&Ii,&v);
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsDiff(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Ii,&v1);
    VecGetValues(vec2,1,&Ii,&v2);
    v = v1 - v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}

// Print out (vec1 - vec2) with 15 significant figures.
void printVecsSum(Vec vec1,Vec vec2)
{
  PetscInt Ii,Istart,Iend;
  PetscScalar v1,v2,v;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(vec1,1,&Ii,&v1);
    VecGetValues(vec2,1,&Ii,&v2);
    v = v1 + v2;
    PetscPrintf(PETSC_COMM_WORLD,"%.15e\n",v);
  }
}


// Write vec to the file loc in binary format.
// Note that due to a memory problem in PETSc, looping over this many
// times will result in an error.
PetscErrorCode writeVec(Vec vec,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode writeMat(Mat mat,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
  ierr = MatView(mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}


// Print all entries of 2D DMDA global vector to stdout, including which
// processor each entry lives on, and the corresponding subscripting
// indices
PetscErrorCode printf_DM_2d(const Vec gvec, const DM dm)
{
    PetscErrorCode ierr = 0;
#if VERBOSE > 2
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::printf_DM_2d in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  DMDAGetCorners(dm,&mStart,&nStart,0,&m,&n,0);

  PetscScalar **gxArr;
  DMDAVecGetArray(dm,gvec,&gxArr);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      PetscPrintf(PETSC_COMM_SELF,"%i: gxArr[%i][%i] = %g\n",
        rank,j,i,gxArr[j][i]);
    }
  }
  DMDAVecRestoreArray(dm,gvec,&gxArr);

#if VERBOSE > 2
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::printf_DM_2d in fault.cpp.\n");
#endif
  return ierr;
}



// computes || vec1 - vec2||_mat
double computeNormDiff_Mat(const Mat& mat,const Vec& vec1,const Vec& vec2)
{
  PetscErrorCode ierr = 0;

  Vec diff;
  ierr = VecDuplicate(vec1,&diff);CHKERRQ(ierr);
  ierr = VecWAXPY(diff,-1.0,vec1,vec2);CHKERRQ(ierr);

  PetscScalar diffErr = computeNorm_Mat(mat,diff);
  PetscScalar vecErr = computeNorm_Mat(mat,vec1);

  PetscScalar err = diffErr/vecErr;

  VecDestroy(&diff);

  return err;
}

// computes || vec ||_mat
double computeNorm_Mat(const Mat& mat,const Vec& vec)
{
  PetscErrorCode ierr = 0;

  Vec Matxvec;
  ierr = VecDuplicate(vec,&Matxvec);CHKERRQ(ierr);
  ierr = MatMult(mat,vec,Matxvec);CHKERRQ(ierr);

  PetscScalar err;
  ierr = VecDot(vec,Matxvec,&err);CHKERRQ(ierr);

  return err;
}


// computes || vec1 - vec2 ||_2
double computeNormDiff_2(const Vec& vec1,const Vec& vec2)
{
  PetscErrorCode ierr = 0;

  Vec diff;
  ierr = VecDuplicate(vec1,&diff);CHKERRQ(ierr);
  ierr = VecWAXPY(diff,-1.0,vec1,vec2);CHKERRQ(ierr);

  PetscScalar err;
  ierr = VecNorm(diff,NORM_2,&err);CHKERRQ(ierr);

  PetscInt len;
  ierr = VecGetSize(vec1,&len);CHKERRQ(ierr);

  err = err/sqrt(len);

  VecDestroy(&diff);

  return err;
}



// loads a PETSc Vec from a binary file
// Note: memory for out MUST be allocated before calling this function
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "loadFieldsFromFiles";
  string fileName = "genFuncs.cpp";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
#endif

  string vecSourceFile = inputDir + fieldName;
  PetscViewer inv;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);

  ierr = VecLoad(out,inv);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif
  return ierr;
}


// loads a std library vector from a list in the input file
PetscErrorCode loadVectorFromInputFile(const string& str,vector<double>& vec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  size_t pos = 0; // position of delimiter in string
  string delim = " "; // delimiter between values in list (whitespace sensitive)
  string remstr; // holds remaining string as str is parsed through
  double val; // holds values


  //~PetscPrintf(PETSC_COMM_WORLD,"About to start loading aVals:\n");
  //~PetscPrintf(PETSC_COMM_WORLD,"input str = %s\n\n",str.c_str());

  // holds remainder as str is parsed through (with beginning and ending brackets removed)
  pos = str.find("]");
  remstr = str.substr(1,pos-1);
  //~PetscPrintf(PETSC_COMM_WORLD,"remstr = %s\n",remstr.c_str());

  pos = remstr.find(delim);
  while (pos != remstr.npos) {
    pos = remstr.find(delim);
    val = atof( remstr.substr(0,pos).c_str() );
    remstr = remstr.substr(pos + delim.length());
    //~PetscPrintf(PETSC_COMM_WORLD,"val = %g  |  remstr = %s\n",val,remstr.c_str());
    vec.push_back(val);
  }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

