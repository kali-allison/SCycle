#include "genFuncs.hpp"



// check if file exists
bool doesFileExist(const string fileName)
{
    std::ifstream infile(fileName.c_str());
    return infile.good();
};


// clean up a C++ std library vector of PETSc Vecs
void destroyVector(std::vector<Vec>& vec)
{
  for(std::vector<Vec>::size_type i = 0; i != vec.capacity(); i++) {
      VecDestroy(&vec[i]);
  }
}


void destroyVector(std::map<string,Vec>& vec)
{
  for (map<string,Vec>::iterator it = vec.begin(); it!=vec.end(); it++ ) {
    VecDestroy(&vec[it->first]);
  }
}

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
  PetscPrintf(PETSC_COMM_WORLD,"\n");
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
PetscErrorCode writeMat(Mat& mat,std::string str)
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}



// Write vec to the file loc in binary format.
// Note that due to a memory problem in PETSc, looping over this many
// times will result in an error.
PetscErrorCode writeVec(Vec vec,std::string str)
{
return writeVec(vec,str.c_str());
}
PetscErrorCode writeVecAppend(Vec vec,std::string str)
{
return writeVecAppend(vec,str.c_str());
}

PetscErrorCode writeVec(Vec vec,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode writeVecAppend(Vec vec,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_APPEND,&viewer);
  ierr = VecView(vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode writeMat(Mat& mat,const char * loc)
{
  PetscErrorCode ierr = 0;
  PetscViewer    viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
  ierr = MatView(mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  return ierr;
}


PetscViewer initiateViewer(std::string str)
{
  PetscViewer vw;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&vw);
  return vw;
}
PetscErrorCode appendViewer(PetscViewer& vw,const std::string str)
{
  PetscErrorCode ierr = 0;
  ierr = PetscViewerDestroy(&vw); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_APPEND,&vw);
  CHKERRQ(ierr);
  return ierr;
}
PetscErrorCode appendViewers(map<string,PetscViewer>& vwL,const std::string dir)
{
  PetscErrorCode ierr = 0;
  for (map<string,PetscViewer>::iterator it=vwL.begin(); it!=vwL.end(); it++ ) {
    ierr = PetscViewerDestroy(&vwL[it->first]); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(dir + it->first).c_str(),FILE_MODE_APPEND,&vwL[it->first]);
    CHKERRQ(ierr);
  }
  return ierr;
}

PetscErrorCode io_initiateWriteAppend(std::map <string,std::pair<PetscViewer,string> >& vwL, const std::string key,const Vec& vec, const std::string dir)
{
  PetscErrorCode ierr = 0;

  // initiate viewer
  PetscViewer vw;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,dir.c_str(),FILE_MODE_WRITE,&vw);
  vwL[key].first = vw;
  vwL[key].second = dir;

  // write vec out
  ierr = VecView(vec,vw); CHKERRQ(ierr);

  // change viewer to append mode
  ierr = PetscViewerDestroy(&vw); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,dir.c_str(),FILE_MODE_APPEND,&vwL[key].first); CHKERRQ(ierr);

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

  VecDestroy(&Matxvec);
  return err;
}


// computes || vec1 - vec2 ||_2 / sqrt(length(vec1))
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

// computes || vec1 - vec2 ||_2 / || vec1 ||_2
double computeNormDiff_L2_scaleL2(const Vec& vec1,const Vec& vec2)
{
  PetscErrorCode ierr = 0;

  Vec diff;
  ierr = VecDuplicate(vec1,&diff);CHKERRQ(ierr);
  ierr = VecWAXPY(diff,-1.0,vec1,vec2);CHKERRQ(ierr);

  PetscScalar err;
  ierr = VecNorm(diff,NORM_2,&err);CHKERRQ(ierr);

  PetscScalar len;
  ierr = VecNorm(vec1,NORM_2,&len);CHKERRQ(ierr);

  err = err/len;

  VecDestroy(&diff);

  return err;
}

// out = vecL' x A x vecR
double multVecMatsVec(const Vec& vecL, const Mat& A, const Vec& vecR)
{
  PetscErrorCode ierr = 0;
  double out = 0;

  Vec temp;
  VecDuplicate(vecL,&temp);
  ierr = MatMult(A,vecR,temp); CHKERRQ(ierr);
  ierr = VecDot(vecL,temp,&out); CHKERRQ(ierr);
  VecDestroy(&temp);
  return out;
}

// out = vecL' x A x B x vecR
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Vec& vecR)
{
  PetscErrorCode ierr = 0;
  double out = 0;

  Mat AB;
  MatMatMult(A,B,MAT_INITIAL_MATRIX,1.0,&AB);

  Vec temp;
  VecDuplicate(vecL,&temp);

  ierr = MatMult(AB,vecR,temp); CHKERRQ(ierr);
  ierr = VecDot(vecL,temp,&out); CHKERRQ(ierr);

  VecDestroy(&temp);
  MatDestroy(&AB);
  return out;
}

// out = vecL' x A x B x C x vecR
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Mat& C, const Vec& vecR)
{
  PetscErrorCode ierr = 0;
  double out = 0;

  Mat ABC;
  MatMatMatMult(A,B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ABC);

  Vec temp;
  VecDuplicate(vecL,&temp);

  ierr = MatMult(ABC,vecR,temp); CHKERRQ(ierr);
  ierr = VecDot(vecL,temp,&out); CHKERRQ(ierr);

  VecDestroy(&temp);
  MatDestroy(&ABC);
  return out;
}

// out = vecL' x A x B x C x vecR
double multVecMatsVec(const Vec& vecL, const Mat& A, const Mat& B, const Mat& C, const Mat& D, const Vec& vecR)
{
  PetscErrorCode ierr = 0;
  double out = 0;

  Mat ABC,ABCD;
  MatMatMatMult(A,B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ABC);
  MatMatMult(ABC,D,MAT_INITIAL_MATRIX,1.0,&ABCD);
  MatDestroy(&ABC);

  Vec temp;
  VecDuplicate(vecL,&temp);

  ierr = MatMult(ABCD,vecR,temp); CHKERRQ(ierr);
  ierr = VecDot(vecL,temp,&out); CHKERRQ(ierr);

  VecDestroy(&temp);
  MatDestroy(&ABCD);
  return out;
}

// out = A x B x vecR; assumes vecR is DIFFERENT from out
PetscErrorCode multMatsVec(Vec& out, const Mat& A, const Mat& B, const Vec& vecR)
{
  PetscErrorCode ierr = 0;
  Vec BvecR;
  VecDuplicate(vecR,&BvecR);
  ierr = MatMult(B,vecR,BvecR); CHKERRQ(ierr);
  ierr = MatMult(A,BvecR,out); CHKERRQ(ierr);
  ierr = VecDestroy(&BvecR);
  return ierr;
}

// out = A x B x vecR; assumes vecR is IDENTICAL to out
PetscErrorCode multMatsVec(const Mat& A, const Mat& B, Vec& vecR)
{
  PetscErrorCode ierr = 0;
  Vec BvecR;
  VecDuplicate(vecR,&BvecR);
  ierr = MatMult(B,vecR,BvecR); CHKERRQ(ierr);
  ierr = MatMult(A,BvecR,vecR); CHKERRQ(ierr);
  ierr = VecDestroy(&BvecR);
  return ierr;
}

// log10(out) = a*log10(vec1) + b*log10(vec2)
// out may not be the same as vec1 or vec2
PetscErrorCode MyVecLog10AXPBY(Vec& out,const double a, const Vec& vec1, const double b, const Vec& vec2)
{

  if (out == NULL) { VecDuplicate(vec1,&out); }

  // compute effective viscosity
  PetscScalar *outA;
  PetscScalar const *vec1A,*vec2A;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(vec1,&Istart,&Iend);
  VecGetArrayRead(vec1,&vec1A);
  VecGetArrayRead(vec2,&vec2A);
  VecGetArray(out,&outA);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
   PetscScalar log10Out = a*log10(vec1A[Jj]) + b*log10(vec2A[Jj]);
    outA[Jj] = pow(10.,log10Out);
    Jj++;
  }
  VecRestoreArrayRead(vec1,&vec1A);
  VecRestoreArrayRead(vec2,&vec2A);
  VecRestoreArray(out,&outA);

  return 0;
}


// loads a PETSc Vec from a binary file
// Note: memory for out MUST be allocated before calling this function
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName)
{
  PetscErrorCode ierr = 0;
  bool fileExists = 0;
  ierr = loadVecFromInputFile(out,inputDir,fieldName,fileExists); CHKERRQ(ierr);
  return ierr;
}

// loads a PETSc Vec from a binary file
// Note: memory for out MUST be allocated before calling this function
PetscErrorCode loadVecFromInputFile(Vec& out,const string inputDir, const string fieldName, bool& fileExists)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "loadFieldsFromFiles";
    string fileName = "genFuncs.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Attempting to load: %s%s\n",inputDir.c_str(),fieldName.c_str());CHKERRQ(ierr);
  #endif

  string vecSourceFile = inputDir + fieldName;

  fileExists = doesFileExist(vecSourceFile);
  if (fileExists) {
    PetscPrintf(PETSC_COMM_WORLD,"Note: Loading Vec from file: %s\n",vecSourceFile.c_str());
    PetscViewer inv;
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);

    ierr = VecLoad(out,inv);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: File not found: %s\n",vecSourceFile.c_str());
  }

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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadVectorFromInputFile in genFuncs.cpp.\n");CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadVectorFromInputFile in genFuncs.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode loadVectorFromInputFile(const string& str,vector<int>& vec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  size_t pos = 0; // position of delimiter in string
  string delim = " "; // delimiter between values in list (whitespace sensitive)
  string remstr; // holds remaining string as str is parsed through
  int val; // holds values


  //~ PetscPrintf(PETSC_COMM_WORLD,"About to start loading timeIntInds:\n");
  //~ PetscPrintf(PETSC_COMM_WORLD,"input str = %s\n\n",str.c_str());

  // holds remainder as str is parsed through (with beginning and ending brackets removed)
  pos = str.find("]");
  remstr = str.substr(1,pos-1);
  //~ PetscPrintf(PETSC_COMM_WORLD,"remstr = %s\n",remstr.c_str());

  pos = remstr.find(delim);
  val = atoi( remstr.substr(0,pos).c_str() );
  vec.push_back(val);
  remstr = remstr.substr(pos + delim.length());
  while (pos != remstr.npos) {
    pos = remstr.find(delim);
    val = atoi( remstr.substr(0,pos).c_str() );
    remstr = remstr.substr(pos + delim.length());
    //~ PetscPrintf(PETSC_COMM_WORLD,"val = %i  |  remstr = %s\n",val,remstr.c_str());
    vec.push_back(val);
  }
  //~ PetscPrintf(PETSC_COMM_WORLD,"val = %i  |  remstr = %s\n",val,remstr.c_str());

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode loadVectorFromInputFile(const string& str,vector<string>& vec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  size_t pos = 0; // position of delimiter in string
  string delim = " "; // delimiter between values in list (whitespace sensitive)
  string remstr; // holds remaining string as str is parsed through
  string val; // holds values


  //~ PetscPrintf(PETSC_COMM_WORLD,"About to start loading timeIntInds:\n");
  //~ PetscPrintf(PETSC_COMM_WORLD,"input str = %s\n\n",str.c_str());

  // holds remainder as str is parsed through (with beginning and ending brackets removed)
  pos = str.find("]");
  remstr = str.substr(1,pos-1);
  //~ PetscPrintf(PETSC_COMM_WORLD,"remstr = %s\n",remstr.c_str());

  pos = remstr.find(delim);
  val = remstr.substr(0,pos).c_str();
  vec.push_back(val);
  remstr = remstr.substr(pos + delim.length());
  while (pos != remstr.npos) {
    pos = remstr.find(delim);
    val = remstr.substr(0,pos).c_str();
    remstr = remstr.substr(pos + delim.length());
    //~ PetscPrintf(PETSC_COMM_WORLD,"val = %s  |  remstr = %s\n",val.c_str(),remstr.c_str());
    vec.push_back(val);
  }
  //~ PetscPrintf(PETSC_COMM_WORLD,"val = %s  |  remstr = %s\n",val.c_str(),remstr.c_str());

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

// creates a string containing the contents of C++ std library vector
string vector2str(const vector<double> vec)
{
  ostringstream ss;
  for (vector<double>::const_iterator Ii=vec.begin(); Ii != vec.end(); Ii++) {
    ss << " " << *Ii;
  }
  string str = "[" + ss.str() + "]";

  return str;
}

// creates a string containing the contents of C++ std library vector
string vector2str(const vector<int> vec)
{
  ostringstream ss;
  for (vector<int>::const_iterator Ii=vec.begin(); Ii != vec.end(); Ii++) {
    ss << " " << *Ii;
  }
  string str = "[" + ss.str() + "]";

  return str;
}

// creates a string containing the contents of C++ std library vector
string vector2str(const vector<string> vec)
{
  ostringstream ss;
  for (vector<string>::const_iterator Ii=vec.begin(); Ii != vec.end(); Ii++) {
    ss << " " << *Ii;
  }
  string str = "[" + ss.str() + "]";

  return str;
}




// prints an array to a single line in std out
PetscErrorCode printArray(const PetscScalar * arr,const PetscScalar len)
{
  PetscErrorCode ierr = 0;
  string funcName = "genFuncs::printArray";
  string fileName = "genFuncs.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  std::cout << "[";
  for(int i=0; i<len; i++) {
    std::cout << arr[i] << ",";
  }
  std::cout << "]" << std::endl;


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}


double MMS_test(const double y,const double z) { return y*10.0 + z; }
double MMS_test(const double z) { return z; }




// Fills vec with the linear interpolation between the pairs of points (vals,depths).
PetscErrorCode setVec(Vec& vec, const Vec& coord, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;


  VecSet(vec,vals[0]);

  PetscInt N;
  VecGetSize(coord,&N);
  if (N == 1) { return ierr; }


  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar *vecA;
  const PetscScalar *coordA;
  PetscInt Jj = 0;
  VecGetArray(vec,&vecA);
  VecGetArrayRead(coord,&coordA);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    z = coordA[Jj];
    for (size_t ind = 0; ind < vecLen-1; ind++) {
      z0 = depths[0+ind];
      z1 = depths[0+ind+1];
      v0 = vals[0+ind];
      v1 = vals[0+ind+1];
      if (z>=z0 && z<=z1) {
        v = (v1 - v0)/(z1-z0) * (z-z0) + v0;
        vecA[Jj] = v;
      }
      else if (z>z1 && ind == vecLen-2) {
        v = (v1 - v0)/(z1-z0) * (z-z0) + v0;
        vecA[Jj] = v;
      }
    }
    Jj++;
  }
  VecRestoreArray(vec,&vecA);
  VecRestoreArrayRead(coord,&coordA);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const Vec& yV, const double t)
{
  PetscErrorCode ierr = 0;
  PetscScalar const *y;
  PetscScalar *v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArrayRead(yV,&y);
  ierr = VecGetArray(vec,&v);
  PetscInt Jj = 0;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v[Jj] = func(y[Jj],t);
    Jj++;
  }
  VecRestoreArrayRead(yV,&y);
  VecRestoreArray(vec,&v);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double),const Vec& yV)
{
  PetscErrorCode ierr = 0;
  PetscScalar const *y;
  PetscScalar *v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArrayRead(yV,&y);
  ierr = VecGetArray(vec,&v);
  PetscInt Jj = 0;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v[Jj] = func(y[Jj]);
    Jj++;
  }
  VecRestoreArrayRead(yV,&y);
  VecRestoreArray(vec,&v);
  return ierr;
}


PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const Vec& yV,const Vec& zV, const double t)
{
  PetscErrorCode ierr = 0;
  PetscScalar const *y,*z;
  PetscScalar *v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArrayRead(yV,&y);
  ierr = VecGetArrayRead(zV,&z);
  ierr = VecGetArray(vec,&v);
  PetscInt Jj = 0;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v[Jj] = func(y[Jj],z[Jj],t);
    Jj++;
  }
  VecRestoreArrayRead(yV,&y);
  VecRestoreArrayRead(zV,&z);
  VecRestoreArray(vec,&v);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const Vec& yV,const Vec& zV)
{
  PetscErrorCode ierr = 0;
  PetscScalar const *y,*z;
  PetscScalar *v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArrayRead(yV,&y);
  ierr = VecGetArrayRead(zV,&z);
  ierr = VecGetArray(vec,&v);
  PetscInt Jj = 0;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v[Jj] = func(y[Jj],z[Jj]);
    Jj++;
  }
  VecRestoreArrayRead(yV,&y);
  VecRestoreArrayRead(zV,&z);
  VecRestoreArray(vec,&v);
  return ierr;
}


// Map a function that acts on scalars to a 2D DMDA Vec
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const int N, const double dy, const double dz,DM da)
{
  // assumes vec has already been created and it's size has been allocated
  PetscErrorCode ierr = 0;

  PetscInt zS,yS,zn,yn;
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt zE = zS + zn;
  PetscInt yE = yS + yn;

  PetscScalar** arr;
  ierr = DMDAVecGetArray(da, vec, &arr);CHKERRQ(ierr);

  PetscInt yI,zI;
  PetscScalar y,z;
    for (yI = yS; yI < yE; yI++) {
      for (zI = zS; zI < zE; zI++) {
        y = yI * dy;
        z = zI * dz;
        arr[yI][zI] = func(y,z);
      }
    }

  ierr = DMDAVecRestoreArray(da, vec, &arr);CHKERRQ(ierr);
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);

  return ierr;
}

// Map a function that acts on scalars to a 1DD DMDA Vec
PetscErrorCode mapToVec(Vec& vec, double(*func)(double),
  const int N, const double dz,DM da)
{
  // assumes vec has already been created and it's size has been allocated
  PetscErrorCode ierr = 0;

  PetscInt zS,zn;
  DMDAGetCorners(da, &zS, 0, 0, &zn, 0, 0);
  PetscInt zE = zS + zn;

  PetscScalar* arr;
  ierr = DMDAVecGetArray(da, vec, &arr);CHKERRQ(ierr);

  PetscInt zI;
  PetscScalar z;
  for (zI = zS; zI < zE; zI++) {
    z = zI * dz;
    arr[zI] = func(z);
  }

  ierr = DMDAVecRestoreArray(da, vec, &arr);CHKERRQ(ierr);
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);

  return ierr;
}

// Map a function that acts on scalars to a 2D DMDA Vec
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const int N, const double dy, const double dz,const double t,DM da)
{
  // assumes vec has already been created and it's size has been allocated
  PetscErrorCode ierr = 0;

  PetscInt zS,yS,zn,yn;
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt zE = zS + zn;
  PetscInt yE = yS + yn;

  PetscScalar** arr;
  ierr = DMDAVecGetArray(da, vec, &arr);CHKERRQ(ierr);

  PetscInt yI,zI;
  PetscScalar y,z;
    for (yI = yS; yI < yE; yI++) {
      for (zI = zS; zI < zE; zI++) {
        y = yI * dy;
        z = zI * dz;
        arr[yI][zI] = func(y,z,t);
      }
    }

  ierr = DMDAVecRestoreArray(da, vec, &arr);CHKERRQ(ierr);
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);

  return ierr;
}


// Print out a vector with 15 significant figures.
void printVec(const Vec vec,const DM da)
{
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscInt zS,yS,zn,yn;
  DMDAGetCorners(da, &zS, &yS, 0, &zn, &yn, 0);
  PetscInt zE = zS + zn;
  PetscInt yE = yS + yn;

  PetscScalar** arr;
  DMDAVecGetArray(da, vec, &arr);

  PetscInt yI,zI;
    for (yI = yS; yI < yE; yI++) {
      for (zI = zS; zI < zE; zI++) {
        PetscPrintf(PETSC_COMM_SELF,"%i: f(%i,%i) = %.2f\n",rank,yI,zI,arr[yI][zI]);
      }
    }

  DMDAVecRestoreArray(da, vec, &arr);
  VecAssemblyBegin(vec);
  VecAssemblyEnd(vec);
}



// repmat for vecs (i.e. vec -> [vec vec])
// Note: out must already be allocated onto processors
// n = # of repeats
PetscErrorCode repVec(Vec& out, const Vec& in, const PetscInt n)
{
  PetscErrorCode ierr = 0;
  PetscInt N,Istart,Iend;
  PetscScalar v = 0.0;
  PetscScalar vals[n];
  PetscInt    inds[n];

  VecGetSize(in,&N);
  VecGetOwnershipRange(in,&Istart,&Iend);
  for (PetscInt Ii=Istart; Ii<Iend; Ii++ ) {
    ierr = VecGetValues(in,1,&Ii,&v);CHKERRQ(ierr);
    for (int i=0; i<n; i++) {
      vals[i] = v;
      inds[i] = Ii + i*N;
    }
    ierr = VecSetValues(out,n,inds,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(out);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(out);CHKERRQ(ierr);

  return ierr;
}

// undoes repmat for vecs (i.e. [vec vec] -> vec)
// Note: out must already be allocated onto processors
// gIstart,gIend = global indices of in to store in out
PetscErrorCode sepVec(Vec& out, const Vec& in, const PetscInt gIstart, const PetscInt gIend)
{
  PetscErrorCode ierr = 0;
  PetscScalar v = 0.0;

  PetscInt Istart, Iend;
  VecGetOwnershipRange(in,&Istart,&Iend);
  for (PetscInt Ii=Istart; Ii<Iend; Ii++ ) {
    if (Ii >= gIstart && Ii < gIend) {
      ierr = VecGetValues(in,1,&Ii,&v);CHKERRQ(ierr);
      PetscInt Jj = Ii - gIstart;
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ii = %i, Jj = %i\n",Ii,Jj);
      ierr = VecSetValue(out,Jj,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(out);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(out);CHKERRQ(ierr);

  return ierr;
}

// maps vec to bigger vec (i.e. vec -> [0 0 0, vec, 0 0 0])
// Note: out must already be allocated onto processors
// Istart,Iend = global indices of out to put the values of in into
PetscErrorCode distributeVec(Vec& out, const Vec& in, const PetscInt gIstart, const PetscInt gIend)
{
  PetscErrorCode ierr = 0;
  PetscScalar v = 0.0;

  PetscInt Istart, Iend;
  VecGetOwnershipRange(in,&Istart,&Iend);
  for (PetscInt Ii=Istart; Ii<Iend; Ii++ ) {
      ierr = VecGetValues(in,1,&Ii,&v);CHKERRQ(ierr);
      PetscInt Jj = Ii + gIstart;
      //~ PetscPrintf(PETSC_COMM_WORLD,"Ii = %i, Jj = %i\n",Ii,Jj);
      ierr = VecSetValue(out,Jj,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(out);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(out);CHKERRQ(ierr);

  return ierr;
}
