#include "genFuncs.hpp"



// check if file exists
bool doesFileExist(const string fileName)
{
    std::ifstream infile(fileName.c_str());
    return infile.good();
};

// load matlab-style file
PetscErrorCode loadFileIfExists_matlab(const string fileName, Vec& vec)
{
  PetscErrorCode ierr = 0;

  bool fileExists = doesFileExist(fileName);
  if (fileExists) {
    PetscViewer inv;
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecLoad(vec,inv);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: File not found: %s\n",fileName.c_str());
  }
  return ierr;
}


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

PetscErrorCode writeMat(Mat mat,const char * loc)
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

  //~ Mat AB;
  //~ MatMatMult(A,B,MAT_INITIAL_MATRIX,1.0,&AB); // has memory leak!!
  //~ ierr = MatMult(AB,vecR,out); CHKERRQ(ierr);
  //~ MatDestroy(&AB);

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

  //~ Mat AB;
  //~ MatMatMult(A,B,MAT_INITIAL_MATRIX,1.0,&AB);
  //~ Vec temp;
  //~ VecDuplicate(vecR,&temp);
  //~ ierr = MatMult(AB,vecR,temp); CHKERRQ(ierr);
  //~ ierr = VecCopy(temp,vecR); CHKERRQ(ierr);
  //~ VecDestroy(&temp);
  //~ MatDestroy(&AB);

  Vec BvecR;
  VecDuplicate(vecR,&BvecR);
  ierr = MatMult(B,vecR,BvecR); CHKERRQ(ierr);
  ierr = MatMult(A,BvecR,vecR); CHKERRQ(ierr);
  ierr = VecDestroy(&BvecR);

  return ierr;
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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"  Attempting to load: %s%s\n",inputDir.c_str(),fieldName.c_str());CHKERRQ(ierr);
#endif

  string vecSourceFile = inputDir + fieldName;

  bool fileExists = doesFileExist(vecSourceFile);
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


//======================================================================
//                  MMS Functions

/*
// version 1
double MMS_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double MMS_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double MMS_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double MMS_f_z(const double y,const double z) { return cos(y)*cos(z); }
double MMS_f_zz(const double y,const double z) { return -cos(y)*sin(z); }


double MMS_g(const double t) { return exp(-t/60.0) - exp(-t/3e7) + exp(-t/3e9); }
double MMS_uA(const double y,const double z,const double t) { return MMS_f(y,z)*MMS_g(t); }
double MMS_uA_y(const double y,const double z,const double t) { return MMS_f_y(y,z)*MMS_g(t); }
double MMS_uA_yy(const double y,const double z,const double t) { return MMS_f_yy(y,z)*MMS_g(t); }
double MMS_uA_z(const double y,const double z,const double t) { return MMS_f_z(y,z)*MMS_g(t); }
double MMS_uA_zz(const double y,const double z,const double t) { return MMS_f_zz(y,z)*MMS_g(t); }
double MMS_uA_t(const double y,const double z,const double t) {
  return MMS_f(y,z)*((-1.0/60)*exp(-t/60.0) - (-1.0/3e7)*exp(-t/3e7) +   (-1.0/3e9)*exp(-t/3e9));
}

double MMS_mu(const double y,const double z) { return sin(y)*sin(z) + 30; }
double MMS_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double MMS_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double MMS_sigmaxy(const double y,const double z,const double t) { return MMS_mu(y,z)*MMS_uA_y(y,z,t); }
double MMS_sigmaxz(const double y,const double z, const double t) { return MMS_mu(y,z)*MMS_uA_z(y,z,t); }


// specific MMS functions
double MMS_visc(const double y,const double z) { return cos(y)*cos(z) + 2e10; }
double MMS_invVisc(const double y,const double z) { return 1.0/MMS_visc(y,z); }
double MMS_invVisc_y(const double y,const double z) { return sin(y)*cos(z)/pow( cos(y)*cos(z)+2e10, 2.0); }
double MMS_invVisc_z(const double y,const double z) { return cos(y)*sin(z)/pow( cos(y)*cos(z)+2e10 ,2.0); }

double MMS_gxy(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fy = MMS_f_y(y,z);
  //~ return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fy/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fy/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double MMS_gxy_y(const double y,const double z,const double t)
{
  //~return 0.5 * MMS_uA_yy(y,z,t);
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double Ay = MMS_mu_y(y,z)*MMS_invVisc(y,z) + MMS_mu(y,z)*MMS_invVisc_y(y,z);
  double fy = MMS_f_y(y,z);
  double fyy = MMS_f_yy(y,z);

  double T1 = 60, T2 = 3e7, T3 = 3e9;
  double d1 = T1*A-1, d2 = T2*A-1, d3 = T3*A-1;
  double out1 = -pow(T1,2.0)*A*Ay*fy/pow(d1,2.0)*(exp(-t/T1)-exp(-A*t))  + T1*fy*Ay/d1 *(exp(-t/T1)-exp(-A*t))
      +T1*A*Ay*fy*exp(-A*t)*t/d1 + T1*A*fyy/d1*(exp(-t/T1)-exp(-A*t));
  double out2 = pow(T2,2.0)*A*Ay*fy/pow(d2,2.0)*(exp(-t/T2)-exp(-A*t)) - T2*fy*Ay/d2 *(exp(-t/T2)-exp(-A*t))
       -T2*A*Ay*fy*exp(-A*t)*t/d2 - T2*A*fyy/d2*(exp(-t/T2)-exp(-A*t));
  double out3 = -pow(T3,2.0)*A*Ay*fy/pow(d3,2.0)*(exp(-t/T3)-exp(-A*t))  + T3*fy*Ay/d3 *(exp(-t/T3)-exp(-A*t))
       +T3*A*Ay*fy*exp(-A*t)*t/d3 + T3*A*fyy/d3*(exp(-t/T3)-exp(-A*t));
  return out1 + out2 + out3;

}
double MMS_gxy_t(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fy = MMS_f_y(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fy/(T1*A-1)*((-1.0/T1)*exp(-t/T1)+A*exp(-A*t))
       - T2*A*fy/(T2*A-1)*((-1.0/T2)*exp(-t/T2)+A*exp(-A*t))
       + T3*A*fy/(T3*A-1)*((-1.0/T3)*exp(-t/T3)+A*exp(-A*t));
}

double MMS_gxz(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fz = MMS_f_z(y,z);
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fz/(T1*A-1)*(exp(-t/T1)-exp(-A*t))
       - T2*A*fz/(T2*A-1)*(exp(-t/T2)-exp(-A*t))
       + T3*A*fz/(T3*A-1)*(exp(-t/T3)-exp(-A*t));
}
double MMS_gxz_z(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double Az = MMS_mu_z(y,z)*MMS_invVisc(y,z) + MMS_mu(y,z)*MMS_invVisc_z(y,z);
  double fz = MMS_f_z(y,z);
  double fzz = MMS_f_zz(y,z);
  //~ double den = A-1.0, B = exp(-t)-exp(-A*t);
  //~ return t*A*Az*fz*exp(-A*t)/den - A*fz*Az*B/pow(den,2.0) + fz*Az*B/den + A*fzz*B/den;

  double T1 = 60, T2 = 3e7, T3 = 3e9;
  double d1 = T1*A-1, d2 = T2*A-1, d3 = T3*A-1;
  double out1 = -pow(T1,2.0)*A*Az*fz/pow(d1,2.0)*(exp(-t/T1)-exp(-A*t))  + T1*fz*Az/d1 *(exp(-t/T1)-exp(-A*t))
      +T1*A*Az*fz*exp(-A*t)*t/d1 + T1*A*fzz/d1*(exp(-t/T1)-exp(-A*t));
  double out2 = pow(T2,2.0)*A*Az*fz/pow(d2,2.0)*(exp(-t/T2)-exp(-A*t)) - T2*fz*Az/d2 *(exp(-t/T2)-exp(-A*t))
       -T2*A*Az*fz*exp(-A*t)*t/d2 - T2*A*fzz/d2*(exp(-t/T2)-exp(-A*t));
  double out3 = -pow(T3,2.0)*A*Az*fz/pow(d3,2.0)*(exp(-t/T3)-exp(-A*t))  + T3*fz*Az/d3 *(exp(-t/T3)-exp(-A*t))
       +T3*A*Az*fz*exp(-A*t)*t/d3 + T3*A*fzz/d3*(exp(-t/T3)-exp(-A*t));
  return out1 + out2 + out3;



}
double MMS_gxz_t(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fz = MMS_f_z(y,z);
  //~ return A*fz/(A-1.0)*(-exp(-t) + A*exp(-A*t));
  double T1 = 60, T2 = 3e7, T3 = 3e9;
  return T1*A*fz/(T1*A-1)*((-1.0/T1)*exp(-t/T1)+A*exp(-A*t))
       - T2*A*fz/(T2*A-1)*((-1.0/T2)*exp(-t/T2)+A*exp(-A*t))
       + T3*A*fz/(T3*A-1)*((-1.0/T3)*exp(-t/T3)+A*exp(-A*t));
}

// source terms for viscous strain rates
double MMS_max_gxy_t_source(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double uy = MMS_uA_y(y,z,t);
  double g = MMS_gxy(y,z,t);

  return MMS_gxy_t(y,z,t) - A*(uy - g);
}
double MMS_max_gxz_t_source(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double uz = MMS_uA_z(y,z,t);
  double g = MMS_gxz(y,z,t);

  return MMS_gxz_t(y,z,t) - A*(uz - g);
}

double MMS_gSource(const double y,const double z,const double t)
{
  PetscScalar mu = MMS_mu(y,z);
  PetscScalar mu_y = MMS_mu_y(y,z);
  PetscScalar mu_z = MMS_mu_z(y,z);
  PetscScalar gxy = MMS_gxy(y,z,t);
  PetscScalar gxz = MMS_gxz(y,z,t);
  PetscScalar gxy_y = MMS_gxy_y(y,z,t);
  PetscScalar gxz_z = MMS_gxz_z(y,z,t);
  return -mu*(gxy_y + gxz_z) - mu_y*gxy - mu_z*gxz; // full answer
}

// specific to power law
double MMS_A(const double y,const double z) { return cos(y)*cos(z) + 398; }
double MMS_B(const double y,const double z) { return sin(y)*sin(z) + 4.28e4; }
double MMS_T(const double y,const double z) { return sin(y)*cos(z) + 800; }
double MMS_n(const double y,const double z) { return cos(y)*sin(z) + 3.0; }
double MMS_pl_sigmaxy(const double y,const double z,const double t) { return MMS_mu(y,z)*(MMS_uA_y(y,z,t) - MMS_gxy(y,z,t)); }
double MMS_pl_sigmaxz(const double y,const double z, const double t) { return MMS_mu(y,z)*(MMS_uA_z(y,z,t) - MMS_gxz(y,z,t)); }
double MMS_sigmadev(const double y,const double z,const double t)
{
  return sqrt( pow(MMS_pl_sigmaxy(y,z,t),2.0) + pow(MMS_pl_sigmaxz(y,z,t),2.0) );
}


// source terms for viscous strain rates
double MMS_pl_gxy_t_source(const double y,const double z,const double t)
{
  double A = MMS_A(y,z);
  double B = MMS_B(y,z);
  double n = MMS_n(y,z);
  double T = MMS_T(y,z);
  double sigmadev = MMS_sigmadev(y,z,t) * 1.0;
  double sigmaxy = MMS_pl_sigmaxy(y,z,t);
  double effVisc = 1.0/( A*pow(sigmadev,n-1.0)*exp(-B/T) ) * 1e-3;
  double v = sigmaxy/effVisc;

  return MMS_gxy_t(y,z,t) - v;
}
double MMS_pl_gxz_t_source(const double y,const double z,const double t)
{
  double A = MMS_A(y,z);
  double B = MMS_B(y,z);
  double n = MMS_n(y,z);
  double T = MMS_T(y,z);
  double sigmadev = MMS_sigmadev(y,z,t);
  double sigmaxz = MMS_pl_sigmaxz(y,z,t);
  double effVisc = 1.0/( A*pow(sigmadev,n-1.0)*exp(-B/T) ) * 1e-3;
  double v = sigmaxz/effVisc;

  return MMS_gxz_t(y,z,t) - v;
}

double MMS_uSource(const double y,const double z,const double t)
{
  PetscScalar mu = MMS_mu(y,z);
  PetscScalar mu_y = MMS_mu_y(y,z);
  PetscScalar mu_z = MMS_mu_z(y,z);
  PetscScalar u_y = MMS_uA_y(y,z,t);
  PetscScalar u_yy = MMS_uA_yy(y,z,t);
  PetscScalar u_z = MMS_uA_z(y,z,t);
  PetscScalar u_zz = MMS_uA_zz(y,z,t);
  return mu*(u_yy + u_zz) + mu_y*u_y + mu_z*u_z;
}*/



//======================================================================
// 1D MMS
// version 1
//~ double MMS_f1D(const double y) { return cos(y) + 2; } // helper function for uA
//~ double MMS_f_y1D(const double y) { return -sin(y); }
//~ double MMS_f_yy1D(const double y) { return -cos(y); }
//~ double MMS_f_z1D(const double y) { return 0; }
//~ double MMS_f_zz1D(const double y) { return 0; }

//~ double MMS_uA1D(const double y,const double t) { return MMS_f1D(y)*exp(-t); }
//~ double MMS_uA_y1D(const double y,const double t) { return MMS_f_y1D(y)*exp(-t); }
//~ double MMS_uA_yy1D(const double y,const double t) { return MMS_f_yy1D(y)*exp(-t); }
//~ double MMS_uA_z1D(const double y,const double t) { return 0; }
//~ double MMS_uA_zz1D(const double y,const double t) { return 0; }
//~ double MMS_uA_t1D(const double y,const double t) { return -MMS_f1D(y)*exp(-t); }

//~ double MMS_mu1D(const double y) { return sin(y) + 2.0; }
//~ double MMS_mu_y1D(const double y) { return cos(y); }
//~ double MMS_mu_z1D(const double y) { return 0; }

//~ double MMS_sigmaxy1D(const double y,const double t) { return MMS_mu1D(y)*MMS_uA_y1D(y,t); }


// specific MMS functions
/*
double MMS_visc1D(const double y) { return cos(y) + 20.0; }
double MMS_invVisc1D(const double y) { return 1.0/(cos(y) + 20.0); }
double MMS_invVisc_y1D(const double y) { return sin(y)/pow( cos(y)+20.0, 2.0); }
double MMS_invVisc_z1D(const double y) { return 0; }

double MMS_gxy1D(const double y,const double t)
{
  double A = MMS_mu1D(y)*MMS_invVisc1D(y);
  double fy = MMS_f_y1D(y);
  return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
}
double MMS_gxy_y1D(const double y,const double t)
{
  double A = MMS_mu1D(y)*MMS_invVisc1D(y);
  double Ay = MMS_mu_y1D(y)*MMS_invVisc1D(y) + MMS_mu1D(y)*MMS_invVisc_y1D(y);
  double fy = MMS_f_y1D(y);
  double fyy = MMS_f_yy1D(y);
  double den = A-1.0, B = exp(-t)-exp(-A*t);
  return t*A*Ay*fy*exp(-A*t)/den - A*fy*Ay*B/pow(den,2.0) + fy*Ay*B/den + A*fyy*B/den;
}
double MMS_gxy_t1D(const double y,const double t)
{
  double A = MMS_mu1D(y)*MMS_invVisc1D(y);
  double fy = MMS_f_y1D(y);
  return A*fy*(-exp(-t) + A*exp(-A*t))/(A-1.0);
}

double MMS_gxz1D(const double y,const double t)
{
  //~ double A = MMS_mu(y)*MMS_invVisc(y);
  //~ double fz = MMS_f_z(y);
  return 0;
}
double MMS_gxz_z1D(const double y,const double t)
{
  //~ double A = MMS_mu(y)*MMS_invVisc(y);
  //~ double Az = MMS_mu_z(y)*MMS_invVisc(y) + MMS_mu(y)*MMS_invVisc_z(y);
  //~ double fz = MMS_f_z(y);
  //~ double fzz = MMS_f_zz(y);
  //~ double den = A-1.0, B = exp(-t)-exp(-A*t);
  return 0;
}
double MMS_gxz_t1D(const double y,const double t)
{
  //~ double A = MMS_mu(y)*MMS_invVisc(y);
  //~ double fz = MMS_f_z(y);
  return 0;
}

double MMS_gSource1D(const double y,const double t)
{
  PetscScalar mu = MMS_mu1D(y);
  PetscScalar mu_y = MMS_mu_y1D(y);
  //~ PetscScalar mu_z = MMS_mu_z1D(y);
  PetscScalar gxy = MMS_gxy1D(y,t);
  //~ PetscScalar gxz = MMS_gxz1D(y,t);
  PetscScalar gxy_y = MMS_gxy_y1D(y,t);
  //~ PetscScalar gxz_z = MMS_gxz_z1D(y,t);
  //~ return -mu*(gxy_y + gxz_z) - mu_y*gxy - mu_z*gxz; // full answer
  return -mu*gxy_y - mu_y*gxy;
}



// specific to power law
double MMS_A1D(const double y) { return cos(y) + 1e-9; }
double MMS_B1D(const double y) { return sin(y) + 1.44e4; }
double MMS_T1D(const double y) { return sin(y) + 600; }
double MMS_n1D(const double y) { return cos(y) + 3.0; }
double MMS_pl_sigmaxy1D(const double y,const double t) { return MMS_mu1D(y)*(MMS_uA_y1D(y,t) - MMS_gxy1D(y,t)); }
double MMS_pl_sigmaxz1D(const double y,const double t) { return 0; }
double MMS_sigmadev1D(const double y,const double t) { return sqrt( pow(MMS_pl_sigmaxy1D(y,t),2.0)); }


// source terms for viscous strain rates
double MMS_pl_gxy_t_source1D(const double y,const double t)
{
  double A = MMS_A1D(y);
  double B = MMS_B1D(y);
  double n = MMS_n1D(y);
  double T = MMS_T1D(y);
  double sigmadev = MMS_sigmadev1D(y,t);
  double sigmaxy = MMS_pl_sigmaxy1D(y,t);
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxy*1e-3;

  return MMS_gxy_t1D(y,t) - v;
}
double MMS_pl_gxz_t_source1D(const double y,const double t)
{
  double A = MMS_A1D(y);
  double B = MMS_B1D(y);
  double n = MMS_n1D(y);
  double T = MMS_T1D(y);
  double sigmadev = MMS_sigmadev1D(y,t);
  double sigmaxz = MMS_pl_sigmaxz1D(y,t);
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxz*1e-3;

  return MMS_gxz_t1D(y,t) - v;
}

//~ double MMS_uSource1D(const double y,const double t)
//~ {
  //~ PetscScalar mu = MMS_mu1D(y);
  //~ PetscScalar mu_y = MMS_mu_y1D(y);
  //~ PetscScalar mu_z = MMS_mu_z1D(y);
  //~ PetscScalar u_y = MMS_uA_y1D(y,t);
  //~ PetscScalar u_yy = MMS_uA_yy1D(y,t);
  //~ PetscScalar u_z = MMS_uA_z1D(y,t);
  //~ PetscScalar u_zz = MMS_uA_zz1D(y,t);
  //~ return mu*(u_yy + u_zz) + mu_y*u_y + mu_z*u_z;
//~ }
*/









//======================================================================
// 2D MMS for heat equation
// version 1
//~ double MMS_he1_rho(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he1_c(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he1_k(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he1_h(const double y,const double z)
//~ {
  //~ return 0.0;
//~ }

//~ double MMS_he1_T(const double y,const double z, const double t)
//~ {
  //~ return sin(y)*cos(z)*exp(-2*t);
//~ }
//~ double MMS_he1_T_t(const double y,const double z, const double t)
//~ {
  //~ return -2.0*sin(y)*cos(z)*exp(-2.0*t);
//~ }
//~ double MMS_he1_T_y(const double y,const double z, const double t)
//~ {
  //~ return cos(y)*cos(z)*exp(-2.0*t);
//~ }
//~ double MMS_he1_T_z(const double y,const double z, const double t)
//~ {
  //~ return -sin(y)*sin(z)*exp(-2.0*t);
//~ }


//~ // version 2
//~ double MMS_he2_rho(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he2_c(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he2_k(const double y,const double z)
//~ {
  //~ return 1.0;
//~ }

//~ double MMS_he2_h(const double y,const double z)
//~ {
  //~ return 0.0;
//~ }

//~ double MMS_he2_f(const double t)
//~ {
  //~ return 5.*sin(y)*cos(z) + 100.;
//~ }
//~ double MMS_he2_f_y(const double t)
//~ {
  //~ return 5.*cos(y)*cos(z);
//~ }
//~ double MMS_he2_f_z(const double t)
//~ {
  //~ return -5.*sin(y)*sin(z);
//~ }
//~ double MMS_he2_g(const double t)
//~ {
  //~ return sin(2.*t);
//~ }
//~ double MMS_he2_g_t(const double t)
//~ {
  //~ return 2.*cos(2.*t);
//~ }
//~ double MMS_he2_T(const double y,const double z, const double t)
//~ {
  //~ return MMS_he2_f(y,z)*MMS_he2_g(t);
//~ }
//~ double MMS_he2_T_t(const double y,const double z, const double t)
//~ {
  //~ return MMS_he2_f(y,z)*MMS_he2_g_t(t);
//~ }
//~ double MMS_he2_T_y(const double y,const double z, const double t)
//~ {
  //~ return MMS_he2_f_y(y,z)*MMS_he2_g(t);
//~ }
//~ double MMS_he2_T_z(const double y,const double z, const double t)
//~ {
  //~ return MMS_he2_f_z(y,z)*MMS_he2_g(t);
//~ }






PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const Vec& yV, const double t)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    ierr = VecGetValues(yV,1,&Ii,&y);CHKERRQ(ierr);
    //~ y = dy*(Ii/N);
    //~ z = dz*(Ii-N*(Ii/N));
    v = func(y,t);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double),const Vec& yV)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    ierr = VecGetValues(yV,1,&Ii,&y);CHKERRQ(ierr);
    v = func(y);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}


PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const Vec& yV,const Vec& zV, const double t)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    //~ y = dy*(Ii/N);
    //~ z = dz*(Ii-N*(Ii/N));
    ierr = VecGetValues(yV,1,&Ii,&y);CHKERRQ(ierr);
    ierr = VecGetValues(zV,1,&Ii,&z);CHKERRQ(ierr);
    v = func(y,z,t);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const Vec& yV,const Vec& zV)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    //~ y = dy*(Ii/N);
    //~ z = dz*(Ii-N*(Ii/N));
    ierr = VecGetValues(yV,1,&Ii,&y);CHKERRQ(ierr);
    ierr = VecGetValues(zV,1,&Ii,&z);CHKERRQ(ierr);
    v = func(y,z);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}







//~ // scalars args to func are: spatial variable, spatial variable
//~ PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  //~ const int N, const double h1, const double h2)
//~ {
  //~ PetscErrorCode ierr = 0;

  //~ if (N == 1) { mapToVec1Dt(vec,func,N,h1,h2); } // h2 is time
  //~ else { mapToVec2D(vec,func,N,h1,h2); } // h2 is spatial variable
  //~ return ierr;
//~ }

//~ // scalars args to func are: spatial variable, spatial variable
//~ PetscErrorCode mapToVec2D(Vec& vec, double(*func)(double,double),
  //~ const int N, const double dy, const double dz)
//~ {
  //~ PetscErrorCode ierr = 0;
  //~ PetscScalar y,z,v;
  //~ PetscInt Ii,Istart,Iend;
  //~ ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  //~ for (Ii=Istart; Ii<Iend; Ii++) {
    //~ y = dy*(Ii/N);
    //~ z = dz*(Ii-N*(Ii/N));
    //~ v = func(y,z);
    //~ ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  //~ }
  //~ ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  //~ return ierr;
//~ }

//~ // scalars args to func are: spatial variable, time
//~ PetscErrorCode mapToVec1Dt(Vec& vec, double(*func)(double,double),
  //~ const int N, const double dy, const double t)
//~ {
  //~ PetscErrorCode ierr = 0;
  //~ PetscScalar y,v;
  //~ PetscInt Ii,Istart,Iend;
  //~ ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  //~ for (Ii=Istart; Ii<Iend; Ii++) {
    //y = dy*(Ii/N);
    //~ v = func(y,t);
    //~ ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  //~ }
  //~ ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  //~ return ierr;
//~ }

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
