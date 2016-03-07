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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"  Attempting to load: %s%s\n",inputDir.c_str(),fieldName.c_str());CHKERRQ(ierr);
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


double MMS_f(const double y,const double z) { return cos(y)*sin(z); } // helper function for uA
double MMS_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double MMS_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double MMS_f_z(const double y,const double z) { return cos(y)*cos(z); }
double MMS_f_zz(const double y,const double z) { return -cos(y)*sin(z); }

double MMS_uA(const double y,const double z,const double t) { return MMS_f(y,z)*exp(-t); }
double MMS_uA_y(const double y,const double z,const double t) { return MMS_f_y(y,z)*exp(-t); }
double MMS_uA_yy(const double y,const double z,const double t) { return MMS_f_yy(y,z)*exp(-t); }
double MMS_uA_z(const double y,const double z,const double t) { return MMS_f_z(y,z)*exp(-t); }
double MMS_uA_zz(const double y,const double z,const double t) { return MMS_f_zz(y,z)*exp(-t); }
double MMS_uA_t(const double y,const double z,const double t) { return -MMS_f(y,z)*exp(-t); }

double MMS_mu(const double y,const double z) { return sin(y)*sin(z) + 2.0; }
double MMS_mu_y(const double y,const double z) { return cos(y)*sin(z); }
double MMS_mu_z(const double y,const double z) { return sin(y)*cos(z); }

double MMS_sigmaxy(const double y,const double z,const double t) { return MMS_mu(y,z)*MMS_uA_y(y,z,t); }
double MMS_sigmaxz(const double y,const double z, const double t) { return MMS_mu(y,z)*MMS_uA_z(y,z,t); }


// specific MMS functions
double MMS_visc(const double y,const double z) { return cos(y)*cos(z) + 20.0; }
double MMS_invVisc(const double y,const double z) { return 1.0/(cos(y)*cos(z) + 20.0); }
double MMS_invVisc_y(const double y,const double z) { return sin(y)*cos(z)/pow( cos(y)*cos(z)+20.0, 2.0); }
double MMS_invVisc_z(const double y,const double z) { return cos(y)*sin(z)/pow( cos(y)*cos(z)+20.0 ,2.0); }

double MMS_gxy(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fy = MMS_f_y(y,z);
  return A*fy/(A-1.0)*(exp(-t) - exp(-A*t));
}
double MMS_gxy_y(const double y,const double z,const double t)
{
  //~return 0.5 * MMS_uA_yy(y,z,t);
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double Ay = MMS_mu_y(y,z)*MMS_invVisc(y,z) + MMS_mu(y,z)*MMS_invVisc_y(y,z);
  double fy = MMS_f_y(y,z);
  double fyy = MMS_f_yy(y,z);
  double den = A-1.0, B = exp(-t)-exp(-A*t);
  return t*A*Ay*fy*exp(-A*t)/den - A*fy*Ay*B/pow(den,2.0) + fy*Ay*B/den + A*fyy*B/den;
}
double MMS_gxy_t(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fy = MMS_f_y(y,z);
  return A*fy*(-exp(-t) + A*exp(-A*t))/(A-1.0);
}

double MMS_gxz(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fz = MMS_f_z(y,z);
  return A*fz/(A-1.0)*(exp(-t) - exp(-A*t));
}
double MMS_gxz_z(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double Az = MMS_mu_z(y,z)*MMS_invVisc(y,z) + MMS_mu(y,z)*MMS_invVisc_z(y,z);
  double fz = MMS_f_z(y,z);
  double fzz = MMS_f_zz(y,z);
  double den = A-1.0, B = exp(-t)-exp(-A*t);
  return t*A*Az*fz*exp(-A*t)/den - A*fz*Az*B/pow(den,2.0) + fz*Az*B/den + A*fzz*B/den;
}
double MMS_gxz_t(const double y,const double z,const double t)
{
  double A = MMS_mu(y,z)*MMS_invVisc(y,z);
  double fz = MMS_f_z(y,z);
  return A*fz/(A-1.0)*(-exp(-t) + A*exp(-A*t));
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
double MMS_A(const double y,const double z) { return cos(y)*cos(z) + 3.0; }
double MMS_B(const double y,const double z) { return sin(y)*sin(z) + 2.0; }
double MMS_T(const double y,const double z) { return sin(y)*cos(z) + 2.0; }
double MMS_n(const double y,const double z) { return cos(y)*sin(z) + 2.0; }
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
  double sigmadev = MMS_sigmadev(y,z,t);
  double sigmaxy = MMS_pl_sigmaxy(y,z,t);
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxy*1e-3;

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
  double v = A*pow(sigmadev,n-1.0)*exp(-B/T)*sigmaxz*1e-3;

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
}



// Map a function that acts on scalars to a 2D Vec
PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double,double),
  const int N, const double dy, const double dz, const double t)
{
  //~PetscPrintf(PETSC_COMM_WORLD,"N = %i, dy = %g, dz = %g, t = %g\n",N,dy,dz,t);

  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    y = dy*(Ii/N);
    z = dz*(Ii-N*(Ii/N));
    //~PetscPrintf(PETSC_COMM_WORLD,"%i: (y,z) = (%e,%e)\n",Ii,y,z);
    v = func(y,z,t);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode mapToVec(Vec& vec, double(*func)(double,double),
  const int N, const double dy, const double dz)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    y = dy*(Ii/N);
    z = dz*(Ii-N*(Ii/N));
    v = func(y,z);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
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
