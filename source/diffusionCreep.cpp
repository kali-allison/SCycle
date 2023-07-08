#include "diffusionCreep.hpp"

#define FILENAME "diffusionCreep.cpp"

using namespace std;


//======================================================================
// diffusion creep class


DiffusionCreep::DiffusionCreep(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim)
  : _file(file),_delim(delim),_inputDir("unspecified"),_y(&y),_z(&z),
  _A(NULL),_n(NULL),_QR(NULL),_m(NULL),_invEffVisc(NULL)
{
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::DiffusionCreep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings();
  checkInput();
  setMaterialParameters();
  if (!D._restartFromChkpt && !D._restartFromChkptSS) {
    loadFieldsFromFiles();
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

DiffusionCreep::~DiffusionCreep()
{
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::~DiffusionCreep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_QR);
  VecDestroy(&_m);
  VecDestroy(&_invEffVisc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode DiffusionCreep::loadSettings()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "DiffusionCreep::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  ifstream infile( _file );
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);
    rhs = "";
    if (line.length() > (pos + _delim.length())) {
      rhs = line.substr(pos+_delim.length(),line.npos);
    }
    rhsFull = rhs; // everything after _delim

    // interpret everything after the appearance of a space on the line as a comment
    pos = rhs.find(" ");
    rhs = rhs.substr(0,pos);

    if (var.compare("inputDir") == 0) { _inputDir = rhs; }
    else if (var.compare("diff_AVals")==0) { loadVectorFromInputFile(rhsFull,_AVals); }
    else if (var.compare("diff_ADepths")==0) { loadVectorFromInputFile(rhsFull,_ADepths); }
    else if (var.compare("diff_QRVals")==0) { loadVectorFromInputFile(rhsFull,_QRVals); }
    else if (var.compare("diff_QRDepths")==0) { loadVectorFromInputFile(rhsFull,_QRDepths); }
    else if (var.compare("diff_nVals")==0) { loadVectorFromInputFile(rhsFull,_nVals); }
    else if (var.compare("diff_nDepths")==0) { loadVectorFromInputFile(rhsFull,_nDepths); }
    else if (var.compare("diff_mVals")==0) { loadVectorFromInputFile(rhsFull,_mVals); }
    else if (var.compare("diff_mDepths")==0) { loadVectorFromInputFile(rhsFull,_mDepths); }

  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DiffusionCreep::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    assert(_AVals.size() >= 2);
    assert(_QRVals.size() >= 2);
    assert(_nVals.size() >= 2);
    assert(_mVals.size() >= 2);
    assert(_AVals.size() == _ADepths.size() );
    assert(_QRVals.size() == _QRDepths.size() );
    assert(_nVals.size() == _nDepths.size() );
    assert(_mVals.size() == _mDepths.size() );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DiffusionCreep::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(*_y,&_A);  setVec(_A,*_z,_AVals,_ADepths); PetscObjectSetName((PetscObject) _A, "A");
  VecDuplicate(_A,&_QR); setVec(_QR,*_z,_QRVals,_QRDepths); PetscObjectSetName((PetscObject) _QR, "QR");
  VecDuplicate(_A,&_n);  setVec(_n,*_z,_nVals,_nDepths); PetscObjectSetName((PetscObject) _n, "n");
  VecDuplicate(_A,&_m);  setVec(_m,*_z,_mVals,_mDepths); PetscObjectSetName((PetscObject) _m, "m");

  VecDuplicate(_A,&_invEffVisc); VecSet(_invEffVisc,1.0); PetscObjectSetName((PetscObject) _invEffVisc, "invEffVisc");

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DiffusionCreep::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_A,_inputDir,"diff_A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"diff_QR"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"diff_n"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_m,_inputDir,"diff_m"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_invEffVisc,_inputDir,"diff_invEffVisc"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DiffusionCreep::writeContext(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");    CHKERRQ(ierr);
  ierr = VecView(_A, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_QR, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_n, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_m, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DiffusionCreep::loadCheckpoint(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/diffusionCreep");    CHKERRQ(ierr);
  ierr = VecLoad(_A, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_QR, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_n, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_m, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// estimate 1 / (effective viscosity) from a reference strain rate
// dg is in 1e-3 s^-1
// let A = A exp(-Q/RT) d^-m
// dg = A s^n -> s = (dg/A)^(1/n)
// dg = s / v -> v = s/dg OR 1/v = dg/s
PetscErrorCode DiffusionCreep::guessInvEffVisc(const Vec& Temp,const double dg,const Vec& grainSize)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::guessInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *A,*QR,*n,*T,*d,*m;
  PetscScalar *invEffVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(grainSize,&d);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&QR);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(_m,&m);
  VecGetArrayRead(Temp,&T);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar temp = A[Jj] * exp(-QR[Jj]/T[Jj]) * pow(d[Jj],-m[Jj]);
    PetscScalar s = pow( dg/temp, 1.0/n[Jj] );
    invEffVisc[Jj] = dg / s;
    Jj++;
  }
  VecRestoreArrayRead(grainSize,&d);
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&QR);
  VecRestoreArrayRead(_n,&n);
  VecRestoreArrayRead(_m,&m);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute 1 / (effective viscosity)
PetscErrorCode DiffusionCreep::computeInvEffVisc(const Vec& Temp,const Vec& sdev,const Vec& grainSize)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DiffusionCreep::computeInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *s,*A,*QR,*n,*T,*d,*m;
  PetscScalar *invEffVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(sdev,&s);
  VecGetArrayRead(grainSize,&d);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&QR);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(_m,&m);
  VecGetArrayRead(Temp,&T);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    assert(!std::isnan(d[Jj]));
    assert(!std::isnan(s[Jj]));
    invEffVisc[Jj] = 1e3 * A[Jj] * pow(s[Jj],n[Jj]-1.0) * exp(-QR[Jj]/T[Jj]) * pow(d[Jj],-m[Jj]);
    Jj++;
  }
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArrayRead(grainSize,&d);
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&QR);
  VecRestoreArrayRead(_n,&n);
  VecRestoreArrayRead(_m,&m);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
