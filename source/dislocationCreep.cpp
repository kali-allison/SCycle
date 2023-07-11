#include "dislocationCreep.hpp"

#define FILENAME "dislocationCreep.cpp"

using namespace std;


//======================================================================
// dislocation creep class

DislocationCreep::DislocationCreep(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim,const string prefix)
  : _file(file),_delim(delim),_inputDir("unspecified"),_prefix(prefix),_y(&y),_z(&z),
  _A(NULL),_n(NULL),_QR(NULL),_invEffVisc(NULL)
{
  #if VERBOSE > 1
    string funcName = "DislocationCreep::DislocationCreep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings();
  checkInput();
  setMaterialParameters();
  if (!D._restartFromChkpt && !D._restartFromChkptSS) {
    loadFieldsFromFiles(_prefix);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

DislocationCreep::~DislocationCreep()
{
  #if VERBOSE > 1
    string funcName = "DislocationCreep::~DislocationCreep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_A);
  VecDestroy(&_n);
  VecDestroy(&_QR);
  VecDestroy(&_invEffVisc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode DislocationCreep::loadSettings()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "DislocationCreep::loadSettings()";
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
    else if (var.compare("disl" + _prefix + "_AVals")==0) { loadVectorFromInputFile(rhsFull,_AVals); }
    else if (var.compare("disl" + _prefix + "_ADepths")==0) { loadVectorFromInputFile(rhsFull,_ADepths); }
    else if (var.compare("disl" + _prefix + "_QRVals")==0) { loadVectorFromInputFile(rhsFull,_QRVals); }
    else if (var.compare("disl" + _prefix + "_QRDepths")==0) { loadVectorFromInputFile(rhsFull,_QRDepths); }
    else if (var.compare("disl" + _prefix + "_nVals")==0) { loadVectorFromInputFile(rhsFull,_nVals); }
    else if (var.compare("disl" + _prefix + "_nDepths")==0) { loadVectorFromInputFile(rhsFull,_nDepths); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DislocationCreep::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    assert(_AVals.size() >= 2);
    assert(_QRVals.size() >= 2);
    assert(_nVals.size() >= 2);
    assert(_AVals.size() == _ADepths.size() );
    assert(_QRVals.size() == _QRDepths.size() );
    assert(_nVals.size() == _nDepths.size() );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DislocationCreep::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(*_y,&_A);  setVec(_A,*_z,_AVals,_ADepths); PetscObjectSetName((PetscObject) _A, "A");
  VecDuplicate(*_y,&_QR);  setVec(_QR,*_z,_QRVals,_QRDepths); PetscObjectSetName((PetscObject) _QR, "QR");
  VecDuplicate(*_y,&_n);  setVec(_n,*_z,_nVals,_nDepths); PetscObjectSetName((PetscObject) _n, "n");

  VecDuplicate(*_y,&_invEffVisc); VecSet(_invEffVisc,1.0);
  PetscObjectSetName((PetscObject) _invEffVisc, "invEffVisc");

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DislocationCreep::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_A,_inputDir,"disl_A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"disl_QR"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,_inputDir,"disl_n"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_invEffVisc,_inputDir,"disl_invEffVisc"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// load input binary files that follow form:
// fileName = _inputDir + "disl" + prefix + "_" + <name of field>
PetscErrorCode DislocationCreep::loadFieldsFromFiles(const string prefix)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::loadFieldsFromFiles(string prefix)";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string fullPrefix = _inputDir + "disl" + prefix + "_";

  ierr = loadVecFromInputFile(_A,fullPrefix,"A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,fullPrefix,"QR"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_n,fullPrefix,"n"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_invEffVisc,fullPrefix,"invEffVisc"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode DislocationCreep::writeContext(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string groupName = "/momBal/dislocationCreep" + _prefix;

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, groupName.c_str());           CHKERRQ(ierr);
  ierr = VecView(_A, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_QR, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_n, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DislocationCreep::loadCheckpoint(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string groupName = "/momBal/dislocationCreep" + _prefix;

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, groupName.c_str());           CHKERRQ(ierr);
  ierr = VecLoad(_A, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_QR, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_n, viewer);                                           CHKERRQ(ierr);
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
PetscErrorCode DislocationCreep::guessInvEffVisc(const Vec& Temp, const double dg)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::guessInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *A,*QR,*n,*T;
  PetscScalar *invEffVisc;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&QR);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(Temp,&T);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar temp = A[Jj] * exp(-QR[Jj]/T[Jj]);
    PetscScalar s = pow( dg/temp, 1.0/n[Jj] );
    invEffVisc[Jj] = dg / s;
    Jj++;
  }
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&QR);
  VecRestoreArrayRead(_n,&n);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArray(_invEffVisc,&invEffVisc);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute 1 / (effective viscosity)
PetscErrorCode DislocationCreep::computeInvEffVisc(const Vec& Temp,const Vec& sdev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DislocationCreep::computeInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *s,*A,*QR,*n,*T;
  PetscScalar *invEffVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(sdev,&s);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&QR);
  VecGetArrayRead(_n,&n);
  VecGetArrayRead(Temp,&T);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    assert(!std::isnan(s[Jj]));
    invEffVisc[Jj] = 1e3 * A[Jj] * pow(s[Jj],n[Jj]-1.0) * exp(-QR[Jj]/T[Jj]);
    Jj++;
  }
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&QR);
  VecRestoreArrayRead(_n,&n);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
