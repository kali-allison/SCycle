#include "dissolutionPrecipitationCreep.hpp"

#define FILENAME "dissolutionPrecipitationCreep.cpp"

using namespace std;


//======================================================================
// dissolution-precipitation creep class

DissolutionPrecipitationCreep::DissolutionPrecipitationCreep(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim)
  : _file(file),_delim(delim),_inputDir("unspecified"),_y(&y),_z(&z),_R(8.3144e-3),
  _B(NULL),_D(NULL),_c(NULL),_Vs(NULL),_m(NULL),_invEffVisc(NULL)
{
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::DissolutionPrecipitationCreep";
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

DissolutionPrecipitationCreep::~DissolutionPrecipitationCreep()
{
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::~DissolutionPrecipitationCreep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_B);
  VecDestroy(&_D);
  VecDestroy(&_c);
  VecDestroy(&_Vs);
  VecDestroy(&_m);
  VecDestroy(&_invEffVisc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode DissolutionPrecipitationCreep::loadSettings()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::loadSettings()";
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
    else if (var.compare("dp_BVals")==0) { loadVectorFromInputFile(rhsFull,_BVals); }
    else if (var.compare("dp_BDepths")==0) { loadVectorFromInputFile(rhsFull,_BDepths); }
    else if (var.compare("dp_DVals")==0) { loadVectorFromInputFile(rhsFull,_DVals); }
    else if (var.compare("dp_DDepths")==0) { loadVectorFromInputFile(rhsFull,_DDepths); }
    else if (var.compare("dp_cVals")==0) { loadVectorFromInputFile(rhsFull,_cVals); }
    else if (var.compare("dp_cDepths")==0) { loadVectorFromInputFile(rhsFull,_cDepths); }
    else if (var.compare("dp_VsVals")==0) { loadVectorFromInputFile(rhsFull,_VsVals); }
    else if (var.compare("dp_VsDepths")==0) { loadVectorFromInputFile(rhsFull,_VsDepths); }
    else if (var.compare("dp_mVals")==0) { loadVectorFromInputFile(rhsFull,_mVals); }
    else if (var.compare("dp_mDepths")==0) { loadVectorFromInputFile(rhsFull,_mDepths); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DissolutionPrecipitationCreep::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    assert(_BVals.size() >= 2);
    assert(_DVals.size() >= 2);
    assert(_cVals.size() >= 2);
    assert(_VsVals.size() >= 2);
    assert(_mVals.size() >= 2);
    assert(_BVals.size() == _BDepths.size() );
    assert(_DVals.size() == _DDepths.size() );
    assert(_cVals.size() == _cDepths.size() );
    assert(_VsVals.size() == _VsDepths.size() );
    assert(_mVals.size() == _mDepths.size() );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DissolutionPrecipitationCreep::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(*_y,&_B);  setVec(_B,*_z,_BVals,_BDepths); PetscObjectSetName((PetscObject) _B, "B");
  VecDuplicate(*_y,&_D);  setVec(_D,*_z,_DVals,_DDepths); PetscObjectSetName((PetscObject) _D, "D");
  VecDuplicate(*_y,&_c);  setVec(_c,*_z,_cVals,_cDepths); PetscObjectSetName((PetscObject) _c, "c");
  VecDuplicate(*_y,&_Vs);  setVec(_Vs,*_z,_VsVals,_VsDepths); PetscObjectSetName((PetscObject) _Vs, "Vs");
  VecDuplicate(*_y,&_m);  setVec(_m,*_z,_mVals,_mDepths); PetscObjectSetName((PetscObject) _m, "m");

  VecDuplicate(*_y,&_invEffVisc); VecSet(_invEffVisc,1.0);
  PetscObjectSetName((PetscObject) _invEffVisc, "invEffVisc");

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DissolutionPrecipitationCreep::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_B,_inputDir,"dp_B"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_D,_inputDir,"dp_D"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_c,_inputDir,"dp_c"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_Vs,_inputDir,"dp_Vs"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_m,_inputDir,"dp_m"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_invEffVisc,_inputDir,"dp_invEffVisc"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode DissolutionPrecipitationCreep::writeContext(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");  CHKERRQ(ierr);
  ierr = VecView(_B, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_D, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Vs, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_m, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode DissolutionPrecipitationCreep::loadCheckpoint(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/dissolutionPrecipitationCreep");  CHKERRQ(ierr);
  ierr = VecLoad(_B, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_D, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Vs, viewer);                                          CHKERRQ(ierr);
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
// let A = A fh2o^rd^-m
// let Q = 3*Vs / (R*T)
// dg = A s^n -> s = (dg/A)^(1/n)
// dg = s / v -> v = s/dg OR 1/v = dg/s
PetscErrorCode DissolutionPrecipitationCreep::guessInvEffVisc(const Vec& Temp,const double dg,const Vec& grainSize, const Vec& WetDistribution)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::guessInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *B,*D,*c,*Vs,*m,*d,*T,*wetDist;
  PetscScalar *invEffVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(_B,&B);
  VecGetArrayRead(_Vs,&Vs);
  VecGetArrayRead(_D,&D);
  VecGetArrayRead(_c,&c);
  VecGetArrayRead(_m,&m);
  VecGetArrayRead(grainSize,&d);
  VecGetArrayRead(Temp,&T);
  VecGetArrayRead(WetDistribution,&wetDist);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {

    PetscScalar A = std::sqrt(3)*B[Jj] * D[Jj] * c[Jj] * Vs[Jj] * pow(d[Jj],-m[Jj]) * wetDist[Jj];
    // first 3.0 from equation from Manon and Sandra
    // 3e3 converts to sdev in MPa
    // sqrt(3) is for conversion from differential stress to dev stress
    PetscScalar Q = 3.0*std::sqrt(3.0)*1e3*Vs[Jj] / (_R*T[Jj]); // everything in the exponential except deviatoric stress

    PetscScalar s = (1.0/Q) * log((1.0/A) * dg + 1.0); // log = natural log
    invEffVisc[Jj] = dg / s;

    Jj++;
  }
  VecRestoreArrayRead(_B,&B);
  VecRestoreArrayRead(_Vs,&Vs);
  VecRestoreArrayRead(_D,&D);
  VecRestoreArrayRead(_c,&c);
  VecRestoreArrayRead(_m,&m);
  VecRestoreArrayRead(grainSize,&d);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArrayRead(WetDistribution,&wetDist);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute 1 / (effective viscosity)
PetscErrorCode DissolutionPrecipitationCreep::computeInvEffVisc(const Vec& Temp,const Vec& sdev,const Vec& grainSize, const Vec& WetDistribution)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "DissolutionPrecipitationCreep::computeInvEffVisc";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *s,*B,*D,*c,*Vs,*m,*d,*T,*wetDist;
  PetscScalar *invEffVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(sdev,&s);
  VecGetArrayRead(_B,&B);
  VecGetArrayRead(_Vs,&Vs);
  VecGetArrayRead(_D,&D);
  VecGetArrayRead(_c,&c);
  VecGetArrayRead(_m,&m);
  VecGetArrayRead(grainSize,&d);
  VecGetArrayRead(Temp,&T);
  VecGetArrayRead(WetDistribution,&wetDist);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    assert(!std::isnan(s[Jj]));
    PetscScalar num = 3.0*std::sqrt(3.0)*1e3 *Vs[Jj]*s[Jj];
    PetscScalar RT = _R*T[Jj];
    PetscScalar expVal = exp(num/RT);
    assert(~std::isnan(expVal));
    assert(~std::isinf(expVal));
    invEffVisc[Jj] = 1e3 * 2.0 * std::sqrt(3.0) * B[Jj] * D[Jj] * c[Jj] * Vs[Jj] * pow(d[Jj],-m[Jj]) * wetDist[Jj] * (expVal - 1.0);
    Jj++;
  }
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArrayRead(_B,&B);
  VecRestoreArrayRead(_Vs,&Vs);
  VecRestoreArrayRead(_D,&D);
  VecRestoreArrayRead(_c,&c);
  VecRestoreArrayRead(_m,&m);
  VecRestoreArrayRead(grainSize,&d);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArrayRead(WetDistribution,&wetDist);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
