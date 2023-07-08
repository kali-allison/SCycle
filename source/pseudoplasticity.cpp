#include "pseudoplasticity.hpp"

#define FILENAME "pseudoplasticity.cpp"

using namespace std;

//======================================================================
// pseudoplasticity class

Pseudoplasticity::Pseudoplasticity(Domain& D, const Vec& y, const Vec& z, const char *file, const string delim)
  : _file(file),_delim(delim),_inputDir("unspecified"),_y(&y),_z(&z)
{
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::Pseudoplasticity";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings();
  checkInput();
  setMaterialParameters();
  //~ loadFieldsFromFiles();
  if (!D._restartFromChkpt && !D._restartFromChkptSS) {
    loadFieldsFromFiles();
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

Pseudoplasticity::~Pseudoplasticity()
{
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::~Pseudoplasticity";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_yieldStress);
  VecDestroy(&_invEffVisc);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode Pseudoplasticity::loadSettings()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "Pseudoplasticity::loadSettings()";
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
    else if (var.compare("yieldStressVals")==0) { loadVectorFromInputFile(rhsFull,_yieldStressVals); }
    else if (var.compare("yieldStressDepths")==0) { loadVectorFromInputFile(rhsFull,_yieldStressDepths); }

  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Pseudoplasticity::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    assert(_yieldStressVals.size() >= 2);
    assert(_yieldStressVals.size() == _yieldStressDepths.size() );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Pseudoplasticity::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(*_y,&_yieldStress);
  ierr = setVec(_yieldStress,*_z,_yieldStressVals,_yieldStressDepths); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _yieldStress, "yieldStress");

  VecDuplicate(*_y,&_invEffVisc);
  VecSet(_invEffVisc,1.0);
  PetscObjectSetName((PetscObject) _invEffVisc, "invEffVisc");

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Pseudoplasticity::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    ierr = loadVecFromInputFile(_yieldStress,_inputDir,"plasticity_yieldStress"); CHKERRQ(ierr);
    ierr = loadVecFromInputFile(_invEffVisc,_inputDir,"plasticity_invEffVisc"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Pseudoplasticity::writeContext(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/pseudoplasticity");          CHKERRQ(ierr);
  ierr = VecView(_yieldStress, viewer);                                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Pseudoplasticity::loadCheckpoint(PetscViewer &viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/momBal/pseudoplasticity");          CHKERRQ(ierr);
  ierr = VecLoad(_yieldStress, viewer);                                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute 1 / (effective viscosity) based on assumed strain rate
// v = sy / dg -> 1/v = dg/sy
PetscErrorCode Pseudoplasticity::guessInvEffVisc(const double dg)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::guessInvEffVisc()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(_yieldStress,_invEffVisc);
  VecReciprocal(_invEffVisc);
  VecScale(_invEffVisc,dg);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute 1 / (effective viscosity)
PetscErrorCode Pseudoplasticity::computeInvEffVisc(const Vec& dgdev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Pseudoplasticity::computeInvEffVisc()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar const *dg=0,*sy=0;
  PetscScalar *invEffVisc=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_invEffVisc,&Istart,&Iend);
  VecGetArrayRead(_yieldStress,&sy);
  VecGetArrayRead(dgdev,&dg);
  VecGetArray(_invEffVisc,&invEffVisc);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    invEffVisc[Jj] = dg[Jj] / sy[Jj];
    Jj++;
  }
  VecRestoreArrayRead(dgdev,&dg);
  VecRestoreArray(_invEffVisc,&invEffVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
