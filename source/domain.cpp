#include "domain.hpp"

#define FILENAME "domain.cpp"

using namespace std;

// member function definitions including constructor
// first type of constructor with 1 parameter
Domain::Domain(const char *file)
  : _file(file),_delim(" = "),_inputDir("unspecified_"),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),
  _momentumBalanceType("quasidynamic"),_systemEvolutionType("transient"),
  _operatorType("matrix-based"),_sbpCompatibilityType("fullyCompatible"),
  _gridSpacingType("variableGridSpacing"),_isMMS(0),_computeGreensFunction_fault(0),_computeGreensFunction_offFault(0),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),_vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(1),_dr(1),
  _bCoordTrans(-1),
  _saveChkpts(1), _restartFromChkpt(1),_restartFromChkptSS(0),_outputFileMode(FILE_MODE_WRITE),_prevChkptTimeStep1D(0),_prevChkptTimeStep2D(0),
  _outFileMode(FILE_MODE_APPEND)
{
  #if VERBOSE > 1
    string funcName = "Domain::Domain(const char *file)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif
  //~ _ckpt(0), _ckptNumber(0), _interval(1e4),
  loadSettings(_file);
  checkInput();

  // grid spacing for logical coordinates
  if (_Ny > 1) { _dq = 1.0 / (_Ny - 1.0); }
  if (_Nz > 1) { _dr = 1.0 / (_Nz - 1.0); }

  #if VERBOSE > 2 // each processor prints loaded values to screen
    PetscMPIInt rank,size;
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    for (int Ii = 0; Ii < size; Ii++) {
      view(Ii);
    }
  #endif

  allocateFields();
  if (_restartFromChkpt) { loadCheckpoint(); }
  else if (_restartFromChkptSS) { loadCheckpointSS(); }
  if (!_restartFromChkpt && !_restartFromChkptSS) { setFields(); }
  setScatters();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

}


// second type of constructor with 3 parameters
Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
  : _file(file),_delim(" = "),_inputDir("unspecified_"),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_momentumBalanceType("quasidynamic"),_systemEvolutionType("transient"),
  _operatorType("matrix-based"),_sbpCompatibilityType("fullyCompatible"),
  _gridSpacingType("variableGridSpacing"),_isMMS(0),_computeGreensFunction_fault(0),_computeGreensFunction_offFault(0),
  _order(4),_Ny(Ny),_Nz(Nz),_Ly(-1),_Lz(-1),_vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(1),_dr(1),
  _bCoordTrans(-1),
  _saveChkpts(1), _restartFromChkpt(1),_restartFromChkptSS(1),_outputFileMode(FILE_MODE_APPEND),_prevChkptTimeStep1D(0),_prevChkptTimeStep2D(0),
  _outFileMode(FILE_MODE_WRITE)
{
  #if VERBOSE > 1
    string funcName = "Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();

  _Ny = Ny;
  _Nz = Nz;

  if (_Ny > 1) { _dq = 1.0/(_Ny-1.0); }
  if (_Nz > 1) { _dr = 1.0/(_Nz-1.0); }

  #if VERBOSE > 2 // each processor prints loaded values to screen
    PetscMPIInt rank,size;
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    for (int Ii = 0; Ii < size; Ii++) {
      view(Ii);
    }
  #endif

  allocateFields();
  if (_restartFromChkpt == 1) { loadCheckpoint(); }
  if (_restartFromChkpt == 0) { setFields(); }
  setScatters();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// destructor
Domain::~Domain()
{
  #if VERBOSE > 1
    string funcName = "Domain::~Domain";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  // free memory
  VecDestroy(&_q);
  VecDestroy(&_r);
  VecDestroy(&_y);
  VecDestroy(&_z);
  VecDestroy(&_y0);
  VecDestroy(&_z0);

  // set map iterator, free memory from VecScatter
  map<string,VecScatter>::iterator it;
  for (it = _scatters.begin(); it != _scatters.end(); it++) {
    VecScatterDestroy(&it->second);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// load settings from input file
PetscErrorCode Domain::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
  PetscMPIInt rank,size;

  #if VERBOSE > 1
    string funcName = "Domain::loadData";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  MPI_Comm_size(PETSC_COMM_WORLD,&size); // global number of processors
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); // current processor number

  // read file inputs
  ifstream infile(file);
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

    // set variables, convert string to ints/floats
    if (var.compare("order") == 0) { _order = atoi(rhs.c_str()); }
    else if (var.compare("Ny") == 0 && _Ny < 0) { _Ny = atoi(rhs.c_str()); }
    else if (var.compare("Nz") == 0 && _Nz < 0) { _Nz = atoi(rhs.c_str()); }
    else if (var.compare("Ly") == 0) { _Ly = atof(rhs.c_str()); }
    else if (var.compare("Lz") == 0) { _Lz = atof(rhs.c_str()); }

    else if (var.compare("inputDir") == 0) { _inputDir = rhs; }
    else if (var.compare("outputDir")==0) { _outputDir =  rhs; }

    else if (var.compare("operatorType")==0) { _operatorType = rhs; }
    else if (var.compare("sbpCompatibilityType")==0) { _sbpCompatibilityType = rhs; }
    else if (var.compare("gridSpacingType")==0) { _gridSpacingType = rhs; }
    else if (var.compare("bulkDeformationType")==0) { _bulkDeformationType = rhs; }
    else if (var.compare("momentumBalanceType")==0) { _momentumBalanceType = rhs; }
    else if (var.compare("systemEvolutionType")==0) { _systemEvolutionType = rhs; }
    else if (var.compare("isMMS") == 0) { _isMMS = atoi(rhs.c_str()); }
    else if (var.compare("computeGreensFunction_fault") == 0) { _computeGreensFunction_fault = atoi(rhs.c_str()); }
    else if (var.compare("computeGreensFunction_offFault") == 0) { _computeGreensFunction_offFault = atoi(rhs.c_str()); }


    else if (var.compare("bCoordTrans")==0) { _bCoordTrans = atof( rhs.c_str() ); }
    else if (var.compare("vL")==0) { _vL = atof( rhs.c_str() ); }

    else if (var.compare("saveChkpts") == 0) { _saveChkpts = atoi(rhs.c_str()); }
    else if (var.compare("restartFromChkpt") == 0) { _restartFromChkpt = atoi(rhs.c_str()); }
    else if (var.compare("restartFromChkptSS") == 0) { _restartFromChkptSS = atoi(rhs.c_str()); }

    //~ else if (var.compare("enableCheckpointing") == 0) { _ckpt = atoi(rhs.c_str()); }
    //~ else if (var.compare("interval") == 0) { _interval = (int)atof(rhs.c_str()); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Specified processor prints scalar/string data members to std::out.
PetscErrorCode Domain::view(PetscMPIInt rank)
{
  PetscErrorCode ierr = 0;
  PetscMPIInt localRank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);

  if (localRank==rank) {
    #if VERBOSE > 1
      string funcName = "Domain::view";
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
    #endif

    // start printing all the inputs
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n\nrank=%i in Domain::view\n",rank); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"order = %i\n",_order);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ny = %i\n",_Ny);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Nz = %i\n",_Nz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ly = %e\n",_Ly);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Lz = %e\n",_Lz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"isMMS = %i\n",_isMMS);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"momBalType = %s\n",_momentumBalanceType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bulkDeformationType = %s\n",_bulkDeformationType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"operatorType = %s\n",_operatorType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"sbpCompatibilityType = %s\n",_sbpCompatibilityType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"gridSpacingType = %s\n",_gridSpacingType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    #if VERBOSE > 1
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
    #endif
  }
  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode Domain::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_systemEvolutionType.compare("transient") == 0 ||
    _systemEvolutionType.compare("steadyStateIts") == 0);

  assert(_bulkDeformationType.compare("linearElastic") == 0 ||
    _bulkDeformationType.compare("powerLaw") == 0);

  assert(_sbpCompatibilityType.compare("fullyCompatible") == 0 ||
    _sbpCompatibilityType.compare("compatible") == 0);

  if (_bCoordTrans > 0.0) {
    _gridSpacingType = "variableGridSpacing";
  }

  assert(_gridSpacingType.compare("variableGridSpacing") == 0 ||
     _gridSpacingType.compare("constantGridSpacing") == 0);

  assert(_momentumBalanceType.compare("quasidynamic") == 0 ||
    _momentumBalanceType.compare("dynamic") == 0 ||
    _momentumBalanceType.compare("quasidynamic_and_dynamic") == 0 ||
    _momentumBalanceType.compare("steadyStateIts") == 0);

  assert(_order == 2 || _order == 4);
  assert(_Ly > 0 && _Lz > 0);
  assert(_dq > 0 && !std::isnan(_dq));
  assert(_dr > 0 && !std::isnan(_dr));

  //~ assert(_ckpt >= 0 && _ckptNumber >= 0);
  //~ assert(_interval >= 0);

  if (_systemEvolutionType == "steadyStateIts" && _restartFromChkpt == 1) {
    _restartFromChkpt = 0;
    _restartFromChkptSS = 1;
  }
  if (_systemEvolutionType == "steadyStateIts") {
    _outputFileMode = FILE_MODE_WRITE;
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Save all scalar fields to text file named domain.txt in output directory.
// Also writes out coordinate systems q, r, y, z into respective files in output directory
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode Domain::write(PetscViewer& viewer_hdf5)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::write";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  string str = _outputDir + "domain.txt";

  PetscViewer    viewer;
  // write into file using PetscViewer
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII); CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, str.c_str()); CHKERRQ(ierr);

  // domain properties
  ierr = PetscViewerASCIIPrintf(viewer,"order = %i\n",_order);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Ny = %i\n",_Ny);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Nz = %i\n",_Nz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Ly = %g # (km)\n",_Ly);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Lz = %g # (km)\n",_Lz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"isMMS = %i\n",_isMMS);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeGreensFunction_fault = %i\n",_computeGreensFunction_fault);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeGreensFunction_offFault = %i\n",_computeGreensFunction_offFault);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBalType = %s\n",_momentumBalanceType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bulkDeformationType = %s\n",_bulkDeformationType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"operatorType = %s\n",_operatorType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"gridSpacingType = %s\n",_gridSpacingType.c_str());CHKERRQ(ierr);

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"bCoordTrans = %.15e\n",_bCoordTrans);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // checkpoint settings
  //~ ierr = PetscViewerASCIIPrintf(viewer,"checkpoint_enabled = %i\n",_ckpt);CHKERRQ(ierr);
  //~ ierr = PetscViewerASCIIPrintf(viewer,"checkpoint_number = %i\n",_ckptNumber);CHKERRQ(ierr);
  //~ ierr = PetscViewerASCIIPrintf(viewer,"checkpoint_interval = %i\n",_interval);CHKERRQ(ierr);

  // get number of processors
  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  // output coordinate system vectors
  writeHDF5(viewer_hdf5);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// output using HDF5 format
PetscErrorCode Domain::writeHDF5(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::writeHDF5";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/domain");            CHKERRQ(ierr);
  ierr = VecView(_q, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_r, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_y, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// output all information needed for checkpoint
PetscErrorCode Domain::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::writeCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/domain");                    CHKERRQ(ierr);
  ierr = VecView(_q, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_r, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_y, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode Domain::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::allocateFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // generate vector _y with size _Ny*_Nz
  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);

  ierr = VecDuplicate(_y,&_z); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_q); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_r); CHKERRQ(ierr);


  // name vectors for output
  ierr = PetscObjectSetName((PetscObject) _q, "q");                     CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _r, "r");                     CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y");                     CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _z, "z");                     CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// construct coordinate transform, setting vectors q, r, y, z
PetscErrorCode Domain::setFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // construct q and r
  {
    PetscInt Ii,Istart,Iend;
    ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);
    PetscScalar *q,*r;
    VecGetArray(_q,&q);
    VecGetArray(_r,&r);
    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      q[Jj] = _dq*(Ii/_Nz);
      r[Jj] = _dr*(Ii-_Nz*(Ii/_Nz));
      Jj++;
    }
    VecRestoreArray(_q,&q);
    VecRestoreArray(_r,&r);
  }


  // construct y
  bool fileExists = 0;
  loadVecFromInputFile(_y, _inputDir, "y",fileExists);
  if (fileExists==0) {
    if (_bCoordTrans <= 0) {
      VecCopy(_q,_y);
      VecScale(_y,_Ly);
    }
    else {
      PetscInt Ii,Istart,Iend;
      ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);
      const PetscScalar *q;
      PetscScalar *y;
      VecGetArrayRead(_q,&q);
      VecGetArray(_y,&y);
      PetscInt Jj = 0;
      for (Ii=Istart;Ii<Iend;Ii++) {
         y[Jj] = _Ly * sinh(_bCoordTrans*q[Jj])/sinh(_bCoordTrans);
        Jj++;
      }
      VecRestoreArrayRead(_q,&q);
      VecRestoreArray(_y,&y);
    }
  }


  // construct z
  fileExists = 0;
  loadVecFromInputFile(_z, _inputDir, "z",fileExists);
  if (fileExists==0) {
    VecCopy(_r,_z);
    VecScale(_z,_Lz);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Domain::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::allocateFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  bool fileExists = 0;
  fileExists = doesFileExist(fileName);
  if (fileExists && _restartFromChkpt == 1) {
    _outputFileMode = FILE_MODE_APPEND;
    PetscPrintf(PETSC_COMM_WORLD,"Note: will start simulation from previous checkpoint.\n");

    // load saved checkpoint data
    PetscViewer viewer_prev_checkpoint;

    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer_prev_checkpoint);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer_prev_checkpoint, "/domain");   CHKERRQ(ierr);
    ierr = VecLoad(_q,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_r,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_y,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_z,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer_prev_checkpoint);               CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(viewer_prev_checkpoint, "/time1D");   CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer_prev_checkpoint, "time1D", "chkptTimeStep", PETSC_INT, NULL, &_prevChkptTimeStep1D); CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer_prev_checkpoint);               CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer_prev_checkpoint, "/time2D");   CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer_prev_checkpoint, "time2D", "chkptTimeStep", PETSC_INT, NULL, &_prevChkptTimeStep2D); CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer_prev_checkpoint);               CHKERRQ(ierr);

    PetscViewerDestroy(&viewer_prev_checkpoint);

    // check if data_1D and data_2D exist
    fileName = _outputDir + "data_1D.h5";
  fileExists = doesFileExist(fileName);
  if (!fileExists) {_prevChkptTimeStep1D = 0;};

  fileName = _outputDir + "data_2D.h5";
  fileExists = doesFileExist(fileName);
  if (!fileExists) {_prevChkptTimeStep2D = 0;};
  }
  else {
    _restartFromChkpt = 0;
    _outputFileMode = FILE_MODE_WRITE;
    PetscPrintf(PETSC_COMM_WORLD,"Note: not restarting from previous checkpoint.\n\n");
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode Domain::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::allocateFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string fileName1 = _outputDir + "data_steadyState.h5";
  string fileName2 = _outputDir + "data_context.h5";

  bool fileExists = 0;
  fileExists = doesFileExist(fileName1) && doesFileExist(fileName2);
  if (fileExists && _restartFromChkptSS == 1) {
    _restartFromChkpt = 0;
    _outputFileMode = FILE_MODE_WRITE;
    PetscPrintf(PETSC_COMM_WORLD,"Note: will start from previous steady-state iteration.\n");

    // load saved checkpoint data
    PetscViewer viewer_prev_checkpoint;

    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName2.c_str(), FILE_MODE_READ, &viewer_prev_checkpoint);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer_prev_checkpoint, "/domain");   CHKERRQ(ierr);
    ierr = VecLoad(_q,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_r,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_y,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = VecLoad(_z,viewer_prev_checkpoint);                            CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer_prev_checkpoint);               CHKERRQ(ierr);

    PetscViewerDestroy(&viewer_prev_checkpoint);
  }
  else {
    _restartFromChkptSS = 0;
    _outputFileMode = FILE_MODE_WRITE;
    PetscPrintf(PETSC_COMM_WORLD,"Note: not restarting from previous steady-state iteration.\n\n");
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// scatters values from one vector to another
// used to get slip on the fault from the displacement vector, i.e., slip = u(1:Nz); shear stress on the fault from the stress vector sxy; surface displacement; surface heat flux
PetscErrorCode Domain::setScatters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Domain::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // some example 1D vectors
  VecCreate(PETSC_COMM_WORLD,&_y0); VecSetSizes(_y0,PETSC_DECIDE,_Nz); VecSetFromOptions(_y0); VecSet(_y0,0.0);
  VecCreate(PETSC_COMM_WORLD,&_z0); VecSetSizes(_z0,PETSC_DECIDE,_Ny); VecSetFromOptions(_z0); VecSet(_z0,0.0);


  { // set up scatter context to take values for y=0 from body field and put them on a Vec of size Nz
    PetscInt *indices; PetscMalloc1(_Nz,&indices);
    for (PetscInt Ii=0; Ii<_Nz; Ii++) { indices[Ii] = Ii; }
    IS is;
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, indices, PETSC_COPY_VALUES, &is);
    ierr = VecScatterCreate(_y, is, _y0, is, &_scatters["body2L"]); CHKERRQ(ierr);
    PetscFree(indices);
    ISDestroy(&is);
  }

  { // set up scatter context to take values for y=Ly from body field and put them on a Vec of size Nz
    // indices to scatter from
    PetscInt *fi; PetscMalloc1(_Nz,&fi);
    for (PetscInt Ii=0; Ii<_Nz; Ii++) { fi[Ii] = Ii + (_Ny*_Nz-_Nz); }
    IS isf; ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, fi, PETSC_COPY_VALUES, &isf);

    // indices to scatter to
    PetscInt *ti; PetscMalloc1(_Nz,&ti);
    for (PetscInt Ii=0; Ii<_Nz; Ii++) { ti[Ii] = Ii; }
    IS ist; ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, ti, PETSC_COPY_VALUES, &ist);

    ierr = VecScatterCreate(_y, isf, _y0, ist, &_scatters["body2R"]); CHKERRQ(ierr);
    PetscFree(fi); PetscFree(ti);
    ISDestroy(&isf); ISDestroy(&ist);
  }

  { // set up scatter context to take values for z=0 from body field and put them on a Vec of size Ny
    // indices to scatter from
    IS isf; ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, 0, _Nz, &isf);

    // indices to scatter to
    PetscInt *ti; PetscMalloc1(_Ny,&ti);
    for (PetscInt Ii=0; Ii<_Ny; Ii++) { ti[Ii] = Ii; }
    IS ist; ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti, PETSC_COPY_VALUES, &ist);

    ierr = VecScatterCreate(_y, isf, _z0, ist, &_scatters["body2T"]); CHKERRQ(ierr);
    PetscFree(ti);
    ISDestroy(&isf); ISDestroy(&ist);
  }

  { // set up scatter context to take values for z=Lz from body field and put them on a Vec of size Ny
    // indices to scatter from
    IS isf; ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, _Nz-1, _Nz, &isf);

    // indices to scatter to
    PetscInt *ti; PetscMalloc1(_Ny,&ti);
    for (PetscInt Ii=0; Ii<_Ny; Ii++) { ti[Ii] = Ii; }
    IS ist; ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti, PETSC_COPY_VALUES, &ist);

    ierr = VecScatterCreate(_y, isf, _z0, ist, &_scatters["body2B"]); CHKERRQ(ierr);
    PetscFree(ti);
    ISDestroy(&isf); ISDestroy(&ist);
  }


  VecScatterBegin(_scatters["body2T"], _y, _z0, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["body2T"], _y, _z0, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(_scatters["body2L"], _z, _y0, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["body2L"], _z, _y0, INSERT_VALUES, SCATTER_FORWARD);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// create example vector for testing purposes
PetscErrorCode Domain::testScatters() {
  PetscErrorCode ierr = 0;

  Vec body;
  ierr = VecDuplicate(_y,&body); CHKERRQ(ierr);
  PetscInt      Istart,Iend,Jj = 0;
  PetscScalar   *bodyA;
  ierr = VecGetOwnershipRange(body,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArray(body,&bodyA); CHKERRQ(ierr);

  for (PetscInt Ii = Istart; Ii<Iend; Ii++) {
    PetscInt Iy = Ii/_Nz;
    PetscInt Iz = (Ii-_Nz*(Ii/_Nz));
    bodyA[Jj] = 10.*Iy + Iz;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %g\n",Iy,Iz,bodyA[Jj]); CHKERRQ(ierr);
    Jj++;
  }
  ierr = VecRestoreArray(body,&bodyA); CHKERRQ(ierr);

  // test various mappings
  // y = 0: mapping to L
  Vec out;
  ierr = VecDuplicate(_y0,&out); CHKERRQ(ierr);
  ierr = VecScatterBegin(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecDestroy(&out); CHKERRQ(ierr);

  // y = Ly: mapping to R
  Vec out1;
  ierr = VecDuplicate(_y0,&out1); CHKERRQ(ierr);
  ierr = VecSet(out1,-1.); CHKERRQ(ierr);
  ierr = VecScatterBegin(_scatters["body2R"], body, out1, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(_scatters["body2R"], body, out1, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecDestroy(&out1); CHKERRQ(ierr);

  // z=0: mapping to T
  Vec out2;
  ierr = VecDuplicate(_z0,&out2); CHKERRQ(ierr);
  ierr = VecSet(out2,-1.); CHKERRQ(ierr);
  ierr = VecScatterBegin(_scatters["body2T"], body, out2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(_scatters["body2T"], body, out2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // z=Lz: mapping to B
  ierr = VecScatterBegin(_scatters["body2B"], body, out2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(_scatters["body2B"], body, out2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // z=Lz: mapping from B to body
  ierr = VecScatterBegin(_scatters["body2T"], out2, body, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(_scatters["body2T"], out2, body, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

  ierr = VecDestroy(&out2); CHKERRQ(ierr);
  ierr = VecDestroy(&body); CHKERRQ(ierr);

  return ierr;
}
