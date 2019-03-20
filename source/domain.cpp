#include "domain.hpp"

#define FILENAME "domain.cpp"

using namespace std;

// member function definitions including constructor
// first type of constructor with 1 parameter
Domain::Domain(const char *file)
  : _file(file),_delim(" = "),_inputDir("unspecified"),_outputDir(" "),
  _bulkDeformationType("linearElastic"),
  _momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),_operatorType("matrix-based"),
  _sbpCompatibilityType("fullyCompatible"),
  _gridSpacingType("variableGridSpacing"),_isMMS(0),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),_vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(-1), _ckpt(0), _ckptNumber(0), _interval(500), _maxStepCount(1e8)
{
  #if VERBOSE > 1
    string funcName = "Domain::Domain(const char *file)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  // load data from file
  loadSettings(_file);
  if (_ckpt > 0) {
    loadValueFromCheckpoint(_outputDir, "ckptNumber", _ckptNumber);
    _maxStepCount = _interval;
  }
  
  // check domain size and set grid spacing in y direction
  if (_Ny > 1) {
    _dq = 1.0 / (_Ny - 1.0);
  }
  else {
    _dq = 1;
  }

  // set grid spacing in z-direction
  if (_Nz > 1) {
    _dr = 1.0 / (_Nz - 1.0);
  }
  else {
    _dr = 1;
  }

  #if VERBOSE > 2 // each processor prints loaded values to screen
    PetscMPIInt rank,size;
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    for (int Ii = 0; Ii < size; Ii++) {
      view(Ii);
    }
  #endif

  checkInput(); // perform some basic value checking to prevent NaNs
  setFields();
  setScatters();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

}


// second type of constructor with 3 parameters
Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
  : _file(file),_delim(" = "),_inputDir("unspecified"),_outputDir(" "),
  _bulkDeformationType("linearElastic"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),_operatorType("matrix-based"),
  _sbpCompatibilityType("fullyCompatible"),
  _gridSpacingType("variableGridSpacing"),_isMMS(0),
  _order(4),_Ny(Ny),_Nz(Nz),_Ly(-1),_Lz(-1),_vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_y0(NULL),_z0(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(-1), _ckpt(0), _ckptNumber(0), _interval(500), _maxStepCount(1e8)
{
  #if VERBOSE > 1
    string funcName = "Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  if (_ckpt > 0) {
    loadValueFromCheckpoint(_outputDir, "ckptNumber", _ckptNumber);
    _maxStepCount = _interval;
  }
  
  _Ny = Ny;
  _Nz = Nz;

  if (_Ny > 1) {
    _dq = 1.0/(_Ny-1.0);
  }
  else {
    _dq = 1;
  }
  
  if (_Nz > 1) {
    _dr = 1.0/(_Nz-1.0);
  }
  else {
    _dr = 1;
  }

  #if VERBOSE > 2 // each processor prints loaded values to screen
    PetscMPIInt rank,size;
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    for (int Ii = 0; Ii < size; Ii++) {
      view(Ii);
    }
  #endif

  checkInput(); // perform some basic value checking to prevent NaNs  
  setFields();
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

  // determines size of the group associated with a communicator
  MPI_Comm_size(PETSC_COMM_WORLD,&size); 
  // determines rank of the calling processes in the communicator
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

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
    if (var.compare("order") == 0) {
      _order = atoi(rhs.c_str());
    }
    else if (var.compare("Ny") == 0 && _Ny < 0) {
      _Ny = atoi(rhs.c_str());
    }
    else if (var.compare("Nz") == 0 && _Nz < 0) {
      _Nz = atoi(rhs.c_str());
    }
    else if (var.compare("Ly") == 0) {
      _Ly = atof(rhs.c_str());
    }
    else if (var.compare("Lz") == 0) {
      _Lz = atof(rhs.c_str());
    }
    // _isMMS must be 0 or 1
    else if (var.compare("isMMS") == 0) {
      _isMMS = atoi(rhs.c_str());
    }
    else if (var.compare("sbpType")==0) {
      _sbpType = rhs;
    }
    else if (var.compare("operatorTYpe")==0) {
      _operatorType = rhs;
    }
    else if (var.compare("sbpCompatibilityType")==0) {
      _sbpCompatibilityType = rhs;
    }
    else if (var.compare("gridSpacingType")==0) {
      _gridSpacingType = rhs;
    }
    else if (var.compare("bulkDeformationType")==0) {
      _bulkDeformationType = rhs;
    }
    else if (var.compare("momentumBalanceType")==0) {
      _momentumBalanceType = rhs;
    }
    else if (var.compare("inputDir") == 0) {
      _inputDir = rhs;
    }
    else if (var.compare("outputDir")==0) {
      _outputDir =  rhs;
    }
    else if (var.compare("bCoordTrans")==0) {
      _bCoordTrans = atof( rhs.c_str() );
    }
    else if (var.compare("vL")==0) {
      _vL = atof( rhs.c_str() );
    }
    else if (var.compare("maxStepCount") == 0) {
      _maxStepCount = (int)atof(rhs.c_str());
    }
    else if (var.compare("ckpt") == 0) {
      _ckpt = atoi(rhs.c_str());
    }
    else if (var.compare("interval") == 0) {
      _interval = (int)atof(rhs.c_str());
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Specified processor prints scalar/string data members to stdout.
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
    ierr = PetscPrintf(PETSC_COMM_SELF,"sbpType = %s\n",_sbpType.c_str());CHKERRQ(ierr);
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
  assert(_dq > 0 && !isnan(_dq));
  assert(_dr > 0 && !isnan(_dr));

  assert(_ckpt >= 0 && _ckptNumber >= 0);
  assert(_interval >= 0);
  assert(_maxStepCount > 0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Save all scalar fields to text file named domain.txt in output directory.
// Also writes out coordinate systems q, r, y, z into respective files in output directory
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode Domain::write()
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
  ierr = PetscViewerASCIIPrintf(viewer,"momBalType = %s\n",_momentumBalanceType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bulkDeformationType = %s\n",_bulkDeformationType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sbpType = %s\n",_sbpType.c_str());CHKERRQ(ierr);

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"bCoordTrans = %.15e\n",_bCoordTrans);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // get number of processors
  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);
  // free viewer for domain.txt
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output q
  PetscViewer view1;
  str =  _outputDir + "q";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&view1);CHKERRQ(ierr);
  ierr = VecView(_q,view1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view1);CHKERRQ(ierr);

  // output r
  PetscViewer view2;
  str =  _outputDir + "r";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&view2);CHKERRQ(ierr);
  ierr = VecView(_r,view2);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view2);CHKERRQ(ierr);

  // output y
  PetscViewer view3;
  str =  _outputDir + "y";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&view3);CHKERRQ(ierr);
  ierr = VecView(_y,view3);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view3);CHKERRQ(ierr);

  // output z
  PetscViewer view4;
  str =  _outputDir + "z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&view4);CHKERRQ(ierr);
  ierr = VecView(_z,view4);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view4);CHKERRQ(ierr);
 
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

  // generate vector _y with size _Ny*_Nz
  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y"); CHKERRQ(ierr);

  // duplicate _y into _z, _q, _r
  ierr = VecDuplicate(_y,&_z); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _z, "z"); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_q); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _q, "q"); CHKERRQ(ierr);
  ierr = VecDuplicate(_y,&_r); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _r, "r"); CHKERRQ(ierr);

  if (_ckptNumber > 0) {
    loadVecFromInputFile(_y, _outputDir, "y");
    loadVecFromInputFile(_z, _outputDir, "z");
    loadVecFromInputFile(_q, _outputDir, "q");
    loadVecFromInputFile(_r, _outputDir, "r");    
  }
  else {
    // construct coordinate transform
    PetscInt Ii,Istart,Iend,Jj = 0;
    PetscScalar *y,*z,*q,*r;
    ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);

    // return pointers to local data arrays (the processor's portion of vector data)
    ierr = VecGetArray(_y,&y); CHKERRQ(ierr);
    ierr = VecGetArray(_z,&z); CHKERRQ(ierr);
    ierr = VecGetArray(_q,&q); CHKERRQ(ierr);
    ierr = VecGetArray(_r,&r); CHKERRQ(ierr);

    // set vector entries for q, r (coordinate transform) and y, z (no transform)
    for (Ii=Istart; Ii<Iend; Ii++) {
      q[Jj] = _dq*(Ii/_Nz);
      r[Jj] = _dr*(Ii-_Nz*(Ii/_Nz));

      // matrix-based, fully compatible, allows curvilinear coordinate transformation
      if (_sbpType.compare("mfc_coordTrans") ) {
	y[Jj] = (_dq*_Ly)*(Ii/_Nz);
	z[Jj] = (_dr*_Lz)*(Ii-_Nz*(Ii/_Nz));
      }
      else {
	// hardcoded transformation (not available for z)
	if (_bCoordTrans > 0) {
	  y[Jj] = _Ly * sinh(_bCoordTrans * q[Jj]) / sinh(_bCoordTrans);
	}
	// no transformation
	y[Jj] = q[Jj]*_Ly;
	z[Jj] = r[Jj]*_Lz;
      }
      Jj++;
    }

    // restore arrays
    ierr = VecRestoreArray(_y,&y); CHKERRQ(ierr);
    ierr = VecRestoreArray(_z,&z); CHKERRQ(ierr);
    ierr = VecRestoreArray(_q,&q); CHKERRQ(ierr);
    ierr = VecRestoreArray(_r,&r); CHKERRQ(ierr);
  }

  // for ice stream forcing coordinate system
  if (_ckptNumber == 0 && _inputDir != "unspecified") {
    loadVecFromInputFile(_y, _inputDir, "y");
    loadVecFromInputFile(_z, _inputDir, "z");
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

  // set _y0 to be zero vector with length _Nz
  ierr = VecCreate(PETSC_COMM_WORLD,&_y0); CHKERRQ(ierr);
  ierr = VecSetSizes(_y0,PETSC_DECIDE,_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y0); CHKERRQ(ierr);
  ierr = VecSet(_y0,0.0); CHKERRQ(ierr);

  // set _z0 to be zero vector with length _Ny
  ierr = VecCreate(PETSC_COMM_WORLD,&_z0); CHKERRQ(ierr);
  ierr = VecSetSizes(_z0,PETSC_DECIDE,_Ny); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_z0); CHKERRQ(ierr);
  ierr = VecSet(_z0,0.0); CHKERRQ(ierr);

  // set up scatter context to take values for y = 0 from body field and put them on a Vec of size Nz
  PetscInt *indices;
  IS is;  // index set
  ierr = PetscMalloc1(_Nz,&indices); CHKERRQ(ierr);

  // we want to scatter from index 0 to _Nz - 1, i.e. take the first _Nz components of the vector to scatter from
  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    indices[Ii] = Ii;
  }

  // creates data structure for an index set containing a list of integers
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, indices, PETSC_COPY_VALUES, &is); CHKERRQ(ierr);

  // creates vector scatter context, scatters values from _y (at indices is) to _y0 (at indices is)
  ierr = VecScatterCreate(_y, is, _y0, is, &_scatters["body2L"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(indices); CHKERRQ(ierr);
  ierr = ISDestroy(&is); CHKERRQ(ierr);

  //===============================================================================
  // set up scatter context to take values for y = Ly from body field and put them on a Vec of size Nz
  // indices to scatter from
  PetscInt *fi;
  IS isf;
  ierr = PetscMalloc1(_Nz,&fi); CHKERRQ(ierr);

  // we want to scatter from index _Ny*_Nz - _Nz to _Ny*_Nz - 1, i.e. the last _Nz entries of the vector to scatter from
  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    fi[Ii] = Ii + (_Ny*_Nz-_Nz);
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, fi, PETSC_COPY_VALUES, &isf); CHKERRQ(ierr);

  // indices to scatter to
  PetscInt *ti;
  IS ist;
  ierr = PetscMalloc1(_Nz,&ti); CHKERRQ(ierr);
  for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
    ti[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, ti, PETSC_COPY_VALUES, &ist); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf, _y0, ist, &_scatters["body2R"]); CHKERRQ(ierr);
  // free memory
  ierr = PetscFree(fi); CHKERRQ(ierr);
  ierr = PetscFree(ti); CHKERRQ(ierr);
  ierr = ISDestroy(&isf); CHKERRQ(ierr);
  ierr = ISDestroy(&ist); CHKERRQ(ierr);

  
  //============================================================================== 
  // set up scatter context to take values for z = 0 from body field and put them on a Vec of size Ny
  // indices to scatter from
  IS isf2;
  /* creates a data structure for an index set with a list of evenly spaced integers
   * locally owned portion of index set has length _Ny
   * first element of locally owned index set is 0
   * change to the next index is _Nz (the stride)
   * takes indices [0, _Nz, 2*_Nz, ..., (_Ny-1)*_Nz]
   */
  ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, 0, _Nz, &isf2); CHKERRQ(ierr);

  // indices to scatter to
  PetscInt *ti2;
  IS ist2;
  ierr = PetscMalloc1(_Ny,&ti2); CHKERRQ(ierr);

  // length _Ny
  for (PetscInt Ii=0; Ii<_Ny; Ii++) {
    ti2[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti2, PETSC_COPY_VALUES, &ist2); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf2, _z0, ist2, &_scatters["body2T"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(ti2); CHKERRQ(ierr);
  ierr = ISDestroy(&isf2); CHKERRQ(ierr);
  ierr = ISDestroy(&ist2); CHKERRQ(ierr);


  //==============================================================================
  // set up scatter context to take values for z = Lz from body field and put them on a Vec of size Ny
  // indices to scatter from
  IS isf3;
  // takes indices [_Nz - 1, 2*_Nz - 1, ..., _Ny*_Nz - 1]
  ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, _Nz - 1, _Nz, &isf3); CHKERRQ(ierr);

  // indices to scatter to
  PetscInt *ti3;
  IS ist3;
  ierr = PetscMalloc1(_Ny,&ti3); CHKERRQ(ierr);
  for (PetscInt Ii = 0; Ii<_Ny; Ii++) {
    ti3[Ii] = Ii;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti3, PETSC_COPY_VALUES, &ist3); CHKERRQ(ierr);
  ierr = VecScatterCreate(_y, isf3, _z0, ist3, &_scatters["body2B"]); CHKERRQ(ierr);

  // free memory
  ierr = PetscFree(ti3); CHKERRQ(ierr);
  ierr = ISDestroy(&isf3); CHKERRQ(ierr);
  ierr = ISDestroy(&ist3); CHKERRQ(ierr);

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
