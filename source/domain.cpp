#include "domain.hpp"

#define FILENAME "sbpOps_fc.cpp"

using namespace std;

// member function definitions including constructor
// first type of constructor with 1 parameter
Domain::Domain(const char *file)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),_operatorType("matrix-based"),_sbpCompatibilityType("fullyCompatible"),_gridSpacingType("variableGridSpacing"),
  _isMMS(0),_loadICs(0), _inputDir("unspecified_"),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),
  _vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(-1)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  // load data from file
  loadData(_file);

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
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),_operatorType("matrix-based"),_sbpCompatibilityType("fullyCompatible"),_gridSpacingType("variableGridSpacing"),
  _isMMS(0),_loadICs(0),_inputDir("unspecified_"),
  _order(4),_Ny(Ny),_Nz(Nz),_Ly(-1),_Lz(-1),
  _vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(-1)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  loadData(_file);

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

  for (int Ii=0;Ii<size;Ii++) {
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
    std::string funcName = "Domain::~Domain";
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
  for (it = _scatters.begin(); it != _scatters.end(); it++ ) {
    VecScatterDestroy(&it->second);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// define loadData function, takes 1 parameter - the filename
PetscErrorCode Domain::loadData(const char *file)
{
  PetscErrorCode ierr = 0;
  PetscMPIInt rank,size;

  #if VERBOSE > 1
    std::string funcName = "Domain::loadData";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // determines size of the group associated with a communicator
  // determines rank of the calling processes in the communicator
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
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
    else if (var.compare("isMMS") == 0) {
      _isMMS = 0;
      std::string temp = rhs;
      if (temp.compare("yes") == 0 || temp.compare("y") == 0) {
	_isMMS = 1;
      }
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
    else if (var.compare("loadICs")==0) {
      _loadICs = (int)atof(rhs.c_str());
    }
    else if (var.compare("inputDir")==0) {
      _inputDir = rhs;
    }
    else if (var.compare("bCoordTrans")==0) {
      _bCoordTrans = atof( rhs.c_str() );
    }
    else if (var.compare("outputDir")==0) {
      _outputDir =  rhs;
    }
    else if (var.compare("vL")==0) {
      _vL = atof( rhs.c_str() );
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
      std::string funcName = "Domain::view";
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
    #endif

    // start printing all the inputs
    PetscPrintf(PETSC_COMM_SELF,"\n\nrank=%i in Domain::view\n",rank);
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
    std::string funcName = "Domain::checkInput";
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


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}



// Save all scalar fields to text file named domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode Domain::write()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::write";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  std::string str = _outputDir + "domain.txt";
  PetscViewer    viewer;

  // write into file using PetscViewer
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

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
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output q
  str =  _outputDir + "q";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_q,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output r
  str =  _outputDir + "r";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_r,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output y
  str =  _outputDir + "y";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_y,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output z
  str =  _outputDir + "z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

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
    std::string funcName = "Domain::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // generate vector _y with size _Ny*_Nz
  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y"); CHKERRQ(ierr);

  // duplicate _y into _z, _q, _r
  VecDuplicate(_y,&_z); PetscObjectSetName((PetscObject) _z, "z");
  VecDuplicate(_y,&_q); PetscObjectSetName((PetscObject) _q, "q");
  VecDuplicate(_y,&_r); PetscObjectSetName((PetscObject) _r, "r");

  // construct coordinate transform
  PetscInt Ii,Istart,Iend,Jj = 0;
  PetscScalar *y,*z,*q,*r;
  ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);

  // return pointers to local data arrays (the processor's portion of vector data)
  VecGetArray(_y,&y);
  VecGetArray(_z,&z);
  VecGetArray(_q,&q);
  VecGetArray(_r,&r);

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
  VecRestoreArray(_y,&y);
  VecRestoreArray(_z,&z);
  VecRestoreArray(_q,&q);
  VecRestoreArray(_r,&r);

  // load y and z instead
  loadVecFromInputFile(_y,_inputDir,"y");
  loadVecFromInputFile(_z,_inputDir,"z");

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
    std::string funcName = "Domain::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set _y0 to be zero vector with length _Nz
  VecCreate(PETSC_COMM_WORLD,&_y0);
  VecSetSizes(_y0,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_y0);
  VecSet(_y0,0.0);  

  // set _z0 to be zero vector with length _Ny
  VecCreate(PETSC_COMM_WORLD,&_z0);
  VecSetSizes(_z0,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_z0);
  VecSet(_z0,0.0);

  { // set up scatter context to take values for y = 0 from body field and put them on a Vec of size Nz
    PetscInt *indices;
    IS is;  // index set
    PetscMalloc1(_Nz,&indices);

    // we want to scatter from index 0 to _Nz - 1, i.e. take the first _Nz components of the vector to scatter from
    for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
      indices[Ii] = Ii;
    }

    // creates data structure for an index set containing a list of integers
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, indices, PETSC_COPY_VALUES, &is);

    // creates vector scatter context, scatters values from _y (at indices is) to _y0 (at indices is)
    ierr = VecScatterCreate(_y, is, _y0, is, &_scatters["body2L"]); CHKERRQ(ierr);

    // free memory
    PetscFree(indices);
    ISDestroy(&is);
  }

  { // set up scatter context to take values for y = Ly from body field and put them on a Vec of size Nz
    // indices to scatter from
    PetscInt *fi;
    IS isf;
    PetscMalloc1(_Nz,&fi);

    // we want to scatter from index _Ny*_Nz - _Nz to _Ny*_Nz - 1, i.e. the last _Nz entries of the vector to scatter from
    for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
      fi[Ii] = Ii + (_Ny*_Nz-_Nz);
    }
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, fi, PETSC_COPY_VALUES, &isf);

    // indices to scatter to
    PetscInt *ti;
    IS ist;
    PetscMalloc1(_Nz,&ti);
    for (PetscInt Ii = 0; Ii<_Nz; Ii++) {
      ti[Ii] = Ii;
    }
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Nz, ti, PETSC_COPY_VALUES, &ist);
    ierr = VecScatterCreate(_y, isf, _y0, ist, &_scatters["body2R"]); CHKERRQ(ierr);

    // free memory
    PetscFree(fi);
    PetscFree(ti);
    ISDestroy(&isf);
    ISDestroy(&ist);
  }

  { // set up scatter context to take values for z = 0 from body field and put them on a Vec of size Ny
    // indices to scatter from
    IS isf;
    /* creates a data structure for an index set with a list of evenly spaced integers
     * locally owned portion of index set has length _Ny
     * first element of locally owned index set is 0
     * change to the next index is _Nz (the stride)
     * takes indices [0, _Nz, 2*_Nz, ..., (_Ny-1)*_Nz]
    */
    ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, 0, _Nz, &isf);

    // indices to scatter to
    PetscInt *ti;
    IS ist;
    PetscMalloc1(_Ny,&ti);

    // length _Ny
    for (PetscInt Ii=0; Ii<_Ny; Ii++) {
      ti[Ii] = Ii;
    }
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti, PETSC_COPY_VALUES, &ist);
    ierr = VecScatterCreate(_y, isf, _z0, ist, &_scatters["body2T"]); CHKERRQ(ierr);

    // free memory
    PetscFree(ti);
    ISDestroy(&isf);
    ISDestroy(&ist);
  }

  { // set up scatter context to take values for z = Lz from body field and put them on a Vec of size Ny
    // indices to scatter from
    IS isf;
    // takes indices [_Nz - 1, 2*_Nz - 1, ..., _Ny*_Nz - 1]
    ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, _Nz - 1, _Nz, &isf);

    // indices to scatter to
    PetscInt *ti;
    IS ist;
    PetscMalloc1(_Ny,&ti);
    for (PetscInt Ii = 0; Ii<_Ny; Ii++) {
      ti[Ii] = Ii;
    }
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti, PETSC_COPY_VALUES, &ist);
    ierr = VecScatterCreate(_y, isf, _z0, ist, &_scatters["body2B"]); CHKERRQ(ierr);

    // free memory
    PetscFree(ti);
    ISDestroy(&isf);
    ISDestroy(&ist);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}

// // create example vector for testing purposes
// PestcErrorCode Domain::testScatters() {
//   Vec body;
//   VecDuplicate(_y,&body);
//   PetscInt      Istart,Iend,Jj = 0;
//   PetscScalar   *bodyA;
//   PetscErrorCode ierr = 0;
//   VecGetOwnershipRange(body,&Istart,&Iend);
//   VecGetArray(body,&bodyA);

//   for (PetscInt Ii = Istart; Ii<Iend; Ii++) {
//     PetscInt Iy = Ii/_Nz;
//     PetscInt Iz = (Ii-_Nz*(Ii/_Nz));
//     bodyA[Jj] = 10.*Iy + Iz;
//     PetscPrintf(PETSC_COMM_WORLD,"%i %i %g\n",Iy,Iz,bodyA[Jj]);
//     Jj++;
//   }
//   VecRestoreArray(body,&bodyA);

//   // test various mappings
//   // y = 0: mapping to L
//   Vec out;
//   VecDuplicate(_y0,&out);
//   VecScatterBegin(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecScatterEnd(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecView(out,PETSC_VIEWER_STDOUT_WORLD);
//   VecDestroy(&out);

//   // y = Ly: mapping to R
//   Vec out;
//   VecDuplicate(_y0,&out); VecSet(out,-1.);
//   VecScatterBegin(_scatters["body2R"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecScatterEnd(_scatters["body2R"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecView(out,PETSC_VIEWER_STDOUT_WORLD);
//   VecDestroy(&out);

//   // z=0: mapping to T
//   Vec out;
//   VecDuplicate(_z0,&out); VecSet(out,-1.);
//   VecScatterBegin(_scatters["body2T"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecScatterEnd(_scatters["body2T"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecView(out,PETSC_VIEWER_STDOUT_WORLD);
//   VecDestroy(&out);

//   // z=Lz: mapping to B
//   Vec out;
//   VecDuplicate(_z0,&out);
//   VecSet(out,-1.);
//   VecScatterBegin(_scatters["body2B"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecScatterEnd(_scatters["body2B"], body, out, INSERT_VALUES, SCATTER_FORWARD);
//   VecView(out,PETSC_VIEWER_STDOUT_WORLD);
//   VecDestroy(&out);

//   // z=Lz: mapping from B to body
//   Vec out;
//   VecDuplicate(_z0,&out);
//   VecSet(out,-1.);
//   VecScatterBegin(_scatters["body2T"], out, body, INSERT_VALUES, SCATTER_REVERSE);
//   VecScatterEnd(_scatters["body2T"], out, body, INSERT_VALUES, SCATTER_REVERSE);
//   VecView(body,PETSC_VIEWER_STDOUT_WORLD);

//   VecDestroy(&out);
//   VecDestroy(&body);
//   assert(0);

//   return ierr;
// }

