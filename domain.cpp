#include "domain.hpp"

#define FILENAME "sbpOps_fc.cpp"

using namespace std;

Domain::Domain(const char *file)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_problemType("strikeSlip"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),
  _isMMS(0),_loadICs(0),_numCycles(1), _inputDir("unspecified"),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),
  _vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(5.0)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  loadData(_file);

  //~ _dy = _Ly/(_Ny-1.0);
  //~ if (_Nz > 1) { _dz = _Lz/(_Nz-1.0); }
  //~ else (_dz = 1);

  if (_Ny > 1) { _dq = 1.0/(_Ny-1.0); }
  else (_dq = 1);
  if (_Nz > 1) { _dr = 1.0/(_Nz-1.0); }
  else (_dr = 1);

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif

  checkInput(); // perform some basic value checking to prevent NaNs
  setFields();

  setScatters();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

}


Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_problemType("strikeSlip"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),
  _isMMS(0),_loadICs(0),_inputDir("unspecified_"),
  _order(4),_Ny(Ny),_Nz(Nz),_Ly(-1),_Lz(-1),
  _vL(1e-9),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(5.0)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  loadData(_file);

  _Ny = Ny;
  _Nz = Nz;

  if (_Ny > 1) { _dq = 1.0/(_Ny-1.0); }
  else (_dq = 1);
  if (_Nz > 1) { _dr = 1.0/(_Nz-1.0); }
  else (_dr = 1);

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif

  checkInput(); // perform some basic value checking to prevent NaNs
  setFields();
  setScatters();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

}



Domain::~Domain()
{
  #if VERBOSE > 1
    std::string funcName = "Domain::~Domain";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_q);
  VecDestroy(&_r);
  VecDestroy(&_y);
  VecDestroy(&_z);

  VecDestroy(&_y0);
  VecDestroy(&_z0);

  map<string,VecScatter>::iterator it;
  for (it = _scatters.begin(); it!=_scatters.end(); it++ ) {
    VecScatterDestroy(&it->second);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



PetscErrorCode Domain::loadData(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::loadData";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ifstream infile( file );
  string line,var;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);

    if (var.compare("order")==0) { _order = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Ny")==0 && _Ny < 0)
    { _Ny = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Nz")==0 && _Nz < 0)
    { _Nz = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Ly")==0) { _Ly = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Lz")==0) { _Lz = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    else if (var.compare("isMMS")==0) {
      _isMMS = 0;
      std::string temp = line.substr(pos+_delim.length(),line.npos);
      if (temp.compare("yes")==0 || temp.compare("y")==0) { _isMMS = 1; }
    }
    else if (var.compare("sbpType")==0) {
      _sbpType = line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("bulkDeformationType")==0) {
      _bulkDeformationType = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("problemType")==0) {
      _problemType = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("momentumBalanceType")==0) {
      _momentumBalanceType = line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("loadICs")==0) {
      _loadICs = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }


    else if (var.compare("inputDir")==0) {
      _inputDir = line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("bCoordTrans")==0) {
       _bCoordTrans = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // output directory
    else if (var.compare("outputDir")==0) {
      _outputDir =  line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("vL")==0) { _vL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("numCycles")==0) { _numCycles = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
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
    PetscPrintf(PETSC_COMM_SELF,"\n\nrank=%i in Domain::view\n",rank);
    ierr = PetscPrintf(PETSC_COMM_SELF,"order = %i\n",_order);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ny = %i\n",_Ny);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Nz = %i\n",_Nz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ly = %e\n",_Ly);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Lz = %e\n",_Lz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"isMMS = %i\n",_isMMS);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"problemType = %s\n",_problemType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"momBalType = %s\n",_momentumBalanceType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bulkDeformationType = %s\n",_bulkDeformationType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"sbpType = %s\n",_sbpType.c_str());CHKERRQ(ierr);

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

  assert(_bulkDeformationType.compare("linearElastic")==0 ||
    _bulkDeformationType.compare("powerLaw")==0 );

  assert(_problemType.compare("strikeSlip")==0 ||
    _problemType.compare("iceStream")==0 );

  assert(_momentumBalanceType.compare("quasidynamic")==0 ||
    _momentumBalanceType.compare("dynamic")==0 ||
    _momentumBalanceType.compare("quasidynamic_and_dynamic")==0 ||
    _momentumBalanceType.compare("steadyStateIts")==0 ||
    _momentumBalanceType.compare("switching")==0 );

  assert( _order==2 || _order==4 );
  assert( _Ly > 0 && _Lz > 0);
  assert( _dq > 0 && !isnan(_dq) );
  assert( _dr > 0 && !isnan(_dr) );


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
  ierr = PetscViewerASCIIPrintf(viewer,"problemType = %s\n",_problemType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBalType = %s\n",_momentumBalanceType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bulkDeformationType = %s\n",_bulkDeformationType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sbpType = %s\n",_sbpType.c_str());CHKERRQ(ierr);


  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"bCoordTrans = %.15e\n",_bCoordTrans);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);


  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  //~ // output shear modulus
  //~ str =  _outputDir + "muPlus";
  //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ ierr = VecView(_muVecP,viewer);CHKERRQ(ierr);
  //~ ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

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





PetscErrorCode Domain::setFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscScalar alphay, alphaz;
  if (_order == 2 ) { alphay = 0.5 * _Ly * _dq; alphaz = 0.5 * _Lz * _dr; }
  if (_order == 4 ) { alphay = 0.4567e4/0.14400e5 * _Ly * _dq; alphaz = 0.4567e4/0.14400e5 * _Lz * _dr; }

  if (_sbpType.compare("mfc_coordTrans") == 0){alphay /= _Ly; alphaz /= _Lz;}

  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y"); CHKERRQ(ierr);

  VecDuplicate(_y,&_z); PetscObjectSetName((PetscObject) _z, "z");
  VecDuplicate(_y,&_q); PetscObjectSetName((PetscObject) _q, "q");
  VecDuplicate(_y,&_r); PetscObjectSetName((PetscObject) _r, "r");

  VecDuplicate(_y, &_alphay); VecSet(_alphay, alphay);
  VecDuplicate(_y, &_alphaz); VecSet(_alphaz, alphaz);

  // construct coordinate transform
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_q,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar *y,*z,*q,*r;
  VecGetArray(_y,&y);
  VecGetArray(_z,&z);
  VecGetArray(_q,&q);
  VecGetArray(_r,&r);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    q[Jj] = _dq*(Ii/_Nz);
    r[Jj] = _dr*(Ii-_Nz*(Ii/_Nz));
    if (_sbpType.compare("mfc_coordTrans") ) { // no coordinate transform
      y[Jj] = (_dq*_Ly)*(Ii/_Nz);
      z[Jj] = (_dr*_Lz)*(Ii-_Nz*(Ii/_Nz));
    }
    else {
      // no transformation
      y[Jj] = q[Jj]*_Ly;
      z[Jj] = r[Jj]*_Lz;

      // y[Jj] = _Ly * sinh(_bCoordTrans*q[Jj])/sinh(_bCoordTrans);
    }

    Jj++;
  }
  VecRestoreArray(_y,&y);
  VecRestoreArray(_z,&z);
  VecRestoreArray(_q,&q);
  VecRestoreArray(_r,&r);

  // load y and z instead
  if (_inputDir.compare("unspecified") != 0) {
    ierr = loadVecFromInputFile(_y,_inputDir,"y"); CHKERRQ(ierr);
    ierr = loadVecFromInputFile(_z,_inputDir,"z"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}

PetscErrorCode Domain::setScatters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::setFields";
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

  { // set up scatter context to take values for z=0 from body field and put them on a Vec of size Ny
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

  // create example vector for testing purposes
  //~ Vec body; VecDuplicate(_y,&body);
  //~ PetscInt       Istart,Iend;
  //~ PetscScalar   *bodyA;
  //~ VecGetOwnershipRange(body,&Istart,&Iend);
  //~ VecGetArray(body,&bodyA);
  //~ PetscInt Jj = 0;
  //~ for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    //~ PetscInt Iy = Ii/_Nz;
    //~ PetscInt Iz = (Ii-_Nz*(Ii/_Nz));
    //~ bodyA[Jj] = 10.*Iy + Iz;
    //~ PetscPrintf(PETSC_COMM_WORLD,"%i %i %g\n",Iy,Iz,bodyA[Jj]);
    //~ Jj++;
  //~ }
  //~ VecRestoreArray(body,&bodyA);


  // test various mappings

  // y = 0: mapping to L
  //~ Vec out; VecDuplicate(_y0,&out);
  //~ VecScatterBegin(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecScatterEnd(_scatters["body2L"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecView(out,PETSC_VIEWER_STDOUT_WORLD);

  //~ // y = Ly: mapping to R
  //~ Vec out; VecDuplicate(_y0,&out); VecSet(out,-1.);
  //~ VecScatterBegin(_scatters["body2R"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecScatterEnd(_scatters["body2R"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecView(out,PETSC_VIEWER_STDOUT_WORLD);

  //~ // z=0: mapping to T
  //~ Vec out; VecDuplicate(_z0,&out); VecSet(out,-1.);
  //~ VecScatterBegin(_scatters["body2T"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecScatterEnd(_scatters["body2T"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecView(out,PETSC_VIEWER_STDOUT_WORLD);

  //~ // z=Lz: mapping to B
  //~ Vec out; VecDuplicate(_z0,&out); VecSet(out,-1.);
  //~ VecScatterBegin(_scatters["body2B"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecScatterEnd(_scatters["body2B"], body, out, INSERT_VALUES, SCATTER_FORWARD);
  //~ VecView(out,PETSC_VIEWER_STDOUT_WORLD);

  // z=Lz: mapping from B to body
  //~ Vec out; VecDuplicate(_z0,&out); VecSet(out,-1.);
  //~ VecScatterBegin(_scatters["body2T"], out, body, INSERT_VALUES, SCATTER_REVERSE);
  //~ VecScatterEnd(_scatters["body2T"], out, body, INSERT_VALUES, SCATTER_REVERSE);
  //~ VecView(body,PETSC_VIEWER_STDOUT_WORLD);

  //~ VecDestroy(&body);
  //~ assert(0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}
