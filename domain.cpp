#include "domain.hpp"

using namespace std;

Domain::Domain(const char *file)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_problemType("strikeSlip"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),
  _isMMS(0),_loadICs(0),_inputDir("unspecified"),
  _order(4),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),
  _yInputDir("unspecified"),_zInputDir("unspecified"),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(5.0),
  _da(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file)";
    std::string fileName = "domain.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
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

  //~ DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
    //~ DMDA_STENCIL_BOX,_Nz,_Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &_da);
  //~ PetscInt zn,yn;
  //~ DMDAGetCorners(_da, &_zS, &_yS, 0, &zn, &yn, 0);
  //~ _zE = _zS + zn;
  //~ _yE = _yS + yn;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

}


Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
: _file(file),_delim(" = "),_outputDir("data/"),
  _bulkDeformationType("linearElastic"),_problemType("strikeSlip"),_momentumBalanceType("quasidynamic"),
  _sbpType("mfc_coordTrans"),
  _isMMS(0),_loadICs(0),_inputDir("unspecified"),
  _order(4),_Ny(Ny),_Nz(Nz),_Ly(-1),_Lz(-1),
  _yInputDir("unspecified"),_zInputDir("unspecified"),
  _q(NULL),_r(NULL),_y(NULL),_z(NULL),_dq(-1),_dr(-1),
  _bCoordTrans(5.0),
  _da(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)";
    std::string fileName = "domain.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
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

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

}



Domain::~Domain()
{
  #if VERBOSE > 1
    std::string funcName = "Domain::~Domain";
    std::string fileName = "domain.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif

  VecDestroy(&_q);
  VecDestroy(&_r);
  VecDestroy(&_y);
  VecDestroy(&_z);

  DMDestroy(&_da);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
}



PetscErrorCode Domain::loadData(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::loadData";
    std::string fileName = "domain.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
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
    else if (var.compare("zInputDir")==0) {
      _zInputDir = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("yInputDir")==0) {
      _yInputDir = line.substr(pos+_delim.length(),line.npos);
    }

    else if (var.compare("bCoordTrans")==0) {
       _bCoordTrans = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // output directory
    else if (var.compare("outputDir")==0) {
      _outputDir =  line.substr(pos+_delim.length(),line.npos);
    }
  }
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}



/*
// load shear modulus structure from input file
PetscErrorCode Domain::loadShearModSettings(ifstream& infile)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::loadShearModSettings in domain.cpp.\n");CHKERRQ(ierr);
  #endif


  string line,var;
  size_t pos = 0;

  // load settings for distribution type (order of lines non significant)
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of _delimiter
    var = line.substr(0,pos);

    if (line.compare(_endBlock)==0)
    {
      //~PetscPrintf(PETSC_COMM_WORLD,"\n\nfound _endBlock in loadShearModulusSettings\n");
      break; // done loading block, exit while loop
    }

    else if (var.compare("problem")==0)
    {
      _problemType = line.substr(pos+_delim.length(),line.npos); // symmetric or full
    }

    else if (_shearDistribution.compare("basin")==0)
    {
      if (var.compare("muInPlus")==0) { _muInPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("muOutPlus")==0) { _muOutPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoInPlus")==0) { _rhoInPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoOutPlus")==0) { _rhoOutPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("muInMinus")==0) { _muInMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("muOutMinus")==0) { _muOutMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoInMinus")==0) { _rhoInMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoOutMinus")==0) { _rhoOutMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("depth")==0) { _depth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("width")==0) { _width = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      // look for mu, rho
      if (var.compare("muPlus")==0) { _muValPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoPlus")==0) { _rhoValPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("muMinus")==0) { _muValMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoMinus")==0) { _rhoValMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("depth")==0) { _depth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("width")==0) { _width = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      _muValPlus = 1.0;
      // look for rho, mu will be prescribed
      if (var.compare("rhoPlus")==0) { _rhoValPlus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoMinus")==0) { _rhoValMinus = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("depth")==0) { _depth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("width")==0) { _width = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else if (_shearDistribution.compare("CVM")==0 )
    {
      // needed for depth-dependent friction
      if (var.compare("depth")==0) { _depth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("width")==0) { _width = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else { // print error message and fail
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: shearDistribution type not understood\n");CHKERRQ(ierr);
      assert(0>1);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::loadShearModSettings in domain.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}
*/


// Specified processor prints scalar/string data members to stdout.
PetscErrorCode Domain::view(PetscMPIInt rank)
{
  PetscErrorCode ierr = 0;
  PetscMPIInt localRank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&localRank);
  if (localRank==rank) {
  #if VERBOSE > 1
    std::string funcName = "Domain::view";
    std::string fileName = "domain.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
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
    std::string fileName = "domain.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  assert(_bulkDeformationType.compare("linearElastic")==0 ||
    _bulkDeformationType.compare("powerLaw")==0 );

  assert(_problemType.compare("strikeSlip")==0 ||
    _problemType.compare("iceStream")==0 );

  assert(_momentumBalanceType.compare("quasidynamic")==0 ||
    _momentumBalanceType.compare("dynamic")==0 ||
    _momentumBalanceType.compare("quasidynamic_and_dynamic")==0 ||
    _momentumBalanceType.compare("steadyStateIts")==0 );

  assert( _order==2 || _order==4 );
  assert( _Ly > 0 && _Lz > 0);
  assert( _dq > 0 && !isnan(_dq) );
  assert( _dr > 0 && !isnan(_dr) );


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
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
    std::string fileName = "domain.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}





PetscErrorCode Domain::setFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Domain::setFields";
    std::string fileName = "domain.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  ierr = VecCreate(PETSC_COMM_WORLD,&_y); CHKERRQ(ierr);
  ierr = VecSetSizes(_y,PETSC_DECIDE,_Ny*_Nz); CHKERRQ(ierr);
  ierr = VecSetFromOptions(_y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _y, "y"); CHKERRQ(ierr);

  VecDuplicate(_y,&_z); PetscObjectSetName((PetscObject) _z, "z");
  VecDuplicate(_y,&_q); PetscObjectSetName((PetscObject) _q, "q");
  VecDuplicate(_y,&_r); PetscObjectSetName((PetscObject) _r, "r");

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
      //~ y[Jj] = q[Jj]*_Ly;
      z[Jj] = r[Jj]*_Lz;

      y[Jj] = _Ly * sinh(_bCoordTrans*q[Jj])/sinh(_bCoordTrans);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
return ierr;
}
