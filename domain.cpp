#include "domain.hpp"

using namespace std;

Domain::Domain(const char *file)
: _file(file),_delim(" = "),_startBlock("{"),_endBlock("}"),
 _shearDistribution("basin"),_linSolver("AMG"),
  _timeControlType("P"),_timeIntegrator("FEuler"),_outputDir("data/")
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in domain.cpp.\n");
#endif

  loadData(_file);

  assert(_Ny>1);
  _dy = _Ly/(_Ny-1.0);
  if (_Nz > 1) { _dz = _Lz/(_Nz-1.0); }
  else (_dz = 1);

  if (_initDeltaT<_minDeltaT) {_initDeltaT = _minDeltaT; }
  _f0=0.6;
  _v0=1e-6;
  //~_vp=1e-9;
  //~_vp=1e-4;

  //~_csOut = sqrt(_muOut/_rhoOut);

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif


MatCreate(PETSC_COMM_WORLD,&_mu);
setFields();


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in domain.cpp.\n");
#endif

}


Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
: _file(file),_delim(" = "),_startBlock("{"),_endBlock("}"),
  _shearDistribution("basin"),_linSolver("AMG"),
  _timeControlType("P"),_timeIntegrator("FEuler"),_outputDir("data/")
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor:Domain(const char *file,PetscInt Ny, PetscInt Nz) in domain.cpp.\n");
#endif

  loadData(_file);

  _Ny = Ny;
  _Nz = Nz;


  assert(_Ny>1);
  _dy = _Ly/(_Ny-1.0);
  if (_Nz > 1) { _dz = _Lz/(_Nz-1.0); }
  else (_dz = 1);

  if (_initDeltaT<_minDeltaT) {_initDeltaT = _minDeltaT; }
  _f0=0.6;
  _v0=1e-6;
  _vp=1e-9;
  //~_vp=1e-4;

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif

MatCreate(PETSC_COMM_WORLD,&_mu);
setFields();


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor:Domain(const char *file,PetscInt Ny, PetscInt Nz) in domain.cpp.\n");
#endif
}



Domain::~Domain()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in domain.cpp.\n");
#endif

  PetscFree(_muArr);
  PetscFree(_rhoArr);
  PetscFree(_csArr);

  MatDestroy(&_mu);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in domain.cpp.\n");
#endif
}



PetscErrorCode Domain::loadData(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in domain.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
#endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // temporary char arrays used to set std::string data members
  int charSize = 200;
  char *outputDir = (char *) malloc(sizeof(char)*charSize+1);
  char *linSolver = (char *) malloc(sizeof(char)*charSize+1);
  char *shearDistribution = (char *) malloc(sizeof(char)*charSize+1);
  char *timeIntegrator = (char *) malloc(sizeof(char)*charSize+1);
  char *timeControlType = (char *) malloc(sizeof(char)*charSize+1);

  // 1 processor loads settings from file, communicates variables to all other processors
  if (rank==0) {
    ifstream infile( file );
    string line,var;
    //~string delim = " = ";
    size_t pos = 0;
    while (getline(infile, line))
    {
      istringstream iss(line);
      pos = line.find(_delim); // find position of delimiter
      var = line.substr(0,pos);

      if (var.compare("order")==0) { _order = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("Ny")==0) { _Ny = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("Nz")==0) { _Nz = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("Ly")==0) { _Ly = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("Lz")==0) { _Lz = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("Dc")==0) { _Dc = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      //fault properties
      else if (var.compare("seisDepth")==0) { _seisDepth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("bAbove")==0) { _bAbove = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("bBelow")==0) { _bBelow = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("sigma_N")==0) { _sigma_N_val = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      else if (var.compare("vp")==0) { _vp = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      // sedimentary basin properties
      else if (var.compare("shearDistribution")==0) {
        _shearDistribution = line.substr(pos+_delim.length(),line.npos);
        strcpy(shearDistribution,_shearDistribution.c_str());
        loadShearModulusSettings(infile);
      }

      // linear solver settings
      else if (var.compare("linSolver")==0) {
        _linSolver = line.substr(pos+_delim.length(),line.npos);
        strcpy(linSolver,_linSolver.c_str());
      }
      else if (var.compare("kspTol")==0) { _kspTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      // time integration properties
      else if (var.compare("timeIntegrator")==0) {
        _timeIntegrator = line.substr(pos+_delim.length(),line.npos);
        strcpy(timeIntegrator,_timeIntegrator.c_str());
      }
      else if (var.compare("timeControlType")==0) {
        _timeControlType = line.substr(pos+_delim.length(),line.npos);
        strcpy(timeControlType,_timeControlType.c_str());
      }
      else if (var.compare("strideLength")==0){ _strideLength = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("initTime")==0) { _initTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("maxTime")==0) { _maxTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("minDeltaT")==0) { _minDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("atol")==0) { _atol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      // other tolerances
      else if (var.compare("rootTol")==0) { _rootTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

      // output directory
      else if (var.compare("outputDir")==0) {
        _outputDir =  line.substr(pos+_delim.length(),line.npos);
        strcpy(outputDir,_outputDir.c_str());
      }
    }
  }

  // send loaded values to all other processors
  MPI_Bcast(&_order,1,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_Ny,1,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_Nz,1,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_Ly,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_Lz,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&_Dc,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&_seisDepth,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_bAbove,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_bBelow,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_sigma_N_val,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_vp,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&shearDistribution[0],sizeof(char)*charSize,MPI_CHAR,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_muVal,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_rhoVal,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_muIn,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_muOut,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_rhoIn,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_rhoOut,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_depth,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_width,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&linSolver[0],sizeof(char)*charSize,MPI_CHAR,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_kspTol,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&timeIntegrator[0],sizeof(char)*charSize,MPI_CHAR,0,PETSC_COMM_WORLD);
  MPI_Bcast(&timeControlType[0],sizeof(char)*charSize,MPI_CHAR,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_strideLength,1,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_maxStepCount,1,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_initTime,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_maxTime,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_minDeltaT,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_maxDeltaT,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_initDeltaT,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(&_atol,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&_rootTol,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  MPI_Bcast(&outputDir[0],sizeof(char)*charSize,MPI_CHAR,0,PETSC_COMM_WORLD);

  _outputDir = outputDir;
  _linSolver = linSolver;
  _shearDistribution = shearDistribution;
  _timeIntegrator = timeIntegrator;
  _timeControlType = timeControlType;


  free(outputDir);
  free(linSolver);
  free(shearDistribution);
  free(timeIntegrator);
  free(timeControlType);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode Domain::loadShearModulusSettings(ifstream& infile)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadShearModulusSettings in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  string line,var;
  size_t pos = 0;
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
    else if (_shearDistribution.compare("basin")==0)
    {
      if (var.compare("muIn")==0) { _muIn = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("muOut")==0) { _muOut = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoIn")==0) { _rhoIn = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rhoOut")==0) { _rhoOut = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("depth")==0) { _depth = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("width")==0) { _width = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      // look for mu, rho
      if (var.compare("mu")==0) { _muVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
      else if (var.compare("rho")==0) { _rhoVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      // look for rho, mu will be prescribed
      if (var.compare("rho")==0) { _rhoVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    }
    else { // print error message and fail
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: shearDistribution type not understood\n");CHKERRQ(ierr);
      assert(0>1);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadShearModulusSettings in domain.cpp.\n");CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_SELF,"Starting view in domain.cpp.\n");CHKERRQ(ierr);
    #endif

    PetscPrintf(PETSC_COMM_SELF,"\n\nrank=%i in Domain::view\n",rank);
    ierr = PetscPrintf(PETSC_COMM_SELF,"order = %i\n",_order);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ny = %i\n",_Ny);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Nz = %i\n",_Nz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ly = %e\n",_Ly);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Lz = %e\n",_Lz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"dy = %.15e\n",_dy);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"dz = %.15e\n",_dz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"Dc = %.15e\n",_Dc);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // fault properties
    ierr = PetscPrintf(PETSC_COMM_SELF,"seisDepth = %f\n",_seisDepth);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bAbove = %f\n",_bAbove);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bBelow = %f\n",_bBelow);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"sigma_N_val = %f\n",_sigma_N_val);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"vp = %f\n",_vp);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // sedimentary basin properties
    ierr = PetscPrintf(PETSC_COMM_SELF,"shearDistribution = %s\n",_shearDistribution.c_str());CHKERRQ(ierr);
    if (_shearDistribution.compare("basin")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"muIn = %f\n",_muIn);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"muOut = %f\n",_muOut);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoIn = %f\n",_rhoIn);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoOut = %f\n",_rhoOut);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"depth = %f\n",_depth);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"width = %f\n",_width);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"mu = %f\n",_muVal);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rho = %f\n",_rhoVal);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"rho = %f\n",_rhoVal);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // linear solve settings
    ierr = PetscPrintf(PETSC_COMM_SELF,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // time monitering
    ierr = PetscPrintf(PETSC_COMM_SELF,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"strideLength = %i\n",_strideLength);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"initTime = %.15e\n",_initTime);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"maxTime = %.15e\n",_maxTime);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"atol = %.15e\n",_atol);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"minDeltaT = %.15e\n",_minDeltaT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"maxDeltaT = %.15e\n",_maxDeltaT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"initDeltaT = %.15e\n",_initDeltaT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // tolerance nonlinear solve (for vel)
    ierr = PetscPrintf(PETSC_COMM_SELF,"rootTol = %.15e\n",_rootTol);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_SELF,"Ending view in domain.cpp.\n");CHKERRQ(ierr);
#endif
  }
  return ierr;
}

// Save all scalar fields to text file named domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
// Also save the shear modulus matrix.
PetscErrorCode Domain::write()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting write in domain.cpp.\n");CHKERRQ(ierr);
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
  ierr = PetscViewerASCIIPrintf(viewer,"Ly = %g\n",_Ly);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Lz = %g\n",_Lz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"dy = %.15e\n",_dy);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"dz = %.15e\n",_dz);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Dc = %15e\n",_Dc);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // fault properties
  ierr = PetscViewerASCIIPrintf(viewer,"seisDepth = %.15e\n",_seisDepth);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bAbove = %.15e\n",_bAbove);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bBelow = %.15e\n",_bBelow);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sigma_N_val = %.15e\n",_sigma_N_val);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"vp = %.15e\n",_vp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // sedimentary basin properties
  ierr = PetscViewerASCIIPrintf(viewer,"shearDistribution = %s\n",_shearDistribution.c_str());CHKERRQ(ierr);
  if (_shearDistribution.compare("basin")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"muIn = %.15e\n",_muIn);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"muOut = %.15e\n",_muOut);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rhoIn = %.15e\n",_rhoIn);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rhoOut = %.15e\n",_rhoOut);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"depth = %.15e\n",_depth);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"width = %.15e\n",_width);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"mu = %.15e\n",_muVal);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rho = %.15e\n",_rhoVal);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"rho = %.15e\n",_rhoVal);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %g\n",_kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"strideLength = %i\n",_strideLength);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"minDeltaT = %.15e\n",_minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxDeltaT = %.15e\n",_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initDeltaT = %.15e\n",_initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"atol = %.15e\n",_atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // tolerance for nonlinear solve (for vel)
  ierr = PetscViewerASCIIPrintf(viewer,"rootTol = %g\n",_rootTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"f0 = %g\n",_f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);


  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output shear modulus matrix
  str =  _outputDir + "mu";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_mu,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending write in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}





PetscErrorCode Domain::setFields()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFields in domain.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar    v,y,z;

  Vec muVec;
  PetscInt *muInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_muArr);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_rhoArr);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_csArr);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);

  PetscScalar r = 0;
  PetscScalar rbar = 0.25*_width*_width;
  PetscScalar rw = 1+0.25*_width*_width/_depth/_depth;
  for (Ii=0;Ii<_Ny*_Nz;Ii++) {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    y = _dy*(Ii/_Nz);
    r=y*y+(0.25*_width*_width/_depth/_depth)*z*z;

    if (_shearDistribution.compare("basin")==0) {
      v = 0.5*(_rhoOut-_rhoIn)*(tanh((double)(r-rbar)/rw)+1) + _rhoIn;
      _rhoArr[Ii] = v;

      _csIn = sqrt(_muIn/_rhoIn);
      _csOut = sqrt(_muOut/_rhoOut);
      v = 0.5*(_csOut-_csIn)*(tanh((double)(r-rbar)/rw)+1) + _csIn;
      _csArr[Ii] = v;

      v = 0.5*(_muOut-_muIn)*(tanh((double)(r-rbar)/rw)+1) + _muIn;
    }
    else if (_shearDistribution.compare("constant")==0) {
      _rhoArr[Ii] = _rhoVal;
      _csArr[Ii] = sqrt(_muVal/_rhoVal);
      v = _muVal;
    }
    else if (_shearDistribution.compare("gradient")==0) {
      _rhoArr[Ii] = _rhoVal;
       _csArr[Ii] = sqrt(_muVal/_rhoVal);
      v = Ii+2;
    }
    else if (_shearDistribution.compare("mms")==0) {
      _rhoArr[Ii] = _rhoVal;
       _csArr[Ii] = sqrt(_muVal/_rhoVal);
      v = sin(y+z) + 2.0;
    }
    else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: shearDistribution type not understood\n");CHKERRQ(ierr);
      assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
    }
    _muArr[Ii] = v;
    muInds[Ii] = Ii;
  }

  ierr = VecSetValues(muVec,_Ny*_Nz,muInds,_muArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(_mu,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_mu);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_mu,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_mu,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_mu);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_mu,muVec,INSERT_VALUES);CHKERRQ(ierr);
  //~ierr = MatView(_mu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  VecDestroy(&muVec);
  PetscFree(muInds);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFields in domain.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}
