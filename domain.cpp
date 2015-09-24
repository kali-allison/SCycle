#include "domain.hpp"

using namespace std;

Domain::Domain(const char *file)
: _file(file),_delim(" = "),_startBlock("{"),_endBlock("}"),
  _order(0),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),_dy(-1),_dz(-1),_Dc(-1),
  _bcTType("unspecified"),_bcRType("unspecified"),_bcBType("unspecified"),
  _bcLType("unspecified"),
  _sigma_N_min(-1),_sigma_N_max(-1),_sigma_N(NULL),
  _shearDistribution("unspecified"),_problemType("unspecificed"),_inputDir("unspecified"),
  _muValPlus(-1),_rhoValPlus(-1),_muInPlus(-1),_muOutPlus(-1),
  _rhoInPlus(-1),_rhoOutPlus(-1),_depth(-1),_width(-1),
  _muArrPlus(NULL),_csArrPlus(NULL),_sigmaNArr(NULL),_muP(NULL),
  _muValMinus(-1),_rhoValMinus(-1),_muInMinus(-1),_muOutMinus(-1),
  _rhoInMinus(-1),_rhoOutMinus(-1),
  _muArrMinus(NULL),_csArrMinus(NULL),_muM(NULL),
  _viscDistribution("unspecified"),_visc(NULL),
  _linSolver("unspecified"),_kspTol(-1),
  _timeControlType("unspecified"),_timeIntegrator("unspecified"),
  _strideLength(-1),_maxStepCount(-1),_initTime(-1),_maxTime(-1),
  _minDeltaT(-1),_maxDeltaT(-1),_initDeltaT(_minDeltaT),
  _atol(-1),_outputDir("unspecified"),_f0(0.6),_v0(1e-6),_vL(-1)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::Domain in domain.cpp.\n");
#endif

  loadData(_file);

  _dy = _Ly/(_Ny-1.0);
  if (_Nz > 1) { _dz = _Lz/(_Nz-1.0); }
  else (_dz = 1);

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {_initDeltaT = _minDeltaT; }

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif

  checkInput(); // perform some basic value checking to prevent NaNs

  MatCreate(PETSC_COMM_WORLD,&_muP);

  // if loading fields from source vecs
  if (_shearDistribution.compare("CVM")==0 ) {
    loadFieldsFromFiles();
  }

  // if setting fields from input values
  if (_shearDistribution.compare("basin")==0 ||
         _shearDistribution.compare("constant")==0 ||
         _shearDistribution.compare("gradient")==0 ||
         _shearDistribution.compare("mms")==0 )
  {
    setFieldsPlus();
    setNormalStress();
    if (_problemType.compare("full")==0) {
      MatCreate(PETSC_COMM_WORLD,&_muM);
      setFieldsMinus();
    }
  }

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::Domain in domain.cpp.\n");
#endif

}


Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz)
: _file(file),_delim(" = "),_startBlock("{"),_endBlock("}"),
  _order(0),_Ny(-1),_Nz(-1),_Ly(-1),_Lz(-1),_dy(-1),_dz(-1),_Dc(-1),
  _bcTType("unspecified"),_bcRType("unspecified"),_bcBType("unspecified"),
  _bcLType("unspecified"),
  _sigma_N_min(-1),_sigma_N_max(-1),_sigma_N(NULL),
  _shearDistribution("unspecified"),_problemType("unspecificed"),_inputDir("unspecified"),
  _muValPlus(-1),_rhoValPlus(-1),_muInPlus(-1),_muOutPlus(-1),
  _rhoInPlus(-1),_rhoOutPlus(-1),_depth(-1),_width(-1),
  _muArrPlus(NULL),_csArrPlus(NULL),_sigmaNArr(NULL),_muP(NULL),
  _muValMinus(-1),_rhoValMinus(-1),_muInMinus(-1),_muOutMinus(-1),
  _rhoInMinus(-1),_rhoOutMinus(-1),
  _muArrMinus(NULL),_csArrMinus(NULL),_muM(NULL),
  _viscDistribution("unspecified"),_visc(NULL),
  _linSolver("unspecified"),_kspTol(-1),
  _timeControlType("unspecified"),_timeIntegrator("unspecified"),
  _strideLength(-1),_maxStepCount(-1),_initTime(-1),_maxTime(-1),
  _minDeltaT(-1),_maxDeltaT(-1),_initDeltaT(_minDeltaT),
  _atol(-1),_outputDir("unspecified"),_f0(0.6),_v0(1e-6),_vL(-1)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz) in domain.cpp.\n");
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

#if VERBOSE > 2 // each processor prints loaded values to screen
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  for (int Ii=0;Ii<size;Ii++) { view(Ii); }
#endif

  MatCreate(PETSC_COMM_WORLD,&_muP);

  // if loading fields from source vecs
  if (_shearDistribution.compare("CVM")==0 ) {
    loadFieldsFromFiles();
  }

  // if setting fields from input values
  if (_shearDistribution.compare("basin")==0 ||
         _shearDistribution.compare("constant")==0 ||
         _shearDistribution.compare("gradient")==0 ||
         _shearDistribution.compare("mms")==0 )
  {
    setFieldsPlus();
    setNormalStress();
    if (_problemType.compare("full")==0) {
      MatCreate(PETSC_COMM_WORLD,&_muM);
      setFieldsMinus();
    }
  }


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::Domain(const char *file,PetscInt Ny, PetscInt Nz) in domain.cpp.\n");
#endif
}



Domain::~Domain()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::~Domain in domain.cpp.\n");
#endif

  VecDestroy(&_sigma_N);

  PetscFree(_muArrPlus);
  PetscFree(_csArrPlus);
  PetscFree(_sigmaNArr);
  PetscFree(_muArrMinus);
  PetscFree(_csArrMinus);

  MatDestroy(&_muP);
  MatDestroy(&_muM);

  VecDestroy(&_visc);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::~Domain in domain.cpp.\n");
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


  ifstream infile( file );
  string line,var;
  //~string delim = " = ";
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);

    if (var.compare("order")==0) { _order = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Ny")==0) { _Ny = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Nz")==0) { _Nz = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Ly")==0) { _Ly = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("Lz")==0) { _Lz = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    else if (var.compare("Dc")==0) { _Dc = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    // boundary condition types
    else if (var.compare("bcT")==0) { _bcTType = line.substr(pos+_delim.length(),line.npos); }
    else if (var.compare("bcR")==0) { _bcRType = line.substr(pos+_delim.length(),line.npos); }
    else if (var.compare("bcB")==0) { _bcBType = line.substr(pos+_delim.length(),line.npos); }
    else if (var.compare("bcL")==0) { _bcLType = line.substr(pos+_delim.length(),line.npos); }

    //fault properties
    else if (var.compare("sigma_N_min")==0) { _sigma_N_min = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("sigma_N_max")==0) { _sigma_N_max = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    else if (var.compare("vL")==0) { _vL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("aVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_aVals);
      _aLen = _aVals.size();
    }
    else if (var.compare("aDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_aDepths);
    }
    else if (var.compare("bVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_bVals);
    }
    else if (var.compare("bDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_bDepths);
    }

    // material properties
    else if (var.compare("shearDistribution")==0) {
      _shearDistribution = line.substr(pos+_delim.length(),line.npos);
      loadShearModSettings(infile);
    }
    else if (var.compare("inputDir")==0) {
      _inputDir = line.substr(pos+_delim.length(),line.npos);
    }

    // viscosity for asthenosphere
    else if (var.compare("viscDistribution")==0) { _viscDistribution = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("viscVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscVals);
      }
    else if (var.compare("viscDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscDepths);
      }


    // linear solver settings
    else if (var.compare("linSolver")==0) {
      _linSolver = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("kspTol")==0) { _kspTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) {
      _timeIntegrator = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("timeControlType")==0) {
      _timeControlType = line.substr(pos+_delim.length(),line.npos);
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
    }
  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Domain::loadVectorFromInputFile(const string& str,vector<double>& vec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  size_t pos = 0; // position of delimiter in string
  string delim = " "; // delimiter between values in list (whitespace sensitive)
  string remstr; // holds remaining string as str is parsed through
  double val; // holds values


  //~PetscPrintf(PETSC_COMM_WORLD,"About to start loading aVals:\n");
  //~PetscPrintf(PETSC_COMM_WORLD,"input str = %s\n\n",str.c_str());

  // holds remainder as str is parsed through (with beginning and ending brackets removed)
  pos = str.find("]");
  remstr = str.substr(1,pos-1);
  //~PetscPrintf(PETSC_COMM_WORLD,"remstr = %s\n",remstr.c_str());

  pos = remstr.find(delim);
  while (pos != remstr.npos) {
    pos = remstr.find(delim);
    val = atof( remstr.substr(0,pos).c_str() );
    remstr = remstr.substr(pos + delim.length());
    //~PetscPrintf(PETSC_COMM_WORLD,"val = %g  |  remstr = %s\n",val,remstr.c_str());
    vec.push_back(val);
  }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::loadVectorFromInputFile in domain.cpp.\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


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

// creates a string containing the contents of C++ std library vector
string Domain::vector2str(const vector<double> vec)
{
  ostringstream ss;
  for (vector<double>::const_iterator Ii=vec.begin(); Ii != vec.end(); Ii++) {
    ss << " " << *Ii;
  }
  string str = "[" + ss.str() + "]";
  //~PetscPrintf(PETSC_COMM_WORLD,"%s\n",str.c_str());

  return str;
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

    // boundary conditions
    ierr = PetscPrintf(PETSC_COMM_SELF,"bcT = %s\n",_bcTType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bcR = %s\n",_bcRType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bcB = %s\n",_bcBType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"bcL = %s\n",_bcLType.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // fault properties
    ierr = PetscPrintf(PETSC_COMM_SELF,"sigma_N_min = %f\n",_sigma_N_min);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"sigma_N_max = %f\n",_sigma_N_max);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"vp = %.15e\n",_vL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    // sedimentary basin properties
    ierr = PetscPrintf(PETSC_COMM_SELF,"inputDir = %s\n",_inputDir.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"shearDistribution = %s\n",_shearDistribution.c_str());CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"problemType = %s\n",_problemType.c_str());CHKERRQ(ierr);
    // y>0 properties
    if (_shearDistribution.compare("basin")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"muInPlus = %f\n",_muInPlus);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"muOutPlus = %f\n",_muOutPlus);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoInPlus = %f\n",_rhoInPlus);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoOutPlus = %f\n",_rhoOutPlus);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"muPlus = %f\n",_muValPlus);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoPlus = %f\n",_rhoValPlus);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      ierr = PetscPrintf(PETSC_COMM_SELF,"rhoPlus = %f\n",_rhoValPlus);CHKERRQ(ierr);
    }
    if (_problemType.compare("full")==0)
    {
      if (_shearDistribution.compare("basin")==0)
      {
        ierr = PetscPrintf(PETSC_COMM_SELF,"muInMinus = %f\n",_muInMinus);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"muOutMinus = %f\n",_muOutMinus);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"rhoInMinus = %f\n",_rhoInMinus);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"rhoOutMinus = %f\n",_rhoOutMinus);CHKERRQ(ierr);
      }
      else if (_shearDistribution.compare("constant")==0)
      {
        ierr = PetscPrintf(PETSC_COMM_SELF,"muMinus = %f\n",_muValMinus);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"rhoMinus = %f\n",_rhoValMinus);CHKERRQ(ierr);
      }
      else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
      {
        ierr = PetscPrintf(PETSC_COMM_SELF,"rhoMinus = %f\n",_rhoValMinus);CHKERRQ(ierr);
      }
      else if (_shearDistribution.compare("CVM")==0 )
      {
        ierr = PetscPrintf(PETSC_COMM_SELF,"inputDir = %s\n",_inputDir.c_str());CHKERRQ(ierr);
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"depth = %f\n",_depth);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"width = %f\n",_width);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_SELF,"visc = %.15e\n",_visc);CHKERRQ(ierr);

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


// Check that required fields have been set by the input file

PetscErrorCode Domain::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::checkInputPlus in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  assert( _order==2 || _order==4 );
  //~assert( _Ny > 3 && _Nz > 0 );
  assert( _Ly > 0 && _Lz > 0);
  assert( _dy > 0 && !isnan(_dy) );
  assert( _dz > 0 && !isnan(_dz) );

  assert(_Dc > 0 );
  assert(_aVals.size() == _aDepths.size() );
  assert(_bVals.size() == _bDepths.size() );
  assert(_sigma_N_min > 0);
  assert(_sigma_N_max > 0);
  assert(_sigma_N_max >= _sigma_N_min);

  assert(_vL > 0);

  assert(_viscVals.size() == _viscDepths.size() );


  assert(_timeIntegrator.compare("FEuler")==0 || _timeIntegrator.compare("RK32")==0);
  assert(_timeControlType.compare("P")==0 ||
         _timeControlType.compare("PI")==0 ||
         _timeControlType.compare("PID")==0 );
  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_strideLength >= 1);
  assert(_atol >= 1e-14);
  assert(_minDeltaT >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT > _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);

  assert(_rootTol >= 1e-14);

  assert(_linSolver.compare("MUMPSCHOLESKY") == 0 ||
         _linSolver.compare("MUMPSLU") == 0 ||
         _linSolver.compare("PCG") == 0 ||
         _linSolver.compare("AMG") == 0 );
  assert(_kspTol >= 1e-14);


    assert(_problemType.compare("full")==0 || _problemType.compare("symmetric")==0);
    assert(_shearDistribution.compare("basin")==0 ||
         _shearDistribution.compare("constant")==0 ||
         _shearDistribution.compare("gradient")==0 ||
         _shearDistribution.compare("mms")==0 ||
         _shearDistribution.compare("CVM")==0 );

  if (_shearDistribution.compare("constant")==0 ||
      _shearDistribution.compare("gradient")==0 ||
      _shearDistribution.compare("mms")==0 )
  {
    assert(_muValPlus>=1e-14);
    assert(_rhoValPlus>=1e-14);
    if (_problemType.compare("full")==0) {
      assert(_muValMinus>=1e-14);
      assert(_rhoValMinus>=1e-14);
    }
  }
  else if (_shearDistribution.compare("basin")==0) {
    assert(_muInPlus>=1e-14);
    assert(_muOutPlus>=1e-14);
    assert(_rhoInPlus>=1e-14);
    assert(_rhoOutPlus>=1e-14);
    assert(_depth>=1e-14);
    assert(_width>=1e-14);
    if (_problemType.compare("full")==0) {
      assert(_muInMinus>=1e-14);
      assert(_muOutMinus>=1e-14);
      assert(_rhoInMinus>=1e-14);
      assert(_rhoOutMinus>=1e-14);
    }
  }
  else if (_shearDistribution.compare("CVM")==0) {
    assert(_inputDir.compare("unspecified") != 0); // input dir must be specified
  }


  assert(_shearDistribution.compare("CVM")!=0 || _problemType.compare("full")!=0 );

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::checkInputPlus in domain.cpp.\n");CHKERRQ(ierr);
#endif
  //~}
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

  // boundary conditions
  ierr = PetscViewerASCIIPrintf(viewer,"bcT = %s\n",_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcR = %s\n",_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcB = %s\n",_bcBType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcL = %s\n",_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // fault properties
  ierr = PetscViewerASCIIPrintf(viewer,"_aVals = %s\n",vector2str(_aVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"_aDepths = %s\n",vector2str(_aDepths).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"_bVals = %s\n",vector2str(_bVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"_bDepths = %s\n",vector2str(_bDepths).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sigma_N_min = %.15e\n",_sigma_N_min);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"sigma_N_max = %.15e\n",_sigma_N_max);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"vp = %.15e\n",_vL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"visc = %.15e\n",_visc);CHKERRQ(ierr);

  // material properties
  ierr = PetscViewerASCIIPrintf(viewer,"shearDistribution = %s\n",_shearDistribution.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"problemType = %s\n",_problemType.c_str());CHKERRQ(ierr);
  // y>0 properties
  if (_shearDistribution.compare("basin")==0)
  {
    ierr = PetscViewerASCIIPrintf(viewer,"muInPlus = %.15e\n",_muInPlus);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"muOutPlus = %.15e\n",_muOutPlus);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"rhoInPlus = %.15e\n",_rhoInPlus);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"rhoOutPlus = %.15e\n",_rhoOutPlus);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"depth = %.15e\n",_depth);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"width = %.15e\n",_width);CHKERRQ(ierr);
  }
  else if (_shearDistribution.compare("constant")==0)
  {
    ierr = PetscViewerASCIIPrintf(viewer,"muPlus = %.15e\n",_muValPlus);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"rhoPlus = %.15e\n",_rhoValPlus);CHKERRQ(ierr);
  }
  else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
  {
    ierr = PetscViewerASCIIPrintf(viewer,"rhoPlus = %.15e\n",_rhoValPlus);CHKERRQ(ierr);
  }
  if (_problemType.compare("full")==0)
  {
    if (_shearDistribution.compare("basin")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"muInMinus = %.15e\n",_muInMinus);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"muOutMinus = %.15e\n",_muOutMinus);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rhoInMinus = %.15e\n",_rhoInMinus);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rhoOutMinus = %.15e\n",_rhoOutMinus);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("constant")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"muMinus = %.15e\n",_muValMinus);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"rhoMinus = %.15e\n",_rhoValMinus);CHKERRQ(ierr);
    }
    else if (_shearDistribution.compare("gradient")==0 || _shearDistribution.compare("mms")==0)
    {
      ierr = PetscViewerASCIIPrintf(viewer,"rhoMinus = %.15e\n",_rhoValMinus);CHKERRQ(ierr);
    }
  }
  if (_shearDistribution.compare("CVM")==0 )
  {
    ierr = PetscViewerASCIIPrintf(viewer,"inputDir = %s\n",_inputDir.c_str());CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"_viscVals = %s\n",vector2str(_viscVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"_viscDepths = %s\n",vector2str(_viscDepths).c_str());CHKERRQ(ierr);

  // linear solve settings
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol = %.15e\n",_kspTol);CHKERRQ(ierr);
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
  ierr = PetscViewerASCIIPrintf(viewer,"rootTol = %e\n",_rootTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"outputDir = %s\n",_outputDir.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"f0 = %e\n",_f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);


  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output shear modulus matrix
  str =  _outputDir + "muPlus";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(_muP,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output viscosity vector
  str =  _outputDir + "visc";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_visc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output normal stress vector
  str =  _outputDir + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (_problemType.compare("full")==0)
  {
    str =  _outputDir + "muMinus";
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(_muM,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending write in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}





PetscErrorCode Domain::setFieldsPlus()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFieldsPlus in domain.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar    v,y,z,csIn,csOut;

  Vec muVec;
  PetscInt *muInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_muArrPlus);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_csArrPlus);CHKERRQ(ierr);


  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);

  // set viscosity
  ierr = VecDuplicate(muVec,&_visc);CHKERRQ(ierr);
  PetscInt Istart,Iend;
  ierr = VecGetOwnershipRange(_visc,&Istart,&Iend);CHKERRQ(ierr);

  // layered viscosity
  //~// control on transition of viscosity (somewhat arbitrary choice)
  /*PetscScalar transToLowCrust = 5 + _depthUpToLowCrust; // km
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    if (z<_depthUpToLowCrust) { visc = _viscUpCrust; }
    else if (z <= transToLowCrust){
      //~//visc = (z - _depthUpToLowCrust)*(_viscUpCrust-_viscLowCrust)/(_depthUpToLowCrust-transToLowCrust)
             //~//+ _viscUpCrust;
      visc = (z - _depthUpToLowCrust)*(log10(_viscUpCrust)-log10(_viscLowCrust))/(_depthUpToLowCrust-transToLowCrust)
             + log10(_viscUpCrust);
      visc = pow(10,visc);
    }
    else { visc = _viscLowCrust; }
    ierr = VecSetValues(_visc,1,&Ii,&visc,INSERT_VALUES);CHKERRQ(ierr);
  }*/

  //~// gradient starting at lower crust
  //~for (Ii=Istart;Ii<Iend;Ii++)
  //~{
    //~z = _dz*(Ii-_Nz*(Ii/_Nz));
    //~if (z<_depthUpToLowCrust) { visc = _viscUpCrust; }
    //~else {
      //~visc = (z - _depthUpToLowCrust)*(log10(_viscUpCrust)-log10(_viscLowCrust))/(_depthUpToLowCrust-_Lz)
             //~+ log10(_viscUpCrust);
      //~visc = pow(10,visc);
    //~}
    //~ierr = VecSetValues(_visc,1,&Ii,&visc,INSERT_VALUES);CHKERRQ(ierr);
  //~}
  //~ierr = VecAssemblyBegin(_visc);CHKERRQ(ierr);
  //~ierr = VecAssemblyEnd(_visc);CHKERRQ(ierr);

  //~VecView(_visc,PETSC_VIEWER_STDOUT_WORLD);

  // build viscosity structure for generalized input
  size_t vecLen = _viscDepths.size();
  PetscScalar z0,z1,v0,v1;
  for (Ii=Istart;Ii<Iend;Ii++)
  {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    //~PetscPrintf(PETSC_COMM_WORLD,"1: Ii = %i, z = %g\n",Ii,z);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
        z0 = _viscDepths[0+ind];
        z1 = _viscDepths[0+ind+1];
        v0 = log10(_viscVals[0+ind]);
        v1 = log10(_viscVals[0+ind+1]);
        //~PetscPrintf(PETSC_COMM_WORLD,"  ind=%i: z0 = %g | z1 = %g | v0 = %g  | v1 = %g\n",ind,z0,z1,v0,v1);
        if (z>=z0 && z<=z1) {
          v = (v1 - v0)/(z1-z0) * (z-z0) + v0;
          v = pow(10,v);
          }
        ierr = VecSetValues(_visc,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_visc);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_visc);CHKERRQ(ierr);




  // set shear modulus, shear wave speed, and density
  // controls on transition in shear modulus
  PetscScalar r = 0;
  PetscScalar rbar = 0.25*_width*_width;
  PetscScalar rw = 1+0.25*_width*_width/_depth/_depth;
  for (Ii=0;Ii<_Ny*_Nz;Ii++) {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    y = _dy*(Ii/_Nz);
    r=y*y+(0.25*_width*_width/_depth/_depth)*z*z;

    if (_shearDistribution.compare("basin")==0) {
      v = 0.5*(_rhoOutPlus-_rhoInPlus)*(tanh((double)(r-rbar)/rw)+1) + _rhoInPlus;

      csIn = sqrt(_muInPlus/_rhoInPlus);
      csOut = sqrt(_muOutPlus/_rhoOutPlus);
      v = 0.5*(csOut-csIn)*(tanh((double)(r-rbar)/rw)+1) + csIn;
      _csArrPlus[Ii] = v;

      v = 0.5*(_muOutPlus-_muInPlus)*(tanh((double)(r-rbar)/rw)+1) + _muInPlus;
    }
    else if (_shearDistribution.compare("constant")==0) {
      _csArrPlus[Ii] = sqrt(_muValPlus/_rhoValPlus);
      v = _muValPlus;
    }
    else if (_shearDistribution.compare("gradient")==0) {
       _csArrPlus[Ii] = sqrt(_muValPlus/_rhoValPlus);
      v = Ii+2;
    }
    else if (_shearDistribution.compare("mms")==0) {
       _csArrPlus[Ii] = sqrt(_muValPlus/_rhoValPlus);
      v = sin(y)*sin(z) + 2.0;
    }
    else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: shearDistribution type not understood\n");CHKERRQ(ierr);
      assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
    }
    _muArrPlus[Ii] = v;
    muInds[Ii] = Ii;
  }

  ierr = VecSetValues(muVec,_Ny*_Nz,muInds,_muArrPlus,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(_muP,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_muP);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_muP,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_muP,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_muP);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_muP,muVec,INSERT_VALUES);CHKERRQ(ierr);

  VecDestroy(&muVec);
  PetscFree(muInds);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFieldsPlus in domain.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}


// Arrays start at fault and move out to remote boundaries.
PetscErrorCode Domain::setFieldsMinus()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFields in domain.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar    v,y,z,csIn,csOut;

  Vec muVec;
  PetscInt *muInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);

  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_muArrMinus);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_csArrMinus);CHKERRQ(ierr);


  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);

  PetscScalar r = 0;
  PetscScalar rbar = 0.25*_width*_width;
  PetscScalar rw = 1+0.25*_width*_width/_depth/_depth;
  for (Ii=0;Ii<_Ny*_Nz;Ii++) {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    y = -_Ly + _dy*(Ii/_Nz);
    //~y = - _dy*(Ii/_Nz);
    r=y*y+(0.25*_width*_width/_depth/_depth)*z*z;

    if (_shearDistribution.compare("basin")==0) {
      v = 0.5*(_rhoOutMinus-_rhoInMinus)*(tanh((double)(r-rbar)/rw)+1) + _rhoInMinus;

      csIn = sqrt(_muInMinus/_rhoInMinus);
      csOut = sqrt(_muOutMinus/_rhoOutMinus);
      v = 0.5*(csOut-csIn)*(tanh((double)(r-rbar)/rw)+1) + csIn;
      _csArrMinus[Ii] = v;

      v = 0.5*(_muOutMinus-_muInMinus)*(tanh((double)(r-rbar)/rw)+1) + _muInMinus;
    }
    else if (_shearDistribution.compare("constant")==0) {
      _csArrMinus[Ii] = sqrt(_muValMinus/_rhoValMinus);
      v = _muValMinus;
    }
    else if (_shearDistribution.compare("gradient")==0) {
       _csArrMinus[Ii] = sqrt(_muValMinus/_rhoValMinus);
      v = Ii+2;
    }
    else if (_shearDistribution.compare("mms")==0) {
       _csArrMinus[Ii] = sqrt(_muValMinus/_rhoValMinus);
      v = sin(y+z) + 2.0;
    }
    else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: shearDistribution type not understood\n");CHKERRQ(ierr);
      assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
    }
    _muArrMinus[Ii] = v;
    muInds[Ii] = Ii;
  }
  ierr = VecSetValues(muVec,_Ny*_Nz,muInds,_muArrMinus,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(_muM,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_muM);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_muM,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_muM,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_muM);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_muM,muVec,INSERT_VALUES);CHKERRQ(ierr);

  VecDestroy(&muVec);
  PetscFree(muInds);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFields in domain.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}



PetscErrorCode Domain::setNormalStress()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setNormalStress in domain.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar z = 0, g=9.8;
  PetscScalar rhoIn = _muArrPlus[0]/(_csArrPlus[0]*_csArrPlus[0]);
  PetscScalar rhoOut = _muArrPlus[_Nz-1]/(_csArrPlus[_Nz-1]*_csArrPlus[_Nz-1]);
  PetscInt    *sigmaInds;
  ierr = PetscMalloc(_Nz*sizeof(PetscScalar),&_sigmaNArr);CHKERRQ(ierr);
  ierr = PetscMalloc(_Nz*sizeof(PetscInt),&sigmaInds);CHKERRQ(ierr);
  for (Ii=0;Ii<_Nz;Ii++)
  {
    sigmaInds[Ii] = Ii;

    z = ((double) Ii)*_dz;
    // gradient following lithostatic - hydrostatic
    if (Ii<=_depth/_dz) {
      _sigmaNArr[Ii] = rhoIn*g*z - g*z;
    }
    else if (Ii>_depth/_dz) {
      _sigmaNArr[Ii] = rhoOut*g*(z-_depth) + rhoIn*g*_depth - g*z;
    }

    // normal stress is > 0 at Earth's surface
    _sigmaNArr[Ii] += _sigma_N_min;

    // cap to represent fluid overpressurization (Lapusta and Rice, 2000)
    // (in the paper, the max is 50 MPa)
    _sigmaNArr[Ii] =(PetscScalar) min((double) _sigmaNArr[Ii],_sigma_N_max);
  }
  VecCreate(PETSC_COMM_WORLD,&_sigma_N);
  VecSetSizes(_sigma_N,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_sigma_N);     PetscObjectSetName((PetscObject) _sigma_N, "_sigma_N");
  ierr = VecSetValues(_sigma_N,_Nz,sigmaInds,_sigmaNArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sigma_N);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sigma_N);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setNormalStress in domain.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}


// parse input file and load values into data members
PetscErrorCode Domain::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadFieldsFromFiles in domain.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt *muInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_muArrPlus);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscScalar),&_csArrPlus);CHKERRQ(ierr);


  // load normal stress: _sigma_N
  string vecSourceFile = _inputDir + "sigma_N";
  PetscViewer inv;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&_sigma_N);CHKERRQ(ierr);
  ierr = VecSetSizes(_sigma_N,PETSC_DECIDE,_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_sigma_N);     PetscObjectSetName((PetscObject) _sigma_N, "_sigma_N");
  ierr = VecLoad(_sigma_N,inv);CHKERRQ(ierr);


  // Create a local vector containing cs on each processor (may not work in parallel!!)
  // and put resulting data into array.
  Vec  localCs;
  PetscScalar cs;
  vecSourceFile = _inputDir + "cs";
  ierr = VecCreateSeq(PETSC_COMM_SELF,_Ny*_Nz,&localCs);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_SELF,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);
  ierr = VecLoad(localCs,inv);

  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(localCs,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(localCs,1,&Ii,&cs);CHKERRQ(ierr);
    _csArrPlus[Ii] = cs;
  }


  // load shear modulus distribution from file, put into _muArrPlus AND _muP
  Vec  localMu;
  PetscScalar mu;
  vecSourceFile = _inputDir + "shear";
  ierr = VecCreateSeq(PETSC_COMM_SELF,_Ny*_Nz,&localMu);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_SELF,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);
  ierr = VecLoad(localMu,inv);

  ierr = VecGetOwnershipRange(localMu,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(localMu,1,&Ii,&mu);CHKERRQ(ierr);
    _muArrPlus[Ii] = mu;
    muInds[Ii] = Ii;
  }

  Vec muVec;
  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);
  ierr = VecSetValues(muVec,_Ny*_Nz,muInds,_muArrPlus,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(_muP,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_muP);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_muP,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_muP,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_muP);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_muP,muVec,INSERT_VALUES);CHKERRQ(ierr);



  // load viscosity from input file
  vecSourceFile = _inputDir + "visc";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&_visc);CHKERRQ(ierr);
  ierr = VecSetSizes(_visc,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_visc);     PetscObjectSetName((PetscObject) _visc, "_visc");
  ierr = VecLoad(_visc,inv);CHKERRQ(ierr);




#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadFieldsFromFiles in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}

