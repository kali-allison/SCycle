#include "grainSizeEvolution.hpp"

#define FILENAME "grainSizeEvolution.cpp"


//======================================================================
// power-law rheology class

GrainSizeEvolution::GrainSizeEvolution(Domain& D)
: _D(&D),_file(D._file),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _grainSizeEvType("transient"),_grainSizeEvTypeSS("steadyState"),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _A(NULL),_QR(NULL),_p(NULL),_f(NULL),_gamma(NULL),_piez_A(NULL),_piez_n(NULL),_d(NULL),_d_t(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::GrainSizeEvolution";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  allocateFields(); // initialize fields
  setMaterialParameters();
  loadFieldsFromFiles(); // load from previous simulation

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

GrainSizeEvolution::~GrainSizeEvolution()
{
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::~GrainSizeEvolution";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_A);
  VecDestroy(&_QR);
  VecDestroy(&_p);
  VecDestroy(&_f);
  VecDestroy(&_gamma);
  VecDestroy(&_piez_A);
  VecDestroy(&_piez_n);
  VecDestroy(&_d);
  VecDestroy(&_d_t);

  for (map<string,pair<PetscViewer,string> >::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first].first);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// loads settings from the input text file
PetscErrorCode GrainSizeEvolution::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  ifstream infile( file );
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

    // static grain growth parameters
    if (var.compare("grainSizeEv_AVals")==0) { loadVectorFromInputFile(rhsFull,_AVals); }
    else if (var.compare("grainSizeEv_ADepths")==0) { loadVectorFromInputFile(rhsFull,_ADepths); }
    else if (var.compare("grainSizeEv_QRVals")==0) { loadVectorFromInputFile(rhsFull,_QRVals); }
    else if (var.compare("grainSizeEv_QRDepths")==0) { loadVectorFromInputFile(rhsFull,_QRDepths); }
    else if (var.compare("grainSizeEv_pVals")==0) { loadVectorFromInputFile(rhsFull,_pVals); }
    else if (var.compare("grainSizeEv_pDepths")==0) { loadVectorFromInputFile(rhsFull,_pDepths); }

    // (GJ/m^2) specific surface energy
    else if (var.compare("grainSizeEv_gammaVals")==0) { loadVectorFromInputFile(rhsFull,_gammaVals); }
    else if (var.compare("grainSizeEv_gammaDepths")==0) { loadVectorFromInputFile(rhsFull,_gammaDepths); }

    else if (var.compare("grainSizeEv_c")==0) { _c = atof( rhs.c_str() ); }

    // partitioning of mechanical work parameter
    else if (var.compare("grainSizeEv_fVals")==0) { loadVectorFromInputFile(rhsFull,_fVals); }
    else if (var.compare("grainSizeEv_fDepths")==0) { loadVectorFromInputFile(rhsFull,_fDepths); }

    // optional piezometer inputs
    else if (var.compare("grainSizeEv_piez_AVals")==0) { loadVectorFromInputFile(rhsFull,_piez_AVals); }
    else if (var.compare("grainSizeEv_piez_ADepths")==0) { loadVectorFromInputFile(rhsFull,_piez_ADepths); }
    else if (var.compare("grainSizeEv_piez_nVals")==0) { loadVectorFromInputFile(rhsFull,_piez_nVals); }
    else if (var.compare("grainSizeEv_piez_nDepths")==0) { loadVectorFromInputFile(rhsFull,_piez_nDepths); }

    // initial values for grain size
    else if (var.compare("grainSizeEv_grainSizeVals")==0) { loadVectorFromInputFile(rhsFull,_dVals); }
    else if (var.compare("grainSizeEv_grainSizeDepths")==0) { loadVectorFromInputFile(rhsFull,_dDepths); }

    if (var.compare("grainSizeEv_grainSizeEvType")==0) { _grainSizeEvType = rhs.c_str(); }
    if (var.compare("grainSizeEv_grainSizeEvTypeSS")==0) { _grainSizeEvTypeSS = rhs.c_str(); }

  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode GrainSizeEvolution::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

    assert(_AVals.size() >= 2);
    assert(_AVals.size() == _ADepths.size() );

    assert(_QRVals.size() >= 2);
    assert(_QRVals.size() == _QRDepths.size() );

    assert(_pVals.size() >= 2);
    assert(_pVals.size() == _pDepths.size() );

    assert(_fVals.size() >= 2);
    assert(_fVals.size() == _fDepths.size() );

    assert(_gammaVals.size() >= 2);
    assert(_gammaVals.size() == _gammaDepths.size() );

    assert(_dVals.size() >= 2);
    assert(_dVals.size() == _dDepths.size() );

    assert(_c > 0);

    assert(_piez_AVals.size() == _piez_ADepths.size() );
    assert(_piez_nVals.size() == _piez_nDepths.size() );

    assert(_grainSizeEvType.compare("transient")==0 ||
      _grainSizeEvType.compare("steadyState")==0 ||
      _grainSizeEvType.compare("piezometer")==0 );

    assert(_grainSizeEvTypeSS.compare("transient")==0 ||
      _grainSizeEvTypeSS.compare("steadyState")==0 ||
      _grainSizeEvTypeSS.compare("piezometer")==0 );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// allocate space for member fields
PetscErrorCode GrainSizeEvolution::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDuplicate(*_z,&_A); VecSet(_A,0.0);
  VecDuplicate(_A,&_QR); VecSet(_QR,0.0);
  VecDuplicate(_A,&_p); VecSet(_p,0.0);
  VecDuplicate(_A,&_f); VecSet(_f,0.0);
  VecDuplicate(_A,&_gamma); VecSet(_gamma,0.0);
  VecDuplicate(_A,&_d); VecSet(_d,0.0);
  VecDuplicate(_A,&_d_t); VecSet(_d_t,0.0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode GrainSizeEvolution::setMaterialParameters()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::setMaterialParameters";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set each field using it's vals and depths std::vectors
  ierr = setVec(_A,*_z,_AVals,_ADepths);                                CHKERRQ(ierr);
  ierr = setVec(_QR,*_z,_QRVals,_QRDepths);                             CHKERRQ(ierr);
  ierr = setVec(_p,*_z,_pVals,_pDepths);                                CHKERRQ(ierr);
  ierr = setVec(_f,*_z,_fVals,_fDepths);                                CHKERRQ(ierr);
  ierr = setVec(_gamma,*_z,_gammaVals,_gammaDepths);                    CHKERRQ(ierr);
  ierr = setVec(_d,*_z,_dVals,_dDepths);                                CHKERRQ(ierr);
  VecSet(_d_t,0.);

  // if user provided piezometric relation
  if (_grainSizeEvType == "piezometer" || _grainSizeEvTypeSS == "piezometer") {
    VecDuplicate(*_z,&_piez_A); VecSet(_piez_A,0.0);
    VecDuplicate(*_z,&_piez_n); VecSet(_piez_n,0.0);
    ierr = setVec(_piez_A,*_z,_piez_AVals,_piez_ADepths);                                CHKERRQ(ierr);
    ierr = setVec(_piez_n,*_z,_piez_nVals,_piez_nDepths);                                CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
return ierr;
}


//parse input file and load values into data members
PetscErrorCode GrainSizeEvolution::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_A,_inputDir,"grainSizeEv_A"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_QR,_inputDir,"grainSizeEv_QR"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_p,_inputDir,"grainSizeEv_p"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_f,_inputDir,"grainSizeEv_f"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_gamma,_inputDir,"grainSizeEv_gamma"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_d,_inputDir,"grainSizeEv_d"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_d_t,_inputDir,"grainSizeEv_d_t"); CHKERRQ(ierr);

  if (_grainSizeEvType == "piezometer" || _grainSizeEvTypeSS == "piezometer") {
    ierr = loadVecFromInputFile(_piez_A,_inputDir,"grainSizeEv_piez_A"); CHKERRQ(ierr);
    ierr = loadVecFromInputFile(_piez_n,_inputDir,"grainSizeEv_piez_n"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode GrainSizeEvolution::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // add deep copy of grain size to integrated variables, stored in _var
  if (_grainSizeEvType == "transient") {
    if (varEx.find("grainSize") != varEx.end() ) { VecCopy(_d,varEx["grainSize"]); }
    else { Vec var; VecDuplicate(_d,&var); VecCopy(_d,var); varEx["grainSize"] = var; }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode GrainSizeEvolution::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("grainSize")->second,_d);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode GrainSizeEvolution::d_dt(Vec& grainSizeEv_t,const Vec& grainSize,const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  const PetscScalar *A,*B,*p,*T,*f,*g,*s,*dgdev,*d;
  PetscScalar *d_t;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_d,&Istart,&Iend);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&B);
  VecGetArrayRead(_p,&p);
  VecGetArrayRead(Temp,&T);
  VecGetArrayRead(_f,&f);
  VecGetArrayRead(_gamma,&g);
  VecGetArrayRead(sdev,&s);
  VecGetArrayRead(dgdev_disl,&dgdev);
  VecGetArrayRead(grainSize,&d);
  VecGetArray(_d_t,&d_t);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar cc = f[Jj] / (g[Jj] *_c);

    // static grain growth rate
    PetscScalar growth = A[Jj] * exp(-B[Jj]/T[Jj]) * (1.0/p[Jj]) * pow(d[Jj], 1.0-p[Jj]);

    // grain size reduction from work done by dislocation creep
    PetscScalar w = s[Jj]*0.5*dgdev[Jj]; // work, 0.5 to convert from engineering strain rate to geophysics strain rate
    PetscScalar red = - cc * d[Jj]*d[Jj] * w;
    d_t[Jj] = growth + red;
    if (std::isinf(red)) {
      PetscPrintf(PETSC_COMM_WORLD,"%i: cc = %.15e, d = %.15e, s = %.15e, dgdev = %.15e\n",Jj,cc,d[Jj],s[Jj],dgdev[Jj]);
    }

    assert(!std::isnan(dgdev[Jj]));
    assert(!std::isinf(dgdev[Jj]));
    assert(!std::isnan(growth));
    assert(!std::isinf(growth));
    assert(!std::isnan(red));
    assert(!std::isinf(red));
    assert(!std::isnan(d_t[Jj]));
    assert(!std::isinf(d_t[Jj]));

    Jj++;
  }
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&B);
  VecRestoreArrayRead(_p,&p);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArrayRead(_f,&f);
  VecRestoreArrayRead(_gamma,&g);
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArrayRead(dgdev_disl,&dgdev);
  VecRestoreArrayRead(grainSize,&d);
  VecRestoreArray(_d_t,&d_t);

  VecCopy(_d_t,grainSizeEv_t);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute grain size based on piezometric relation
PetscErrorCode GrainSizeEvolution::computeGrainSizeFromPiez(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::computeGrainSizeFromPiez";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_d,&Istart,&Iend);
  const PetscScalar *s;
  VecGetArrayRead(sdev,&s);

  PetscScalar *d;
  VecGetArray(_d,&d);

  const PetscScalar *A,*n;
  VecGetArrayRead(_piez_A,&A);
  VecGetArrayRead(_piez_n,&n);

  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    d[Jj] = A[Jj] * pow(s[Jj],n[Jj]);

    // impose floor and ceiling to grain size
    d[Jj] = max(d[Jj],1e-7);
    d[Jj] = min(d[Jj],10.0);

    assert(!std::isnan(d[Jj]));
    assert(!std::isinf(d[Jj]));

    Jj++;
  }
  VecRestoreArrayRead(_piez_A,&A);
  VecRestoreArrayRead(_piez_n,&n);
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArray(_d,&d);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


//======================================================================
// Steady state functions
//======================================================================



// compute steady-state grain size from Austin and Evans (2007)
PetscErrorCode GrainSizeEvolution::computeSteadyStateGrainSize(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::computeSteadyStateGrainSize";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_d,&Istart,&Iend);
  const PetscScalar *s;
  VecGetArrayRead(sdev,&s);

  PetscScalar *d;
  VecGetArray(_d,&d);


  const PetscScalar *A,*B,*p,*T,*f,*g,*dgdev;

  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&B);
  VecGetArrayRead(_p,&p);
  VecGetArrayRead(Temp,&T);
  VecGetArrayRead(_f,&f);
  VecGetArrayRead(_gamma,&g);
  VecGetArrayRead(dgdev_disl,&dgdev);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar AA = A[Jj] * exp(-B[Jj]/T[Jj]) * (1.0/p[Jj]);
    PetscScalar BB = f[Jj] / (g[Jj] *_c) * 0.5 * dgdev[Jj]; // 0.5 to convert dgdev from engineering to geophysics convention
    PetscScalar a = 1.0 - p[Jj];
    PetscScalar b = 1.0;
    PetscScalar c = 2.0;

    if ( std::isinf( pow(BB/AA,1.0/(a-c)) ) ) {
      d[Jj] = 1e-8;
    }
    else {
      d[Jj] = pow(BB/AA,1.0/(a-c)) * pow(s[Jj],b/(a-c));
    }

    if ( std::isnan(d[Jj]) ) {

      PetscPrintf(PETSC_COMM_WORLD,"A = %.15e, QR = %.15e, p = %.15e, T = %.15e\n", A[Jj], B[Jj], p[Jj], T[Jj]);
      PetscPrintf(PETSC_COMM_WORLD,"AA = %.15e, BB = %.15e, a = %.15e, b = %.15e, c = %.15e\n", AA, BB, a, b, c);
      PetscPrintf(PETSC_COMM_WORLD,"pow(BB/AA,1.0/(a-c)) = %.15e\n", pow(BB/AA,1.0/(a-c)));
      PetscPrintf(PETSC_COMM_WORLD,"b/(a-c) = %.15e\n", b/(a-c));
      PetscPrintf(PETSC_COMM_WORLD,"sdev = %.15e\n", s[Jj]);
    }
    if ( std::isinf(d[Jj]) ) {

      PetscPrintf(PETSC_COMM_WORLD,"A = %.15e, QR = %.15e, p = %.15e, T = %.15e\n", A[Jj], B[Jj], p[Jj], T[Jj]);
      PetscPrintf(PETSC_COMM_WORLD,"AA = %.15e, BB = %.15e, a = %.15e, b = %.15e, c = %.15e\n", AA, BB, a, b, c);
      PetscPrintf(PETSC_COMM_WORLD,"pow(BB/AA,1.0/(a-c)) = %.15e\n", pow(BB/AA,1.0/(a-c)));
      PetscPrintf(PETSC_COMM_WORLD,"b/(a-c) = %.15e\n", b/(a-c));
      PetscPrintf(PETSC_COMM_WORLD,"sdev = %.15e\n", s[Jj]);
    }

    assert(!std::isnan(d[Jj]));
    assert(!std::isinf(d[Jj]));

    Jj++;
  }
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&B);
  VecRestoreArrayRead(_p,&p);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArrayRead(_f,&f);
  VecRestoreArrayRead(_gamma,&g);
  VecRestoreArrayRead(dgdev_disl,&dgdev);
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArray(_d,&d);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// put viscous strains etc in varSS
PetscErrorCode GrainSizeEvolution::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::initiateVarSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  varSS["grainSizeEv_d"] = _d;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


//======================================================================
// IO functions
//======================================================================

PetscErrorCode GrainSizeEvolution::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Grain Size Evolution Runtime Summary:\n");CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode GrainSizeEvolution::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "GrainSizeEvolution::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif


  ierr = writeVec(_A,outputDir + "grainSizeEv_A");                      CHKERRQ(ierr);
  ierr = writeVec(_QR,outputDir + "grainSizeEv_QR");                    CHKERRQ(ierr);
  ierr = writeVec(_p,outputDir + "grainSizeEv_p");                      CHKERRQ(ierr);
  ierr = writeVec(_f,outputDir + "grainSizeEv_f");                      CHKERRQ(ierr);
  ierr = writeVec(_gamma,outputDir + "grainSizeEv_gamma");              CHKERRQ(ierr);

  if (_grainSizeEvType == "piezometer" || _grainSizeEvTypeSS == "piezometer") {
    ierr = writeVec(_piez_A,outputDir + "grainSizeEv_piez_A");          CHKERRQ(ierr);
    ierr = writeVec(_piez_n,outputDir + "grainSizeEv_piez_n");          CHKERRQ(ierr);
  }


  // output scalar fields
  std::string str = _outputDir + "grainSizeEv_context.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"grainSizeEvType = %s\n",_grainSizeEvType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"grainSizeEvTypeSS = %s\n",_grainSizeEvTypeSS.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"c = %g\n",_c);CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode GrainSizeEvolution::writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "GrainSizeEvolution::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  //~ double startTime = MPI_Wtime();

  if (_viewers.empty()) {
    ierr = io_initiateWriteAppend(_viewers, "grainSizeEv_d", _d, outputDir + "grainSizeEv_d"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "grainSizeEv_d_t", _d_t, outputDir + "grainSizeEv_d_t"); CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_d,_viewers["grainSizeEv_d"].first); CHKERRQ(ierr);
    ierr = VecView(_d_t,_viewers["grainSizeEv_d_t"].first); CHKERRQ(ierr);
  }

  //~ _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at time %g\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


