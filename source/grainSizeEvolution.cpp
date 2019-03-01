#include "grainSizeEvolution.hpp"

#define FILENAME "grainSizeEvolution.cpp"


//======================================================================
// power-law rheology class

GrainSizeEvolution::GrainSizeEvolution(Domain& D)
: _D(&D),_file(D._file),_delim(D._delim),_inputDir(D._inputDir),_outputDir(D._outputDir),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_y(&D._y),_z(&D._z),
  _A(NULL),_QR(NULL),_p(NULL),_f(NULL),_gamma(NULL),_d(NULL),_d_t(NULL)
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

  // boundary conditions
  VecDestroy(&_A);
  VecDestroy(&_QR);
  VecDestroy(&_p);
  VecDestroy(&_f);
  VecDestroy(&_gamma);

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

    // initial values for grain size
    else if (var.compare("grainSizeEv_grainSizeVals")==0) { loadVectorFromInputFile(rhsFull,_dVals); }
    else if (var.compare("grainSizeEv_grainSizeDepths")==0) { loadVectorFromInputFile(rhsFull,_dDepths); }

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

// set off-fault material properties
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


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}





PetscErrorCode GrainSizeEvolution::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  // add deep copies of viscous strains to integrated variables, stored in _var
  if (varEx.find("grainSize") != varEx.end() ) { VecCopy(_d,varEx["grainSize"]); }
  else { Vec var; VecDuplicate(_d,&var); VecCopy(_d,var); varEx["grainSize"] = var; }

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

  // if integrating viscous strains in time
  VecCopy(varEx.find("grainSizeEv_d")->second,_d);

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

  VecCopy(grainSize,_d);


  const PetscScalar *A,*B,*p,*T,*f,*s,*dgdev,*g;
  PetscScalar *dg;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_d,&Istart,&Iend);
  VecGetArrayRead(_A,&A);
  VecGetArrayRead(_QR,&B);
  VecGetArrayRead(_p,&p);
  VecGetArrayRead(Temp,&T);
  VecGetArrayRead(_f,&f);
  VecGetArrayRead(sdev,&s);
  VecGetArrayRead(dgdev_disl,&dgdev);
  VecGetArrayRead(grainSize,&g);
  VecGetArray(_d_t,&dg);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar growth = A[Jj] * exp(-B[Jj]/T[Jj]) * (1.0/p[Jj]) * pow(g[Jj], 1.0-p[Jj]); // static grain growth rate
    PetscScalar red = - f[Jj] * g[Jj]*g[Jj] * s[Jj]*dgdev[Jj]; // size reduction from disl. creep
    dg[Jj] = growth + red;
    Jj++;
  }
  VecRestoreArrayRead(_A,&A);
  VecRestoreArrayRead(_QR,&B);
  VecRestoreArrayRead(_p,&p);
  VecRestoreArrayRead(Temp,&T);
  VecRestoreArrayRead(_f,&f);
  VecRestoreArrayRead(sdev,&s);
  VecRestoreArrayRead(dgdev_disl,&dgdev);
  VecRestoreArrayRead(grainSize,&g);
  VecRestoreArray(_d_t,&dg);

  VecCopy(_d_t,grainSizeEv_t);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


//======================================================================
// Steady state functions
//======================================================================



// compute steady-state grain size
PetscErrorCode GrainSizeEvolution::computeSteadyStateGrainSize(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "GrainSizeEvolution::computeSteadyStateGrainSize";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  const PetscScalar *A,*B,*p,*T,*f,*g,*s,*dgdev;
  PetscScalar *d;
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
  VecGetArray(_d,&d);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar cc = f[Jj] / (g[Jj] *_c);
    PetscScalar temp = A[Jj]*exp(-B[Jj]/T[Jj]) / (p[Jj]*cc*s[Jj]*dgdev[Jj]);
    PetscScalar n = 1.0 / (1.0 + p[Jj]);
    d[Jj] = pow(temp, n);
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


PetscErrorCode GrainSizeEvolution::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "GrainSizeEvolution::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif


  ierr = writeVec(_A,outputDir + "grainSizeEv_A");                        CHKERRQ(ierr);
  ierr = writeVec(_QR,outputDir + "grainSizeEv_QR");                      CHKERRQ(ierr);
  ierr = writeVec(_p,outputDir + "grainSizeEv_p");                        CHKERRQ(ierr);
  ierr = writeVec(_f,outputDir + "grainSizeEv_f");                        CHKERRQ(ierr);
  ierr = writeVec(_gamma,outputDir + "grainSizeEv_gamma");                CHKERRQ(ierr);

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

  if (stepCount == 0) {
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
