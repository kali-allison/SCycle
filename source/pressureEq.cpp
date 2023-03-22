#include "pressureEq.hpp"

#define FILENAME "pressureEq.cpp"

using namespace std;

// constructor for PressureEq
// by default, uses AMG for preconditioning
// default explicit time integration, permeability not slip- or pressure dependent
// bottom boundary condition Q
PressureEq::PressureEq(Domain &D)
: _D(&D), _p(NULL), _permSlipDependent("no"), _permPressureDependent("no"),
  _file(D._file), _delim(D._delim),
  _outputDir(D._outputDir), _isMMS(D._isMMS),
  _hydraulicTimeIntType("explicit"),_guessSteadyStateICs(1),
  _initTime(0.0), _initDeltaT(1e-3),
  _order(D._order), _N(D._Nz), _L(D._Lz), _h(D._dr), _z(NULL),
  _n_p(NULL), _beta_p(NULL), _k_p(NULL), _eta_p(NULL), _rho_f(NULL), _g(9.8),
  _bcB_ratio(1.0), _bcB_type("Q"),
  _maxBeIteration(1), _minBeDifference(0.01),
  _linSolver("AMG"), _ksp(NULL), _kspTol(1e-10), _sbp(NULL), _linSolveCount(0),
  _writeTime(0), _linSolveTime(0), _ptTime(0), _startTime(0), _miscTime(0), _invTime(0)
{
  #if VERBOSE > 1
    string funcName = "PressureEq::PressureEq";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  loadSettings(_file);
  checkInput();
  setFields(D);

  if (_D->_restartFromChkpt) {
    assert(0);
    _guessSteadyStateICs = 0;
  }
  else if (_D->_restartFromChkptSS) {
    assert(0);
    _guessSteadyStateICs = 0;
  }

  setUpSBP();

  // Backward Eular updates
  if (_hydraulicTimeIntType.compare("implicit") == 0) {
    setUpBe(D);
  }

  // compute initial conditions
  // TODO: I think this logic can be improved
  if (_isMMS == 0) {
    if (_guessSteadyStateICs == 1) {
      if (_permPressureDependent == "no") {
        _maxBeIteration = 1;
      }
      if (_permPressureDependent == "yes" && _permSlipDependent == "yes") {
        VecCopy(_k_p, _k_slip);
      }
      for (int i = 0; i < _maxBeIteration; i++) {
        computeInitialSteadyStatePressure(D);
        if (_permPressureDependent == "yes") {
          updatePermPressureDependent();
          Vec coeff;
          computeVariableCoefficient(coeff);
          _sbp->updateVarCoeff(coeff);
          updateBoundaryCoefficient(coeff);
          VecDestroy(&coeff);
        }
      }
      if (_permPressureDependent == "yes" && _permSlipDependent == "yes") {
        VecCopy(_k_slip, _k_p);
      }
    }
  }

  if (_D->_restartFromChkpt) { loadCheckpoint(); }
  else { loadFieldsFromFiles(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
}


// destructor, free memory
PressureEq::~PressureEq()
{
  #if VERBOSE > 1
    string funcName = "PressureEq::~PressureEq";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // fields that exist on the fault
  VecDestroy(&_n_p);
  VecDestroy(&_beta_p);
  VecDestroy(&_k_p);
  VecDestroy(&_eta_p);
  VecDestroy(&_rho_f);

  // permeability
  VecDestroy(&_k_slip);
  VecDestroy(&_kL_p);
  VecDestroy(&_kT_p);
  VecDestroy(&_kmin_p);
  VecDestroy(&_kmax_p);
  VecDestroy(&_k_press);
  VecDestroy(&_kmin2_p);
  VecDestroy(&_sigma_p);

  VecDestroy(&_p);
  VecDestroy(&_bcL);
  VecDestroy(&_bcT);
  VecDestroy(&_bcB);
  VecDestroy(&_p_t);
  VecDestroy(&_z);
  VecDestroy(&_bcB_gravity);
  VecDestroy(&_bcB_impose);
  VecDestroy(&_sN);
  KSPDestroy(&_ksp);

  delete _sbp;

  KSPDestroy(&_ksp);
  VecScatterDestroy(&_scatters);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
}


// return pressure: copy _p to P
PetscErrorCode PressureEq::getPressure(Vec &P)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::getPressure()";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(_p, P);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);

  #endif
  return ierr;
}


// set pressure, copies P to output _p
PetscErrorCode PressureEq::setPressure(const Vec &P)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::setPressure()";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(P, _p);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// return permeability: copy _k_p to K
PetscErrorCode PressureEq::getPermeability(Vec &K)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::getPermeability()";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(_k_p, K);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// set permeability: copy K to _k_p
PetscErrorCode PressureEq::setPremeability(const Vec &K)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::setPremeability()";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(K, _k_p);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// load field from input file
PetscErrorCode PressureEq::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::loadSettings";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  PetscMPIInt rank, size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ifstream infile(file);
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  while (getline(infile, line)) {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0, pos);
    rhs = "";
    if (line.length() > (pos + _delim.length())) {
      rhs = line.substr(pos + _delim.length(), line.npos);
    }
    rhsFull = rhs; // everything after _delim

    // interpret everything after the appearance of a space on the line as a comment
    pos = rhs.find(" ");
    rhs = rhs.substr(0, pos);

    if (var.compare("guessSteadyStateICs") == 0) { _guessSteadyStateICs = atoi(rhs.c_str()); }
    else if (var.compare("linSolver") == 0) { _linSolver = rhs.c_str(); }
    else if (var.compare("hydraulicTimeIntType") == 0) { _hydraulicTimeIntType = rhs.c_str(); }
    else if (var.compare("bcB_ratio") == 0) { _bcB_ratio = atof(rhs.c_str()); }
    else if (var.compare("bcB_type") == 0) { _bcB_type = rhs.c_str(); }

    // loading vector inputs in the input file
    else if (var.compare("sNVals") == 0) { loadVectorFromInputFile(rhsFull, _sigmaNVals); }
    else if (var.compare("sNDepths") == 0) { loadVectorFromInputFile(rhsFull, _sigmaNDepths); }
    else if (var.compare("n_pVals") == 0) { loadVectorFromInputFile(rhsFull, _n_pVals); }
    else if (var.compare("n_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _n_pDepths); }
    else if (var.compare("beta_pVals") == 0) { loadVectorFromInputFile(rhsFull, _beta_pVals); }
    else if (var.compare("beta_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _beta_pDepths); }
    else if (var.compare("k_pVals") == 0) { loadVectorFromInputFile(rhsFull, _k_pVals); }
    else if (var.compare("k_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _k_pDepths); }
    else if (var.compare("eta_pVals") == 0) { loadVectorFromInputFile(rhsFull, _eta_pVals); }
    else if (var.compare("eta_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _eta_pDepths); }
    else if (var.compare("rho_fVals") == 0) { loadVectorFromInputFile(rhsFull, _rho_fVals); }
    else if (var.compare("rho_fDepths") == 0) { loadVectorFromInputFile(rhsFull, _rho_fDepths); }
    else if (var.compare("g") == 0) { _g = atof(rhs.c_str()); }
    else if (var.compare("pVals") == 0) { loadVectorFromInputFile(rhsFull, _pVals); }
    else if (var.compare("pDepths") == 0) { loadVectorFromInputFile(rhsFull, _pDepths); }
    else if (var.compare("vL") == 0) { _vL = atof(rhs.c_str()); }

    // permability evolution parameters - slip-dependent
    else if (var.compare("permSlipDependent") == 0) { _permSlipDependent = rhs.c_str(); }
    else if (var.compare("kL_pVals") == 0) { loadVectorFromInputFile(rhsFull, _kL_pVals); }
    else if (var.compare("kL_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _kL_pDepths); }
    else if (var.compare("kT_pVals") == 0) { loadVectorFromInputFile(rhsFull, _kT_pVals); }
    else if (var.compare("kT_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _kT_pDepths); }
    else if (var.compare("kmin_pVals") == 0) { loadVectorFromInputFile(rhsFull, _kmin_pVals); }
    else if (var.compare("kmin_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _kmin_pDepths); }
    else if (var.compare("kmax_pVals") == 0) { loadVectorFromInputFile(rhsFull, _kmax_pVals); }
    else if (var.compare("kmax_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _kmax_pDepths); }

    // permeability evolution parameters - pressure-dependent
    else if (var.compare("permPressureDependent") == 0) { _permPressureDependent = rhs.c_str(); }
    else if (var.compare("kmin2_pVals") == 0) { loadVectorFromInputFile(rhsFull, _kmin2_pVals); }
    else if (var.compare("kmin2_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _kmin2_pDepths); }
    else if (var.compare("sigma_pVals") == 0) { loadVectorFromInputFile(rhsFull, _sigma_pVals); }
    else if (var.compare("sigma_pDepths") == 0) { loadVectorFromInputFile(rhsFull, _sigma_pDepths); }
    else if (var.compare("maxBeIteration") == 0) { _maxBeIteration = (int)atof(rhs.c_str()); }
    else if (var.compare("minBeDifference") == 0) { _minBeDifference = atof(rhs.c_str()); }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// initialize vector fields
PetscErrorCode PressureEq::setFields(Domain &D)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::setFields";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // allocate memory and partition accross processors
  VecCreate(PETSC_COMM_WORLD, &_p);
  VecSetSizes(_p, PETSC_DECIDE, _N);
  VecSetFromOptions(_p);
  PetscObjectSetName((PetscObject)_p, "_p");
  VecSet(_p, 0.0);

  VecDuplicate(_p, &_p_t);
  PetscObjectSetName((PetscObject)_p_t, "p_t");
  VecSet(_p_t, 0.0);

  // extract local z from D._z (which is the full 2D field)
  VecDuplicate(_p, &_z);
  PetscInt Istart, Iend;
  PetscScalar z = 0;
  VecGetOwnershipRange(D._z, &Istart, &Iend);

  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    if (Ii < _N) {
      VecGetValues(D._z, 1, &Ii, &z);
      VecSetValue(_z, Ii, z, INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_z);
  VecAssemblyEnd(_z);

  VecDuplicate(_p, &_n_p);
  PetscObjectSetName((PetscObject)_n_p, "n_p");
  VecSet(_n_p, 0.0);

  VecDuplicate(_p, &_beta_p);
  PetscObjectSetName((PetscObject)_beta_p, "beta_p");
  VecSet(_beta_p, 0.0);

  VecDuplicate(_p, &_k_p);
  PetscObjectSetName((PetscObject)_k_p, "k_p");
  VecSet(_k_p, 0.0);

  VecDuplicate(_p, &_k_slip);
  PetscObjectSetName((PetscObject)_k_slip, "k_slip");
  VecSet(_k_slip, 0.0);

  VecDuplicate(_p, &_k_press);
  PetscObjectSetName((PetscObject)_k_press, "k_press");
  VecSet(_k_press, 0.0);

  VecDuplicate(_p, &_eta_p);
  PetscObjectSetName((PetscObject)_eta_p, "eta_p");
  VecSet(_eta_p, 0.0);

  VecDuplicate(_p, &_rho_f);
  PetscObjectSetName((PetscObject)_rho_f, "rho_f");
  VecSet(_rho_f, 0.0);

  VecDuplicate(_p, &_sN);
  PetscObjectSetName((PetscObject)_sN, "sN");
  VecSet(_sN, 0.0);

  ierr = setVec(_n_p, _z, _n_pVals, _n_pDepths); CHKERRQ(ierr);
  ierr = setVec(_beta_p, _z, _beta_pVals, _beta_pDepths); CHKERRQ(ierr);
  ierr = setVec(_k_p, _z, _k_pVals, _k_pDepths); CHKERRQ(ierr);

  VecCopy(_k_p, _k_slip);
  VecCopy(_k_p, _k_press);

  ierr = setVec(_eta_p, _z, _eta_pVals, _eta_pDepths); CHKERRQ(ierr);
  ierr = setVec(_rho_f, _z, _rho_fVals, _rho_fDepths); CHKERRQ(ierr);
  ierr = setVec(_sN, _z, _sigmaNVals, _sigmaNDepths); CHKERRQ(ierr);

  // for slip-dependent permeability
  if ( _permSlipDependent.compare("yes") == 0 ) {
    VecDuplicate(_p, &_kL_p);
    PetscObjectSetName((PetscObject)_kL_p, "kL_p");
    VecSet(_kL_p, 0.0);

    VecDuplicate(_p, &_kT_p);
    PetscObjectSetName((PetscObject)_kT_p, "kT_p");
    VecSet(_kT_p, 0.0);

    VecDuplicate(_p, &_kmin_p);
    PetscObjectSetName((PetscObject)_kmin_p, "kmin_p");
    VecSet(_kmin_p, 0.0);

    VecDuplicate(_p, &_kmax_p);
    PetscObjectSetName((PetscObject)_kmax_p, "kmax_p");
    VecSet(_kmax_p, 0.0);

    VecDuplicate(_p, &_kT_p);
    PetscObjectSetName((PetscObject)_kT_p, "kT_p");
    VecSet(_kT_p, 0.0);

    VecDuplicate(_p, &_kmin_p);
    PetscObjectSetName((PetscObject)_kmin_p, "kmin_p");
    VecSet(_kmin_p, 0.0);

    VecDuplicate(_p, &_kmax_p);
    PetscObjectSetName((PetscObject)_kmax_p, "kmax_p");
    VecSet(_kmax_p, 0.0);

    ierr = setVec(_kL_p, _z, _kL_pVals, _kL_pDepths); CHKERRQ(ierr);
    ierr = setVec(_kT_p, _z, _kT_pVals, _kT_pDepths); CHKERRQ(ierr);
    ierr = setVec(_kmin_p, _z, _kmin_pVals, _kmin_pDepths); CHKERRQ(ierr);
    ierr = setVec(_kmax_p, _z, _kmax_pVals, _kmax_pDepths); CHKERRQ(ierr);
  }

  // for pressure-dependent permeability
  if ( _permPressureDependent.compare("yes") == 0 ) {
    VecDuplicate(_p, &_kmin2_p);
    PetscObjectSetName((PetscObject)_kmin2_p, "meanP_p");
    VecSet(_kmin2_p, 0.0);

    VecDuplicate(_p, &_sigma_p);
    PetscObjectSetName((PetscObject)_sigma_p, "sigma_p");
    VecSet(_sigma_p, 0.0);

    ierr = setVec(_kmin2_p, _z, _kmin2_pVals, _kmin2_pDepths); CHKERRQ(ierr);
    ierr = setVec(_sigma_p, _z, _sigma_pVals, _sigma_pDepths); CHKERRQ(ierr);
  }

  // boundary conditions
  VecDuplicate(_p, &_bcL);
  PetscObjectSetName((PetscObject)_bcL, "bcL");
  VecSet(_bcL, 0.0);

  VecCreate(PETSC_COMM_WORLD, &_bcT);
  VecSetSizes(_bcT, PETSC_DECIDE, 1);
  VecSetFromOptions(_bcT);
  PetscObjectSetName((PetscObject)_bcT, "bcT");

  VecDuplicate(_bcT, &_bcB);
  PetscObjectSetName((PetscObject)_bcB, "bcB");
  VecSet(_bcB, 0);

  VecDuplicate(_bcT, &_bcB_gravity);
  PetscObjectSetName((PetscObject)_bcB_gravity, "bcB_gravity");
  VecSet(_bcB_gravity, 0);

  VecDuplicate(_bcT, &_bcB);
  PetscObjectSetName((PetscObject)_bcB, "bcB");
  VecSet(_bcB, 0);

  VecDuplicate(_bcT, &_bcB_gravity);
  PetscObjectSetName((PetscObject)_bcB_gravity, "bcB_gravity");
  VecSet(_bcB_gravity, 0);

  VecDuplicate(_bcT, &_bcB_impose);
  PetscObjectSetName((PetscObject)_bcB_impose, "bcB_impose");
  VecSet(_bcB_impose, 0);

  // scatter for boundary
  PetscInt *fi;
  PetscMalloc1(1, &fi);
  fi[0] = _N - 1;
  IS isf;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, fi, PETSC_COPY_VALUES, &isf);
  PetscInt *ti;
  PetscMalloc1(1, &ti);
  ti[0] = 0;

  IS ist;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, ti, PETSC_COPY_VALUES, &ist);
  ierr = VecScatterCreate(_p, isf, _bcB, ist, &_scatters); CHKERRQ(ierr);

  // free memory
  PetscFree(fi);
  PetscFree(ti);
  ISDestroy(&isf);
  ISDestroy(&ist);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// load from previous simulations
PetscErrorCode PressureEq::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  string funcName = "PressureEq::loadFieldsFromFiles()";
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  CHKERRQ(ierr);
#endif

  //~ if (_D->_ckptNumber > 0) {
    //~ ierr = loadVecFromInputFile(_p, _outputDir, "p_ckpt");
    //~ ierr = loadVecFromInputFile(_p_t, _outputDir, "p_t_ckpt");
    //~ ierr = loadVecFromInputFile(_k_p, _outputDir, "k_ckpt");
    //~ ierr = loadVecFromInputFile(_k_slip, _outputDir, "k_slip_ckpt");
    //~ ierr = loadVecFromInputFile(_k_press, _outputDir, "k_press_ckpt");

    //~ // other material properties files to load (not checkpoint files)
    //~ ierr = loadVecFromInputFile(_beta_p, _outputDir, "p_n"); CHKERRQ(ierr);
    //~ ierr = loadVecFromInputFile(_beta_p, _outputDir, "p_beta"); CHKERRQ(ierr);
    //~ ierr = loadVecFromInputFile(_eta_p, _outputDir, "p_eta"); CHKERRQ(ierr);
    //~ ierr = loadVecFromInputFile(_rho_f, _outputDir, "p_rho_f"); CHKERRQ(ierr);
  //~ }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  CHKERRQ(ierr);
#endif

  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode PressureEq::checkInput()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::checkInput";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  assert(_pVals.size()       == _pDepths.size());
  assert(_n_pVals.size()     == _n_pDepths.size());
  assert(_beta_pVals.size()  == _beta_pDepths.size());
  assert(_k_pVals.size()     == _k_pDepths.size());
  assert(_eta_pVals.size()   == _eta_pDepths.size());
  assert(_rho_fVals.size()   == _rho_fDepths.size());
  assert(_kL_pVals.size()    == _kL_pDepths.size());
  assert(_kT_pVals.size()    == _kT_pDepths.size());
  assert(_kmin_pVals.size()  == _kmin_pDepths.size());
  assert(_kmax_pVals.size()  == _kmax_pDepths.size());
  assert(_kmin2_pVals.size() == _kmin2_pDepths.size());
  assert(_sigma_pVals.size() == _sigma_pDepths.size());
  assert(_sigmaNVals.size()  == _sigmaNDepths.size());

  assert(_pVals.size()       != 0);
  assert(_n_pVals.size()     != 0);
  assert(_beta_pVals.size()  != 0);
  assert(_k_pVals.size()     != 0);
  assert(_eta_pVals.size()   != 0);
  assert(_rho_fVals.size()   != 0);
  assert(_kL_pVals.size()    != 0);
  assert(_kT_pVals.size()    != 0);
  assert(_kmin_pVals.size()  != 0);
  assert(_kmax_pVals.size()  != 0);
  assert(_kmin2_pVals.size() != 0);
  assert(_sigma_pVals.size() != 0);
  assert(_sigmaNVals.size()  != 0);
  assert(_g >= 0);

  assert(_hydraulicTimeIntType.compare("explicit") == 0 || _hydraulicTimeIntType.compare("implicit") == 0);
  assert(_permSlipDependent.compare("no") == 0 || _permSlipDependent.compare("yes") == 0 );
  assert(_permPressureDependent.compare("no") == 0 || _permPressureDependent.compare("yes") == 0 );
  assert(_bcB_type.compare("Q") == 0 || _bcB_type.compare("Dp") == 0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// compute the variable coefficient vector
PetscErrorCode PressureEq::computeVariableCoefficient(Vec &coeff)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::computeVariableCoefficient";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(_p, &coeff);

  // coeff = rho_f * k_p / eta_p
  VecSet(coeff, 1.0); //g
  VecPointwiseMult(coeff, coeff, _rho_f);
  VecPointwiseMult(coeff, coeff, _k_p);     //rho^2*g * k
  VecPointwiseDivide(coeff, coeff, _eta_p); //rhog = rho^2*g * k/eta

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// update coefficient vector's boundary terms
PetscErrorCode PressureEq::updateBoundaryCoefficient(const Vec &coeff)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::be";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  Vec coeff_rho_g;
  VecDuplicate(coeff, &coeff_rho_g);
  VecCopy(coeff, coeff_rho_g);
  VecPointwiseMult(coeff_rho_g, coeff_rho_g, _rho_f);

  Vec tmp;
  VecDuplicate(coeff, &tmp);

  // add gradient instead of flux
  if ( _bcB_type.compare("Dp") == 0 ) {
    VecSet(tmp, _g * (1.0 + _bcB_ratio)); //g
  }
  else if ( _bcB_type.compare("Q") == 0 ) {
    VecSet(tmp, _g * 1.0); //g
  }

  VecPointwiseMult(coeff_rho_g, coeff_rho_g, tmp);
  VecScatterBegin(_scatters, coeff_rho_g, _bcB, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters, coeff_rho_g, _bcB, INSERT_VALUES, SCATTER_FORWARD);

  if ( _bcB_type.compare("Q") == 0 ) {
    VecAXPY(_bcB, 1.0, _bcB_impose);
  }

  // free memory
  VecDestroy(&coeff_rho_g);
  VecDestroy(&tmp);

  _ptTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// set up the SBP matrix
PetscErrorCode PressureEq::setUpSBP()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::setUpSBP";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up variable coefficient
  Vec coeff;
  computeVariableCoefficient(coeff);

  // Set up linear system
  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    _sbp = new SbpOps_m_constGrid(_order, 1, _N, 1, _L, coeff);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    _sbp = new SbpOps_m_varGrid(_order, 1, _N, 1, _L, coeff);
    _sbp->setGrid(NULL, &_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }

  // set boundary conditions for SBP operator
  _sbp->setBCTypes("Dirichlet", "Dirichlet", "Dirichlet", "Neumann"); //bcR, bcT, bcL, bcB
  VecSet(_bcL, 0);
  VecSet(_bcT, 0);
  VecSet(_bcB, 0);
  VecSet(_bcB_gravity, _g * _rho_fVals.back() * _rho_fVals.back() * _k_pVals.back() / _eta_pVals.back());
  VecSet(_bcB_impose, _g * _rho_fVals.back() * _rho_fVals.back() * _k_pVals.back() / _eta_pVals.back() * _bcB_ratio);
  VecAXPY(_bcB, 1.0, _bcB_gravity);
  VecAXPY(_bcB, 1.0, _bcB_impose);
  // VecSet(_bcB, _g * _rho_fVals.back() * _k_pVals.back() / _eta_pVals.back() * (1 + _bcB_ratio));

  if (_hydraulicTimeIntType.compare("explicit") == 0) {
    _sbp->setMultiplyByH(0);
  }
  else {
    _sbp->setMultiplyByH(1);
  }

  _sbp->setCompatibilityType(_D->_sbpCompatibilityType);
  _sbp->setLaplaceType("z");
  _sbp->computeMatrices(); // actually create the matrices

  VecDestroy(&coeff);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// set up KSP, matrices, boundary conditions for the hydralic problem
PetscErrorCode PressureEq::setUpBe(Domain &D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::setUpBe";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  Mat D2;
  _sbp->getA(D2);
  Mat H;
  _sbp->getH(H);

  // set up boundary terms
  Vec rhs;
  VecDuplicate(_p, &rhs);
  _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);

  // rho_n_beta = 1/(rho * n * beta)
  Vec rho_n_beta;
  VecDuplicate(_p, &rho_n_beta);
  VecSet(rho_n_beta, 1);
  VecPointwiseDivide(rho_n_beta, rho_n_beta, _rho_f);
  VecPointwiseDivide(rho_n_beta, rho_n_beta, _n_p);
  VecPointwiseDivide(rho_n_beta, rho_n_beta, _beta_p);

  Mat Diag_rho_n_beta;
  MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &Diag_rho_n_beta);
  MatDiagonalSet(Diag_rho_n_beta, rho_n_beta, INSERT_VALUES);

  Mat D2_rho_n_beta;
  MatMatMult(Diag_rho_n_beta, D2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D2_rho_n_beta); // 1/(rho * n * beta) D2

  MatScale(D2_rho_n_beta, -_initDeltaT);

  // MatShift(D2_rho_n_beta, 1); // I - dt/(rho*n*beta)*D2
  MatAXPY(D2_rho_n_beta, 1, H, SUBSET_NONZERO_PATTERN); // H - dt/(rho*n*beta)*D2

  setupKSP(D2_rho_n_beta);

  // free memory
  MatDestroy(&Diag_rho_n_beta);
  MatDestroy(&D2_rho_n_beta);
  VecDestroy(&rho_n_beta);
  VecDestroy(&rhs);
  // TODO: check D2 and H are destroyed elsewhere

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// set up linear solver context
PetscErrorCode PressureEq::setupKSP(const Mat &A)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEquation::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // set up Krylov Subspace solver KSP
  PC pc;
  KSPCreate(PETSC_COMM_WORLD, &_ksp);
  ierr = KSPSetType(_ksp, KSPRICHARDSON); CHKERRQ(ierr);
  ierr = KSPSetOperators(_ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(_ksp, PETSC_FALSE); CHKERRQ(ierr);

  // algebraic multigrid
  // set up preconditioner, using the boomerAMG PC from Hypre
  ierr = KSPGetPC(_ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCHYPRE); CHKERRQ(ierr);
  //~ ierr = PCHYPRESetType(pc, "boomeramg"); CHKERRQ(ierr); //!!! THIS IS NEEDED
  ierr = KSPSetTolerances(_ksp, _kspTol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc, 4); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE); CHKERRQ(ierr);

  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(_ksp); CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  double startTime = MPI_Wtime();
  ierr = KSPSetUp(_ksp); CHKERRQ(ierr);
  _ptTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// compute initial steady state pressure
PetscErrorCode PressureEq::computeInitialSteadyStatePressure(Domain &D)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::computeInitialSteadyStatePressure";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // permeability for plate loading velocity
  // k = (V/L*kmax + 1.0/T*kmin)/(V/L + 1.0/T)

  // set up linear solver context
  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD, &ksp);

  Mat D2;
  _sbp->getA(D2);

  ierr = KSPSetType(ksp, KSPRICHARDSON); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, D2, D2); CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(ksp, PETSC_FALSE); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCHYPRE); CHKERRQ(ierr);
  //~ ierr = PCHYPRESetType(pc, "boomeramg"); CHKERRQ(ierr); //!!! THIS IS NEEDED
  ierr = KSPSetTolerances(ksp, _kspTol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc, 4); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);

  // set up boundary conditions
  Vec rhs;
  VecDuplicate(_p, &rhs);
  _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);

  // add source term from gravity
  Vec rhog, rhog_y, temp;
  VecDuplicate(_p, &rhog);
  VecDuplicate(_p, &temp);
  VecSet(rhog, _g);
  VecPointwiseMult(rhog, rhog, _rho_f);
  VecPointwiseMult(rhog, rhog, _rho_f);
  VecPointwiseMult(rhog, rhog, _k_p);
  VecPointwiseDivide(rhog, rhog, _eta_p);
  VecDuplicate(_p, &rhog_y);
  _sbp->Dz(rhog, rhog_y);

  // variable grid spacing -> coordinate transform
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J, Jinv, qy, rz, yq, zr;
    ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr); CHKERRQ(ierr);

    Vec tmp;
    VecDuplicate(_p, &tmp);
    ierr = MatMult(J, rhog_y, tmp);
    VecCopy(tmp, rhog_y);
    VecDestroy(&tmp);
  }

  if (_hydraulicTimeIntType.compare("implicit") == 0) {
    _sbp->H(rhog_y, temp);
    VecAXPY(rhs, 1.0, temp);
  }
  else {
    VecAXPY(rhs, 1.0, rhog_y);
  }

  double startTime = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, _p); CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // free memory
  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&temp);
  VecDestroy(&rhs);
  KSPDestroy(&ksp);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// set up integration, put puressure and permeability terms into varEx, varIm
PetscErrorCode PressureEq::initiateIntegrand(const PetscScalar time, map<string, Vec> &varEx, map<string, Vec> &varIm)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // make shallow copy of pressure
  Vec p;
  VecDuplicate(_p, &p);
  VecCopy(_p, p);

  // put variable to be integrated explicitly into varEx
  // pressure is explicitly integrated
  if (_hydraulicTimeIntType.compare("explicit") == 0) {
    varEx["pressure"] = p;
  }

  // put variable to be integrated implicity into varIm
  // pressure is also implicitly integrated
  else if (_hydraulicTimeIntType.compare("implicit") == 0) {
    varIm["pressure"] = p;
  }

  // permeability is explicitly integrated
  if (_permSlipDependent.compare("yes") == 0 || _permPressureDependent.compare("yes") == 0) {
    Vec k_p;
    VecDuplicate(_p, &k_p);
    VecCopy(_k_p, k_p);
    varEx["permeability"] = k_p;
    // free memory
    VecDestroy(&k_p);
  }

  // free memory
  VecDestroy(&p);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


//  update pressure and permeability for explicit time stepping method
// update _p and _k_p from values that are stored in varEx
PetscErrorCode PressureEq::updateFields(const PetscScalar time, const map<string, Vec> &varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  if (_hydraulicTimeIntType.compare("explicit") == 0 && varEx.find("pressure") != varEx.end()) {
    VecCopy(varEx.find("pressure")->second, _p);
  }

  if (varEx.find("permeability") != varEx.end()) {
    VecCopy(varEx.find("permeability")->second, _k_p);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// update pressure and permeability for implicit time stepping method
// update _p and _k_p from values stored in varEx and varIm
PetscErrorCode PressureEq::updateFields(const PetscScalar time, const map<string, Vec> &varEx, const map<string, Vec> &varIm)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  if (_hydraulicTimeIntType.compare("explicit") == 0 && varEx.find("pressure") != varEx.end()) {
    VecCopy(varEx.find("pressure")->second, _p);
  }
  else if (_hydraulicTimeIntType.compare("implicit") == 0 && varIm.find("pressure") != varIm.end()) {
    VecCopy(varIm.find("pressure")->second, _p);
  }

  if (varEx.find("permeability") != varEx.end()) {
    VecCopy(varEx.find("permeability")->second, _k_p);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// time stepping function, for pressure-dependent permeability
PetscErrorCode PressureEq::updatePermPressureDependent()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::updatePermPressureDependent";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // k = (k0 - kmin2) / exp( (sN-p)/sigma_p ) + kmin2
  Vec tmp1, tmp2;
  VecDuplicate(_k_p, &tmp1);
  VecCopy(_k_slip, tmp1);
  VecDuplicate(_sN, &tmp2);
  VecCopy(_sN, tmp2);

  ierr = VecAXPY(tmp2, -1.0, _p); // sN - p
  ierr = VecPointwiseDivide(tmp2, tmp2, _sigma_p); // (sN-p)/sigma_p
  ierr = VecExp(tmp2); // exp( (sN-p)/sigma_p)
  ierr = VecAXPY(tmp1, -1.0, _kmin2_p);
  ierr = VecPointwiseDivide(tmp1, tmp1, tmp2);
  ierr = VecAXPY(tmp1, 1.0, _kmin2_p);
  ierr = VecCopy(tmp1, _k_p);

  VecCopy(_k_p, _k_press);

  // free memory
  VecDestroy(&tmp1);
  VecDestroy(&tmp2);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// update slip-dependent permeability
PetscErrorCode PressureEq::dk_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::dk_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  Vec vel_abs;
  VecDuplicate(dvarEx.find("slip")->second, &vel_abs);
  VecCopy(dvarEx.find("slip")->second, vel_abs);
  ierr = VecAbs(vel_abs);
  Vec dk = dvarEx["permeability"];

  // dk_dt = - |V|/L * (k - kmax) - 1/T * (k - kmin)
  Vec tmp;
  VecDuplicate(_p, &tmp);
  // - |V|/L * (k - kmax)
  VecWAXPY(tmp, -1.0, _kmax_p, _k_p);
  VecPointwiseMult(tmp, tmp, vel_abs);
  VecPointwiseDivide(tmp, tmp, _kL_p);
  VecAXPY(dk, -1.0, tmp);
  // - 1/T * (k - kmin)
  VecWAXPY(tmp, -1.0, _kmin_p, _k_p);
  VecPointwiseDivide(tmp, tmp, _kT_p);
  VecAXPY(dk, -1.0, tmp);

  VecDestroy(&tmp);
  VecDestroy(&vel_abs);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// update rate of change of slip-dependent permeability
PetscErrorCode PressureEq::dk_dt(const PetscScalar time, const Vec slipVel, const Vec &K, Vec &dKdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::dk_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  Vec vel_abs;
  VecDuplicate(slipVel, &vel_abs);
  VecCopy(slipVel, vel_abs);
  ierr = VecAbs(vel_abs);

  // dk_dt = - |V|/L * (k - kmax) - 1/T * (k - kmin)
  Vec tmp;
  VecDuplicate(_p, &tmp);

  // - |V|/L * (k - kmax)
  VecWAXPY(tmp, -1.0, _kmax_p, _k_p);
  VecPointwiseMult(tmp, tmp, vel_abs);
  VecPointwiseDivide(tmp, tmp, _kL_p);
  VecAXPY(dKdt, -1.0, tmp);
  // - 1/T * (k - kmin)
  VecWAXPY(tmp, -1.0, _kmin_p, _k_p);
  VecPointwiseDivide(tmp, tmp, _kT_p);
  VecAXPY(dKdt, -1.0, tmp);

  VecDestroy(&tmp);
  VecDestroy(&vel_abs);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// purely explicit time integration
PetscErrorCode PressureEq::d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::d_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  if (_permSlipDependent.compare("yes") == 0) {
    ierr = dk_dt(time, varEx, dvarEx);
    CHKERRQ(ierr);
  }

  if (_hydraulicTimeIntType.compare("explicit") == 0) {
    if (_isMMS) {
      ierr = d_dt_mms(time, varEx, dvarEx); CHKERRQ(ierr);
    }
    else {
      ierr = dp_dt(time, varEx, dvarEx); CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// implicit/explicit time integration
PetscErrorCode PressureEq::d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::d_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  if (_permSlipDependent.compare("yes") == 0) {
    ierr = dk_dt(time, varEx, dvarEx); CHKERRQ(ierr);
  }

  if (_hydraulicTimeIntType.compare("explicit") == 0) {
    ierr = dp_dt(time, varEx, dvarEx); CHKERRQ(ierr);
  }
  else {
    if (_isMMS) {
      ierr = be_mms(time, varEx, dvarEx, varIm, varImo, dt); CHKERRQ(ierr);
    }
    else {
      ierr = be(time, varEx, dvarEx, varIm, varImo, dt); CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  return ierr;
}


// purely explicit time integration
PetscErrorCode PressureEq::dp_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::dp_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  if (_permPressureDependent.compare("yes") == 0 || _permSlipDependent.compare("yes") == 0) {
    if (_permPressureDependent.compare("yes") == 0) {
      updatePermPressureDependent();
    }
    // this part is computationally expensive due to update at every single time integration step
    Vec coeff;
    computeVariableCoefficient(coeff);
    _sbp->updateVarCoeff(coeff);
    updateBoundaryCoefficient(coeff);
    VecDestroy(&coeff);
  }

  Vec p_t = dvarEx["pressure"]; // to make this code slightly easier to read

  // source term from gravity: d/dz ( rho*k/eta * g )
  Vec rhog, rhog_y;
  VecDuplicate(_p, &rhog);
  VecSet(rhog, _g);                       //g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho*g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho^2*g
  VecPointwiseMult(rhog, rhog, _k_p);     // rho^2*g*k
  VecPointwiseDivide(rhog, rhog, _eta_p); // rho^2*g *k/eta
  VecDuplicate(_p, &rhog_y);
  _sbp->Dz(rhog, rhog_y); //Dz(rho^2*g*k/eta)

  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2, _p, p_t);
  CHKERRQ(ierr);

  // set up boundary terms
  Vec rhs;
  VecDuplicate(_k_p, &rhs);
  _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);

  // d/dt p = (D2*p - Dz(rho^2*g*k/eta) - rhs) / (rho * n * beta)

  VecAXPY(p_t, -1.0, rhs);

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J, Jinv, qy, rz, yq, zr;
    ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr);
    CHKERRQ(ierr);
    Vec temp;
    VecDuplicate(p_t, &temp);
    ierr = MatMult(Jinv, p_t, temp);
    VecCopy(temp, p_t);
    VecDestroy(&temp);
  }

  VecAXPY(p_t, -1.0, rhog_y);

  VecPointwiseDivide(p_t, p_t, _rho_f);
  VecPointwiseDivide(p_t, p_t, _n_p);
  VecPointwiseDivide(p_t, p_t, _beta_p);

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&rhs);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// this is doing the same thing as the above function, except for dPdt
// IMEX
PetscErrorCode PressureEq::dp_dt(const PetscScalar time, const Vec &P, Vec &dPdt)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::dp_dt";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  if (_permPressureDependent.compare("yes") == 0 || _permSlipDependent.compare("yes") == 0)
  {
    if (_permPressureDependent.compare("yes") == 0) {
      updatePermPressureDependent();
    }
    Vec coeff;
    computeVariableCoefficient(coeff);
    _sbp->updateVarCoeff(coeff);
    updateBoundaryCoefficient(coeff);
    VecDestroy(&coeff);
  }

  // source term from gravity: d/dz ( rho*k/eta * g )
  Vec rhog, rhog_y;
  VecDuplicate(_p, &rhog);
  VecSet(rhog, _g);                       //g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho*g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho^2*g
  VecPointwiseMult(rhog, rhog, _k_p);     // rho^2*g*k
  VecPointwiseDivide(rhog, rhog, _eta_p); // rho^2*g *k/eta
  VecDuplicate(_p, &rhog_y);
  _sbp->Dz(rhog, rhog_y); //Dz(rho^2*g*k/eta)

  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2, _p, dPdt);
  CHKERRQ(ierr);

  // set up boundary terms
  Vec rhs;
  VecDuplicate(_k_p, &rhs);
  _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);
  // VecView(_bcB, PETSC_VIEWER_STDOUT_WORLD);

  // d/dt p = (D2*p - Dz(rho^2*g*k/eta) - rhs) / (rho * n * beta)

  VecAXPY(dPdt, -1.0, rhs);
  // VecAXPY(p_t,-1.0,rhog_y);

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0)
  {
    Mat J, Jinv, qy, rz, yq, zr;
    ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr);
    CHKERRQ(ierr);
    Vec temp;
    VecDuplicate(dPdt, &temp);
    ierr = MatMult(Jinv, dPdt, temp);
    VecCopy(temp, dPdt);
    VecDestroy(&temp);
  }

  VecAXPY(dPdt, -1.0, rhog_y);

  VecPointwiseDivide(dPdt, dPdt, _rho_f);
  VecPointwiseDivide(dPdt, dPdt, _n_p);
  VecPointwiseDivide(dPdt, dPdt, _beta_p);

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&rhs);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// purely explicit time integration for MMS test
PetscErrorCode PressureEq::d_dt_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::d_dt_mms";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  Vec p_t = dvarEx["pressure"]; // to make this code slightly easier to read

  // source term from gravity: d/dz ( rho*k/eta * g )
  Vec rhog, rhog_y;
  VecDuplicate(_p, &rhog);
  VecSet(rhog, _g);                       //g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho*g
  VecPointwiseMult(rhog, rhog, _rho_f);   // rho^2*g
  VecPointwiseMult(rhog, rhog, _k_p);     // rho^2*g*k
  VecPointwiseDivide(rhog, rhog, _eta_p); // rho^2*g *k/eta
  VecDuplicate(_p, &rhog_y);
  _sbp->Dz(rhog, rhog_y); //Dz(rho^2*g*k/eta)

  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2, _p, p_t);
  CHKERRQ(ierr);

  // set up boundary terms
  Vec rhs;
  VecDuplicate(_k_p, &rhs);
  _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);

  // d/dt p = (D2*p - Dz(rho^2*g*k/eta) - rhs) / (rho * n * beta)

  VecAXPY(p_t, -1.0, rhs);
  // VecAXPY(p_t,-1.0,rhog_y);

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0)
  {
    Mat J, Jinv, qy, rz, yq, zr;
    ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr);
    CHKERRQ(ierr);
    Vec temp;
    VecDuplicate(p_t, &temp);
    ierr = MatMult(Jinv, p_t, temp);
    VecCopy(temp, p_t);
    VecDestroy(&temp);
  }

  VecAXPY(p_t, -1.0, rhog_y);

  VecPointwiseDivide(p_t, p_t, _rho_f);
  VecPointwiseDivide(p_t, p_t, _n_p);
  VecPointwiseDivide(p_t, p_t, _beta_p);

  Vec source;
  VecDuplicate(_p, &source);
  mapToVec(source, zzmms_pSource1D, _z, time);

  VecAXPY(p_t, 1.0, source);

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&rhs);
  VecDestroy(&source);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// TODO: check with Weiqiang all the commented out codes, and go through this entire file with him. There is also too much code duplication - this can definitely be condensed a lot
// backward Euler implicit solve
// new result goes in varIm
PetscErrorCode PressureEq::be(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "PressureEq::be";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  if (_permSlipDependent.compare("yes") == 0) {
    Vec coeff;
    computeVariableCoefficient(coeff);
    _sbp->updateVarCoeff(coeff);
    updateBoundaryCoefficient(coeff);
    VecDestroy(&coeff);
    if (_permPressureDependent.compare("yes") == 0) {
      VecCopy(_k_p, _k_slip); // used in permPressureDependent
    }
  }

  VecCopy(varImo.find("pressure")->second, _p);

  Vec rhog, rhog_y;
  VecDuplicate(_p, &rhog);
  VecDuplicate(_p, &rhog_y);

  Vec rhs;
  VecDuplicate(_p, &rhs);

  Vec temp;
  VecDuplicate(_p, &temp);

  Vec p_prev;
  VecDuplicate(_p, &p_prev);
  VecCopy(_p, p_prev);

  Vec rho_n_beta; // rho_n_beta = 1/(rho * n * beta)
  VecDuplicate(_p, &rho_n_beta);

  Mat Diag_rho_n_beta = NULL;
  Mat D2_rho_n_beta = NULL;

  Mat H;
  _sbp->getH(H);
  MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &Diag_rho_n_beta);
  Mat D2;
  _sbp->getA(D2);
  MatMatMult(Diag_rho_n_beta, D2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D2_rho_n_beta);

  Vec Hxp;
  VecDuplicate(_p, &Hxp);

  Vec tmp1;
  VecDuplicate(_p, &tmp1);
  Mat tmp2;
  Mat J, Jinv, qy, rz, yq, zr;
  ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr); CHKERRQ(ierr);
  MatMatMult(Jinv, D2_rho_n_beta, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp2);

  if (_permPressureDependent.compare("no") == 0){
    _maxBeIteration = 1;
  }

  for (int i = 0; i < _maxBeIteration; i++) {
    double tmpTime = MPI_Wtime();
    if (_permPressureDependent.compare("yes") == 0)
    {
      updatePermPressureDependent();
      Vec coeff;
      computeVariableCoefficient(coeff);
      _sbp->updateVarCoeff(coeff);
      updateBoundaryCoefficient(coeff);
      VecDestroy(&coeff);
    }
    _miscTime += MPI_Wtime() - tmpTime;

    // source term from gravity: d/dz ( rho*k/eta * g )
    VecSet(rhog, _g);                       //g
    VecPointwiseMult(rhog, rhog, _rho_f);   //rho*g
    VecPointwiseMult(rhog, rhog, _rho_f);   //rho^2*g
    VecPointwiseMult(rhog, rhog, _k_p);     //rho^2*g * k
    VecPointwiseDivide(rhog, rhog, _eta_p); //rhog = rho^2*g * k/eta
    _sbp->Dz(rhog, rhog_y); //rhog_y = D1(rho^2*g * k/eta)

    Mat D2;
    _sbp->getA(D2);

    Mat H;
    _sbp->getH(H);

    // set up boundary terms
    _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);
    ierr = VecScale(rhs, -1.0); CHKERRQ(ierr);

    // solve Mx = rhs
    VecSet(rho_n_beta, 1);
    VecPointwiseDivide(rho_n_beta, rho_n_beta, _rho_f);
    VecPointwiseDivide(rho_n_beta, rho_n_beta, _n_p);
    VecPointwiseDivide(rho_n_beta, rho_n_beta, _beta_p);
    MatDiagonalSet(Diag_rho_n_beta, rho_n_beta, INSERT_VALUES);

    MatMatMult(Diag_rho_n_beta, D2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &D2_rho_n_beta); // 1/(rho * n * beta) D2

    if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
      Mat J, Jinv, qy, rz, yq, zr;
      ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr); CHKERRQ(ierr);

      ierr = MatMult(Jinv, rhs, tmp1);
      VecCopy(tmp1, rhs);

      MatMatMult(Jinv, D2_rho_n_beta, MAT_REUSE_MATRIX, PETSC_DEFAULT, &tmp2);
      MatCopy(tmp2, D2_rho_n_beta, SAME_NONZERO_PATTERN);
    }

    _sbp->H(rhog_y, temp);
    VecAXPY(rhs, -1.0, temp); // - D1(rho^2*g * k/eta) + SAT

    MatScale(D2_rho_n_beta, -dt);

    MatAXPY(D2_rho_n_beta, 1, H, SUBSET_NONZERO_PATTERN); // H - dt/(rho*n*beta)*D2

    VecPointwiseMult(rhs, rhs, rho_n_beta); //1/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT)

    VecScale(rhs, dt); // dt/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) + dt * src

    ierr = _sbp->H(varImo.find("pressure")->second, Hxp); // H * p(t) + dt/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) +  dt * H * src

    VecAXPY(rhs, 1, Hxp);

    tmpTime = MPI_Wtime();
    ierr = KSPSetOperators(_ksp, D2_rho_n_beta, D2_rho_n_beta); CHKERRQ(ierr);
    ierr = KSPSolve(_ksp, rhs, _p); CHKERRQ(ierr);

    // calculate relative error
    PetscReal err=0.0, s=0.0;
    Vec errVec;
    VecDuplicate(_p, &errVec);
    VecSet(errVec,0.0);
    ierr = VecWAXPY(errVec, -1.0, p_prev, _p); CHKERRQ(ierr);
    VecNorm(errVec, NORM_2, &err);
    VecDestroy(&errVec);
    VecNorm(_p, NORM_2, &s);
    err = err / s;

    VecCopy(_p, p_prev);

    _invTime += MPI_Wtime() - tmpTime;
  }

  VecCopy(_p, varIm["pressure"]);
  if (_permPressureDependent.compare("yes") == 0 && _permSlipDependent.compare("yes") == 0) {
    VecCopy(_k_slip, _k_p); // combine slip dependent and pressure dependent
  }

  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // free memory
  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&temp);
  VecDestroy(&rhs);
  VecDestroy(&rho_n_beta);
  VecDestroy(&Hxp);
  VecDestroy(&p_prev);
  MatDestroy(&Diag_rho_n_beta);
  MatDestroy(&D2_rho_n_beta);
  VecDestroy(&tmp1);
  MatDestroy(&tmp2);

  _ptTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


// TODO: check why is everything commented out here
// backward Euler implicit solve for MMS test
// new result goes in varIm
PetscErrorCode PressureEq::be_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::be_mms";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // double startTime = MPI_Wtime(); // time this section

  // if (_permSlipDependent.compare("yes") == 0) {
  //   Vec coeff;
  //   computeVariableCoefficient(coeff);
  //   _sbp->updateVarCoeff(coeff);
  //   updateBoundaryCoefficient(coeff);
  //   VecDestroy(&coeff);
  // }

  // Vec rho_k_eta_g;
  // VecDuplicate(_p, &rho_k_eta_g);
  // VecSet(rho_k_eta_g, _g * _bcB_ratio); //g
  // VecPointwiseMult(rho_k_eta_g, rho_k_eta_g, _rho_f);
  // VecPointwiseMult(rho_k_eta_g, rho_k_eta_g, _rho_f);   //rho^2*g
  // VecPointwiseMult(rho_k_eta_g, rho_k_eta_g, _k_p);     //rho^2*g * k
  // VecPointwiseDivide(rho_k_eta_g, rho_k_eta_g, _eta_p); //rhog = rho^2*g * k/eta

  // { // set up scatter context to take values for y=Ly from body field and put them on a Vec of size Nz
  //   // indices to scatter from
  //   VecScatter _scatters;
  //   PetscInt *fi;
  //   PetscMalloc1(1, &fi);
  //   // for (PetscInt Ii=0; Ii<1; Ii++) { fi[Ii] = Ii + _N-1; }
  //   fi[0] = _N - 1;
  //   IS isf;
  //   ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, fi, PETSC_COPY_VALUES, &isf);

  //   // indices to scatter to
  //   PetscInt *ti;
  //   PetscMalloc1(1, &ti);
  //   // for (PetscInt Ii=0; Ii<1; Ii++) { ti[Ii] = Ii; };
  //   ti[0] = 0;
  //   IS ist;
  //   ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, ti, PETSC_COPY_VALUES, &ist);

  //   ierr = VecScatterCreate(rho_k_eta_g, isf, _bcB, ist, &_scatters);
  //   CHKERRQ(ierr);
  //   PetscFree(fi);
  //   PetscFree(ti);
  //   ISDestroy(&isf);
  //   ISDestroy(&ist);
  //   VecScatterBegin(_scatters, rho_k_eta_g, _bcB, INSERT_VALUES, SCATTER_FORWARD);
  //   VecScatterEnd(_scatters, rho_k_eta_g, _bcB, INSERT_VALUES, SCATTER_FORWARD);
  //   VecScatterDestroy(&_scatters);
  // }

  // VecDestroy(&rho_k_eta_g);

  // // VecShift(_bcB, _g*_rho_fVals.back()*_k_pVals.back()/_eta_pVals.back()*0.5);

  // // Vec _p = varImo.find("pressure")->second;
  // VecCopy(varImo.find("pressure")->second, _p);

  // Vec rhog, rhog_y;
  // VecDuplicate(_p, &rhog);

  // Vec rhs;
  // VecDuplicate(_k_p, &rhs);

  // Vec temp;
  // VecDuplicate(_p, &temp);

  // Vec rho_n_beta; // rho_n_beta = 1/(rho * n * beta)
  // VecDuplicate(_p, &rho_n_beta);

  // Mat Diag_rho_n_beta;
  // Mat D2_rho_n_beta;

  // Vec source, Hxsource;
  // VecDuplicate(_p, &source);
  // VecDuplicate(_p, &Hxsource);

  // Vec Hxp;
  // VecDuplicate(_p, &Hxp);

  // // assert(0);
  // if (_permPressureDependent.compare("no") == 0)
  // {
  //   _maxBeIteration = 1;
  // }

  // for (int i = 0; i < _maxBeIteration; i++)
  // {

  //   double tmpTime = MPI_Wtime();
  //   if (_permPressureDependent.compare("no") != 0) {
  //     delete _sbp;
  //     updatePermPressureDependent();
  //     setUpSBP();
  //   }
  //   _miscTime += MPI_Wtime() - tmpTime;

  //   // source term from gravity: d/dz ( rho*k/eta * g )

  //   VecSet(rhog, _g);                       //g
  //   VecPointwiseMult(rhog, rhog, _rho_f);   //rho*g
  //   VecPointwiseMult(rhog, rhog, _rho_f);   //rho^2*g
  //   VecPointwiseMult(rhog, rhog, _k_p);     //rho^2*g * k
  //   VecPointwiseDivide(rhog, rhog, _eta_p); //rhog = rho^2*g * k/eta
  //   VecDuplicate(_p, &rhog_y);
  //   _sbp->Dz(rhog, rhog_y); //rhog_y = D1(rho^2*g * k/eta)

  //   Mat D2;
  //   _sbp->getA(D2);

  //   Mat H;
  //   _sbp->getH(H);

  //   // set up boundary terms
  //   _sbp->setRhs(rhs, _bcL, _bcL, _bcT, _bcB);
  //   ierr = VecScale(rhs, -1.0);
  //   CHKERRQ(ierr);

  //   // solve Mx = rhs
  //   // M = I - dt/(rho*n*beta)*D2
  //   // rhs = p + dt/(rho*n*beta) *( -D1(k/eta*rho^2*g) + SAT + source )

  //   // _sbp->H(rhog_y,temp);

  //   // VecAXPY(rhs, -1.0, rhog_y); // - D1(rho^2*g * k/eta) + SAT
  //   // VecAXPY(rhs, -1.0, temp); // - D1(rho^2*g * k/eta) + SAT

  //   // compute MMS source

  //   // PetscPrintf(PETSC_COMM_WORLD,"Time: %f", time);

  //   mapToVec(source, zzmms_pSource1D, _z, time);

  //   ierr = _sbp->H(source, Hxsource);

  //   //~ writeVec(source,_outputDir + "mms_pSource");

  //   // d/dt p = (D2*p - rhs + source) / (rho * n * beta)
  //   // VecAXPY(p_t,1.0,source);

  //   VecSet(rho_n_beta, 1);
  //   VecPointwiseDivide(rho_n_beta, rho_n_beta, _rho_f);
  //   VecPointwiseDivide(rho_n_beta, rho_n_beta, _n_p);
  //   VecPointwiseDivide(rho_n_beta, rho_n_beta, _beta_p);

  //   MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &Diag_rho_n_beta);
  //   MatDiagonalSet(Diag_rho_n_beta, rho_n_beta, INSERT_VALUES);

  //   // MatDuplicate(D2, MAT_DO_NOT_COPY_VALUES, &D2_rho_n_beta);
  //   MatMatMult(Diag_rho_n_beta, D2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D2_rho_n_beta); // 1/(rho * n * beta) D2

  //   if (_D->_gridSpacingType.compare("variableGridSpacing")==0)
  //   {

  //     Mat J, Jinv, qy, rz, yq, zr;
  //     ierr = _sbp->getCoordTrans(J, Jinv, qy, rz, yq, zr);
  //     CHKERRQ(ierr);

  //     Vec tmp1;
  //     VecDuplicate(_p, &tmp1);
  //     ierr = MatMult(Jinv, rhs, tmp1);
  //     VecCopy(tmp1, rhs);
  //     VecDestroy(&tmp1);

  //     Mat tmp2;
  //     // MatDuplicate(D2, MAT_DO_NOT_COPY_VALUES, &tmp2);
  //     MatMatMult(Jinv, D2_rho_n_beta, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp2);
  //     MatCopy(tmp2, D2_rho_n_beta, SAME_NONZERO_PATTERN);
  //     MatDestroy(&tmp2);
  //   }

  //   _sbp->H(rhog_y, temp);
  //   VecAXPY(rhs, -1.0, temp); // - D1(rho^2*g * k/eta) + SAT
  //   // VecAXPY(rhs, 1.0, rhog_y); // - D1(rho^2*g * k/eta) + SAT

  //   MatScale(D2_rho_n_beta, -dt);

  //   // MatShift(D2_rho_n_beta, 1); // I - dt/(rho*n*beta)*D2
  //   MatAXPY(D2_rho_n_beta, 1, H, SUBSET_NONZERO_PATTERN); // H - dt/(rho*n*beta)*D2

  //   VecPointwiseMult(rhs, rhs, rho_n_beta); //1/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT)

  //   // VecAXPY(rhs, 1.0, source); // 1/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) + src

  //   // correct
  //   VecAXPY(rhs, 1.0, Hxsource); // 1/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) + H * src

  //   // VecView(source, PETSC_VIEWER_STDOUT_WORLD);
  //   VecScale(rhs, dt); // dt/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) + dt * src

  //   ierr = _sbp->H(varImo.find("pressure")->second, Hxp); // H * p(t) + dt/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) +  dt * H * src

  //   // VecAXPY(rhs, 1, varImo.find("pressure")->second); // p(t) + dt/(rho * n * beta) * ( - D1(rho^2*g * k/eta) + SAT ) + dt * src
  //   VecAXPY(rhs, 1, Hxp);

  //   tmpTime = MPI_Wtime();
  //   ierr = KSPSetOperators(_ksp, D2_rho_n_beta, D2_rho_n_beta);
  //   CHKERRQ(ierr);
  //   // KSPSetUp(_ksp);
  //   // ierr = KSPSolve(solver, rhs, varIm["pressure"]);CHKERRQ(ierr);
  //   // ierr = KSPSolve(_ksp, rhs, varIm["pressure"]);CHKERRQ(ierr);
  //   ierr = KSPSolve(_ksp, rhs, _p);
  //   CHKERRQ(ierr);
  //   _invTime += MPI_Wtime() - tmpTime;
  // }

  // // VecView(_p, PETSC_VIEWER_STDOUT_WORLD);

  // // varIm["pressure"] = _p;
  // VecCopy(_p, varIm["pressure"]);

  // _linSolveTime += MPI_Wtime() - startTime;
  // _linSolveCount++;

  // //~ mapToVec(dvarEx["pressure"], zzmms_pt1D, _z, time);
  // VecDestroy(&rhog);
  // VecDestroy(&rhog_y);
  // VecDestroy(&temp);
  // VecDestroy(&rhs);
  // VecDestroy(&rho_n_beta);
  // VecDestroy(&source);
  // VecDestroy(&Hxsource);
  // VecDestroy(&Hxp);

  // MatDestroy(&Diag_rho_n_beta);
  // MatDestroy(&D2_rho_n_beta);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}

// =====================================================================
// IO commands

PetscErrorCode PressureEq::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "-------------------------------\n\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "PressureEq Runtime Summary:\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "   pressure rate time (s): %g\n", _ptTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "   %% integration time spent computing pressure rate: %g\n", _ptTime / totRunTime * 100.); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "   delete and create SBP (s): %g\n", _miscTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "   inversion (s): %g\n", _invTime); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode PressureEq::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::writeContext";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  CHKERRQ(ierr);

  PetscViewer viewer_ascii;

  // write out scalar info
  string str = outputDir + "p_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer_ascii);
  PetscViewerSetType(viewer_ascii, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer_ascii, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer_ascii, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer_ascii, "g = %.15e\n", _g); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer_ascii, "hydraulicTimeIntType = %s\n", _hydraulicTimeIntType.c_str()); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_ascii); CHKERRQ(ierr);

  // write material parameters
  // these are also written out with writeStep, so do they actually count as context variables??
  //~ ierr = PetscViewerHDF5PushGroup(viewer, "/pressureEq");               CHKERRQ(ierr);
  //~ ierr = VecView(_n_p, viewer);                                         CHKERRQ(ierr);
  //~ ierr = VecView(_beta_p, viewer);                                      CHKERRQ(ierr);
  //~ ierr = VecView(_k_slip, viewer);                                      CHKERRQ(ierr);
  //~ ierr = VecView(_k_press, viewer);                                     CHKERRQ(ierr);
  //~ ierr = VecView(_eta_p, viewer);                                       CHKERRQ(ierr);
  //~ ierr = VecView(_rho_f, viewer);                                       CHKERRQ(ierr);
  //~ ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PressureEq::writeStep(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::writeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/pressureEq");               CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);

  ierr = VecView(_n_p, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_beta_p, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_press, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_eta_p, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_rho_f, viewer);                                       CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

/*
  if (_isMMS) {
    Vec pA;
    VecDuplicate(_p, &pA);
    mapToVec(pA, zzmms_pA1D, _z, time);
    if (stepCount == 0) {
      ierr = io_initiateWriteAppend(_viewers, "pA", pA, outputDir + "p_pA"); CHKERRQ(ierr);
    }
    else {
      ierr = VecView(pA, _viewers["pA"].first); CHKERRQ(ierr);
    }
    VecDestroy(&pA);
  }


  if (_viewers.empty()) {
    ierr = initiate_appendVecToOutput(_viewers, "p", _p, outputDir + "p",_D->_outFileMode); CHKERRQ(ierr);
    ierr = initiate_appendVecToOutput(_viewers, "p_t", _p_t, outputDir + "p_t",_D->_outFileMode); CHKERRQ(ierr);
    ierr = initiate_appendVecToOutput(_viewers, "k", _k_p, outputDir + "k",_D->_outFileMode); CHKERRQ(ierr);
    ierr = initiate_appendVecToOutput(_viewers, "k_slip", _k_slip, outputDir + "k_slip",_D->_outFileMode); CHKERRQ(ierr);
    ierr = initiate_appendVecToOutput(_viewers, "k_press", _k_press, outputDir + "k_press",_D->_outFileMode); CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_p, _viewers["p"].first); CHKERRQ(ierr);
    ierr = VecView(_p_t, _viewers["p_t"].first); CHKERRQ(ierr);
    ierr = VecView(_k_p, _viewers["k"].first); CHKERRQ(ierr);
    ierr = VecView(_k_slip, _viewers["k_slip"].first); CHKERRQ(ierr);
    ierr = VecView(_k_press, _viewers["k_press"].first); CHKERRQ(ierr);
  }*/


    // write checkpoint files
    //~ if (stepCount == _maxStepCount && _D->_ckpt > 0) {
      //~ ierr = writeVec(_p, outputDir + "p_ckpt"); CHKERRQ(ierr);
      //~ ierr = writeVec(_p_t, outputDir + "p_t_ckpt"); CHKERRQ(ierr);
      //~ ierr = writeVec(_k_p, outputDir + "k_ckpt"); CHKERRQ(ierr);
      //~ ierr = writeVec(_k_slip, outputDir + "k_slip_ckpt"); CHKERRQ(ierr);
      //~ ierr = writeVec(_k_press, outputDir + "k_press_ckpt"); CHKERRQ(ierr);
    //~ }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::writeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/pressureEq");               CHKERRQ(ierr);

  ierr = VecView(_n_p, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_beta_p, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_press, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_eta_p, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_rho_f, viewer);                                       CHKERRQ(ierr);

  ierr = VecView(_n_p, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_beta_p, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_k_press, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_eta_p, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_rho_f, viewer);                                       CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PressureEq::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "pressureEq");               CHKERRQ(ierr);

  ierr = VecLoad(_n_p, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_beta_p, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_k_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_k_press, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_eta_p, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_rho_f, viewer);                                       CHKERRQ(ierr);

  ierr = VecLoad(_n_p, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_beta_p, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_k_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_k_press, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_eta_p, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_rho_f, viewer);                                       CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ending %s in %s\n", funcName.c_str(), FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// MMS functions
// test convergence to analytical solution
PetscErrorCode PressureEq::measureMMSError(const double totRunTime)
{
  Vec pA;
  VecDuplicate(_p, &pA);
  mapToVec(pA, zzmms_pA1D, _z, totRunTime);
  double err2pA = computeNormDiff_2(_p, pA);
  PetscPrintf(PETSC_COMM_WORLD, "%i  %3i %.4e %.4e % .15e\n", _order, _N, _h, err2pA, log2(err2pA));
  VecDestroy(&pA);
  return 0;
};


double PressureEq::zzmms_pt1D(const double z, const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;
  PetscScalar delta_p = 50;
  PetscScalar omega = 2 * PI / T0;
  PetscScalar kz = 2.5 * PI / 30.;
  PetscScalar p_t = delta_p * sin(kz * z) * omega * cos(omega * t); // correct
  return p_t;
}


double PressureEq::zzmms_pA1D(const double z, const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;
  PetscScalar delta_p = 50;
  PetscScalar omega = 2.0 * PI / T0;
  PetscScalar kz = 2.5 * PI / 30.;
  PetscScalar p_src = delta_p * sin(kz * z) * sin(omega * t); // correct
  return p_src;
}

double PressureEq::zzmms_pSource1D(const double z, const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;
  PetscScalar delta_p = 50;
  PetscScalar omega = 2. * PI / T0;
  PetscScalar kz = 2.5 * PI / 30.;
  PetscScalar beta0 = 10;
  PetscScalar eta0 = 1e-12;
  PetscScalar n0 = 0.1;
  PetscScalar k0 = 1e-19;
  PetscScalar p_src = delta_p * (beta0 * eta0 * n0 * omega * cos(omega * t) + k0 * kz * kz * sin(omega * t)) * sin(kz * z) / (beta0 * eta0 * n0); // correct
  return p_src;
}
