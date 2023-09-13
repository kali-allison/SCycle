#include "fault.hpp"

#define FILENAME "fault.cpp"

using namespace std;


Fault::Fault(Domain &D, VecScatter& scatter2fault, const int& faultTypeScale)
  : _D(&D),_inputFile(D._file),_delim(D._delim),
    _inputDir(D._inputDir),_outputDir(D._outputDir),
    _stateLaw("agingLaw"),_faultTypeScale(faultTypeScale),_limitSlipVel(0),
    _N(D._Nz),_L(D._Lz),
    _prestressScalar(0.),
    _f0(0.6),_v0(1e-6),
    _sigmaN_cap(1e14),_sigmaN_floor(0.),
    _fw(0.64),_tau_c(3),_D_fh(5),
    _rootTol(1e-12),_rootIts(0),_maxNumIts(1e4),
    _viewer_hdf5(NULL),
    _computeVelTime(0),_stateLawTime(0), _scatterTime(0),
    _body2fault(&scatter2fault)
{
  #if VERBOSE > 1
    std::string funcName = "Fault::Fault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_inputFile);
  checkInput();
  setFields(D);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// read input file and load fields
PetscErrorCode Fault::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in fault.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
  #endif

  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ifstream infile( file );
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  while (getline(infile, line)) {
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


    if (var.compare("DcVals")==0) { loadVectorFromInputFile(rhsFull,_DcVals); }
    else if (var.compare("DcDepths")==0) { loadVectorFromInputFile(rhsFull,_DcDepths); }
    else if (var.compare("sNVals")==0) { loadVectorFromInputFile(rhsFull,_sigmaNVals); }
    else if (var.compare("sNDepths")==0) { loadVectorFromInputFile(rhsFull,_sigmaNDepths); }
    else if (var.compare("sN_cap")==0) { _sigmaN_cap = atof( rhs.c_str() ); }
    else if (var.compare("sN_floor")==0) { _sigmaN_floor = atof( rhs.c_str() ); }
    else if (var.compare("aVals")==0) { loadVectorFromInputFile(rhsFull,_aVals); }
    else if (var.compare("aDepths")==0) { loadVectorFromInputFile(rhsFull,_aDepths); }
    else if (var.compare("bVals")==0) { loadVectorFromInputFile(rhsFull,_bVals); }
    else if (var.compare("bDepths")==0) { loadVectorFromInputFile(rhsFull,_bDepths); }
    else if (var.compare("cohesionVals")==0) { loadVectorFromInputFile(rhsFull,_cohesionVals); }
    else if (var.compare("cohesionDepths")==0) { loadVectorFromInputFile(rhsFull,_cohesionDepths); }
    else if (var.compare("muVals")==0) { loadVectorFromInputFile(rhsFull,_muVals); }
    else if (var.compare("muDepths")==0) { loadVectorFromInputFile(rhsFull,_muDepths); }
    else if (var.compare("rhoVals")==0) { loadVectorFromInputFile(rhsFull,_rhoVals); }
    else if (var.compare("rhoDepths")==0) { loadVectorFromInputFile(rhsFull,_rhoDepths); }
    else if (var.compare("stateVals")==0) { loadVectorFromInputFile(rhsFull,_stateVals); }
    else if (var.compare("stateDepths")==0) { loadVectorFromInputFile(rhsFull,_stateDepths); }
    else if (var.compare("stateLaw")==0) { _stateLaw = rhs.c_str(); }

    // tolerance for nonlinear solve
    else if (var.compare("rootTol")==0) { _rootTol = atof( rhs.c_str() ); }
    else if (var.compare("prestressScalar")==0) { _prestressScalar = atof( rhs.c_str() ); }

    // friction parameters
    else if (var.compare("f0")==0) { _f0 = atof( rhs.c_str() ); }
    else if (var.compare("v0")==0) { _v0 = atof( rhs.c_str() ); }

    // flash heating parameters
    else if (var.compare("fw")==0) { _fw = atof( rhs.c_str() ); }
    else if (var.compare("VwType")==0) { _VwType = rhs.c_str(); }
    else if (var.compare("VwVals")==0) { loadVectorFromInputFile(rhsFull,_VwVals); }
    else if (var.compare("VwDepths")==0) { loadVectorFromInputFile(rhsFull,_VwDepths); }
    else if (var.compare("TwVals")==0) { loadVectorFromInputFile(rhsFull,_TwVals); }
    else if (var.compare("TwDepths")==0) { loadVectorFromInputFile(rhsFull,_TwDepths); }
    else if (var.compare("D")==0) { _D_fh = atof( rhs.c_str() ); }
    else if (var.compare("tau_c")==0) { _tau_c = atof( rhs.c_str() ); }

    // for locking part of the fault
    else if (var.compare("lockedVals")==0) { loadVectorFromInputFile(rhsFull,_lockedVals); }
    else if (var.compare("lockedDepths")==0) { loadVectorFromInputFile(rhsFull,_lockedDepths); }

    else if (var.compare("limitSlipVel")==0) { _limitSlipVel = atoi( rhs.c_str() ); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in fault.cpp.\n");CHKERRQ(ierr);
  #endif

  return ierr;
}


// load vector fields from directory
PetscErrorCode Fault::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
  #endif

  ierr = loadVecFromInputFile(_sNEff,_D->_inputDir,"sNEff"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_psi,_D->_inputDir,"psi"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_slip,_D->_inputDir,"slip"); CHKERRQ(ierr);

  // load shear stress: pre-stress, quasistatic, and full
  ierr = loadVecFromInputFile(_prestress,_D->_inputDir,"prestress"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_tauQSP,_D->_inputDir,"tauQS"); CHKERRQ(ierr);
  //~ VecAXPY(_tauQSP,1.0,_prestress);

  bool loadedTauP = 0;
  ierr = loadVecFromInputFile(_tauP,_D->_inputDir,"tau",loadedTauP); CHKERRQ(ierr);
  if (!loadedTauP) { VecCopy(_tauQSP,_tauP); }
  VecCopy(_tauP,_strength);

  // rate and state parameters
  ierr = loadVecFromInputFile(_a,_D->_inputDir,"fault_a"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_b,_D->_inputDir,"fault_b"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_Dc,_D->_inputDir,"fault_Dc"); CHKERRQ(ierr);
  if (_stateLaw.compare("flashHeating") == 0) {
    ierr = loadVecFromInputFile(_Vw,_D->_inputDir,"fault_Vw"); CHKERRQ(ierr);
    ierr = loadVecFromInputFile(_Tw,_D->_inputDir,"fault_Tw"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
  #endif

  return ierr;
}

// load a checkpoint
PetscErrorCode Fault::loadCheckpoint()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::loadCheckpoint in fault.cpp.\n");CHKERRQ(ierr);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);


  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                    CHKERRQ(ierr);

  ierr = VecLoad(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw == "flashHeating") {
    ierr = VecLoad(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_c, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_T, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecLoad(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  //~ PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
  #endif

  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode Fault::checkInput()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::checkInput";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_DcVals.size() == _DcDepths.size() );
  assert(_aVals.size() == _aDepths.size() );
  assert(_bVals.size() == _bDepths.size() );
  assert(_sigmaNVals.size() == _sigmaNDepths.size() );
  assert(_cohesionVals.size() == _cohesionDepths.size() );
  assert(_rhoVals.size() == _rhoDepths.size() );
  assert(_muVals.size() == _muDepths.size() );
  assert(_stateVals.size() == _stateDepths.size() );
  assert(_DcVals.size() != 0 );
  assert(_aVals.size() != 0 );
  assert(_bVals.size() != 0 );
  assert(_sigmaNVals.size() != 0 );
  assert(_rhoVals.size() != 0 );
  assert(_muVals.size() != 0 );
  assert(_rootTol >= 1e-14);

  assert(_stateLaw.compare("agingLaw")==0
    || _stateLaw.compare("slipLaw")==0
    || _stateLaw.compare("flashHeating")==0
    || _stateLaw.compare("constantState")==0 );

  assert(_v0 > 0);
  assert(_f0 > 0);

  if (_stateLaw.compare("flashHeating") == 0) {
    assert(_TwVals.size() == _TwDepths.size() );
    assert(_TwVals.size() != 0 );
    assert(_VwType.compare("constant")==0 || _VwType.compare("function_of_Tw")==0 );
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// allocate memory for fields, and set some vectors
PetscErrorCode Fault::setFields(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault::setFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDuplicate(D._y0,&_z);
  double scatterStart = MPI_Wtime();
  VecScatterBegin(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  _scatterTime += MPI_Wtime() - scatterStart;
  PetscObjectSetName((PetscObject) _z, "z");

  VecDuplicate(_z,&_tauP);         VecSet(_tauP,0.0);      PetscObjectSetName((PetscObject) _tauP, "tau");
  VecDuplicate(_tauP,&_tauQSP);    VecSet(_tauQSP,0.0);    PetscObjectSetName((PetscObject) _tauQSP, "tauQS");
  VecDuplicate(_tauP,&_strength);  VecSet(_strength,0.0);  PetscObjectSetName((PetscObject) _strength, "strength");
  VecDuplicate(_tauP,&_prestress); VecSet(_prestress, _prestressScalar); PetscObjectSetName((PetscObject) _prestress, "prestress");

  VecDuplicate(_tauP,&_psi);      VecSet(_psi,0.0);      PetscObjectSetName((PetscObject) _psi, "psi");
  VecDuplicate(_tauP,&_slip);     VecSet(_slip,0.0);     PetscObjectSetName((PetscObject) _slip, "slip");
  VecDuplicate(_tauP,&_slipVel);  VecSet(_slipVel,0.0);  PetscObjectSetName((PetscObject) _slipVel, "slipVel");

  VecDuplicate(_tauP,&_Dc);       VecSet(_Dc,0.0);       PetscObjectSetName((PetscObject) _Dc, "Dc");
  VecDuplicate(_tauP,&_a);        VecSet(_a,0.0);        PetscObjectSetName((PetscObject) _a, "a");
  VecDuplicate(_tauP,&_b);        VecSet(_b,0.0);        PetscObjectSetName((PetscObject) _b, "b");
  VecDuplicate(_tauP,&_cohesion); VecSet(_cohesion,0.0); PetscObjectSetName((PetscObject) _cohesion, "cohesion");
  VecDuplicate(_tauP,&_sN);       VecSet(_sN,0.0);       PetscObjectSetName((PetscObject) _sN, "sN");
  VecDuplicate(_tauP,&_sNEff);    VecSet(_sNEff,0.0);    PetscObjectSetName((PetscObject) _sNEff, "sNEff");
  VecDuplicate(_tauP,&_rho);      VecSet(_rho,0.0);      PetscObjectSetName((PetscObject) _rho, "rho");
  VecDuplicate(_tauP,&_mu);       VecSet(_mu,0.0);       PetscObjectSetName((PetscObject) _mu, "mu");
  VecDuplicate(_tauP,&_locked);   VecSet(_locked,0.0);   PetscObjectSetName((PetscObject) _locked, "locked");
  VecDuplicate(_tauP,&_slip0);    VecSet(_slip0, 0.0);   PetscObjectSetName((PetscObject) _slip0, "slip0");


  if (_stateLaw.compare("flashHeating") == 0) {
    VecDuplicate(_tauP,&_T);      VecSet(_T,0.0);       PetscObjectSetName((PetscObject) _T, "T");
    VecDuplicate(_tauP,&_k);      VecSet(_k,0.0);       PetscObjectSetName((PetscObject) _k, "k");
    VecDuplicate(_tauP,&_c);      VecSet(_c,0.0);       PetscObjectSetName((PetscObject) _c, "c");
    VecDuplicate(_tauP,&_Tw);
    ierr = setVec(_Tw,_z,_TwVals,_TwDepths); CHKERRQ(ierr);
    PetscObjectSetName((PetscObject) _Tw, "Tw");
    VecDuplicate(_tauP,&_Vw);
    ierr = setVec(_Vw,_z,_VwVals,_VwDepths); CHKERRQ(ierr);
    PetscObjectSetName((PetscObject) _Vw, "Vw");
  }
  else { _T = NULL; _k = NULL; _c = NULL; _Tw = NULL; _Vw = NULL; }

  // set fields
  ierr = setVec(_a,_z,_aVals,_aDepths); CHKERRQ(ierr);
  ierr = setVec(_b,_z,_bVals,_bDepths); CHKERRQ(ierr);
  ierr = setVec(_sN,_z,_sigmaNVals,_sigmaNDepths); CHKERRQ(ierr);
  ierr = setVec(_Dc,_z,_DcVals,_DcDepths); CHKERRQ(ierr);
  if (_lockedVals.size() > 0 ) { ierr = setVec(_locked,_z,_lockedVals,_lockedDepths); CHKERRQ(ierr); }
  else { VecSet(_locked,0.); }
  if (_cohesionVals.size() > 0 ) { ierr = setVec(_cohesion,_z,_cohesionVals,_cohesionDepths); CHKERRQ(ierr); }

  scatterStart = MPI_Wtime();
  Vec temp1;
  VecDuplicate(_D->_y,&temp1);
  ierr = setVec(temp1,_D->_z,_rhoVals,_rhoDepths); CHKERRQ(ierr);
  VecScatterBegin(*_body2fault, temp1, _rho, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, temp1, _rho, INSERT_VALUES, SCATTER_FORWARD);

  ierr = setVec(temp1,_D->_z,_muVals,_muDepths); CHKERRQ(ierr);
  VecScatterBegin(*_body2fault, temp1, _mu, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, temp1, _mu, INSERT_VALUES, SCATTER_FORWARD);
  VecDestroy(&temp1);

  _scatterTime += MPI_Wtime() - scatterStart;

  if (_stateVals.size() > 0) { ierr = setVec(_psi,_z,_stateVals,_stateDepths); CHKERRQ(ierr); }
  else { ierr = VecSet(_psi,_f0);CHKERRQ(ierr); }

  { // impose floor and ceiling on effective normal stress
    Vec temp; VecDuplicate(_sN,&temp);
    VecSet(temp,_sigmaN_cap); VecPointwiseMin(_sN,_sN,temp);
    VecSet(temp,_sigmaN_floor); VecPointwiseMax(_sN,_sN,temp);
    VecDestroy(&temp);
  }
  VecCopy(_sN,_sNEff);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set fields needed for flash heating from body fields owned by heat equation
PetscErrorCode Fault::setThermalFields(const Vec& T, const Vec& k, const Vec& c)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::setThermalFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  double scatterStart = MPI_Wtime();

  // scatters values from body fields to the fault
  VecScatterBegin(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, k, _k, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, k, _k, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, c, _c, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, c, _c, INSERT_VALUES, SCATTER_FORWARD);

  _scatterTime += MPI_Wtime() - scatterStart;

   #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// update temperature on the fault based on the temperature body field
PetscErrorCode Fault::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::updateTemperature";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  if (_stateLaw.compare("flashHeating") == 0) {
    double scatterStart = MPI_Wtime();
    VecScatterBegin(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);
    _scatterTime += MPI_Wtime() - scatterStart;
  }

   #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// use pore pressure to compute total normal stress
// sNEff = sN - rho*g*z - dp
// sNEff sigma Normal Effective
PetscErrorCode Fault::setSN(const Vec& p)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::setSN";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  ierr = VecWAXPY(_sN,1.,p,_sNEff); CHKERRQ(ierr);

  // impose floor and ceiling on effective normal stress
  Vec temp;
  VecDuplicate(_sN,&temp);
  VecSet(temp,_sigmaN_cap);
  VecPointwiseMin(_sN,_sN,temp);
  VecSet(temp,_sigmaN_floor);
  VecPointwiseMax(_sN,_sN,temp);
  // free memory
  VecDestroy(&temp);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  return ierr;
}


// compute effective normal stress from total and pore pressure:
// sNEff = sN - rho*g*z - dp
PetscErrorCode Fault::setSNEff(const Vec& p)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::setSNEff";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);

  #endif

  ierr = VecWAXPY(_sNEff,-1.,p,_sN); CHKERRQ(ierr);
  // impose floor and ceiling on effective normal stress
  Vec temp;
  VecDuplicate(_sNEff,&temp);
  VecSet(temp,_sigmaN_cap);
  VecPointwiseMin(_sNEff,_sNEff,temp);
  VecSet(temp,_sigmaN_floor);
  VecPointwiseMax(_sNEff,_sNEff,temp);
  VecDestroy(&temp);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  return ierr;
}



PetscErrorCode Fault::imposeSlipVelCeiling()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault::imposeSlipVelCeiling";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscScalar *slipVel;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_slipVel,&Istart,&Iend);
  VecGetArray(_slipVel,&slipVel);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    slipVel[Jj] = min(slipVel[Jj],_D->_vL);
    Jj++;
  }
  VecRestoreArray(_slipVel,&slipVel);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// print runtime summary
PetscErrorCode Fault::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Fault Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   compute slip vel time (s): %g\n",_computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   state law time (s): %g\n",_stateLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   scatter time (s): %g\n",_scatterTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent finding slip vel law: %g\n",(_computeVelTime/totRunTime)*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in state law: %g\n",(_stateLawTime/totRunTime)*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in scatters: %g\n",(_scatterTime/totRunTime)*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}


// write out parameter settings into "fault_context.txt" file in output directory
// also output vector fields into their respective files in output directory
PetscErrorCode Fault::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer    viewer_ascii;
  // write out scalar info
  string str = outputDir + "fault.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer_ascii);
  PetscViewerSetType(viewer_ascii, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer_ascii, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer_ascii, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer_ascii,"rootTol = %.15e\n",_rootTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer_ascii,"f0 = %.15e\n",_f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer_ascii,"v0 = %.15e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer_ascii,"stateEvolutionLaw = %s\n",_stateLaw.c_str());CHKERRQ(ierr);

  // write flash heating parameters if this is enabled
  if (!_stateLaw.compare("flashHeating")) {
    ierr = PetscViewerASCIIPrintf(viewer_ascii,"fw = %.15e\n",_fw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer_ascii,"tau_c = %.15e # (GPa)\n",_tau_c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer_ascii,"D = %.15e # (um)\n",_D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer_ascii,"VwType = %s\n",_VwType.c_str());CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer_ascii);CHKERRQ(ierr);


  // write Vec context fields
  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                    CHKERRQ(ierr);
  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecView(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sN, viewer);                                       CHKERRQ(ierr);

  if (!_stateLaw.compare("flashHeating")) {
    ierr = VecView(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_c, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_T, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// writes out vector fields at each time step (specified by user using stepCount)
PetscErrorCode Fault::writeStep(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::writeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(viewer);                     CHKERRQ(ierr);

    ierr = VecView(_slip, viewer);                                      CHKERRQ(ierr);
    ierr = VecView(_slipVel, viewer);                                   CHKERRQ(ierr);
    ierr = VecView(_tauP, viewer);                                      CHKERRQ(ierr);
    ierr = VecView(_tauQSP, viewer);                                    CHKERRQ(ierr);
    ierr = VecView(_strength, viewer);                                  CHKERRQ(ierr);
    ierr = VecView(_psi, viewer);                                       CHKERRQ(ierr);
    ierr = VecView(_sNEff, viewer);                                       CHKERRQ(ierr);
    ierr = VecView(_sN, viewer);                                       CHKERRQ(ierr);

    if (_stateLaw.compare("flashHeating") == 0) {
      ierr = VecView(_T, viewer);                                       CHKERRQ(ierr);
      ierr = VecView(_Vw, viewer);                                      CHKERRQ(ierr);
    }

    ierr = PetscViewerHDF5PopTimestepping(viewer);                      CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);                             CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// writes out vector fields at each time step (specified by user using stepCount)
PetscErrorCode Fault::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                     CHKERRQ(ierr);

  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecView(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw.compare("flashHeating") == 0) {
    ierr = VecView(_k, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_c, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_T, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecView(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// destructor, frees memory
Fault::~Fault()
{
  #if VERBOSE > 1
    string funcName = "Fault::~Fault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_z);
  VecDestroy(&_tauQSP);
  VecDestroy(&_tauP);
  VecDestroy(&_strength);
  VecDestroy(&_prestress);

  VecDestroy(&_psi);
  VecDestroy(&_slip);
  VecDestroy(&_slip0);
  VecDestroy(&_slipVel);

  VecDestroy(&_locked);
  VecDestroy(&_Dc);
  VecDestroy(&_a);
  VecDestroy(&_b);
  VecDestroy(&_sNEff);
  VecDestroy(&_sN);
  VecDestroy(&_cohesion);
  VecDestroy(&_mu);
  VecDestroy(&_rho);
  VecDestroy(&_Tw);
  VecDestroy(&_Vw);
  VecDestroy(&_k);
  VecDestroy(&_c);
  VecDestroy(&_T);

  //~ for (map<string,pair<PetscViewer,string>>::iterator it = _viewers.begin(); it != _viewers.end(); it++) {
    //~ PetscViewerDestroy(&_viewers[it->first].first);
  //~ }
  PetscViewerDestroy(&_viewer_hdf5);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// estimate steady-state based on velocity vL
PetscErrorCode Fault::guessSS(const PetscScalar vL)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 2
    string funcName = "Fault::guessSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set slip velocity
  VecSet(_slipVel,vL);

  // set state variable
  if (_stateVals.size() == 0) {
    computePsiSS(vL);
  }

  // shear stress
  PetscInt       Istart,Iend;
  PetscScalar   *tauRSV;
  PetscScalar const *sN,*a,*psi;
  VecGetOwnershipRange(_tauP,&Istart,&Iend);
  VecGetArray(_tauP,&tauRSV);
  VecGetArrayRead(_sNEff,&sN);
  VecGetArrayRead(_psi,&psi);
  VecGetArrayRead(_a,&a);

  PetscInt Jj = 0;
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    tauRSV[Jj] = sN[Jj]*a[Jj]*asinh( (double) 0.5*vL*exp(psi[Jj]/a[Jj])/_v0 );
    Jj++;
  }

  VecRestoreArray(_tauP,&tauRSV);
  VecRestoreArrayRead(_sNEff,&sN);
  VecRestoreArrayRead(_psi,&psi);
  VecRestoreArrayRead(_a,&a);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// compute the state vector psi for steady state solution
PetscErrorCode Fault::computePsiSS(const PetscScalar vL)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 2
    string funcName = "Fault::computePsiSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt           Istart,Iend;
  PetscScalar       *psi;
  const PetscScalar *b;
  VecGetOwnershipRange(_psi,&Istart,&Iend);
  VecGetArray(_psi,&psi);
  VecGetArrayRead(_b,&b);

  PetscInt Jj = 0;
  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    psi[Jj] = _f0 - b[Jj]*log(abs(vL)/_v0);
    Jj++;
  }
  VecRestoreArray(_psi,&psi);
  VecRestoreArrayRead(_b,&b);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


//================================================================================
//================= Functions assuming only + side exists ========================
//================================================================================

// constructor of derived class Fault_qd, initializes the same object as Fault
Fault_qd::Fault_qd(Domain &D, VecScatter& scatter2fault, const int& faultTypeScale)
: Fault(D,scatter2fault,faultTypeScale),_eta_rad(NULL)
{
  #if VERBOSE > 1
    string funcName = "Fault_qd::Fault_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // radiation damping parameter: 0.5 * sqrt(mu*rho)
  VecDuplicate(_tauP,&_eta_rad);
  PetscObjectSetName((PetscObject) _eta_rad, "eta_rad");
  VecPointwiseMult(_eta_rad,_mu,_rho);
  VecSqrtAbs(_eta_rad);
  VecScale(_eta_rad,1.0/_faultTypeScale);

  if (_D->_restartFromChkpt) {
    loadCheckpoint();
  }
  else if (_D->_restartFromChkptSS) {
    loadCheckpointSS();
  }
  else {
    loadFieldsFromFiles();
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// destructor, frees memory
Fault_qd::~Fault_qd()
{
  #if VERBOSE > 1
    string funcName = "Fault_qd::~Fault_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_eta_rad);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// initialize variables to be integrated, put them into varEx
PetscErrorCode Fault_qd::initiateIntegrand(const PetscScalar time, map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated explicitly into varEx
  if (varEx.find("psi") != varEx.end() ) {
    VecCopy(_psi,varEx["psi"]);
  }
  else {
    Vec varPsi;
    VecDuplicate(_psi,&varPsi);
    VecCopy(_psi,varPsi);
    varEx["psi"] = varPsi;
  }

  // slip is initialized in the strikeSlip class's initiateIntegrand function

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// update variables in varEx (i.e. state variable psi, as well as slip), copy them into _psi and _slip
PetscErrorCode Fault_qd::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("psi")->second,_psi);
  VecCopy(varEx.find("slip")->second,_slip);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// solve for slip velocity
// assumes right-lateral fault
PetscErrorCode Fault_qd::computeVel()
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initialize struct to solve for the slip velocity
  PetscScalar *slipVelA;
  const PetscScalar *etaA, *tauQSA, *sNA, *psiA, *aA,*bA,*lockedA,*Co;
  ierr = VecGetArray(_slipVel,&slipVelA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_eta_rad,&etaA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_tauQSP,&tauQSA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_sNEff,&sNA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_psi,&psiA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_a,&aA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_b,&bA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_locked,&lockedA); CHKERRQ(ierr);
  ierr = VecGetArrayRead(_cohesion,&Co); CHKERRQ(ierr);

  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;

  // create ComputeVel_qd struct
  ComputeVel_qd temp(N,etaA,tauQSA,sNA,psiA,aA,bA,_v0,_D->_vL,lockedA,Co);
  ierr = temp.computeVel(slipVelA, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  // now limit slipVel to <= vL if desired
  if (_limitSlipVel) { imposeSlipVelCeiling(); }

  ierr = VecRestoreArray(_slipVel,&slipVelA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_eta_rad,&etaA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_tauQSP,&tauQSA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_sNEff,&sNA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_psi,&psiA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_a,&aA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_b,&bA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_locked,&lockedA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_cohesion,&Co); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// time stepping
PetscErrorCode Fault_qd::d_dt(const PetscScalar time, const map<string,Vec>& varEx, map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // add pre-stress to quasi-static shear stress
  VecAXPY(_tauQSP,1.0,_prestress);

  // compute slip velocity
  double startTime = MPI_Wtime();
  ierr = computeVel();CHKERRQ(ierr);
  VecCopy(_slipVel,dvarEx["slip"]);
  _computeVelTime += MPI_Wtime() - startTime;


  // compute rate of state variable
  Vec dstate = dvarEx.find("psi")->second;
  startTime = MPI_Wtime();
  if (_stateLaw.compare("agingLaw") == 0) {
    //~ ierr = agingLaw_theta_Vec(dstate, _theta, _slipVel, _Dc) CHKERRQ(ierr);
    ierr = agingLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc); CHKERRQ(ierr);
  }
  else if (_stateLaw.compare("slipLaw") == 0) {
    //~ ierr = slipLaw_theta_Vec(dstate, _theta, _slipVel, _Dc); CHKERRQ(ierr);
    ierr =  slipLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc); CHKERRQ(ierr);
  }
  else if (_stateLaw.compare("flashHeating") == 0) {
    ierr = flashHeating_psi_Vec(dstate,_psi,_slipVel,_T,_rho,_c,_k,_Vw,_D_fh,_Tw,_tau_c,_fw,_Dc,_a,_b,_f0,_v0,_VwType);
    CHKERRQ(ierr);
  }
  else if (_stateLaw.compare("constantState") == 0) {
    VecSet(dstate,0.);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n");
    assert(0);
  }
  _stateLawTime += MPI_Wtime() - startTime;


  // set tauP = tauQS - eta_rad *slipVel
  VecCopy(_slipVel,_tauP); // V -> tau
  VecPointwiseMult(_tauP,_eta_rad,_tauP); // tau = V * eta_rad
  VecAYPX(_tauP,-1.0,_tauQSP); // tau = tauQS - V*eta_rad

  // compute frictional strength of fault based on slip velocity
  strength_psi_Vec(_strength, _psi, _slipVel, _a, _sNEff, _v0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

// update tauP based on current tauQSP and slipVel
// set tauP = tauQS - eta_rad *slipVel
PetscErrorCode Fault_qd::updateTauP()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::updateTauP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = VecCopy(_slipVel,_tauP);CHKERRQ(ierr); // V -> tau
  ierr = VecPointwiseMult(_tauP,_eta_rad,_tauP);CHKERRQ(ierr); // tau = V * eta_rad
  ierr = VecAYPX(_tauP,-1.0,_tauQSP);CHKERRQ(ierr); // tau = tauQS - V*eta_rad

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// update strength based on current state (slipVel and psi)
PetscErrorCode Fault_qd::updateStrength()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::updateStrength";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = strength_psi_Vec(_strength, _psi, _slipVel, _a, _sNEff, _v0);CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// output vector fields into file, and calls writeContext function in Fault
PetscErrorCode Fault_qd::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Fault::writeContext(outputDir, viewer);

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_qd");                 CHKERRQ(ierr);
  ierr = VecView(_eta_rad, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode Fault_qd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_qd");                 CHKERRQ(ierr);

  ierr = VecLoad(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw == "flashHeating") {
    ierr = VecLoad(_k, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_c, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_T, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecLoad(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = VecLoad(_eta_rad, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Fault_qd::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::loadCheckpointSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer viewer;

  string fileName = _outputDir + "data_context.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                    CHKERRQ(ierr);
  ierr = VecLoad(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sN, viewer);                                          CHKERRQ(ierr);
  if (_stateLaw == "flashHeating") {
    ierr = VecLoad(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_c, viewer);                                         CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_qd");                 CHKERRQ(ierr);
  ierr = VecLoad(_eta_rad, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                    CHKERRQ(ierr);
  ierr = VecLoad(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_psi, viewer);                                         CHKERRQ(ierr);

  if (_stateLaw.compare("flashHeating") == 0) {
    ierr = VecLoad(_T, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_Vw, viewer);                                        CHKERRQ(ierr);
  }

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Fault_qd::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_qd::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_qd");                 CHKERRQ(ierr);

  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecView(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw == "flashHeating") {
    ierr = VecView(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_c, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_T, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecView(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = VecView(_eta_rad, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



//======================================================================
// functions for struct ComputeVel_qd
//======================================================================

// constructor for ComputeVel_qd
ComputeVel_qd::ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0,const PetscScalar& vL,const PetscScalar* locked,const PetscScalar* Co)
: _a(a),_b(b),_sN(sN),_tauQS(tauQS),_eta(eta),_psi(psi),_locked(locked),_Co(Co),_N(N),_v0(v0),_vL(vL)
{ }


// compute slip velocity for quasidynamic setting
PetscErrorCode ComputeVel_qd::computeVel(PetscScalar *slipVelA, const PetscScalar rootTol, PetscInt &rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar left, right, out;
  PetscInt Jj;
  for (Jj = 0; Jj < _N; Jj++) {
    // hold slip velocity at 0
    if (_locked[Jj] > 0.5) {
      slipVelA[Jj] = 0.;
    }
    // force fault to creep at loading velocity
    else if (_locked[Jj] < -0.5) {
      slipVelA[Jj] = _vL;
    }
    else {
      left = 0.;
      right = _tauQS[Jj] / _eta[Jj];

      // check bounds
      if (std::isnan(left)) {
        PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
        PetscPrintf(PETSC_COMM_WORLD,"tauQS = %g, eta = %g, left = %g\n",_tauQS[Jj],_eta[Jj],left);
        assert(0);
      }
      if (std::isnan(right)) {
        PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
        PetscPrintf(PETSC_COMM_WORLD,"tauQS = %g, eta = %g, right = %g\n",_tauQS[Jj],_eta[Jj],right);
        assert(0);
      }

      out = slipVelA[Jj];

      if (abs(left-right)<1e-14) {
        out = left;
      }
      else {
        //Bisect rootFinder(maxNumIts,rootTol);
              //ierr = rootFinder.setBounds(left,right); CHKERRQ(ierr);
        //ierr = rootFinder.findRoot(this,Jj,&out); assert(ierr == 0); CHKERRQ(ierr);
        //rootIts += rootFinder.getNumIts();
        PetscScalar x0 = slipVelA[Jj];
        BracketedNewton rootFinder(maxNumIts,rootTol);
        ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
        ierr = rootFinder.findRoot(this,Jj,x0,&out); CHKERRQ(ierr);
        rootIts += rootFinder.getNumIts();
      }
      slipVelA[Jj] = out;
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeVel_qd::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out)
{
  PetscErrorCode ierr = 0;
  // frictional strength
  PetscScalar strength = strength_psi(_sN[Jj], _psi[Jj], vel, _a[Jj], _v0);
  // stress on fault
  PetscScalar stress =_tauQS[Jj] - _eta[Jj]*vel;

  *out = strength - stress;
  assert(!std::isnan(*out));
  assert(!std::isinf(*out));

  return ierr;
}


// compute residual for equation to find slip velocity
// This form is for root finding algorithms that require the Jacobian, such as the bracketed Newton method.
PetscErrorCode ComputeVel_qd::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar *out,PetscScalar *J)
{
  PetscErrorCode ierr = 0;
  PetscScalar strength = strength_psi(_sN[Jj], _psi[Jj], vel, _a[Jj], _v0); // frictional strength
  PetscScalar stress = _tauQS[Jj] - _eta[Jj]*vel; // stress on fault

  *out = strength - stress;
  PetscScalar A = _a[Jj]*_sN[Jj];
  PetscScalar B = exp(_psi[Jj]/_a[Jj]) / (2.*_v0);

  // derivative with respect to slipVel
  *J = A*vel/sqrt(B*B*vel*vel + 1.) + _eta[Jj];

  if (std::isinf(*out)) {
    PetscPrintf(PETSC_COMM_WORLD,"Jj = %i\n",Jj);
    PetscPrintf(PETSC_COMM_WORLD,"strength = %.9e\n",strength);
    PetscPrintf(PETSC_COMM_WORLD,"stress = %.9e\n",stress);
    PetscPrintf(PETSC_COMM_WORLD,"A = %.9e\n",A);
    PetscPrintf(PETSC_COMM_WORLD,"B = %.9e\n",B);
    PetscPrintf(PETSC_COMM_WORLD,"psi = %.9e, a = %.9e, exp(_psi[Jj]/_a[Jj]) = %.9e\n",_psi[Jj],_a[Jj],exp(_psi[Jj]/_a[Jj]));
  }

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  assert(!std::isnan(*J));
  assert(!std::isinf(*J));

  return ierr;
}


//=========================Back to classes===============================

// constructor for Fault_fd class
Fault_fd::Fault_fd(Domain &D, VecScatter& scatter2fault, const int& faultTypeScale)
: Fault(D, scatter2fault,faultTypeScale),
  _Phi(NULL), _an(NULL), _fricPen(NULL),
  _u(NULL), _uPrev(NULL), _d2u(NULL),_alphay(NULL),
  _timeMode("None")
{
  #if VERBOSE > 1
    string funcName = "Fault_fd::Fault_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // load settings and fields specific to fully-dynamics case
  loadSettings(_inputFile);
  setFields();
  if (_D->_restartFromChkpt) {
    loadCheckpoint();
  }
  else if (_D->_restartFromChkptSS) {
    loadCheckpointSS();
  }
  else {
    loadFieldsFromFiles();
    loadVecFromInputFile(_tau0,_D->_inputDir,"prestress");
  }


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// destructor, frees memory
Fault_fd::~Fault_fd()
{
  #if VERBOSE > 1
    string funcName = "Fault_fd::~Fault_fd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_tau0);
  VecDestroy(&_Phi);
  VecDestroy(&_an);
  VecDestroy(&_fricPen);
  VecDestroy(&_u);
  VecDestroy(&_uPrev);
  VecDestroy(&_d2u);
  VecDestroy(&_alphay);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// load settings from input file
PetscErrorCode Fault_fd::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::~loadSettings";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  ifstream infile( file );
  string line, var, rhs, rhsFull;
  size_t pos = 0;

  while (getline(infile, line)) {
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

    // Tau dynamic parameters
    if (var.compare("tCenterTau")==0) { _tCenterTau = atof( rhs.c_str() ); }
    else if (var.compare("zCenterTau")==0) { _zCenterTau = atof( rhs.c_str() ); }
    else if (var.compare("tStdTau")==0) { _tStdTau = atof( rhs.c_str() ); }
    else if (var.compare("zStdTau")==0) { _zStdTau = atof( rhs.c_str() ); }
    else if (var.compare("ampTau")==0) { _ampTau = atof( rhs.c_str() ); }
    else if (var.compare("timeMode")==0) { _timeMode = rhs; }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// allocate memory for fields
PetscErrorCode Fault_fd::setFields() {
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::setFields";
    PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
  #endif

  // allocate memory for Vec members
  VecDuplicate(_tauP,&_tau0);     VecSet(_tau0, 0.0);                   PetscObjectSetName((PetscObject) _tau0, "tau0");
  VecDuplicate(_tauP,&_Phi);      VecSet(_Phi, 0.0);                    PetscObjectSetName((PetscObject) _Phi, "Phi");
  VecDuplicate(_tauP,&_an);       VecSet(_an, 0.0);                     PetscObjectSetName((PetscObject) _an, "an");
  VecDuplicate(_tauP,&_fricPen);  VecSet(_fricPen, 0.0);                PetscObjectSetName((PetscObject) _fricPen, "fricPen");
  VecDuplicate(_tauP,&_u);        VecSet(_u,0.0);                       PetscObjectSetName((PetscObject) _u, "u");
  VecDuplicate(_tauP,&_uPrev);    VecSet(_uPrev,0.0);                   PetscObjectSetName((PetscObject) _uPrev, "uPrev");
  VecDuplicate(_tauP,&_d2u);      VecSet(_d2u,0.0);                     PetscObjectSetName((PetscObject) _d2u, "d2u");
  VecDuplicate(_tauP,&_alphay);   VecSet(_alphay, 17.0/48.0 / (_N-1));  PetscObjectSetName((PetscObject) _alphay, "alphay");

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// set up integration, put variables to be integrated into varEx
PetscErrorCode Fault_fd::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated explicitly into varEx
  if (varEx.find("psi") != varEx.end() ) { VecCopy(_psi,varEx["psi"]); }
  else { Vec varPsi; VecDuplicate(_psi,&varPsi); VecCopy(_psi,varPsi); varEx["psi"] = varPsi; }

  if (varEx.find("slip") != varEx.end() ) { VecCopy(_slip,varEx["slip"]); }
  else { Vec varSlip; VecDuplicate(_slip,&varSlip); VecCopy(_slip,varSlip); varEx["slip"] = varSlip; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// update variables in varEx (i.e. state variable psi, as well as slip)
PetscErrorCode Fault_fd::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("psi")->second,_psi);
  VecCopy(varEx.find("slip")->second,_slip);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// compute slip velocity
// assumes right-lateral fault
PetscErrorCode Fault_fd::computeVel()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_fd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  double startTime = MPI_Wtime();

  // initialize struct to solve for the slip velocity
  PetscScalar *Phi, *an, *psi, *fricPen, *a,*sneff, *slipVel, *locked;
  VecGetArray(_Phi,&Phi);
  VecGetArray(_an,&an);
  VecGetArray(_psi,&psi);
  VecGetArray(_fricPen,&fricPen);
  VecGetArray(_a,&a);
  VecGetArray(_sNEff,&sneff);
  VecGetArray(_slipVel,&slipVel);
  VecGetArray(_locked, &locked);

  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;

  ComputeVel_fd temp(locked, N,Phi,an,psi,fricPen,a,sneff, _v0, _D->_vL);
  ierr = temp.computeVel(slipVel, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecRestoreArray(_Phi,&Phi);
  VecRestoreArray(_an,&an);
  VecRestoreArray(_psi,&psi);
  VecRestoreArray(_fricPen,&fricPen);
  VecRestoreArray(_a,&a);
  VecRestoreArray(_sNEff,&sneff);
  VecRestoreArray(_slipVel,&slipVel);
  VecRestoreArray(_locked, &locked);

  _computeVelTime = MPI_Wtime() - startTime;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// compute evolution of state variable psi
PetscErrorCode Fault_fd::computeStateEvolution(Vec& psiNext, const Vec& psi, const Vec& psiPrev)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  double startStateLawTime = MPI_Wtime();

  // initialize struct to solve for the slip velocity
  PetscScalar *Dc, *b, *psiNextA, *psiA, *psiPrevA, *slipVel;
  VecGetArray(_Dc,&Dc);
  VecGetArray(_b,&b);
  VecGetArray(psiNext,&psiNextA);
  VecGetArray(psi,&psiA);
  VecGetArray(psiPrev,&psiPrevA);
  VecGetArray(_slipVel,&slipVel);

  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;

  // compute state evolution for the specified aging law
  if (_stateLaw.compare("agingLaw") == 0){
    ComputeAging_fd temp(N,Dc,b,psiNextA,psiA,psiPrevA, slipVel, _v0, _deltaT, _f0);
    ierr = temp.computeLaw(_rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);
  }

  else if (_stateLaw.compare("slipLaw") == 0){
    PetscScalar *a;
    VecGetArray(_a, &a);
    ComputeSlipLaw_fd temp(N,Dc,a, b,psiNextA,psiA,psiPrevA, slipVel, _v0, _deltaT, _f0);
    ierr = temp.computeLaw(_rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);
    VecRestoreArray(_a, &a);
  }

  else if (_stateLaw.compare("flashHeating") == 0){
    PetscScalar *a, * Vw;
    VecGetArray(_a, &a);
    VecGetArray(_Vw, &Vw);
    ComputeFlashHeating_fd temp(N,Dc,a,b,psiNextA,psiA,psiPrevA, slipVel, Vw, _v0, _deltaT, _f0, _fw);
    ierr = temp.computeLaw(_rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);
    VecRestoreArray(_a, &a);
    VecRestoreArray(_Vw, &Vw);
  }

  else{
    assert(0);
  }

  VecRestoreArray(_Dc,&Dc);
  VecRestoreArray(_b,&b);
  VecRestoreArray(psiNext,&psiNextA);
  VecRestoreArray(psi,&psiA);
  VecRestoreArray(psiPrev,&psiPrevA);
  VecRestoreArray(_slipVel,&slipVel);

  _stateLawTime += MPI_Wtime() - startStateLawTime;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// update prestress vector tau0 based on time offset
PetscErrorCode Fault_fd::updatePrestress(const PetscScalar currT)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::updatePrestress";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar *zz, *tau0;
  PetscInt Ii, IBegin, IEnd;
  PetscInt Jj = 0;
  VecGetArray(_z, &zz);
  VecGetArray(_prestress, &tau0);
  VecGetOwnershipRange(_z, &IBegin, &IEnd);

  PetscScalar timeOffset = 1.0;
  PetscScalar exists = 1.0;

  // set timeOffset based on specified time mode
  if(_timeMode.compare("Gaussian") == 0){ timeOffset = exp(-pow((currT - _tCenterTau), 2) / pow(_tStdTau, 2)); }
  else if (_timeMode.compare("Dirac") == 0 && currT > 0){ timeOffset = 0.0; }
  else if (_timeMode.compare("Heaviside") == 0){ timeOffset = 1.0; }
  else if (_timeMode.compare("None") == 0){ exists = 0.0; }

  // update tau0 prestress vector
  for (Ii = IBegin; Ii < IEnd; Ii++){
    tau0[Jj] = exists * (30.0 + _ampTau * exp(-pow((zz[Jj] - _zCenterTau), 2) / (2.0 * pow(_zStdTau, 2))) * timeOffset);
    Jj++;
  }

  VecRestoreArray(_z, &zz);
  VecRestoreArray(_prestress, &tau0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// calculate slip velocity by calling computeVel(), and performs explicit time stepping, updating fields with the new time step
PetscErrorCode Fault_fd::d_dt(const PetscScalar time,const PetscScalar deltaT, map<string,Vec>& varNext,const map<string,Vec>& var,const map<string,Vec>& varPrev)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update fields with new time step
  _deltaT = deltaT;
  VecCopy(var.find("psi")->second,_psi);
  VecCopy(var.find("slip")->second,_slip);

  // uPrev = (slip - slip0)/faultTypeScale
  VecWAXPY(_uPrev,-1.0,_slip0,varPrev.find("slip")->second);
  VecScale(_uPrev,1.0/_faultTypeScale);

  // u = (slip - slip0)/faultTypeScale
  VecWAXPY(_u,-1.0,_slip0,var.find("slip")->second);
  VecScale(_u,1.0/_faultTypeScale);

  // compute slip velocity
  ierr = setPhi(deltaT);

  // computes abs(slipVel)
  ierr = computeVel(); CHKERRQ(ierr);

  PetscInt       Ii,Istart,Iend;
  PetscScalar   *u, *uPrev, *slip, *slipVel; // changed in this loop
  const PetscScalar  *rho, *sNEff, *a, *an, *Phi, *psi, *alphay; // constant in this loop
  ierr = VecGetOwnershipRange(_u,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArray(_u, &u);
  ierr = VecGetArray(_uPrev, &uPrev);
  ierr = VecGetArray(_slip, &slip);
  ierr = VecGetArray(_slipVel, &slipVel);
  ierr = VecGetArrayRead(_an, &an);
  ierr = VecGetArrayRead(_rho, &rho);
  ierr = VecGetArrayRead(_psi, &psi);
  ierr = VecGetArrayRead(_sNEff, &sNEff);
  ierr = VecGetArrayRead(_a, &a);
  ierr = VecGetArrayRead(_Phi, &Phi);
  ierr = VecGetArrayRead(_alphay, &alphay);

  PetscInt Jj = 0;
  for (Ii = Istart; Ii < Iend; Ii++) {
    if (slipVel[Jj] < 1e-14){
      // slipVel[Jj] = 0;
    }
    else {
      PetscScalar fric = strength_psi(sNEff[Jj], psi[Jj], slipVel[Jj], a[Jj], _v0);
      PetscScalar alpha = 1.0 / (rho[Jj] * alphay[Jj]) * fric / slipVel[Jj];
      PetscScalar A = 1.0 + alpha * deltaT;
      slipVel[Jj] = Phi[Jj] / (1. + _deltaT * alpha);
      u[Jj] = (2.*u[Jj]  +  (an[Jj] * deltaT*deltaT / rho[Jj])  +  (_deltaT*alpha-1.)*uPrev[Jj]) / A;
    }
    Jj++;
  }

  ierr = VecRestoreArray(_u, &u);
  ierr = VecRestoreArray(_uPrev, &uPrev);
  ierr = VecRestoreArray(_slip, &slip);
  ierr = VecRestoreArray(_slipVel, &slipVel);
  ierr = VecRestoreArrayRead(_an, &an);
  ierr = VecRestoreArrayRead(_rho, &rho);
  ierr = VecRestoreArrayRead(_psi, &psi);
  ierr = VecRestoreArrayRead(_sNEff, &sNEff);
  ierr = VecRestoreArrayRead(_a, &a);
  ierr = VecRestoreArrayRead(_Phi, &Phi);
  ierr = VecRestoreArrayRead(_alphay, &alphay);

  // update state variable
  computeStateEvolution(varNext["psi"], var.find("psi")->second, varPrev.find("psi")->second);
  VecCopy(varNext["psi"],_psi);

  // assemble slip from u
  VecWAXPY(_slip,_faultTypeScale,_u,_slip0); // slip = 2*u + slip0
  VecCopy(_slip,varNext["slip"]);

  // compute frictional strength of fault
  strength_psi_Vec(_strength, _psi, _slipVel, _a, _sNEff, _v0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// sets value for array Phi
PetscErrorCode Fault_fd::setPhi(const PetscScalar deltaT)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "Fault_fd::setPhi";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Ii,Istart, Iend;
  ierr = VecGetOwnershipRange(_d2u,&Istart,&Iend);CHKERRQ(ierr);

  PetscScalar  *an, *Phi, *fricPen;
  const PetscScalar *u, *uPrev, *d2u, *rho, *tau0, *alphay;

  ierr = VecGetArray(_an, &an);
  ierr = VecGetArray(_Phi, &Phi);
  ierr = VecGetArray(_fricPen, &fricPen);
  ierr = VecGetArrayRead(_u, &u);
  ierr = VecGetArrayRead(_uPrev, &uPrev);
  ierr = VecGetArrayRead(_d2u, &d2u);
  ierr = VecGetArrayRead(_rho, &rho);
  ierr = VecGetArrayRead(_tau0, &tau0);
  ierr = VecGetArrayRead(_alphay, &alphay);

  PetscInt Jj = 0;
  for (Ii = Istart; Ii < Iend; Ii++){
    an[Jj] = d2u[Jj] + tau0[Jj] / alphay[Jj];
    Phi[Jj] = 2.0 / deltaT * (u[Jj] - uPrev[Jj]) + deltaT * an[Jj] / rho[Jj];
    fricPen[Jj] = deltaT / alphay[Jj] / rho[Jj];
    Jj++;
  }

  ierr = VecRestoreArray(_an, &an);
  ierr = VecRestoreArray(_Phi, &Phi);
  ierr = VecRestoreArray(_fricPen, &fricPen);
  ierr = VecGetArrayRead(_u, &u);
  ierr = VecGetArrayRead(_uPrev, &uPrev);
  ierr = VecGetArrayRead(_d2u, &d2u);
  ierr = VecGetArrayRead(_rho, &rho);
  ierr = VecGetArrayRead(_tau0, &tau0);
  ierr = VecGetArrayRead(_alphay, &alphay);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}

PetscErrorCode Fault_fd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_fd::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_fd");           CHKERRQ(ierr);

  ierr = VecLoad(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw.compare("flashHeating") == 0) {
    ierr = VecLoad(_k, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_c, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_T, viewer);                                        CHKERRQ(ierr);
    ierr = VecLoad(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecLoad(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = VecLoad(_tau0, viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_Phi, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_an, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_fricPen, viewer);                                     CHKERRQ(ierr);
  ierr = VecLoad(_u, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_uPrev, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_d2u, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_alphay, viewer);                                      CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Fault_fd::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_fd::loadCheckpointSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer viewer;

  string fileName = _outputDir + "data_context.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                    CHKERRQ(ierr);
  ierr = VecLoad(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecLoad(_sN, viewer);                                          CHKERRQ(ierr);
  if (_stateLaw == "flashHeating") {
    ierr = VecLoad(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecLoad(_c, viewer);                                         CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_qd");                 CHKERRQ(ierr);
  ierr = VecLoad(_mu, viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fault");                 CHKERRQ(ierr);
  ierr = VecLoad(_slip, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_slipVel, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_tauP, viewer);                                      CHKERRQ(ierr);
  ierr = VecLoad(_tauQSP, viewer);                                    CHKERRQ(ierr);
  ierr = VecLoad(_strength, viewer);                                  CHKERRQ(ierr);
  ierr = VecLoad(_psi, viewer);                                       CHKERRQ(ierr);

  if (_stateLaw.compare("flashHeating") == 0) {
    ierr = VecLoad(_T, viewer);                                       CHKERRQ(ierr);
    ierr = VecLoad(_Vw, viewer);                                      CHKERRQ(ierr);
  }

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Fault_fd::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "Fault_fd::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = PetscViewerHDF5PushGroup(viewer, "/fault_fd");                 CHKERRQ(ierr);

  ierr = VecView(_z, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_a, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_b, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Dc, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_cohesion, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_locked, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_prestress, viewer);                                   CHKERRQ(ierr);
  ierr = VecView(_slip0, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sNEff, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_sN, viewer);                                          CHKERRQ(ierr);

  if (_stateLaw == "flashHeating") {
    ierr = VecView(_k, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_c, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_T, viewer);                                         CHKERRQ(ierr);
    ierr = VecView(_Tw, viewer);                                        CHKERRQ(ierr);
    ierr = VecView(_Vw, viewer);                                        CHKERRQ(ierr);
  }
  ierr = VecView(_slip, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_slipVel, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_tauP, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_tauQSP, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_strength, viewer);                                    CHKERRQ(ierr);
  ierr = VecView(_psi, viewer);                                         CHKERRQ(ierr);

  ierr = VecView(_tau0, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_Phi, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_an, viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_fricPen, viewer);                                     CHKERRQ(ierr);
  ierr = VecView(_u, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_uPrev, viewer);                                       CHKERRQ(ierr);
  ierr = VecView(_d2u, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_alphay, viewer);                                      CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// =================== functions for struct ComputeVel_fd ====================
// to handle the computation of the implicit solvers
// ===========================================================================

// constructor
ComputeVel_fd::ComputeVel_fd(const PetscScalar* locked, const PetscInt N,const PetscScalar* Phi, const PetscScalar* an, const PetscScalar* psi, const PetscScalar* fricPen,const PetscScalar* a,const PetscScalar* sneff, const PetscScalar v0, const PetscScalar vL)
: _locked(locked), _Phi(Phi),_an(an),_psi(psi),_fricPen(fricPen),_a(a),_sNEff(sneff),_N(N), _v0(v0), _vL(vL)
{ }

// compute absolute value of slip velocity for fully dynamic case
PetscErrorCode ComputeVel_fd::computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  BracketedNewton rootFinder(maxNumIts,rootTol);
  //~ Bisect rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, out, temp;

  for (PetscInt Jj = 0; Jj<_N; Jj++) {
    if (_locked[Jj] > 0.5) { // if fault is locked, hold slip velocity at 0
      slipVelA[Jj] = 0.;
    }
    else if (_locked[Jj] < -0.5) { // if fault is locked, hold slip velocity at 0
      slipVelA[Jj] = _vL;
    }
    else {
      left = 0.;
      right = abs(_Phi[Jj]);
      // check bounds
      if (std::isnan(left)) {
        PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_fd::computeVel: left bound evaluated to NaN.\n");
        assert(0);
      }
      if (std::isnan(right)) {
        PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_fd::computeVel: right bound evaluated to NaN.\n");
        assert(0);
      }
      // correct for left-lateral fault motion
      if (left > right) {
        temp = right;
        right = left;
        left = temp;
      }

      if (abs(left-right)<1e-14) { out = left; }
      else {
        ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
        //~ ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
        ierr = rootFinder.findRoot(this,Jj,abs(slipVelA[Jj]), &out);CHKERRQ(ierr);
        rootIts += rootFinder.getNumIts();
      }
      slipVelA[Jj] = out;
      // PetscPrintf(PETSC_COMM_WORLD,"%i: left = %g, right = %g, slipVel = %g\n",Jj,left,right,out);
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeVel_fd::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out)
{
  PetscErrorCode ierr = 0;
  PetscScalar strength = strength_psi(_sNEff[Jj], _psi[Jj], vel, _a[Jj] , _v0); // frictional strength
  PetscScalar stress = abs(_Phi[Jj]) - vel; // stress on fault

  *out = _fricPen[Jj] * strength - stress;
  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  return ierr;
}


// compute residual for equation to find slip velocity
// for methods that require a Jacobian, such as bracketed Newton
PetscErrorCode ComputeVel_fd::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;
  PetscScalar constraints = strength_psi(_sNEff[Jj], _psi[Jj], vel, _a[Jj] , _v0); // frictional strength

  constraints = _fricPen[Jj] * constraints;
  PetscScalar Phi_temp = _Phi[Jj];
  if (Phi_temp < 0){
    Phi_temp = -Phi_temp;
  }

  PetscScalar stress = Phi_temp - vel; // stress on fault

  *out = constraints - stress;
  PetscScalar A = _a[Jj] * _sNEff[Jj];
  PetscScalar B = exp(_psi[Jj] / _a[Jj]) / (2. * _v0);

  *J = 1 + _fricPen[Jj] * A * B / sqrt(1. + B * B * vel * vel);

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  return ierr;
}

// ================================================

// struct for aging law for fully dynamic case
ComputeAging_fd::ComputeAging_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev, const PetscScalar* slipVel, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0)
: _Dc(Dc),_b(b),_slipVel(slipVel),_psi(psi),_psiPrev(psiPrev),_psiNext(psiNext),
 _N(N), _v0(v0), _deltaT(deltaT), _f0(f0)
{ }


// perform root finding once contextal variables have been set
PetscErrorCode ComputeAging_fd::computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // RegulaFalsi rootFinder(maxNumIts,rootTol);
  BracketedNewton rootFinder(maxNumIts,rootTol);
  //~ Bisect rootFinder(maxNumIts,rootTol);

  PetscScalar left, right, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {
    left = -2.;
    right = 2.;
    //~ left = 0.;
    //~ right = 2*_psi[Jj];

    // check bounds
    if (std::isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (std::isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
      assert(0);
    }
    // correct for left-lateral fault motion
    if (left > right) {
      temp = right;
      right = left;
      left = temp;
    }

    if (abs(left-right)<1e-14) {
      _psiNext[Jj] = left;
    }
    else {
      ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
      ierr = rootFinder.findRoot(this,Jj,_psi[Jj],&_psiNext[Jj]);CHKERRQ(ierr);
      //~ ierr = rootFinder.findRoot(this,Jj,&_psiNext[Jj]);CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeAging_fd::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = agingLaw_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;
  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  return ierr;
}


// this form if for algorithms that require a Jacobian, such as bracketed Newton
PetscErrorCode ComputeAging_fd::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = agingLaw_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;

  *J = 1 + _deltaT * _v0 / _Dc[Jj] * exp((_f0 - (_psiPrev[Jj] + state)/2.)/_b[Jj]);

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));

  return ierr;
}

// ================================================
// computes slip law for fully dynamic problem

// constructor
ComputeSlipLaw_fd::ComputeSlipLaw_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* a,const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev,const PetscScalar* slipVel, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0)
: _Dc(Dc),_a(a),_b(b),_slipVel(slipVel),_psi(psi),_psiPrev(psiPrev),_psiNext(psiNext), _N(N), _v0(v0), _deltaT(deltaT), _f0(f0)
{ }


// perform root-finding
PetscErrorCode ComputeSlipLaw_fd::computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // RegulaFalsi rootFinder(maxNumIts,rootTol);
  BracketedNewton rootFinder(maxNumIts,rootTol);
  // Bisect rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {
    left = -2.;
    right = 2.;
    //~ left = 0.;
    //~ right = 2*_psi[Jj];

    // check bounds
    if (std::isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (std::isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
      assert(0);
    }
    // correct for left-lateral fault motion
    if (left > right) {
      temp = right;
      right = left;
      left = temp;
    }

    if (abs(left-right)<1e-14) {
      _psiNext[Jj] = left;
    }
    else {
      ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
      ierr = rootFinder.findRoot(this,Jj,_psi[Jj],&_psiNext[Jj]);CHKERRQ(ierr);
      // ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeSlipLaw_fd::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = slipLaw_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _a[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  return ierr;
}


// this is for algorithms that require a Jacobian, such as bracketed Newton
PetscErrorCode ComputeSlipLaw_fd::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = slipLaw_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _a[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;

  PetscScalar A = abs(_slipVel[Jj]) / 2. / _v0 * exp((_psiPrev[Jj] + state) / 2.0 / _a[Jj]);
  *J = 1 + _deltaT * abs(_slipVel[Jj]) / _Dc[Jj] * A / sqrt(1 + A * A);

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));

  return ierr;
}

// ================================================
// struct to compute flash heating

// constructor
ComputeFlashHeating_fd::ComputeFlashHeating_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* a, const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev, const PetscScalar* slipVel, const PetscScalar* Vw,const PetscScalar v0, const PetscScalar deltaT,const PetscScalar f0, const PetscScalar fw)
: _Dc(Dc),_a(a),_b(b),_slipVel(slipVel),
  _Vw(Vw),_psi(psi),_psiPrev(psiPrev),_psiNext(psiNext),
  _N(N), _v0(v0), _deltaT(deltaT), _f0(f0), _fw(fw)
{ }


// find roots
PetscErrorCode ComputeFlashHeating_fd::computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // RegulaFalsi rootFinder(maxNumIts,rootTol);
  BracketedNewton rootFinder(maxNumIts,rootTol);
  // Bisect rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {
    left = -2.;
    right = 2.;
    //~ left = 0.;
    //~ right = 2*_psi[Jj];

    // check bounds
    if (std::isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (std::isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
      assert(0);
    }
    // correct for left-lateral fault motion
    if (left > right) {
      temp = right;
      right = left;
      left = temp;
    }

    if (abs(left-right)<1e-14) {
      _psiNext[Jj] = left;
    }
    else {
      ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
      ierr = rootFinder.findRoot(this,Jj,_psi[Jj],&_psiNext[Jj]);CHKERRQ(ierr);
      // ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeFlashHeating_fd::getResid(const PetscInt Jj, const PetscScalar state, PetscScalar* out)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = flashHeating_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _Vw[Jj], _fw, _Dc[Jj], _a[Jj], _b[Jj], _f0, _v0);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));
  return ierr;
}


// for methods that require a Jacobian, such as bracketed Newton.
PetscErrorCode ComputeFlashHeating_fd::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;

  PetscScalar G = flashHeating_psi((_psiPrev[Jj] + state)/2.0, _slipVel[Jj], _Vw[Jj], _fw, _Dc[Jj], _a[Jj], _b[Jj], _f0, _v0);
  PetscScalar temp = state - _psiPrev[Jj];
  *out = -2 * _deltaT * G + temp;

  PetscScalar A = abs(_slipVel[Jj]) / 2. / _v0 * exp((_psiPrev[Jj] + state) / 2.0 / _a[Jj]);
  *J = 1 + _deltaT * abs(_slipVel[Jj]) / _Dc[Jj] * A / sqrt(1 + A * A);

  assert(!std::isnan(*out));
  assert(!std::isinf(*out));

  return ierr;
}


// common rate-and-state functions

// state evolution law: aging law, state variable: psi
PetscScalar agingLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  PetscScalar A = exp( (double) (f0-psi)/b );
  PetscScalar dstate = 0.;
  if ( !std::isinf(A) && b>1e-3 ) {
    dstate = (PetscScalar) (b*v0/Dc)*( A - slipVel/v0 );
  }
  assert(!std::isnan(dstate));
  assert(!std::isinf(dstate));
  return dstate;
}


// applies the aging law to a Vec
PetscErrorCode agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  const PetscScalar *psiA,*slipVelA,*aA,*bA,*DcA;
  VecGetArray(dstate,&dstateA);
  VecGetArrayRead(psi,&psiA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(a,&aA);
  VecGetArrayRead(b,&bA);
  VecGetArrayRead(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_psi(psiA[Jj], slipVelA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    if ( std::isnan(dstateA[Jj]) || std::isinf(dstateA[Jj]) ) {
      PetscPrintf(PETSC_COMM_WORLD,"[%i]: dpsi = %g, psi = %g, slipVel = %g, a = %g, b = %g, f0 = %g, v0 = %g, Dc = %g\n",
      Jj,dstateA[Jj], psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
      assert(!std::isnan(dstateA[Jj]));
      assert(!std::isinf(dstateA[Jj]));
    }
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArrayRead(psi,&psiA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(a,&aA);
  VecRestoreArrayRead(b,&bA);
  VecRestoreArrayRead(Dc,&DcA);

  return ierr;
}


// state evolution law: aging law, state variable: theta
PetscScalar agingLaw_theta(const PetscScalar& theta, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar dstate = 1. - theta*abs(slipVel)/Dc;

  assert(!std::isnan(dstate));
  assert(!std::isinf(dstate));
  return dstate;
}


// applies the aging law to a Vec
PetscErrorCode agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  const PetscScalar *thetaA,*slipVelA,*DcA;
  VecGetArray(dstate,&dstateA);
  VecGetArrayRead(theta,&thetaA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArrayRead(theta,&thetaA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(Dc,&DcA);

  return ierr;
}

// state evolution law: slip law, state variable: psi
PetscScalar slipLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  if (slipVel == 0) { return 0.0; }

  PetscScalar absV = abs(slipVel);
  PetscScalar fss = f0 + (a-b)*log(absV/v0); // not regularized
  //~ PetscScalar fss =(a-b)*asinh( (double) absV/v0/2.0 * exp(f0/(a-b)));  // regularized

  //~ PetscScalar f = psi + a*log(absV/v0); // not regularized
  PetscScalar f = a*asinh( (double) (absV/2./v0)*exp(psi/a) ); // regularized

  PetscScalar dstate = -absV/Dc * (f - fss);

  assert(!std::isnan(dstate));
  assert(!std::isinf(dstate));
  return dstate;
}


// applies the state law to a Vec
PetscErrorCode slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  const PetscScalar *psiA,*slipVelA,*aA,*bA,*DcA;
  VecGetArray(dstate,&dstateA);
  VecGetArrayRead(psi,&psiA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(a,&aA);
  VecGetArrayRead(b,&bA);
  VecGetArrayRead(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_psi(psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArrayRead(psi,&psiA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(a,&aA);
  VecRestoreArrayRead(b,&bA);
  VecRestoreArrayRead(Dc,&DcA);

  return ierr;
}


// state evolution law: slip law, state variable: theta
PetscScalar slipLaw_theta(const PetscScalar& state, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar A = state*slipVel/Dc;
  PetscScalar dstate = 0.;
  if (A != 0.) { dstate = -A*log(A); }

  assert(!std::isnan(dstate));
  assert(!std::isinf(dstate));
  return dstate;
}

// applies the slip law to a Vec
PetscErrorCode slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  const PetscScalar *thetaA,*slipVelA,*DcA;
  VecGetArray(dstate,&dstateA);
  VecGetArrayRead(theta,&thetaA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArrayRead(theta,&thetaA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(Dc,&DcA);

  return ierr;
}


// flash heating: compute Vw
PetscScalar flashHeating_Vw(const PetscScalar& T, const PetscScalar& rho, const PetscScalar& c, const PetscScalar& k, const PetscScalar& D, const PetscScalar& Tw, const PetscScalar& tau_c)
{
  PetscScalar rc = rho * c;
  PetscScalar ath = k/rc;
  PetscScalar Vw = (M_PI*ath/D) * pow(rc*(Tw-T)/tau_c,2.);
  return Vw;
}

// flash heating state evolution law
PetscScalar flashHeating_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& Vw, const PetscScalar& fw, const PetscScalar& Dc,const PetscScalar& a,const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0)
{
  //~ if (slipVel == 0) { return 0.0; }

  PetscScalar absV = abs(slipVel);

  // compute fss
  PetscScalar fLV = f0 + (a-b)*log(absV/v0); // not regularized
  //~ PetscScalar fLV =(a-b)*asinh( (double) absV/v0/2.0 * exp(f0/(a-b)));  // regularized
  PetscScalar fss = fLV;

  // compute f
  if (absV > Vw) {
    fss = fw + (fLV - fw)*(Vw/absV);
  }

  //~ PetscScalar f = psi + a*log(absV/v0); // not regularized
  PetscScalar f = a*asinh( (double) (absV/2./v0)*exp(psi/a) ); // regularized

  PetscScalar dpsi = -absV/Dc *(f - fss);

  assert(!std::isnan(dpsi));
  assert(!std::isinf(dpsi));
  return dpsi;
}


// applies the flash heating state law to a Vec
PetscErrorCode flashHeating_psi_Vec(Vec &dpsi,const Vec& psi, const Vec& slipVel, const Vec& T, const Vec& rho, const Vec& c, const Vec& k, Vec& Vw, const PetscScalar& D, const Vec& Tw, const PetscScalar& tau_c, const PetscScalar& fw, const Vec& Dc,const Vec& a,const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const string _VwType)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dpsiA,*VwA;
  const PetscScalar *psiA,*slipVelA,*DcA,*TA,*TwA,*rhoA,*cA,*kA,*aA,*bA;
  VecGetArray(dpsi,&dpsiA);
  VecGetArray(Vw,&VwA);
  VecGetArrayRead(psi,&psiA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(T,&TA);
  VecGetArrayRead(rho,&rhoA);
  VecGetArrayRead(c,&cA);
  VecGetArrayRead(k,&kA);
  VecGetArrayRead(Tw,&TwA);
  VecGetArrayRead(Dc,&DcA);
  VecGetArrayRead(a,&aA);
  VecGetArrayRead(b,&bA);

  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index

  if (_VwType.compare("constant") == 0) {
    for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
      dpsiA[Jj] = flashHeating_psi(psiA[Jj],slipVelA[Jj],VwA[Jj],fw,DcA[Jj],aA[Jj],bA[Jj],f0,v0);
      Jj++;
    }
  }
  else if (_VwType.compare("function_of_Tw") == 0) {
    for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
      VwA[Jj] = flashHeating_Vw(TA[Jj], rhoA[Jj],cA[Jj],kA[Jj],D, TwA[Jj], tau_c);
      dpsiA[Jj] = flashHeating_psi(psiA[Jj],slipVelA[Jj],VwA[Jj],fw,DcA[Jj],aA[Jj],bA[Jj],f0,v0);
      Jj++;
    }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"_VwType not understood!\n");
    assert(0);
  }

  VecRestoreArray(dpsi,&dpsiA);
  VecRestoreArray(Vw,&VwA);
  VecRestoreArrayRead(psi,&psiA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(T,&TA);
  VecRestoreArrayRead(rho,&rhoA);
  VecRestoreArrayRead(c,&cA);
  VecRestoreArrayRead(k,&kA);
  VecRestoreArrayRead(Tw,&TwA);
  VecRestoreArrayRead(Dc,&DcA);
  VecRestoreArrayRead(a,&aA);
  VecRestoreArrayRead(b,&bA);

  return ierr;
}


// frictional strength, regularized form, for state variable psi
PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& v0)
{
  PetscScalar strength = (PetscScalar) a*sN*asinh( (double) (0.5*slipVel/v0)*exp(psi/a) );
  return strength;
}


// computes frictional strength by appying strength_psi to a vector
PetscErrorCode strength_psi_Vec(Vec& strength, const Vec& psi, const Vec& slipVel, const Vec& a,  const Vec& sN, const PetscScalar& v0)
{
  PetscErrorCode ierr = 0;

  PetscScalar *strengthA;
  const PetscScalar *psiA,*slipVelA,*aA,*sNA;
  VecGetArray(strength,&strengthA);
  VecGetArrayRead(psi,&psiA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(a,&aA);
  VecGetArrayRead(sN,&sNA);

  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(strength,&Istart,&Iend); // local portion of global Vec index

  for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
    strengthA[Jj] = strength_psi( sNA[Jj], psiA[Jj], slipVelA[Jj], aA[Jj], v0);
    Jj++;
  }

  VecRestoreArray(strength,&strengthA);
  VecRestoreArrayRead(psi,&psiA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(a,&aA);
  VecRestoreArrayRead(sN,&sNA);

  return ierr;
}
