#include "newFault.hpp"

#define FILENAME "newFault.cpp"

using namespace std;


NewFault::NewFault(Domain&D,VecScatter& scatter2fault)
: _D(&D),_inputFile(D._file),_delim(D._delim),_outputDir(D._outputDir),
  _stateLaw("agingLaw"),
  _N(D._Nz),_L(D._Lz),
  _f0(0.6),_v0(1e-6),
  _sigmaN_cap(1e14),_sigmaN_floor(0.),
  _fw(0.64),_Vw(0.12),_tau_c(3),_Tw(1173),_D_fh(5),
  _rootTol(1e-9),_rootIts(0),_maxNumIts(1e4),
  _computeVelTime(0),_stateLawTime(0),
  _body2fault(&scatter2fault)
{
  #if VERBOSE > 1
    std::string funcName = "NewFault::NewFault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_inputFile);
  checkInput();
  setFields(D);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PetscErrorCode NewFault::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in fault.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
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

    if (var.compare("DcVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcVals);
    }
    else if (var.compare("DcDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcDepths);
    }

    else if (var.compare("sNVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNVals);
    }
    else if (var.compare("sNDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNDepths);
    }
    else if (var.compare("sN_cap")==0) {
      _sigmaN_cap = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("sN_floor")==0) {
      _sigmaN_floor = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("aVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_aVals);
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
    else if (var.compare("cohesionVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cohesionVals);
    }
    else if (var.compare("cohesionDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cohesionDepths);
    }
    else if (var.compare("muVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_muVals);
    }
    else if (var.compare("muDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_muDepths);
    }

    else if (var.compare("rhoVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoVals);
    }
    else if (var.compare("rhoDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rhoDepths);
    }


    else if (var.compare("stateLaw")==0) {
      _stateLaw = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // tolerance for nonlinear solve
    else if (var.compare("rootTol")==0) {
      _rootTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // friction parameters
    else if (var.compare("f0")==0) {
      _f0 = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("v0")==0) {
      _v0 = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // flash heating parameters
    else if (var.compare("fw")==0) {
      _fw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("Vw")==0) {
      _Vw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("Tw")==0) {
      _Tw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("D")==0) {
      _D_fh = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("tau_c")==0) {
      _tau_c = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // for locking part of the fault
    else if (var.compare("lockedVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_lockedVals);
    }
    else if (var.compare("lockedDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_lockedDepths);
    }

  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in fault.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// parse input file and load values into data members
PetscErrorCode NewFault::loadFieldsFromFiles(std::string inputDir)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting NewFault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
#endif

  // load normal stress: _sNEff
  //~string vecSourceFile = _inputDir + "sigma_N";
  ierr = loadVecFromInputFile(_sNEff,inputDir,"sNEff"); CHKERRQ(ierr);

  // load state: psi
  ierr = loadVecFromInputFile(_psi,inputDir,"psi"); CHKERRQ(ierr);

  // load slip
  ierr = loadVecFromInputFile(_slip,inputDir,"slip"); CHKERRQ(ierr);

  // load quasi-static shear stress
  ierr = loadVecFromInputFile(_tauQSP,inputDir,"tauQS"); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending NewFault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode NewFault::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::checkInput";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_DcVals.size() == _DcDepths.size() );
  assert(_aVals.size() == _aDepths.size() );
  assert(_bVals.size() == _bDepths.size() );
  assert(_sigmaNVals.size() == _sigmaNDepths.size() );
  assert(_cohesionVals.size() == _cohesionDepths.size() );
  assert(_rhoVals.size() == _rhoDepths.size() );
  assert(_muVals.size() == _muDepths.size() );


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

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode NewFault::setFields(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // Allocate fields. All fields in this class match the parallel structure of Domain's y0 Vec
  VecDuplicate(D._y0,&_tauP);     PetscObjectSetName((PetscObject) _tauP, "tau"); VecSet(_tauP,0.0);
  VecDuplicate(_tauP,&_tauQSP);     PetscObjectSetName((PetscObject) _tauQSP, "tauQS");  VecSet(_tauQSP,0.0);
  VecDuplicate(_tauP,&_psi);      PetscObjectSetName((PetscObject) _psi, "psi"); VecSet(_psi,0.0);
  VecDuplicate(_tauP,&_dPsi);     PetscObjectSetName((PetscObject) _dPsi, "dPsi"); VecSet(_dPsi,0.0);
  VecDuplicate(_tauP,&_slip);     PetscObjectSetName((PetscObject) _slip, "slip"); VecSet(_slip,0.0);
  VecDuplicate(_tauP,&_slipVel);  PetscObjectSetName((PetscObject) _slipVel, "slipVel"); VecSet(_slipVel,0.0);
  VecDuplicate(_tauP,&_Dc);       PetscObjectSetName((PetscObject) _Dc, "Dc");
  VecDuplicate(_tauP,&_a);        PetscObjectSetName((PetscObject) _a, "a");
  VecDuplicate(_tauP,&_b);        PetscObjectSetName((PetscObject) _b, "b");
  VecDuplicate(_tauP,&_cohesion); PetscObjectSetName((PetscObject) _cohesion, "cohesion"); VecSet(_cohesion,0);
  VecDuplicate(_tauP,&_sN);       PetscObjectSetName((PetscObject) _sN, "sN");
  VecDuplicate(_tauP,&_sNEff);    PetscObjectSetName((PetscObject) _sNEff, "sNEff");
  VecDuplicate(_tauP,&_z);        PetscObjectSetName((PetscObject) _z, "z_fault");
  VecDuplicate(_tauP,&_rho);      PetscObjectSetName((PetscObject) _rho, "rho_fault");
  VecDuplicate(_tauP,&_mu);       PetscObjectSetName((PetscObject) _mu, "mu_fault");
  VecDuplicate(_tauP,&_locked);       PetscObjectSetName((PetscObject) _locked, "locked");

  // create z from D._z
  VecScatterBegin(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);


  if (_stateLaw.compare("flashHeating") == 0) {
    VecDuplicate(_tauP,&_T);    PetscObjectSetName((PetscObject) _T, "T_fault");
    VecDuplicate(_tauP,&_k);    PetscObjectSetName((PetscObject) _k, "k_fault");
    VecDuplicate(_tauP,&_c);  PetscObjectSetName((PetscObject) _c, "c_fault");
  }
  else { _T = NULL; _k = NULL; _c = NULL;}

  // set fields
  ierr = setVec(_a,_z,_aVals,_aDepths); CHKERRQ(ierr);
  ierr = setVec(_b,_z,_bVals,_bDepths); CHKERRQ(ierr);
  ierr = setVec(_sN,_z,_sigmaNVals,_sigmaNDepths); CHKERRQ(ierr);
  ierr = setVec(_Dc,_z,_DcVals,_DcDepths); CHKERRQ(ierr);
  if (_lockedVals.size() > 0 ) { ierr = setVec(_locked,_z,_lockedVals,_lockedDepths); CHKERRQ(ierr); }
  else { VecSet(_locked,0.); }
  if (_cohesionVals.size() > 0 ) { ierr = setVec(_cohesion,_z,_cohesionVals,_cohesionDepths); CHKERRQ(ierr); }
  {
    Vec temp; VecDuplicate(_D->_y,&temp);
    ierr = setVec(temp,_D->_z,_rhoVals,_rhoDepths); CHKERRQ(ierr);
    VecScatterBegin(*_body2fault, temp, _rho, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, temp, _rho, INSERT_VALUES, SCATTER_FORWARD);

    ierr = setVec(temp,_D->_z,_muVals,_muDepths); CHKERRQ(ierr);
    VecScatterBegin(*_body2fault, temp, _mu, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, temp, _mu, INSERT_VALUES, SCATTER_FORWARD);

    VecDestroy(&temp);
  }
  ierr = VecSet(_psi,_f0);CHKERRQ(ierr);

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
PetscErrorCode NewFault::setThermalFields(const Vec& T, const Vec& k, const Vec& c)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setThermalFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  VecScatterBegin(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, k, _k, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, k, _k, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, c, _c, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, c, _c, INSERT_VALUES, SCATTER_FORWARD);

   #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// update temperature on the fault based on the temperature body field
PetscErrorCode NewFault::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::updateTemperature";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  VecScatterBegin(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, T, _T, INSERT_VALUES, SCATTER_FORWARD);

   #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// update shear stress on the fault based on the stress sxy body field
PetscErrorCode NewFault::setTauQS(const Vec& sxy)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setTauQS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  VecScatterBegin(*_body2fault, sxy, _tauQSP, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, sxy, _tauQSP, INSERT_VALUES, SCATTER_FORWARD);

   #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// use pore pressure to compute total normal stress
// sNEff = sN - rho*g*z - dp
// sNEff sigma Normal effective
PetscErrorCode NewFault::setSN(const Vec& p)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setSN";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  ierr = VecWAXPY(_sN,1.,p,_sNEff); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute effective normal stress from total and pore pressure:
// sNEff = sN - rho*g*z - dp
PetscErrorCode NewFault::setSNEff(const Vec& p)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::setSNEff";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  ierr = VecWAXPY(_sNEff,-1.,p,_sN); CHKERRQ(ierr);
    //~ sNEff[Jj] = sN[Jj] - p[Jj];

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode NewFault::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"NewFault Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   compute slip vel time (s): %g\n",_computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   state law time (s): %g\n",_stateLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent finding slip vel law: %g\n",(_computeVelTime/totRunTime)*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in state law: %g\n",(_stateLawTime/totRunTime)*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode NewFault::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer    viewer;

  // write out scalar info
  std::string str = outputDir + "fault_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"rootTol = %.15e\n",_rootTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"f0 = %.15e\n",_f0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %.15e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stateEvolutionLaw = %s\n",_stateLaw.c_str());CHKERRQ(ierr);
  if (!_stateLaw.compare("flashHeating")) {
    ierr = PetscViewerASCIIPrintf(viewer,"fw = %.15e\n",_fw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Vw = %.15e\n",_Vw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"tau_c = %.15e # (GPa)\n",_tau_c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Tw = %.15e # (K)\n",_Tw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"D = %.15e # (um)\n",_D);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  // output vector fields

  str = outputDir + "fault_z";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_z,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "fault_a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "fault_b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output normal stress vector
  str =  outputDir + "fault_sNEff";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sNEff,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output critical distance
  str =  outputDir + "fault_Dc";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_Dc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output cohesion
  str =  outputDir + "fault_cohesion";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_cohesion,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output where fault is locked
  str =  outputDir + "fault_locked";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_locked,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode NewFault::writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::writeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (stepCount == 0) {
    ierr = io_initiateWriteAppend(_viewers, "slip", _slip, outputDir + "slip"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "slipVel", _slipVel, outputDir + "slipVel"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "tauQSP", _tauQSP, outputDir + "tauQSP"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "psi", _psi, outputDir + "psi"); CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_slip,_viewers["slip"].first); CHKERRQ(ierr);
    ierr = VecView(_slipVel,_viewers["slipVel"].first); CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_viewers["tauQSP"].first); CHKERRQ(ierr);
    ierr = VecView(_psi,_viewers["psi"].first); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault::writeStep(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::writeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  writeStep(stepCount,time,_outputDir);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


NewFault::~NewFault()
{
  #if VERBOSE > 1
    std::string funcName = "NewFault::~NewFault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // fields that exist on the fault
  VecDestroy(&_tauQSP);
  VecDestroy(&_tauP);
  VecDestroy(&_psi);
  VecDestroy(&_dPsi);
  VecDestroy(&_slip);
  VecDestroy(&_slipVel);
  VecDestroy(&_z);

  // frictional fields
  VecDestroy(&_Dc);
  VecDestroy(&_a);
  VecDestroy(&_b);
  VecDestroy(&_sNEff);
  VecDestroy(&_sN);
  VecDestroy(&_cohesion);

  for (map<string,std::pair<PetscViewer,string> >::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first].first);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PetscErrorCode NewFault::computeTauRS(Vec& tauRS, const PetscScalar vL)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 2
    std::string funcName = "NewFault::computeTauRS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (tauRS == NULL) { VecDuplicate(_slipVel,&tauRS); }

  PetscInt       Istart,Iend;
  PetscScalar   *tauRSV,*sN,*a,*b;
  VecGetOwnershipRange(tauRS,&Istart,&Iend);
  VecGetArray(tauRS,&tauRSV);
  VecGetArray(_sNEff,&sN);
  VecGetArray(_a,&a);
  VecGetArray(_b,&b);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    tauRSV[Jj] = sN[Jj]*a[Jj]*asinh( (double) 0.5*vL*exp(_f0/a[Jj])/_v0 );
    //~ PetscScalar f = _f0 + (a[Jj] - b[Jj]) * log(vL/_v0);
    //~ tauRSV[Jj] = sN[Jj] * f;
    Jj++;
  }
  VecRestoreArray(tauRS,&tauRSV);
  VecRestoreArray(_sNEff,&sN);
  VecRestoreArray(_a,&a);

  VecSet(_slipVel,vL);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



//================= Functions assuming only + side exists ========================
NewFault_qd::NewFault_qd(Domain&D,VecScatter& scatter2fault)
: NewFault(D,scatter2fault)
{
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::NewFault_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // radiation damping parameter: 0.5 * sqrt(mu*rho)
  VecDuplicate(_tauP,&_eta_rad);  PetscObjectSetName((PetscObject) _eta_rad, "eta_rad");
  VecPointwiseMult(_eta_rad,_mu,_rho);
  VecSqrtAbs(_eta_rad);
  VecScale(_eta_rad,0.5);

  if (D._loadICs==1) {
    //~ loadFieldsFromFiles(D._inputDir);
    loadVecFromInputFile(_eta_rad,D._inputDir,"eta_rad");
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

NewFault_qd::~NewFault_qd()
{
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::~NewFault_qd";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecDestroy(&_eta_rad);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PetscErrorCode NewFault_qd::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "NewFault_qd::loadSettings";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // nothing to do yet

  //~ PetscMPIInt rank,size;
  //~ MPI_Comm_size(PETSC_COMM_WORLD,&size);
  //~ MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  //~ ifstream infile( file );
  //~ string line,var;
  //~ size_t pos = 0;
  //~ while (getline(infile, line))
  //~ {
    //~ istringstream iss(line);
    //~ pos = line.find(_delim); // find position of the delimiter
    //~ var = line.substr(0,pos);


  //~ }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_qd::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated explicitly into varEx
  Vec varPsi; VecDuplicate(_psi,&varPsi); VecCopy(_psi,varPsi);
  varEx["psi"] = varPsi;

  // slip is added by the momentum balance equation


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_qd::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("psi")->second,_psi);
  VecCopy(varEx.find("slip")->second,_slip);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// assumes right-lateral fault
PetscErrorCode NewFault_qd::computeVel()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initialize struct to solve for the slip velocity
  PetscScalar *etaA, *tauQSA, *sNA, *psiA, *aA,*bA,*slipVelA,*lockedA;
  VecGetArray(_slipVel,&slipVelA);
  VecGetArray(_eta_rad,&etaA);
  VecGetArray(_tauQSP,&tauQSA);
  VecGetArray(_sNEff,&sNA);
  VecGetArray(_psi,&psiA);
  VecGetArray(_a,&aA);
  VecGetArray(_b,&bA);
  VecGetArray(_locked,&lockedA);
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);

  PetscInt N = Iend - Istart;
  ComputeVel_qd temp(N,etaA,tauQSA,sNA,psiA,aA,bA,_v0,lockedA);
  ierr = temp.computeVel(slipVelA, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecGetArray(_slipVel,&slipVelA);
  VecGetArray(_eta_rad,&etaA);
  VecGetArray(_tauQSP,&tauQSA);
  VecGetArray(_sNEff,&sNA);
  VecGetArray(_psi,&psiA);
  VecGetArray(_a,&aA);
  VecGetArray(_b,&bA);
  VecGetArray(_locked,&lockedA);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode NewFault_qd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // compute rate of state variable
  Vec dstate = dvarEx.find("psi")->second;
  double startTime = MPI_Wtime();
  if (_stateLaw.compare("agingLaw") == 0) {
    //~ ierr = agingLaw_theta_Vec(dstate, _theta, _slipVel, _Dc) CHKERRQ(ierr);
    ierr = agingLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc); CHKERRQ(ierr);
  }
  else if (_stateLaw.compare("slipLaw") == 0) {
    //~ ierr = slipLaw_theta_Vec(dstate, _theta, _slipVel, _Dc); CHKERRQ(ierr);
    ierr =  slipLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc);  CHKERRQ(ierr);
  }
  //~ else if (_stateLaw.compare("flashHeating") == 0) {
    //~ ierr = flashHeating_psi(Ii,psi,dpsi);CHKERRQ(ierr);
  //~ }
  else if (_stateLaw.compare("constantState") == 0) {
    VecSet(dstate,0.);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n");
    assert(0);
  }
  _stateLawTime += MPI_Wtime() - startTime;


  // compute slip velocity
  startTime = MPI_Wtime();
  ierr = computeVel();CHKERRQ(ierr);
  VecCopy(_slipVel,dvarEx["slip"]);
  _computeVelTime += MPI_Wtime() - startTime;



  // set tauP = tauQS - eta_rad *slipVel
  VecCopy(_slipVel,_tauP);
  VecPointwiseMult(_tauP,_eta_rad,_tauP);
  VecAYPX(_tauP,-1.0,_tauQSP);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_qd::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  NewFault::writeContext(outputDir);

  PetscViewer    viewer;

  // output vector fields

  std::string str = outputDir + "fault_eta_rad";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_eta_rad,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}





//======================================================================
// functions for structs
//======================================================================

ComputeVel_qd::ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0,const PetscScalar* locked)
: _a(a),_b(b),_sN(sN),_tauQS(tauQS),_eta(eta),_psi(psi),_N(N),_v0(v0),_locked(locked)
{ }

PetscErrorCode ComputeVel_qd::computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar left, right, out, temp;
  for (PetscInt Jj = 0; Jj< _N; Jj++) {

    if (_locked[Jj] > 0.) { // if fault is locked, hold slip velocity at 0
      slipVelA[Jj] = 0.;
      break;
    }

    left = 0.;
    right = _tauQS[Jj] / _eta[Jj];

    // check bounds
    if (isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      PetscPrintf(PETSC_COMM_WORLD,"tauQS = %g, eta = %g, left = %g\n",_tauQS[Jj],_eta[Jj],left);
      assert(0);
    }
    if (isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
      PetscPrintf(PETSC_COMM_WORLD,"tauQS = %g, eta = %g, right = %g\n",_tauQS[Jj],_eta[Jj],right);
      assert(0);
    }
    // correct for left-lateral fault motion
    if (left > right) {
      temp = right;
      right = left;
      left = temp;
    }

    out = slipVelA[Jj];
    if (abs(left-right)<1e-14) { out = left; }
    else {
      //~ Bisect rootFinder(maxNumIts,rootTol);
      //~ ierr = rootFinder.setBounds(left,right); CHKERRQ(ierr);
      //~ ierr = rootFinder.findRoot(this,Jj,&out); assert(ierr == 0); CHKERRQ(ierr);
      //~ rootIts += rootFinder.getNumIts();

      PetscScalar x0 = slipVelA[Jj];
      BracketedNewton rootFinder(maxNumIts,rootTol);
      ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
      ierr = rootFinder.findRoot(this,Jj,x0,&out); assert(ierr == 0); CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
    slipVelA[Jj] = out;
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

  PetscScalar strength = strength_psi(_sN[Jj], _psi[Jj], vel, _a[Jj], _v0); // frictional strength
  PetscScalar stress = _tauQS[Jj] - _eta[Jj]*vel; // stress on fault

  *out = strength - stress;
  assert(!isnan(*out));
  assert(!isinf(*out));
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
  *J = A*B/sqrt(B*B*vel*vel + 1.) + _eta[Jj]; // derivative with respect to slipVel

  assert(!isnan(*out)); assert(!isinf(*out));
  assert(!isnan(*J)); assert(!isinf(*J));
  return ierr;
}


//================= Functions assuming only + side exists ========================
NewFault_dyn::NewFault_dyn(Domain&D, VecScatter& scatter2fault)
: NewFault(D, scatter2fault), _Phi(NULL), _slipPrev(NULL),_rhoLocal(NULL),
  _alphay(D._alphay), _alphaz(D._alphaz)
{
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::NewFault_dyn";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

  VecDuplicate(_tauQSP,&_slipPrev); PetscObjectSetName((PetscObject) _slipPrev, "slipPrev");VecSet(_slipPrev,0.0);
  VecDuplicate(_tauQSP,&_Phi); PetscObjectSetName((PetscObject) _Phi, "Phi");VecSet(_Phi, 0.0);
  VecDuplicate(_tauQSP,&_constraints_factor); PetscObjectSetName((PetscObject) _constraints_factor, "constraintsFactor");VecSet(_constraints_factor, 0.0);
  VecDuplicate(_tauQSP,&_an); PetscObjectSetName((PetscObject) _an, "an");VecSet(_an, 0.0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

NewFault_dyn::~NewFault_dyn()
{
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::~NewFault_dyn";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // this is covered by the NewFault destructor.

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PetscErrorCode NewFault_dyn::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::~loadSettings";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
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

    // Tau dynamic parameters
    if (var.compare("tCenterTau")==0) {
      _tCenterTau = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("zCenterTau")==0) {
      _zCenterTau = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("tStdTau")==0) {
      _tStdTau = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("zStdTau")==0) {
      _zStdTau = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("ampTau")==0) {
      _ampTau = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated explicitly into varEx
  Vec varPsi; VecDuplicate(_psi,&varPsi); VecCopy(_psi,varPsi);
  varEx["psi"] = varPsi;

  // slip is added by the momentum balance equation


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("psi")->second,_psi);
  VecCopy(varEx.find("slip")->second,_slip);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// assumes right-lateral fault
PetscErrorCode NewFault_dyn::computeVel()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initialize struct to solve for the slip velocity
  PetscScalar *Phi, *an, *psi, *constraints_factor, *a,*sneff, *slipVel;
  VecGetArray(_Phi,&Phi);
  VecGetArray(_an,&an);
  VecGetArray(_psi,&psi);
  VecGetArray(_constraints_factor,&constraints_factor);
  VecGetArray(_a,&a);
  VecGetArray(_sNEff,&sneff);
  VecGetArray(_slipVel,&slipVel);

  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;
  ComputeVel_dyn temp(N,Phi,an,psi,constraints_factor,a,sneff, _v0);
  ierr = temp.computeVel(slipVel, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecRestoreArray(_Phi,&Phi);
  VecRestoreArray(_an,&an);
  VecRestoreArray(_psi,&psi);
  VecRestoreArray(_constraints_factor,&constraints_factor);
  VecRestoreArray(_a,&a);
  VecRestoreArray(_sNEff,&sneff);
  VecRestoreArray(_slipVel,&slipVel);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::computeAgingLaw()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initialize struct to solve for the slip velocity
  PetscScalar *Dc, *b, *psi, *slipVel, *slipPrev;
  VecGetArray(_Dc,&Dc);
  VecGetArray(_b,&b);
  VecGetArray(_psi,&psi);
  VecGetArray(_slipVel,&slipVel);
  VecGetArray(_slipPrev,&slipPrev);
  PetscInt Istart, Iend;

  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;
  ComputeAging_dyn temp(N,Dc,b,psi,slipVel,slipPrev, _v0, _deltaT, _f0);
  ierr = temp.computeAging(_rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecRestoreArray(_Dc,&Dc);
  VecRestoreArray(_b,&b);
  VecRestoreArray(_psi,&psi);
  VecRestoreArray(_slipVel,&slipVel);
  VecRestoreArray(_slipPrev,&slipPrev);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::initiateIntegrand_dyn(map<string,Vec>& varEx, Vec _rhoVec)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

 // put variables to be integrated explicitly into varEx
  Vec u, uPrev, du;
  VecDuplicate(_tauQSP, &u);
  VecDuplicate(_tauQSP, &du);
  VecDuplicate(_tauQSP, &uPrev);
  VecSet(u, 0);
  VecSet(du, 0);
  VecSet(uPrev, 0);
  varEx["uFault"] = u;
  varEx["uPrevFault"] = uPrev;
  varEx["duFault"] = du;

  VecDuplicate(_tauQSP, &_rhoLocal);
  VecSet(_rhoLocal, 0);

  PetscInt indexes[_N];
  PetscInt Ii;
  VecScatter scattu, scattuPrev, scattrho;
  for (Ii=0; Ii<_N; Ii++){indexes[Ii] = Ii;}

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,_N,indexes, PETSC_COPY_VALUES,&_is);

  ierr = VecScatterCreate(varEx["u"], _is, varEx["uFault"], _is, &scattu);
  VecScatterBegin(scattu, varEx["u"], varEx["uFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scattu, varEx["u"], varEx["uFault"], INSERT_VALUES, SCATTER_FORWARD);

  ierr = VecScatterCreate(varEx["uPrev"], _is, varEx["uPrevFault"], _is, &scattuPrev);
  VecScatterBegin(scattuPrev, varEx["uPrev"], varEx["uPrevFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scattuPrev, varEx["uPrev"], varEx["uPrevFault"], INSERT_VALUES, SCATTER_FORWARD);

  ierr = VecScatterCreate(_rhoVec, _is, _rhoLocal, _is, &scattrho);
  VecScatterBegin(scattrho, _rhoVec, _rhoLocal, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scattrho, _rhoVec, _rhoLocal, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterDestroy(&scattu);
  VecScatterDestroy(&scattuPrev);
  VecScatterDestroy(&scattrho);

  // slip is added by the momentum balance equation
  //~ Vec varSlip; VecDuplicate(_slip,&varSlip); VecCopy(_slip,varSlip);
  //~ varEx["slip"] = varSlip;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::updateTau(const PetscScalar currT){
  PetscScalar *zz, *tauQS;

  PetscInt Ii, IBegin, IEnd;
  PetscInt Jj = 0;
  VecGetArray(_z, &zz);
  VecGetArray(_tauQSP, &tauQS);
  VecGetOwnershipRange(_z, &IBegin, &IEnd);

  PetscScalar timeOffset = exp(-pow((currT - _tCenterTau), 2) / pow(_tStdTau, 2));
  for (Ii=IBegin;Ii<IEnd;Ii++){
    tauQS[Jj] = _ampTau * exp(-pow((zz[Jj] - _zCenterTau), 2) / pow(_zStdTau, 2)) * timeOffset;
    Jj++;
  }
  VecRestoreArray(_z, &zz);
  VecRestoreArray(_tauQSP, &tauQS);
  return 0;
}


PetscErrorCode NewFault_dyn::d_dt(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar deltaT)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

// compute slip velocity
ierr = setPhi(varEx, dvarEx, deltaT);
double startTime = MPI_Wtime();
ierr = computeVel();CHKERRQ(ierr);
double intermediateTime = MPI_Wtime();

_computeVelTime += intermediateTime - startTime;
PetscInt       Ii,Istart,Iend;
PetscInt       Jj = 0;
PetscScalar    *u, *uPrev, uTemp, *rho, *sigma_N, *a, *an, *slip, *slipVel, fric, alpha, A, *vel, *Phi, *psi;
ierr = VecGetOwnershipRange(varEx["uFault"],&Istart,&Iend);CHKERRQ(ierr);
ierr = VecGetArray(varEx["uFault"], &u);
ierr = VecGetArray(varEx["uPrevFault"], &uPrev);
ierr = VecGetArray(_an, &an);
ierr = VecGetArray(_slipVel, &slipVel);
ierr = VecGetArray(_rhoLocal, &rho);
ierr = VecGetArray(varEx["psi"], &psi);
ierr = VecGetArray(_sNEff, &sigma_N);
ierr = VecGetArray(_a, &a);
ierr = VecGetArray(_Phi, &Phi);
ierr = VecGetArray(varEx["slip"], &slip);
ierr = VecGetArray(dvarEx["slip"], &vel);
PetscScalar *b;
ierr = VecGetArray(_b, &b);

for (Ii = Istart; Ii<Iend; Ii++){
  if (slipVel[Jj] < 1e-14){
    uTemp = uPrev[Jj];
    uPrev[Jj] = u[Jj];
    u[Jj] = 2 * u[Jj] - uTemp + _deltaT * _deltaT / rho[Jj] * an[Jj];
    vel[Jj] = 0;
    slipVel[Jj] = 0;
  }
  else{
    fric = (PetscScalar) a[Jj]*sigma_N[Jj]*asinh( (double) (slipVel[Jj]/2./_v0)*exp(psi[Jj]/a[Jj]) );
    alpha = 1 / (rho[Jj] * _alphay) * fric / slipVel[Jj];
    A = 1 + alpha * _deltaT;
    uTemp = uPrev[Jj];
    uPrev[Jj] = u[Jj];
    u[Jj] = (2*u[Jj] + _deltaT * _deltaT / rho[Jj] * an[Jj] + (_deltaT*alpha-1)*uTemp) /  A;
    vel[Jj] = Phi[Jj] / (1 + _deltaT*_alphay / rho[Jj] * fric / slipVel[Jj]);
    slipVel[Jj] = vel[Jj];
  }
  slip[Jj] = 2 * u[Jj];
  Jj++;
}
ierr = VecRestoreArray(varEx["uFault"], &u);
ierr = VecRestoreArray(varEx["uPrevFault"], &uPrev);
ierr = VecRestoreArray(_an, &an);
ierr = VecRestoreArray(_slipVel, &slipVel);
ierr = VecRestoreArray(_rhoLocal, &rho);
ierr = VecRestoreArray(varEx["psi"], &psi);
ierr = VecRestoreArray(_sNEff, &sigma_N);
ierr = VecRestoreArray(_a, &a);
ierr = VecRestoreArray(_Phi, &Phi);
ierr = VecRestoreArray(varEx["slip"], &slip);
ierr = VecRestoreArray(dvarEx["slip"], &vel);

VecScatter scattu, scattuPrev;
ierr = VecScatterCreate(varEx["uFault"], _is, varEx["u"], _is, &scattu);
VecScatterBegin(scattu, varEx["uFault"], varEx["u"], INSERT_VALUES, SCATTER_FORWARD);
VecScatterEnd(scattu, varEx["uFault"], varEx["u"], INSERT_VALUES, SCATTER_FORWARD);

ierr = VecScatterCreate(varEx["uPrevFault"], _is, varEx["uPrev"], _is, &scattuPrev);
VecScatterBegin(scattuPrev, varEx["uPrevFault"], varEx["uPrev"], INSERT_VALUES, SCATTER_FORWARD);
VecScatterEnd(scattuPrev, varEx["uPrevFault"], varEx["uPrev"], INSERT_VALUES, SCATTER_FORWARD);

VecScatterDestroy(&scattu);
VecScatterDestroy(&scattuPrev);

// compute state parameter law
double startAgingTime = MPI_Wtime();
computeAgingLaw();
_stateLawTime += MPI_Wtime() - startAgingTime;
varEx["psi"] = _psi;
#if VERBOSE > 1
PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
#endif

return ierr;
}

PetscErrorCode NewFault_dyn::setPhi(map<string,Vec>& varEx, map<string,Vec>& dvarEx, const PetscScalar deltaT)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_dyn::setPhi";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  PetscInt       Ii,IFaultStart, IFaultEnd;

  VecScatter scattdu;
  ierr = VecScatterCreate(dvarEx["u"], _is, varEx["duFault"], _is, &scattdu);
  VecScatterBegin(scattdu, dvarEx["u"], varEx["duFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scattdu, dvarEx["u"], varEx["duFault"], INSERT_VALUES, SCATTER_FORWARD);

  VecScatterDestroy(&scattdu);
  ierr = VecGetOwnershipRange(varEx["uFault"],&IFaultStart,&IFaultEnd);CHKERRQ(ierr);

  PetscScalar *u, *uPrev, *Laplacian, *rho, *psi, *sigma_N, *tauQS, *slipVel, *an, *Phi, *constraints_factor, *slipPrev, *slipVelocity;
  PetscInt Jj = 0;

  ierr = VecGetArray(varEx["uFault"], &u);
  ierr = VecGetArray(varEx["uPrevFault"], &uPrev);
  ierr = VecGetArray(varEx["duFault"], &Laplacian);
  ierr = VecGetArray(_rhoLocal, &rho);
  ierr = VecGetArray(varEx["psi"], &psi);
  ierr = VecGetArray(_sNEff, &sigma_N);
  ierr = VecGetArray(_tauQSP, &tauQS);
  ierr = VecGetArray(dvarEx["slip"], &slipVel);
  ierr = VecGetArray(_slipVel, &slipVelocity);
  ierr = VecGetArray(_slipPrev, &slipPrev);
  ierr = VecGetArray(_an, &an);
  ierr = VecGetArray(_Phi, &Phi);
  ierr = VecGetArray(_constraints_factor, &constraints_factor);

  for (Ii=IFaultStart;Ii<IFaultEnd;Ii++){
    an[Jj] = Laplacian[Jj] + tauQS[Jj] / _alphay;
    Phi[Jj] = 2 / deltaT * (u[Jj] - uPrev[Jj]) + deltaT * an[Jj] / rho[Jj];
    constraints_factor[Jj] = deltaT / _alphay / rho[Jj];
    slipPrev[Jj] = slipVel[Jj];
    slipVelocity[Jj] = slipVel[Jj];
    Jj++;
  }
  // std::cin.get();
  ierr = VecRestoreArray(varEx["uFault"], &u);
  ierr = VecRestoreArray(varEx["uPrevFault"], &uPrev);
  ierr = VecRestoreArray(varEx["duFault"], &Laplacian);
  ierr = VecRestoreArray(_rhoLocal, &rho);
  ierr = VecRestoreArray(varEx["psi"], &psi);
  ierr = VecRestoreArray(_sNEff, &sigma_N);
  ierr = VecRestoreArray(_tauQSP, &tauQS);
  ierr = VecRestoreArray(dvarEx["slip"], &slipVel);
  ierr = VecRestoreArray(_slipVel, &slipVelocity);
  ierr = VecRestoreArray(_slipPrev, &slipPrev);
  ierr = VecRestoreArray(_an, &an);
  ierr = VecRestoreArray(_Phi, &Phi);
  ierr = VecRestoreArray(_constraints_factor, &constraints_factor);

  _deltaT = deltaT;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// New class to handle the computation of the implicit solvers

ComputeVel_dyn::ComputeVel_dyn(const PetscInt N,const PetscScalar* Phi, const PetscScalar* an, const PetscScalar* psi, const PetscScalar* constraints_factor,const PetscScalar* a,const PetscScalar* sneff, const PetscScalar v0)
: _Phi(Phi),_an(an),_psi(psi),_constraints_factor(constraints_factor),_a(a),_sNEff(sneff),_N(N), _v0(v0)
{ }

PetscErrorCode ComputeVel_dyn::computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Bisect rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, out, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {

    left = 0.;
    right = abs(_Phi[Jj]);
    // check bounds
    if (isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
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
      ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
    slipVelA[Jj] = out;
    // PetscPrintf(PETSC_COMM_WORLD,"%i: left = %g, right = %g, slipVel = %g\n",Jj,left,right,out);
  }


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeVel_dyn::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out)
{
  PetscErrorCode ierr = 0;
  PetscScalar constraints = strength_psi(_sNEff[Jj], _psi[Jj], vel, _a[Jj] , _v0); // frictional strength

  constraints = _constraints_factor[Jj] * constraints;
  PetscScalar Phi_temp = _Phi[Jj];
  if (Phi_temp < 0){Phi_temp = -Phi_temp;}

  PetscScalar stress = Phi_temp - vel; // stress on fault

  *out = constraints - stress;
  assert(!isnan(*out));
  assert(!isinf(*out));
  return ierr;
}

// ================================================


ComputeAging_dyn::ComputeAging_dyn(const PetscInt N,const PetscScalar* Dc, const PetscScalar* b, PetscScalar* psi, const PetscScalar* slipVel,const PetscScalar* slipPrev, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0)
: _Dc(Dc),_b(b),_slipVel(slipVel),_slipPrev(slipPrev),_psi(psi), _N(N), _v0(v0), _deltaT(deltaT), _f0(f0)
{ }

PetscErrorCode ComputeAging_dyn::computeAging(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  RegulaFalsi rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, out, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {

    left = -10.;
    right = 10.;

    // check bounds
    if (isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_qd::computeVel: right bound evaluated to NaN.\n");
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
      ierr = rootFinder.findRoot(this,Jj,_psi[Jj],&out);CHKERRQ(ierr);
      rootIts += rootFinder.getNumIts();
    }
    _psi[Jj] = out;
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// Compute residual for equation to find slip velocity.
// This form is for root finding algorithms that don't require a Jacobian such as the bisection method.
PetscErrorCode ComputeAging_dyn::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out)
{
  PetscErrorCode ierr = 0;

  PetscScalar age = agingLaw_psi((_psi[Jj] + state)/2.0, (_slipVel[Jj] + _slipPrev[Jj])/2.0, _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar stress = state - _psi[Jj];

  *out = _deltaT * age - stress;
  assert(!isnan(*out));
  assert(!isinf(*out));
  return ierr;
}


// common rate-and-state functions

// state evolution law: aging law, state variable: psi
PetscScalar agingLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  PetscScalar A = exp( (double) (f0-psi)/b );
  PetscScalar dstate = 0.;
  if ( !isinf(A) && b>1e-3 ) {
    dstate = (PetscScalar) (b*v0/Dc)*( A - abs(slipVel)/v0 );
  }
  //~ assert(!isnan(dstate));
  //~ assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscScalar agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *psiA,*dstateA,*slipVelA,*aA,*bA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(psi,&psiA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(a,&aA);
  VecGetArray(b,&bA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_psi(psiA[Jj], slipVelA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    if ( isnan(dstateA[Jj]) || isinf(dstateA[Jj]) ) {
      PetscPrintf(PETSC_COMM_WORLD,"[%i]: dpsi = %g, psi = %g, slipVel = %g, a = %g, b = %g, f0 = %g, v0 = %g, Dc = %g\n",
      Jj,dstateA[Jj], psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
      assert(!isnan(dstateA[Jj]));
      assert(!isinf(dstateA[Jj]));
    }
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(psi,&psiA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(a,&aA);
  VecRestoreArray(b,&bA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: aging law, state variable: theta
PetscScalar agingLaw_theta(const PetscScalar& theta, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar dstate = 1. - theta*abs(slipVel)/Dc;

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscScalar agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *thetaA,*dstateA,*slipVelA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(theta,&thetaA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(theta,&thetaA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: slip law, state variable: psi
PetscScalar slipLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  PetscScalar fss = f0 + (a-b)*log(slipVel/v0);
  PetscScalar f = psi + a*log(slipVel/v0);
  PetscScalar dstate = -slipVel/Dc *(f - fss);

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the state law to a Vec
PetscScalar slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *psiA,*dstateA,*slipVelA,*aA,*bA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(psi,&psiA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(a,&aA);
  VecGetArray(b,&bA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_psi(psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(psi,&psiA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(a,&aA);
  VecRestoreArray(b,&bA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: slip law, state variable: theta
PetscScalar slipLaw_theta(const PetscScalar& state, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar A = state*slipVel/Dc;
  PetscScalar dstate = 0.;
  if (A != 0.) { dstate = -A*log(A); }

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the state law to a Vec
PetscScalar slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *thetaA,*dstateA,*slipVelA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(theta,&thetaA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(theta,&thetaA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}


// frictional strength, regularized form, for state variable psi
PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& v0)
{
  PetscScalar strength = (PetscScalar) a*sN*asinh( (double) (slipVel/2./v0)*exp(psi/a) );
  return strength;
}

