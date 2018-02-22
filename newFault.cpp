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
  _computeVelTime(0),_stateLawTime(0), _scatterTime(0),
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
  VecDuplicate(_tauP,&_tauQSP);   PetscObjectSetName((PetscObject) _tauQSP, "tauQS");  VecSet(_tauQSP,0.0);
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
  VecDuplicate(_tauP,&_locked);   PetscObjectSetName((PetscObject) _locked, "locked");
  VecDuplicate(_tauP,&_tau0);     PetscObjectSetName((PetscObject) _tau0, "tau0");VecSet(_tau0, 30.0);

  // create z from D._z
  double scatterStart = MPI_Wtime();
  VecScatterBegin(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  _scatterTime += MPI_Wtime() - scatterStart;


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
    double scatterStart = MPI_Wtime();
    Vec temp; VecDuplicate(_D->_y,&temp);
    ierr = setVec(temp,_D->_z,_rhoVals,_rhoDepths); CHKERRQ(ierr);
    VecScatterBegin(*_body2fault, temp, _rho, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, temp, _rho, INSERT_VALUES, SCATTER_FORWARD);

    ierr = setVec(temp,_D->_z,_muVals,_muDepths); CHKERRQ(ierr);
    VecScatterBegin(*_body2fault, temp, _mu, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, temp, _mu, INSERT_VALUES, SCATTER_FORWARD);

    _scatterTime += MPI_Wtime() - scatterStart;

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
  double scatterStart = MPI_Wtime();

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
PetscErrorCode NewFault::updateTemperature(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::updateTemperature";
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

// update shear stress on the fault based on the stress sxy body field
PetscErrorCode NewFault::setTauQS(const Vec& sxy)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setTauQS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

    double scatterStart = MPI_Wtime();
    VecScatterBegin(*_body2fault, sxy, _tauQSP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(*_body2fault, sxy, _tauQSP, INSERT_VALUES, SCATTER_FORWARD);
    _scatterTime += MPI_Wtime() - scatterStart;

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   scatter time (s): %g\n",_scatterTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent finding slip vel law: %g\n",(_computeVelTime/totRunTime)*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in state law: %g\n",(_stateLawTime/totRunTime)*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in scatters: %g\n",(_scatterTime/totRunTime)*100.);CHKERRQ(ierr);

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
    ierr = io_initiateWriteAppend(_viewers, "tauP", _tauP, outputDir + "tauP"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "tauQSP", _tauQSP, outputDir + "tauQSP"); CHKERRQ(ierr);
    ierr = io_initiateWriteAppend(_viewers, "psi", _psi, outputDir + "psi"); CHKERRQ(ierr);
    if (_stateLaw.compare("flashHeating") == 0) {
      ierr = io_initiateWriteAppend(_viewers, "T", _T, outputDir + "fault_T"); CHKERRQ(ierr);
    }
  }
  else {
    ierr = VecView(_slip,_viewers["slip"].first); CHKERRQ(ierr);
    ierr = VecView(_slipVel,_viewers["slipVel"].first); CHKERRQ(ierr);
    ierr = VecView(_tauP,_viewers["tauP"].first); CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_viewers["tauQSP"].first); CHKERRQ(ierr);
    ierr = VecView(_psi,_viewers["psi"].first); CHKERRQ(ierr);
    if (_stateLaw.compare("flashHeating") == 0) {
      ierr = VecView(_T,_viewers["T"].first); CHKERRQ(ierr);
    }
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

  computePsiSS(vL);
  VecSet(_slipVel,vL);

  if (tauRS == NULL) { VecDuplicate(_slipVel,&tauRS); }

  PetscInt       Istart,Iend;
  PetscScalar   *tauRSV;
  PetscScalar const *sN,*a,*psi;
  VecGetOwnershipRange(tauRS,&Istart,&Iend);
  VecGetArray(tauRS,&tauRSV);
  VecGetArrayRead(_sNEff,&sN);
  VecGetArrayRead(_psi,&psi);
  VecGetArrayRead(_a,&a);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    tauRSV[Jj] = sN[Jj]*a[Jj]*asinh( (double) 0.5*vL*exp(psi[Jj]/a[Jj])/_v0 );
    Jj++;
  }
  VecRestoreArray(tauRS,&tauRSV);
  VecRestoreArrayRead(_sNEff,&sN);
  VecRestoreArrayRead(_psi,&psi);
  VecRestoreArrayRead(_a,&a);

  VecCopy(tauRS,_tauQSP);
  VecCopy(tauRS,_tauP);



  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault::computePsiSS(const PetscScalar vL)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 2
    std::string funcName = "NewFault::computePsiSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  PetscInt           Istart,Iend;
  PetscScalar       *psi;
  PetscScalar const *b;
  VecGetOwnershipRange(_psi,&Istart,&Iend);
  VecGetArray(_psi,&psi);
  VecGetArrayRead(_b,&b);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
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

  if (D._inputDir.compare("unspecified") != 0) {
    loadFieldsFromFiles(D._inputDir);
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
  PetscScalar *slipVelA;
  PetscScalar const *etaA, *tauQSA, *sNA, *psiA, *aA,*bA,*lockedA,*Co;
  VecGetArray(_slipVel,&slipVelA);
  VecGetArrayRead(_eta_rad,&etaA);
  VecGetArrayRead(_tauQSP,&tauQSA);
  VecGetArrayRead(_sNEff,&sNA);
  VecGetArrayRead(_psi,&psiA);
  VecGetArrayRead(_a,&aA);
  VecGetArrayRead(_b,&bA);
  VecGetArrayRead(_locked,&lockedA);
  VecGetArrayRead(_cohesion,&Co);
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;

  ComputeVel_qd temp(N,etaA,tauQSA,sNA,psiA,aA,bA,_v0,lockedA,Co);
  ierr = temp.computeVel(slipVelA, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecGetArray(_slipVel,&slipVelA);
  VecGetArrayRead(_eta_rad,&etaA);
  VecGetArrayRead(_tauQSP,&tauQSA);
  VecGetArrayRead(_sNEff,&sNA);
  VecGetArrayRead(_psi,&psiA);
  VecGetArrayRead(_a,&aA);
  VecGetArrayRead(_b,&bA);
  VecGetArrayRead(_locked,&lockedA);
  VecGetArrayRead(_cohesion,&Co);

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
    ierr = flashHeating_psi_Vec(dstate,_psi,_slipVel,_T,_rho,_c,_k,_D_fh,_Tw,_tau_c,_Vw,_fw,_Dc,_a,_b,_f0,_v0);
    CHKERRQ(ierr);
  }
  else if (_stateLaw.compare("constantState") == 0) {
    // dpsi = 0; psi = f0 - b*ln(|V|/v0)
    VecSet(dstate,0.);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n");
    assert(0);
  }
  _stateLawTime += MPI_Wtime() - startTime;


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

ComputeVel_qd::ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0,const PetscScalar* locked,const PetscScalar* Co)
: _a(a),_b(b),_sN(sN),_tauQS(tauQS),_eta(eta),_psi(psi),_locked(locked),_Co(Co),_N(N),_v0(v0)
{ }

PetscErrorCode ComputeVel_qd::computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscScalar left, right, out;
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
  PetscScalar stress = _tauQS[Jj] - _eta[Jj]*vel + _Co[Jj]; // stress on fault

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
  PetscScalar stress = _tauQS[Jj] - _eta[Jj]*vel + _Co[Jj]; // stress on fault

  *out = strength - stress;
  PetscScalar A = _a[Jj]*_sN[Jj];
  PetscScalar B = exp(_psi[Jj]/_a[Jj]) / (2.*_v0);
  *J = A*B/sqrt(B*B*vel*vel + 1.) + _eta[Jj]; // derivative with respect to slipVel

  assert(!isnan(*out)); assert(!isinf(*out));
  assert(!isnan(*J)); assert(!isinf(*J));
  return ierr;
}

NewFault_dyn::NewFault_dyn(Domain&D, VecScatter& scatter2fault)
: NewFault(D, scatter2fault), _Phi(NULL), _slipPrev(NULL),_rhoLocal(NULL),
  _alphay(D._alphay), _alphaz(D._alphaz), _timeMode("Gaussian"), _isLocked("False"),
  _lockLimit(25.0)
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

  loadSettings(_inputFile);

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
    else if (var.compare("timeMode")==0) {
      _timeMode = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("isLocked")==0) {
      _isLocked = line.substr(pos+_delim.length(),line.npos);
    }
    else if (var.compare("lockLimit")==0) {
      _lockLimit = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
  PetscScalar *Phi, *an, *psi, *constraints_factor, *a,*sneff, *slipVel, *locked;
  VecGetArray(_Phi,&Phi);
  VecGetArray(_an,&an);
  VecGetArray(_psi,&psi);
  VecGetArray(_constraints_factor,&constraints_factor);
  VecGetArray(_a,&a);
  VecGetArray(_sNEff,&sneff);
  VecGetArray(_slipVel,&slipVel);
  VecGetArray(_locked, &locked);
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);CHKERRQ(ierr);
  PetscInt N = Iend - Istart;
  ComputeVel_dyn temp(locked, N,Phi,an,psi,constraints_factor,a,sneff, _v0);
  ierr = temp.computeVel(slipVel, _rootTol, _rootIts, _maxNumIts); CHKERRQ(ierr);

  VecRestoreArray(_Phi,&Phi);
  VecRestoreArray(_an,&an);
  VecRestoreArray(_psi,&psi);
  VecRestoreArray(_constraints_factor,&constraints_factor);
  VecRestoreArray(_a,&a);
  VecRestoreArray(_sNEff,&sneff);
  VecRestoreArray(_slipVel,&slipVel);
  VecRestoreArray(_locked, &locked);
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
  VecDuplicate(_tauP, &u);
  VecDuplicate(_tauP, &du);
  VecDuplicate(_tauP, &uPrev);
  VecSet(u, 0);
  VecSet(du, 0);
  VecSet(uPrev, 0);
  varEx["uFault"] = u;
  varEx["uPrevFault"] = uPrev;
  varEx["duFault"] = du;

  VecDuplicate(_tauQSP, &_rhoLocal);
  VecSet(_rhoLocal, 0);

  double scatterStart = MPI_Wtime();
  VecScatterBegin(*_body2fault, varEx["u"], varEx["uFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, varEx["u"], varEx["uFault"], INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, varEx["uPrev"], varEx["uPrevFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, varEx["uPrev"], varEx["uPrevFault"], INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(*_body2fault, _rhoVec, _rhoLocal, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, _rhoVec, _rhoLocal, INSERT_VALUES, SCATTER_FORWARD);
  _scatterTime += MPI_Wtime() - scatterStart;

  // slip is added by the momentum balance equation
  //~ Vec varSlip; VecDuplicate(_slip,&varSlip); VecCopy(_slip,varSlip);
  //~ varEx["slip"] = varSlip;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode NewFault_dyn::updateTau(const PetscScalar currT){
  PetscScalar *zz, *tau0;

  PetscInt Ii, IBegin, IEnd;
  PetscInt Jj = 0;
  VecGetArray(_z, &zz);
  VecGetArray(_tau0, &tau0);
  VecGetOwnershipRange(_z, &IBegin, &IEnd);
  PetscScalar timeOffset = 1.0;
  if(_timeMode.compare("Gaussian") == 0){
    PetscScalar timeOffset = exp(-pow((currT - _tCenterTau), 2) / pow(_tStdTau, 2));
  }
  else if (_timeMode.compare("Dirac") == 0 && currT > 0){
    timeOffset = 0.0;
  }
  else if (_timeMode.compare("Heaviside") == 0){
    timeOffset = 1.0;
  }
  for (Ii=IBegin;Ii<IEnd;Ii++){
    tau0[Jj] = 30.0 + _ampTau * exp(-pow((zz[Jj] - _zCenterTau), 2) / (2.0 * pow(_zStdTau, 2))) * timeOffset;
    Jj++;
  }
  VecRestoreArray(_z, &zz);
  VecRestoreArray(_tau0, &tau0);
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
PetscScalar    *u, *uPrev, uTemp, *rho, *sigma_N, *a, *an, *slip, *slipVel, fric, alpha, A, *vel, *Phi, *psi, *z;
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
ierr = VecGetArray(_z, &z);
PetscScalar *b;
ierr = VecGetArray(_b, &b);

for (Ii = Istart; Ii<Iend; Ii++){

  if (slipVel[Jj] < 1e-14){
    vel[Jj] = 0;
    slipVel[Jj] = 0;
  }
  else{
    fric = (PetscScalar) a[Jj]*sigma_N[Jj]*asinh( (double) (slipVel[Jj]/2./_v0)*exp(psi[Jj]/a[Jj]) );
    alpha = 1 / (rho[Jj] * _alphay) * fric / slipVel[Jj];
    A = 1 + alpha * _deltaT;
    uTemp = uPrev[Jj];
    uPrev[Jj] = u[Jj];
    u[Jj] = (2.*u[Jj] + _deltaT * _deltaT / rho[Jj] * an[Jj] + (_deltaT*alpha-1.)*uTemp) /  A;
    vel[Jj] = Phi[Jj] / (1. + _deltaT/_alphay / rho[Jj] * fric / slipVel[Jj]);
    slipVel[Jj] = vel[Jj];
  }
  slip[Jj] = 2. * u[Jj];
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
ierr = VecRestoreArray(_z, &z);

  double scatterStart = MPI_Wtime();

  VecScatterBegin(*_body2fault, varEx["uFault"], varEx["u"], INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(*_body2fault, varEx["uFault"], varEx["u"], INSERT_VALUES, SCATTER_REVERSE);

  VecScatterBegin(*_body2fault, varEx["uPrevFault"], varEx["uPrev"], INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(*_body2fault, varEx["uPrevFault"], varEx["uPrev"], INSERT_VALUES, SCATTER_REVERSE);

  _scatterTime += MPI_Wtime() - scatterStart;


// compute state parameter law
double startAgingTime = MPI_Wtime();
computeAgingLaw();
_stateLawTime += MPI_Wtime() - startAgingTime;
VecCopy(_psi, varEx["psi"]);
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

  double scatterStart = MPI_Wtime();

  VecScatterBegin(*_body2fault, dvarEx["u"], varEx["duFault"], INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, dvarEx["u"], varEx["duFault"], INSERT_VALUES, SCATTER_FORWARD);
  _scatterTime += MPI_Wtime() - scatterStart;

  ierr = VecGetOwnershipRange(varEx["uFault"],&IFaultStart,&IFaultEnd);CHKERRQ(ierr);

  PetscScalar *u, *uPrev, *Laplacian, *rho, *psi, *sigma_N, *tau0, *slipVel, *an, *Phi, *constraints_factor, *slipPrev, *slipVelocity;
  PetscInt Jj = 0;

  ierr = VecGetArray(varEx["uFault"], &u);
  ierr = VecGetArray(varEx["uPrevFault"], &uPrev);
  ierr = VecGetArray(varEx["duFault"], &Laplacian);
  ierr = VecGetArray(_rhoLocal, &rho);
  ierr = VecGetArray(varEx["psi"], &psi);
  ierr = VecGetArray(_sNEff, &sigma_N);
  ierr = VecGetArray(_tau0, &tau0);
  ierr = VecGetArray(dvarEx["slip"], &slipVel);
  ierr = VecGetArray(_slipVel, &slipVelocity);
  ierr = VecGetArray(_slipPrev, &slipPrev);
  ierr = VecGetArray(_an, &an);
  ierr = VecGetArray(_Phi, &Phi);
  ierr = VecGetArray(_constraints_factor, &constraints_factor);

  for (Ii=IFaultStart;Ii<IFaultEnd;Ii++){
    an[Jj] = Laplacian[Jj] + tau0[Jj] / _alphay;
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
  ierr = VecRestoreArray(_tau0, &tau0);
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

ComputeVel_dyn::ComputeVel_dyn(const PetscScalar* locked, const PetscInt N,const PetscScalar* Phi, const PetscScalar* an, const PetscScalar* psi, const PetscScalar* constraints_factor,const PetscScalar* a,const PetscScalar* sneff, const PetscScalar v0)
: _locked(locked), _Phi(Phi),_an(an),_psi(psi),_constraints_factor(constraints_factor),_a(a),_sNEff(sneff),_N(N), _v0(v0)
{ }

PetscErrorCode ComputeVel_dyn::computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  BracketedNewton rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, out, temp;

  for (PetscInt Jj = 0; Jj<_N; Jj++) {
    if (_locked[Jj] > 0.){
      slipVelA[Jj] = 0.0;
      break;
    }
    left = 0.;
    right = abs(_Phi[Jj]);
    // check bounds
    if (isnan(left)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_dyn::computeVel: left bound evaluated to NaN.\n");
      assert(0);
    }
    if (isnan(right)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError in ComputeVel_dyn::computeVel: right bound evaluated to NaN.\n");
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
      // ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
      ierr = rootFinder.findRoot(this,Jj,slipVelA[Jj], &out);CHKERRQ(ierr);
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

PetscErrorCode ComputeVel_dyn::getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;
  PetscScalar constraints = strength_psi(_sNEff[Jj], _psi[Jj], vel, _a[Jj] , _v0); // frictional strength

  constraints = _constraints_factor[Jj] * constraints;
  PetscScalar Phi_temp = _Phi[Jj];
  if (Phi_temp < 0){Phi_temp = -Phi_temp;}

  PetscScalar stress = Phi_temp - vel; // stress on fault

  *out = constraints - stress;
  PetscScalar A = _a[Jj] * _sNEff[Jj];
  PetscScalar B = exp(_psi[Jj] / _a[Jj]) / (2. * _v0);

  *J = 1 + _constraints_factor[Jj] * A * B / sqrt(1. + B * B * vel * vel);

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

  // RegulaFalsi rootFinder(maxNumIts,rootTol);
  BracketedNewton rootFinder(maxNumIts,rootTol);
  // Bisect rootFinder(maxNumIts,rootTol);
  PetscScalar left, right, out, temp;
  for (PetscInt Jj = 0; Jj<_N; Jj++) {

    // left = -10.;
    // right = 10.;
    left = 0.;
    right = 2*_psi[Jj];

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
      // ierr = rootFinder.findRoot(this,Jj,&out);CHKERRQ(ierr);
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

  PetscScalar age = agingLaw_psi((_psi[Jj] + state)/2.0, _slipVel[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar stress = state - _psi[Jj];
  *out = -2 * _deltaT * age + stress;
  assert(!isnan(*out));
  assert(!isinf(*out));
  return ierr;
}

PetscErrorCode ComputeAging_dyn::getResid(const PetscInt Jj,const PetscScalar state,PetscScalar* out, PetscScalar *J)
{
  PetscErrorCode ierr = 0;

  PetscScalar age = agingLaw_psi((_psi[Jj] + state)/2.0, _slipVel[Jj], _b[Jj], _f0, _v0, _Dc[Jj]);
  PetscScalar stress = state - _psi[Jj];
  *out = -2 * _deltaT * age + stress;

  *J = 1 + _deltaT * _v0 / _Dc[Jj] * exp((_f0 - (_psi[Jj] + state)/2.)/_b[Jj]);

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
    dstate = (PetscScalar) (b*v0/Dc)*( A - slipVel/v0 );
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscErrorCode agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  PetscScalar const *psiA,*slipVelA,*aA,*bA,*DcA;
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
    if ( isnan(dstateA[Jj]) || isinf(dstateA[Jj]) ) {
      PetscPrintf(PETSC_COMM_WORLD,"[%i]: dpsi = %g, psi = %g, slipVel = %g, a = %g, b = %g, f0 = %g, v0 = %g, Dc = %g\n",
      Jj,dstateA[Jj], psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
      assert(!isnan(dstateA[Jj]));
      assert(!isinf(dstateA[Jj]));
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

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscErrorCode agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  PetscScalar const *thetaA,*slipVelA,*DcA;
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
  PetscScalar absV = abs(slipVel);
  if (absV == 0) { absV += 1e-14; }

  PetscScalar fss = f0 + (a-b)*log(absV/v0);

  // not regularized
  //~ PetscScalar f = psi + a*log(absV/v0);

  // regularized
  PetscScalar f = a*asinh( (double) (absV/2./v0)*exp(psi/a) );

  PetscScalar dstate = -absV/Dc *(f - fss);

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the state law to a Vec
PetscErrorCode slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  PetscScalar const *psiA,*slipVelA,*aA,*bA,*DcA;
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

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the slip law to a Vec
PetscErrorCode slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dstateA;
  PetscScalar const *thetaA,*slipVelA,*DcA;
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


// flash heating state evolution law
PetscScalar flashHeating_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& T, const PetscScalar& rho, const PetscScalar& c, const PetscScalar& k, const PetscScalar& D, const PetscScalar& Tw, const PetscScalar& tau_c, const PetscScalar& Vwi, const PetscScalar& fw, const PetscScalar& Dc,const PetscScalar& a,const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0)
{
  PetscScalar absV = abs(slipVel);

  // compute Vw
  PetscScalar rc = rho * c;
  PetscScalar ath = k/rc;
  PetscScalar Vw = (M_PI*ath/D) * pow(rc*(Tw-T)/tau_c,2.);
  //~ PetscScalar Vw = Vwi;

  if (absV == 0.0) { absV += 1e-14; }

  // compute f
  PetscScalar fLV = f0 + (a-b)*log(absV/v0);
  PetscScalar fss = fLV;

  if (absV > Vw) { fss = fw + (fLV - fw)*(Vw/absV); }
  PetscScalar f = psi + a*log(absV/v0);
  PetscScalar dpsi = -absV/Dc *(f - fss);

  assert(!isnan(dpsi));
  assert(!isinf(dpsi));
  return dpsi;
}

// applies the flash heating state law to a Vec
PetscErrorCode flashHeating_psi_Vec(Vec &dpsi,const Vec& psi, const Vec& slipVel, const Vec& T, const Vec& rho, const Vec& c, const Vec& k, const PetscScalar& D, const PetscScalar& Tw, const PetscScalar& tau_c, const PetscScalar& Vwi, const PetscScalar& fw, const Vec& Dc,const Vec& a,const Vec& b, const PetscScalar& f0, const PetscScalar& v0)
{
  PetscErrorCode ierr = 0;

  PetscScalar *dpsiA;
  PetscScalar const *psiA,*slipVelA,*DcA,*TA,*rhoA,*cA,*kA,*aA,*bA;
  VecGetArray(dpsi,&dpsiA);
  VecGetArrayRead(psi,&psiA);
  VecGetArrayRead(slipVel,&slipVelA);
  VecGetArrayRead(T,&TA);
  VecGetArrayRead(rho,&rhoA);
  VecGetArrayRead(c,&cA);
  VecGetArrayRead(k,&kA);
  VecGetArrayRead(Dc,&DcA);
  VecGetArrayRead(a,&aA);
  VecGetArrayRead(b,&bA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dpsiA[Jj] = flashHeating_psi(psiA[Jj],slipVelA[Jj],TA[Jj],rhoA[Jj],cA[Jj],kA[Jj],D,Tw,tau_c,Vwi,fw,DcA[Jj],aA[Jj],bA[Jj],f0,v0);
    Jj++;
  }
  VecRestoreArray(dpsi,&dpsiA);
  VecRestoreArrayRead(psi,&psiA);
  VecRestoreArrayRead(slipVel,&slipVelA);
  VecRestoreArrayRead(T,&TA);
  VecRestoreArrayRead(rho,&rhoA);
  VecRestoreArrayRead(c,&cA);
  VecRestoreArrayRead(k,&kA);
  VecRestoreArrayRead(Dc,&DcA);
  VecRestoreArrayRead(a,&aA);
  VecRestoreArrayRead(b,&bA);

  return ierr;
}

// frictional strength, regularized form, for state variable psi
PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& v0)
{
  PetscScalar strength = (PetscScalar) a*sN*asinh( (double) (slipVel/2./v0)*exp(psi/a) );
  return strength;
}


