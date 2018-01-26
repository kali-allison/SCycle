#include "newFault.hpp"

#define FILENAME "newFault.cpp"

using namespace std;


NewFault::NewFault(Domain&D,VecScatter& scatter2fault)
: _inputFile(D._file),_delim(D._delim),_outputDir(D._outputDir),
  _stateLaw("agingLaw"),
  _N(D._Nz),_L(D._Lz),
  _f0(0.6),_v0(1e-6),
  _sigmaN_cap(1e14),_sigmaN_floor(0.),
  _rootTol(0),_rootIts(0),_maxNumIts(0),
  _computeVelTime(0),_stateLawTime(0),
  _body2fault(&scatter2fault)
{
  #if VERBOSE > 1
    std::string funcName = "NewFault::NewFault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set a, b, normal stress, and Dc
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
    else if (var.compare("mu")==0) {
      _muVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("rho")==0) {
      _rhoVal = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
    //~ else if (var.compare("fw")==0) {
      //~ _fw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    //~ }
    //~ else if (var.compare("Vw")==0) {
      //~ _Vw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    //~ }
    //~ else if (var.compare("Tw")==0) {
      //~ _Tw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    //~ }
    //~ else if (var.compare("D")==0) {
      //~ _D = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    //~ }
    //~ else if (var.compare("tau_c")==0) {
      //~ _tau_c = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    //~ }
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

  // load quasi-static shear stress
  ierr = loadVecFromInputFile(_eta_rad,inputDir,"eta_rad"); CHKERRQ(ierr);

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

  assert(_DcVals.size() != 0 );
  assert(_aVals.size() != 0 );
  assert(_bVals.size() != 0 );
  assert(_sigmaNVals.size() != 0 );

  assert(_rootTol >= 1e-14);

  assert(_stateLaw.compare("agingLaw")==0
    || _stateLaw.compare("slipLaw")==0
    || _stateLaw.compare("flashHeating")==0 );

  assert(_v0 > 0);
  assert(_f0 > 0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  //~}
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
  VecDuplicate(D._y0,&_tauQSP);     PetscObjectSetName((PetscObject) _tauQSP, "tauQS");  VecSet(_tauQSP,0.0);
  VecDuplicate(_tauQSP,&_psi);      PetscObjectSetName((PetscObject) _psi, "psi"); VecSet(_psi,0.0);
  VecDuplicate(_tauQSP,&_dPsi);     PetscObjectSetName((PetscObject) _dPsi, "dPsi"); VecSet(_dPsi,0.0);
  VecDuplicate(_tauQSP,&_slip);     PetscObjectSetName((PetscObject) _slip, "slip"); VecSet(_slip,0.0);
  VecDuplicate(_tauQSP,&_slipVel);  PetscObjectSetName((PetscObject) _slipVel, "slipVel"); VecSet(_slipVel,0.0);
  VecDuplicate(_tauQSP,&_Dc);       PetscObjectSetName((PetscObject) _Dc, "Dc");
  VecDuplicate(_tauQSP,&_eta_rad);  PetscObjectSetName((PetscObject) _eta_rad, "eta_rad");
  VecDuplicate(_tauQSP,&_a);        PetscObjectSetName((PetscObject) _a, "a");
  VecDuplicate(_tauQSP,&_b);        PetscObjectSetName((PetscObject) _b, "b");
  VecDuplicate(_tauQSP,&_cohesion); PetscObjectSetName((PetscObject) _cohesion, "cohesion"); VecSet(_cohesion,0);
  VecDuplicate(_tauQSP,&_sN);       PetscObjectSetName((PetscObject) _sN, "sN");
  VecDuplicate(_tauQSP,&_sNEff);    PetscObjectSetName((PetscObject) _sNEff, "sNEff");
  VecDuplicate(_tauQSP,&_z);    PetscObjectSetName((PetscObject) _z, "z_fault");


  // set fields
  if (_N == 1) {
    VecSet(_a,_aVals[0]);
    VecSet(_b,_bVals[0]);
    VecSet(_sN,_sigmaNVals[0]);
    VecSet(_Dc,_DcVals[0]);
    VecSet(_cohesion,_cohesionVals[0]);
  }
  else {
    ierr = setVecFromVectors(_a,_aVals,_aDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_b,_bVals,_bDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_sN,_sigmaNVals,_sigmaNDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_Dc,_DcVals,_DcDepths);CHKERRQ(ierr);
    if (_cohesionVals.size() > 0 ) { ierr = setVecFromVectors(_cohesion,_cohesionVals,_cohesionDepths); }
  }
  ierr = VecSet(_psi,_f0);CHKERRQ(ierr);
  { // radiation damping parameter: 0.5 * sqrt(mu*rho)
    Vec mu, rho;
    VecDuplicate(_tauQSP,&mu); VecSet(mu,_muVal);
    VecDuplicate(_tauQSP,&rho); VecSet(rho,_rhoVal);
    ierr = VecPointwiseMult(_eta_rad,mu,rho); CHKERRQ(ierr);
    ierr = VecSqrtAbs(_eta_rad); CHKERRQ(ierr);
    ierr = VecScale(_eta_rad,0.5); CHKERRQ(ierr);
    VecDestroy(&mu);
    VecDestroy(&rho);
  }

  { // impose floor and ceiling on effective normal stress
    Vec temp; VecDuplicate(_sN,&temp);
    VecSet(temp,_sigmaN_cap); VecPointwiseMin(_sN,_sN,temp);
    VecSet(temp,_sigmaN_floor); VecPointwiseMax(_sN,_sN,temp);
    VecDestroy(&temp);
  }
  VecCopy(_sN,_sNEff);


  // create z from D._z
  VecScatterBegin(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(*_body2fault, D._z, _z, INSERT_VALUES, SCATTER_FORWARD);
  VecView(_z,PETSC_VIEWER_STDOUT_WORLD);
  assert(0);


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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent finding slip vel law: %g\n",_computeVelTime/totRunTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in state law: %g\n",_stateLawTime/totRunTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode NewFault::writeContext(const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "NewFault_qd::writeContext";
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
    //~ ierr = PetscViewerASCIIPrintf(viewer,"fw = %.15e\n",_fw);CHKERRQ(ierr);
    //~ ierr = PetscViewerASCIIPrintf(viewer,"Vw = %.15e\n",_Vw);CHKERRQ(ierr);
    //~ ierr = PetscViewerASCIIPrintf(viewer,"tau_c = %.15e # (GPa)\n",_tau_c);CHKERRQ(ierr);
    //~ ierr = PetscViewerASCIIPrintf(viewer,"Tw = %.15e # (K)\n",_Tw);CHKERRQ(ierr);
    //~ ierr = PetscViewerASCIIPrintf(viewer,"D = %.15e # (um)\n",_D);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  // output vector fields

  str = outputDir + "fault_a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "fault_b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "fault_imp";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_eta_rad,viewer);CHKERRQ(ierr);
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
  VecDestroy(&_eta_rad);
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

// Fills vec with the linear interpolation between the pairs of points (vals,depths).
PetscErrorCode NewFault::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setVecFromVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(vec,vals[0]);

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(_z,1,&Ii,&z);CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: z = %g\n",Ii,z);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
      z0 = depths[0+ind];
      z1 = depths[0+ind+1];
      v0 = vals[0+ind];
      v1 = vals[0+ind+1];
      if (z>=z0 && z<=z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }
      ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// Fills vec with the linear interpolation between the pairs of points (vals,depths), but always below the specified max value
PetscErrorCode NewFault::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
  const PetscScalar maxVal)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "NewFault::setVecFromVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(_z,1,&Ii,&z);CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: z = %g\n",Ii,z);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
      z0 = depths[0+ind];
      z1 = depths[0+ind+1];
      v0 = vals[0+ind];
      v1 = vals[0+ind+1];
      if (z>=z0 && z<=z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }
      v = min(maxVal,v);
      ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
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
    //~ tauRSV[Jj] = sN[Jj]*a[Jj]*asinh( (double) 0.5*vL*exp(_f0/a[Jj])/_v0 );
    PetscScalar f = _f0 + (a[Jj] - b[Jj]) * log(vL/_v0);
    tauRSV[Jj] = sN[Jj] * f;
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

  if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

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

  // this is covered by the NewFault destructor.

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
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
    //~ ierr = agingLaw_theta(Ii,theta,dtheta);
    //~ ierr = agingLaw_theta_Vec(dstate, _theta, _slipVel, _Dc) CHKERRQ(ierr);
    ierr = agingLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc); CHKERRQ(ierr);
  }
  else if (!_stateLaw.compare("slipLaw")) {
    //~ ierr = slipLaw_theta_Vec(dstate, _theta, _slipVel, _Dc); CHKERRQ(ierr);
    ierr =  slipLaw_psi_Vec(dstate,_psi,_slipVel,_a,_b,_f0,_v0,_Dc);  CHKERRQ(ierr);
  }
  //~ else if (!_stateLaw.compare("flashHeating")) {
    //~ ierr = flashHeating_psi(Ii,psi,dpsi);CHKERRQ(ierr);
  //~ }
  else if (!_stateLaw.compare("constantState")) {
    VecSet(dstate,0.);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n");
    assert(0);
  }
  _stateLawTime += MPI_Wtime() - startTime;


    //~ // compute slip velocity
//~ double startTime = MPI_Wtime();
  //~ ierr = computeVel();CHKERRQ(ierr);
  //~ VecCopy(_slipVel,dvarEx["slip"]);
//~ _computeVelTime += MPI_Wtime() - startTime;


  // set tauP = tauQS - z/2 *slipVel
  //~ VecCopy(_slipVel,_tauP);
  //~ VecPointwiseMult(_tauP,_eta_rad,_tauP);
  //~ VecAYPX(_tauP,-0.5,_tauQSP);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}






//======================================================================
// functions for structs
//======================================================================

ComputeVel_qd::ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0)
: _a(a),_b(b),_sN(sN),_tauQS(tauQS),_eta(eta),_N(N),_v0(v0)
{ }

PetscErrorCode ComputeVel_qd::computeVel(const PetscScalar rootTol, const PetscScalar rootIts, const PetscScalar maxNumits)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode ComputeVel_qd::getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "ComputeVel_qd::getResid";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}
















