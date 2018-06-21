#include "strikeSlip_powerLaw_fd.hpp"

#define FILENAME "strikeSlip_powerLaw_fd.cpp"

using namespace std;


StrikeSlip_PowerLaw_fd::StrikeSlip_PowerLaw_fd(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),
  _deltaT(1e-3), _CFL(0),
  _y(&D._y),_z(&D._z),
  _alphay(D._alphay), _alphaz(D._alphaz),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(1e-9),
  _isFault("true"),_initialConditions("u"),
   _thermalCoupling("no"),_heatEquationType("transient"),_hydraulicCoupling("no"),
  _stride1D(1),_stride2D(1),_maxStepCount(1e8),
  _initTime(0),_currTime(0),_maxTime(1e15),
  _stepCount(0),_atol(1e-8),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0), _propagateTime(0),
  _bcRType("outGoingCharacteristics"),_bcTType("freeSurface"),_bcLType("outGoingCharacteristics"),_bcBType("outGoingCharacteristics"),
  _quadWaveImex(NULL),
  _fault(NULL),_material(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::StrikeSlip_PowerLaw_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  _he = new HeatEquation(D); // heat equation

  assert(_bcBType.compare("freeSurface")==0 || _bcBType.compare("outGoingCharacteristics")==0);
  assert(_bcTType.compare("freeSurface")==0 || _bcTType.compare("outGoingCharacteristics")==0);
  assert(_bcRType.compare("freeSurface")==0 || _bcRType.compare("outGoingCharacteristics")==0);
  assert(_bcLType.compare("freeSurface")==0 || _bcLType.compare("outGoingCharacteristics")==0);
  _mat_bcBType = "Neumann";
  _mat_bcTType = "Neumann";
  _mat_bcRType = "Neumann";
  _mat_bcLType = "Neumann";

  _material = new PowerLaw(D,*_he,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);
  _cs = *(&(_material->_cs));
  _rhoVec = *(&(_material->_rhoVec));
  _muVec = *(&(_material->_muVec));

  if(_D->_sbpType.compare("mfc_coordTrans")==0){
    Mat J,Jinv,qy,rz,yq,zr;
    _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr);
    Vec temp1, temp2;
    VecDuplicate(_alphay, &temp1);
    VecDuplicate(_alphay, &temp2);
    MatMult(yq, _alphay, temp1);
    MatMult(zr, _alphaz, temp2);
    VecCopy(temp1, _alphay);
    VecCopy(temp2, _alphaz);
    VecCopy(temp1, D._alphay);
    VecCopy(temp2, D._alphaz);
    VecDestroy(&temp1);
    VecDestroy(&temp2);
  }

  _fault = new Fault_fd(D, D._scatters["body2L"]); // fault

  if (_thermalCoupling.compare("no")!=0 && _stateLaw.compare("flashHeating")==0) {
    Vec T; VecDuplicate(_D->_y,&T);
    _he->getTemp(T);
    _fault->setThermalFields(T,_he->_k,_he->_c);
  }

    // pressure diffusion equation
  if (_hydraulicCoupling.compare("no")!=0) {
    _p = new PressureEq(D);
  }
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }

  if (_CFL !=0){
    PetscInt max_index;
    PetscScalar max_speed;
    VecMax(_cs,&max_index,&max_speed);
    // Change for variable grid spacing with min y_q 1 / (Ny - 1)
    if (_D->_sbpType.compare("mfc_coordTrans")==0){
      Mat J,Jinv,qy,rz,yq,zr;
      _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr);

      PetscInt index_z, index_y;
      PetscScalar min_z, min_y;
      Vec yy, zz;
      VecDuplicate(*_y, &yy);
      VecDuplicate(*_z, &zz);
      MatGetDiagonal(yq, yy);
      MatGetDiagonal(zr, zz);
      VecMin(yy,&index_y,&min_y);
      VecMin(zz,&index_z,&min_z);

      _deltaT = 0.5 * _CFL / max_speed * min(min_y / (_Ny - 1), min_z / (_Nz - 1));
    }
    else{
      _deltaT = 0.5 * _CFL / max_speed * min(_Ly / (_Ny - 1), _Lz / (_Nz - 1));
    }
  }
  else{
    PetscInt max_index;
    PetscScalar max_speed, theoretical_dT;
    VecMax(_cs,&max_index,&max_speed);
    if (_D->_sbpType.compare("mfc_coordTrans")==0){
      Mat J,Jinv,qy,rz,yq,zr;
      _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr);

      PetscInt index_z, index_y;
      PetscScalar min_z, min_y;
      Vec yy, zz;
      VecDuplicate(*_y, &yy);
      VecDuplicate(*_z, &zz);
      MatGetDiagonal(yq, yy);
      MatGetDiagonal(zr, zz);
      VecMin(yy,&index_y,&min_y);
      VecMin(zz,&index_z,&min_z);

      theoretical_dT = 0.5 * _CFL / max_speed * min(min_y / (_Ny - 1), min_z / (_Nz - 1));
    }
    else{
      theoretical_dT = 0.5 * _CFL / max_speed * min(_Ly / (_Ny - 1), _Lz / (_Nz - 1));
    }
    if (theoretical_dT > _deltaT){
      PetscPrintf(PETSC_COMM_WORLD, "WARNING : The specified deltaT odes not meet the CFL requirements...");
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


StrikeSlip_PowerLaw_fd::~StrikeSlip_PowerLaw_fd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::~StrikeSlip_PowerLaw_fd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }

  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }

  delete _quadWaveImex;      _quadWaveImex = NULL;
  delete _material;    _material = NULL;
  delete _fault;       _fault = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_PowerLaw_fd::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "HeatEquation::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
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

    if (var.compare("stride1D")==0){ _stride1D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("maxTime")==0) { _maxTime = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("deltaT")==0) { _deltaT = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("CFL")==0) { _CFL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("atol")==0) { _atol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }
    else if (var.compare("isFault")==0) { _isFault = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("initialConditions")==0) { _initialConditions = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("loadDir")==0) { _inputDir = line.substr(pos+_delim.length(),line.npos).c_str(); }
    else if (var.compare("timeIntInds")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_timeIntInds);
    }
    else if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hydraulicCoupling")==0) {
      _hydraulicCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("stateLaw")==0) {
      _stateLaw = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    else if (var.compare("vL")==0) { _vL = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() ); }

    // boundary conditions for momentum balance equation
    else if (var.compare("momBal_bcR_dyn")==0) {
      _bcRType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcT_dyn")==0) {
      _bcTType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcL_dyn")==0) {
      _bcLType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("momBal_bcB_dyn")==0) {
      _bcBType = line.substr(pos+_delim.length(),line.npos).c_str();
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_PowerLaw_fd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  if (_loadICs) { assert(_guessSteadyStateICs == 0); }

  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  // assert(_stride1D >= 1);
  // assert(_stride2D >= 1);
  assert(_atol >= 1e-14);

    // check boundary condition types for momentum balance equation
  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcRType.compare("freeSurface")==0 ||
    _bcRType.compare("tau")==0 ||
    _bcRType.compare("remoteLoading")==0 ||
    _bcRType.compare("symm_fault")==0 ||
    _bcRType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcTType.compare("freeSurface")==0 ||
    _bcTType.compare("tau")==0 ||
    _bcTType.compare("remoteLoading")==0 ||
    _bcTType.compare("symm_fault")==0 ||
    _bcTType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcLType.compare("freeSurface")==0 ||
    _bcLType.compare("tau")==0 ||
    _bcLType.compare("remoteLoading")==0 ||
    _bcLType.compare("symm_fault")==0 ||
    _bcLType.compare("rigid_fault")==0 );

  assert(_bcLType.compare("outGoingCharacteristics")==0 ||
    _bcBType.compare("freeSurface")==0 ||
    _bcBType.compare("tau")==0 ||
    _bcBType.compare("remoteLoading")==0 ||
    _bcBType.compare("symm_fault")==0 ||
    _bcBType.compare("rigid_fault")==0 );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode StrikeSlip_PowerLaw_fd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) { _material->setMMSInitialConditions(_initTime); }

  _fault->initiateIntegrand(_initTime,_varEx);

  Vec slip;
  VecDuplicate(_varEx["psi"], &slip); VecSet(slip,0.);
  _varEx["slip"] = slip;
  Vec dslip;
  VecDuplicate(_varEx["psi"], &dslip); VecSet(dslip,0.);
  _varEx["dslip"] = dslip;

  VecDuplicate(*_z, &_varEx["uPrev"]); VecSet(_varEx["uPrev"],0.);
  VecDuplicate(*_z, &_varEx["u"]); VecSet(_varEx["u"], 0.0);

  if (_inputDir.compare("unspecified") != 0){

    ierr = loadFileIfExists_matlab(_inputDir+"u", _varEx["u"]);
    if (ierr == 1){
        PetscInt Ii,Istart,Iend;
        PetscInt Jj = 0;

      if (_initialConditions.compare("u") == 0){
        PetscScalar *u, *uPrev, *y, *z;
        VecGetOwnershipRange(_varEx["u"],&Istart,&Iend);
        VecGetArray(_varEx["u"],&u);
        VecGetArray(_varEx["uPrev"],&uPrev);
        VecGetArray(*_y, &y);
        VecGetArray(*_z, &z);

        for (Ii=Istart;Ii<Iend;Ii++) {
          u[Jj] = 10 * exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
          uPrev[Jj] = 10 *exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
          Jj++;
        }
        VecRestoreArray(*_y,&y);
        VecRestoreArray(*_z,&z);
        VecRestoreArray(_varEx["u"],&u);
        VecRestoreArray(_varEx["uPrev"],&uPrev);
      }
    }

    ierr = loadFileIfExists_matlab(_inputDir+"uPrev", _varEx["uPrev"]);
    if (ierr == 1){
      VecCopy(_varEx["u"], _varEx["uPrev"]);
    }

    // ##### TODO : Propagate the version of switch for the loading ####
    // Vec lastDeltaT;
    // ierr = VecCreate(PETSC_COMM_WORLD,&lastDeltaT); CHKERRQ(ierr);
    // ierr = VecSetSizes(lastDeltaT,PETSC_DECIDE,1); CHKERRQ(ierr);
    // ierr = VecSetFromOptions(lastDeltaT); CHKERRQ(ierr);

    ierr = loadFileIfExists_matlab(_inputDir + "psi", _fault->_psi);
    ierr = loadFileIfExists_matlab(_inputDir + "psi", _fault->_psiPrev);
    // // ierr = loadFileIfExists_matlab(_inputDir + "tau0", _fault->_tau0);
    ierr = loadFileIfExists_matlab(_inputDir + "slipVel", _varEx["dslip"]);
    // ierr = loadFileIfExists_matlab(_inputDir + "deltaT", lastDeltaT);
    // PetscScalar* lastT;
    // PetscScalar tempdeltaT;
    // VecGetArray(lastDeltaT, &lastT);
    // tempdeltaT = lastT[0];
    // VecDestroy(&lastDeltaT);

    // Vec psiComputation;
    // VecDuplicate(_fault->_psi, &psiComputation);
    // VecCopy(_fault->_psi, psiComputation);
    // VecAXPY(psiComputation, -1.0, _fault->_psiPrev);
    // VecAXPY(_fault->_psiPrev, 1.0 - _deltaT / tempdeltaT, psiComputation);
    // VecDestroy(&psiComputation);

    VecCopy(_varEx["u"], _material->_u);
    _material->computeStresses();
    Vec sxy,sxz,sdev;
    ierr = _material->getStresses(sxy,sxz,sdev);
    ierr = _fault->setTauQS(sxy); CHKERRQ(ierr);
    VecCopy(_fault->_tauQSP, _fault->_tau0);

    VecSet(_varEx["u"], 0.0);
    VecSet(_varEx["uPrev"], 0.0);

    // Vec uPrevTemp;
    // VecDuplicate(_varEx["u"], &uPrevTemp);
    // VecCopy(_varEx["u"], uPrevTemp);
    // VecAXPY(uPrevTemp, -1.0, _varEx["uPrev"]);
    // VecAXPY(_varEx["uPrev"], 1.0 - _deltaT / tempdeltaT, uPrevTemp);
    // VecDestroy(&uPrevTemp);

    // VecCopy(_fault->_psi, _varEx["psi"]);

    ierr = 0;
  }
  else{

  PetscInt Ii,Istart,Iend;
  PetscInt Jj = 0;

  if (_initialConditions.compare("u") == 0){
    PetscScalar *u, *uPrev, *y, *z;
    VecGetOwnershipRange(_varEx["u"],&Istart,&Iend);
    VecGetArray(_varEx["u"],&u);
    VecGetArray(_varEx["uPrev"],&uPrev);
    VecGetArray(*_y, &y);
    VecGetArray(*_z, &z);

    for (Ii=Istart;Ii<Iend;Ii++) {
      u[Jj] = 10 * exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
      uPrev[Jj] = 10 *exp(-pow( y[Jj]-0.3*(_Ly), 2) /5) * exp(-pow(z[Jj]-0.8*(_Lz), 2) /5);
      Jj++;
    }
    VecRestoreArray(*_y,&y);
    VecRestoreArray(*_z,&z);
    VecRestoreArray(_varEx["u"],&u);
    VecRestoreArray(_varEx["uPrev"],&uPrev);
  }

  }
  PetscInt Ii,Istart,Iend;
  PetscInt Jj = 0;
    // Create matrix _ay
    VecDuplicate(*_z, &_ay);
    VecSet(_ay, 0.0);
    Vec _ay_temp, _az_temp;
    VecDuplicate(*_z, &_ay_temp);
    VecSet(_ay_temp, 0.0);
    VecDuplicate(*_z, &_az_temp);
    VecSet(_az_temp, 0.0);

    PetscScalar *ay, *alphay, *alphaz;
    VecGetOwnershipRange(*_y,&Istart,&Iend);
    VecGetArray(_ay,&ay);
    VecGetArray(_alphay, &alphay);
    VecGetArray(_alphaz, &alphaz);
    Jj = 0;

    for (Ii=Istart;Ii<Iend;Ii++) {
      ay[Jj] = 0;
      if ((Ii/_Nz == 0) && (_bcLType.compare("outGoingCharacteristics") == 0)){ay[Jj] += 0.5 / alphay[Jj];}
      if ((Ii/_Nz == _Ny-1) && (_bcRType.compare("outGoingCharacteristics") == 0)){ay[Jj] += 0.5 / alphay[Jj];}
      if ((Ii%_Nz == 0) && (_bcTType.compare("outGoingCharacteristics") == 0)){ay[Jj] += 0.5 / alphaz[Jj];}
      if (((Ii+1)%_Nz == 0) && (_bcBType.compare("outGoingCharacteristics") == 0)){ay[Jj] += 0.5 / alphaz[Jj];}
      Jj++;
    }

    VecRestoreArray(_ay,&ay);
    VecRestoreArray(_alphay, &alphay);
    VecRestoreArray(_alphaz, &alphaz);

    ierr = VecPointwiseMult(_ay, _ay, _cs);

  _fault->initiateIntegrand_dyn(_varEx, _rhoVec);
    // if(_D->_sbpType.compare("mfc_coordTrans")==0){
    //   Mat J,Jinv,qy,rz,yq,zr;
    //   ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    //   Vec temp;
    //   VecDuplicate(_material->_rhoVec, &temp);
    //   MatMult(J, _material->_rhoVec, temp);
    //   VecCopy(temp, _material->_rhoVec);
    //   VecDestroy(&temp);
    // }

  if (_thermalCoupling.compare("no")!=0 ) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
     _fault->updateTemperature(_he->_T);
  }

  if (_hydraulicCoupling.compare("no")!=0 ) {
     _p->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// monitoring function for explicit integration
PetscErrorCode StrikeSlip_PowerLaw_fd::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,int& stopIntegration)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::timeMonitor for explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  if ( _stride1D > 0 && stepCount % _stride1D == 0) {
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
  }

  if ( _stride2D > 0 && stepCount % _stride2D == 0) {
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
  }

_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// monitoring function for IMEX integration
PetscErrorCode StrikeSlip_PowerLaw_fd::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm,int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::timeMonitor for IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  _stepCount = stepCount;
  _currTime = time;

  if ( stepCount % _stride1D == 0) {
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
  }

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_PowerLaw_fd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault->view(_integrateTime);
  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Domain Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Nz: %i\n",_Nz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   Ny: %i\n",_Ny);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of processors: %i\n",num_proc);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_PowerLaw_fd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent propagating the wave (s): %g\n",_propagateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode StrikeSlip_PowerLaw_fd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // output scalar fields
  std::string str = _outputDir + "mediator_context.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"deltaT = %.15e # (s)\n",_deltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"atol = %.15e\n",_atol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _material->writeContext(_outputDir);
  _fault->writeContext(_outputDir);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

//======================================================================
// Adaptive time stepping functions
//======================================================================
PetscErrorCode StrikeSlip_PowerLaw_fd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_PowerLaw_fd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  _quadWaveImex = new OdeSolver_WaveImex(_maxStepCount,_initTime,_maxTime,_deltaT);
  ierr = _quadWaveImex->setInitialConds(_varEx, _varIm);CHKERRQ(ierr);

  ierr = _quadWaveImex->integrate(this);CHKERRQ(ierr);

  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// purely explicit time stepping// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_PowerLaw_fd::d_dt(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx,
                                             map<string,Vec>& varIm,map<string,Vec>& varImo)
{
  PetscErrorCode ierr = 0;
  // ierr = _material->_sbp->setRhs(_material->_rhs,_material->_bcL,_material->_bcR,_material->_bcT,_material->_bcB);CHKERRQ(ierr);
  Mat A;
  ierr = _material->_sbp->getA(A);

  double startPropagation = MPI_Wtime();


  // if (varEx.find("pressure") != varEx.end() && _hydraulicCoupling.compare("no")!=0) {
  //   _p->updateFields(time,varEx);
  // }

  if (varImo.find("Temp") != varImo.end() && _thermalCoupling.compare("coupled")==0) {
    _fault->updateTemperature(varImo.find("Temp")->second);
    _material->updateTemperature(varImo.find("Temp")->second);
  }

    // update effective normal stress in fault using pore pressure
  // if (_hydraulicCoupling.compare("coupled")==0) {
  //   _fault->setSNEff(_p->_p);
  // }

  // Update the laplacian
  Vec Laplacian, temp;
  VecDuplicate(*_y, &Laplacian);
  VecDuplicate(*_y, &temp);
  ierr = MatMult(A, varEx["u"], temp);
  ierr = _material->_sbp->Hinv(temp, Laplacian);
  ierr = VecCopy(Laplacian, dvarEx["u"]);
  VecDestroy(&temp);

  if(_D->_sbpType.compare("mfc_coordTrans")==0){
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    Vec temp;
    // MatView(J, PETSC_VIEWER_STDOUT_WORLD);
    VecDuplicate(dvarEx["u"], &temp);
    MatMult(Jinv, dvarEx["u"], temp);
    VecCopy(temp, dvarEx["u"]);
    VecDestroy(&temp);
  }

  Vec uNext, correction, previous, ones;

  VecDuplicate(varEx["u"], &ones);
  VecDuplicate(varEx["u"], &correction);
  VecSet(ones, 1.0);
  VecSet(correction, 0.0);
  ierr = VecAXPY(correction, _deltaT, _ay);
  ierr = VecAXPY(correction, -1.0, ones);
  VecDuplicate(varEx["u"], &previous);
  VecSet(previous, 0.0);
  ierr = VecPointwiseMult(previous, correction, varEx["uPrev"]);

  VecDuplicate(varEx["u"], &uNext);
  VecSet(uNext, 0.0);
  ierr = VecAXPY(uNext, pow(_deltaT, 2), dvarEx["u"]);
  ierr = VecPointwiseDivide(uNext, uNext, _rhoVec);

  ierr = VecAXPY(uNext, 2, varEx["u"]);
  ierr = VecAXPY(uNext, 1, previous);
  ierr = VecAXPY(correction, 2, ones);
  ierr = VecPointwiseDivide(uNext, uNext, correction);

  ierr = VecCopy(varEx["u"], varEx["uPrev"]);
  ierr = VecCopy(uNext, varEx["u"]);
  VecDestroy(&uNext);
  VecDestroy(&ones);
  VecDestroy(&correction);
  VecDestroy(&previous);
  if (_initialConditions.compare("tau")==0){
    PetscScalar currT;
    _quadWaveImex->getCurrT(currT);
    ierr = _fault->updateTau(currT);
  }

  _propagateTime += MPI_Wtime() - startPropagation;

  if (_isFault.compare("true") == 0){
  ierr = _fault->d_dt(time,varEx,dvarEx, _deltaT);CHKERRQ(ierr);
}

  VecCopy(varEx["u"], _material->_u);
  _material->computeStresses();

  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = _fault->setTauQS(sxy); CHKERRQ(ierr);
  VecCopy(_fault->_tauQSP, _fault->_tauP);
  VecAXPY(_fault->_tauP, 1.0, _fault->_tau0);

  // heat equation
  if (varIm.find("Temp") != varIm.end()) {
    Vec sxy,sxz,sdev;
    _material->getStresses(sxy,sxz,sdev);
    Vec V = _fault->_slipVel;
    Vec tau = _fault->_tauP;
    Vec Told = varImo.find("Temp")->second;
    ierr = _he->be(time,V,tau,NULL,NULL,NULL,varIm["Temp"],Told,_deltaT); CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
  }

  return ierr;
}
