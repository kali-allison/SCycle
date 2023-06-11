#include "heatEquation.hpp"

#define FILENAME "heatEquation.cpp"

using namespace std;

HeatEquation::HeatEquation(Domain& D)
: _D(&D),_order(D._order),_Ny(D._Ny),_Nz(D._Nz),_Nz_lab(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(D._dq),_dz(D._dr),_Lz_lab(D._Lz),_y(&D._y),_z(&D._z),
  _heatEquationType("transient"),_isMMS(D._isMMS),_loadICs(0),
  _file(D._file),_inputDir(D._inputDir),_outputDir(D._outputDir),_delim(D._delim),
  _kTz_z0(NULL),_kTz(NULL),_maxdTVec(NULL),
  _bcRType_ss("Dirichlet"),_bcTType_ss("Dirichlet"),_bcLType_ss("Neumann"),_bcBType_ss("Dirichlet"),
  _bcRType_trans("Dirichlet"),_bcTType_trans("Dirichlet"),_bcLType_trans("Neumann"),_bcBType_trans("Dirichlet"),
  _wViscShearHeating("yes"),_wFrictionalHeating("yes"),_wRadioHeatGen("yes"),
  _sbp(NULL),
  _bcR(NULL),_bcT(NULL),_bcL(NULL),_bcB(NULL),
  _linSolver("CG_PCAMG"),_kspTol(1e-11),
  _kspSS(NULL),_kspTrans(NULL),_pc(NULL),
  _I(NULL),_rcInv(NULL),_B(NULL),_pcMat(NULL),_D2ath(NULL),
  _MapV(NULL),_Gw(NULL),_w(NULL),
  _linSolveTime(0),_factorTime(0),_beTime(0),_writeTime(0),_miscTime(0),
  _linSolveCount(0),
  _Tamb(NULL),_dT(NULL),_T(NULL),
  _k(NULL),_rho(NULL),_c(NULL),_Qrad(NULL),_Qfric(NULL),_Qvisc(NULL),_Q(NULL)
{
  #if VERBOSE > 1
    string funcName = "HeatEquation::HeatEquation";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(_file);

  checkInput();
  allocateFields();
  setFields();

  if (_D->_restartFromChkpt) { loadCheckpoint(); }
  else if (_D->_restartFromChkptSS) { loadCheckpointSS(); }
  else { loadFieldsFromFiles(); }

  if (_D->_restartFromChkpt == 0 && _D->_restartFromChkptSS == 0 &&_loadICs == 0 && _isMMS == 0 ) { computeInitialSteadyStateTemp(); }

  if (_heatEquationType == "transient" ) { setUpTransientProblem();}
  else if (_heatEquationType == "steadyState" ) { setUpSteadyStateProblem(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

HeatEquation::~HeatEquation()
{
  KSPDestroy(&_kspSS);
  KSPDestroy(&_kspTrans);
  MatDestroy(&_B);
  MatDestroy(&_rcInv);
  MatDestroy(&_I);
  MatDestroy(&_D2ath);
  MatDestroy(&_pcMat);

  MatDestroy(&_MapV);
  VecDestroy(&_Gw);
  VecDestroy(&_w);
  VecDestroy(&_Qrad);
  VecDestroy(&_Qfric);
  VecDestroy(&_Qvisc);
  VecDestroy(&_Q);

  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);

  VecDestroy(&_Tamb);
  VecDestroy(&_dT);
  VecDestroy(&_T);
  VecDestroy(&_kTz);
  VecDestroy(&_kTz_z0);

  // boundary conditions
  VecDestroy(&_bcR);
  VecDestroy(&_bcT);
  VecDestroy(&_bcL);
  VecDestroy(&_bcB);

  //~ for (map<string,pair<PetscViewer,string> >::iterator it=_viewers2D.begin(); it !=_viewers2D.end(); it++) {
    //~ PetscViewerDestroy(&_viewers2D[it->first].first);
  //~ }
  //~ PetscViewerDestroy(&_viewer1D_hdf5);

  map<string,VecScatter>::iterator it;
  for (it = _scatters.begin(); it!=_scatters.end(); it++ ) {
    VecScatterDestroy(&it->second);
  }

  delete _sbp;
  _sbp = NULL;
}


// return temperature
PetscErrorCode HeatEquation::getTemp(Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::getTemp()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(_T,T);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// set temperature
PetscErrorCode HeatEquation::setTemp(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setTemp()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_T);
  VecWAXPY(_dT,-1.0,_Tamb,_T);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// loads settings from the input text file
PetscErrorCode HeatEquation::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    string funcName = "HeatEquation::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
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

    if (var.compare("heatEquationType")==0) { _heatEquationType = rhs.c_str(); }
    else if (var.compare("withViscShearHeating")==0) { _wViscShearHeating = rhs.c_str(); }
    else if (var.compare("withFrictionalHeating")==0) { _wFrictionalHeating = rhs.c_str();}
    else if (var.compare("withRadioHeatGeneration")==0) { _wRadioHeatGen = rhs.c_str(); }

    // linear solver settings
    else if (var.compare("linSolver_heateq")==0) { _linSolver = rhs.c_str(); }
    else if (var.compare("kspTol_heateq")==0) { _kspTol = atof( rhs.c_str() ); }

    // if values are set by vector
    else if (var.compare("rhoVals")==0) { loadVectorFromInputFile(rhsFull,_rhoVals); }
    else if (var.compare("rhoDepths")==0) { loadVectorFromInputFile(rhsFull,_rhoDepths); }
    else if (var.compare("kVals")==0) { loadVectorFromInputFile(rhsFull,_kVals); }
    else if (var.compare("kDepths")==0) { loadVectorFromInputFile(rhsFull,_kDepths); }
    else if (var.compare("cVals")==0) { loadVectorFromInputFile(rhsFull,_cVals); }
    else if (var.compare("cDepths")==0) { loadVectorFromInputFile(rhsFull,_cDepths); }


    else if (var.compare("Nz_lab")==0) { _Nz_lab = atoi( rhs.c_str() ); }
    else if (var.compare("TVals")==0) { // TVals = [T0 T_lab TN] || [T0 TN]
      loadVectorFromInputFile(rhsFull,_TVals);
    }
    else if (var.compare("TDepths")==0) {
      loadVectorFromInputFile(rhsFull,_TDepths);
      assert(_TDepths.size() >= 2 && _TDepths.size() <= 4);
      if (_TDepths.size()>2) { _Lz_lab = _TDepths[1]; }
      else { _Lz_lab = _TDepths[0]; }
    }

    else if (var.compare("initTime")==0) { _initTime = atof( rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( rhs.c_str() ); }

    // boundary conditions
    else if (var.compare("bcRType_ss")==0) { _bcRType_ss = rhs.c_str(); }
    else if (var.compare("bcTType_ss")==0) { _bcTType_ss = rhs.c_str(); }
    else if (var.compare("bcLType_ss")==0) { _bcLType_ss = rhs.c_str(); }
    else if (var.compare("bcBType_ss")==0) { _bcBType_ss = rhs.c_str(); }
    else if (var.compare("bcRType_trans")==0) { _bcRType_trans = rhs.c_str(); }
    else if (var.compare("bcTType_trans")==0) { _bcTType_trans = rhs.c_str(); }
    else if (var.compare("bcLType_trans")==0) { _bcLType_trans = rhs.c_str(); }
    else if (var.compare("bcBType_trans")==0) { _bcBType_trans = rhs.c_str(); }

    // finite width shear zone
    else if (var.compare("wVals")==0) { loadVectorFromInputFile(rhsFull,_wVals); }
    else if (var.compare("wDepths")==0) { loadVectorFromInputFile(rhsFull,_wDepths); }

    // radioactive heat generation
    else if (var.compare("he_A0Vals")==0) { loadVectorFromInputFile(rhsFull,_A0Vals); }
    else if (var.compare("he_A0Depths")==0) { loadVectorFromInputFile(rhsFull,_A0Depths); }
    else if (var.compare("he_Lrad")==0) { _Lrad = atof( rhs.c_str() ); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

//parse input file and load values into data members
PetscErrorCode HeatEquation::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::loadFieldsFromFiles()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // material properties
  ierr = loadVecFromInputFile(_rho,_inputDir,"rho"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_k,_inputDir,"k"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_Qrad,_inputDir,"h"); CHKERRQ(ierr);
  ierr = loadVecFromInputFile(_c,_inputDir,"c"); CHKERRQ(ierr);

  bool chkTamb = 0, chkT = 0, chkdT = 0;

  // load Tamb (background geotherm)
  loadVecFromInputFile(_Tamb,_inputDir,"Tamb",chkTamb);
  _loadICs = chkTamb;

  // load T
  loadVecFromInputFile(_T,_inputDir,"T",chkT);

  // if Tamb was loaded and T wasn't, copy Tamb into T
  if (chkT!=1 && chkTamb) { VecCopy(_Tamb,_T); }

  // load dT (perturbation from ambient geotherm)
  loadVecFromInputFile(_dT,_inputDir,"dT",chkdT);

  // dT wasn't loaded, compute it from T and Tamb
  if (chkdT!=1 && chkTamb) {
    VecWAXPY(_dT,-1.0,_Tamb,_T);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// allocate memory and parallel data layout for fields
PetscErrorCode HeatEquation::allocateFields()
{
PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::allocateFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // allocate boundary conditions
  VecDuplicate(_D->_z0,&_bcT); VecSet(_bcT,0.);      PetscObjectSetName((PetscObject) _bcT, "bcT");
  VecDuplicate(_D->_z0,&_bcB); VecSet(_bcB,0.);      PetscObjectSetName((PetscObject) _bcB, "bcB");
  VecDuplicate(_D->_y0,&_bcR); VecSet(_bcR,0.0);     PetscObjectSetName((PetscObject) _bcR, "bcR");
  VecDuplicate(_D->_y0,&_bcL); VecSet(_bcL,0.0);     PetscObjectSetName((PetscObject) _bcL, "bcL");

  VecDuplicate(_bcT,&_kTz_z0); VecSet(_kTz_z0,0.0);  PetscObjectSetName((PetscObject) _kTz_z0, "kTz_z0"); // surface heat flux

  VecDuplicate(*_y,&_k);       VecSet(_k,0.0);       PetscObjectSetName((PetscObject) _k, "k"); // conductivity
  VecDuplicate(_k,&_rho);      VecSet(_rho,0.);      PetscObjectSetName((PetscObject) _rho, "rho"); // density
  VecDuplicate(_k,&_c);        VecSet(_c,0.);        PetscObjectSetName((PetscObject) _c, "c"); // heat capacity
  VecDuplicate(_k,&_Q);        VecSet(_Q,0.);        PetscObjectSetName((PetscObject) _Q, "Q");
  VecDuplicate(_k,&_Qrad);     VecSet(_Qrad,0.);     PetscObjectSetName((PetscObject) _Qrad, "Qrad");
  VecDuplicate(_k,&_Qfric);    VecSet(_Qfric,0.);    PetscObjectSetName((PetscObject) _Qfric, "Qfric");
  VecDuplicate(_k,&_Qvisc);    VecSet(_Qvisc,0.);    PetscObjectSetName((PetscObject) _Qvisc, "Qvisc");
  VecDuplicate(_k,&_kTz);      VecSet(_kTz,0.0);     PetscObjectSetName((PetscObject) _kTz, "kTz");
  VecDuplicate(_k,&_T);        VecSet(_T,0.);        PetscObjectSetName((PetscObject) _T, "T");
  VecDuplicate(_k,&_Tamb);     VecSet(_Tamb,0.);     PetscObjectSetName((PetscObject) _Tamb, "Tamb");
  VecDuplicate(_k,&_dT);       VecSet(_dT,0.);       PetscObjectSetName((PetscObject) _dT, "dT");

  // initiate Vec to hold max temperature for output
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_maxdTVec);
  VecSetBlockSize(_maxdTVec, 1);
  PetscObjectSetName((PetscObject) _maxdTVec, "maxdT"); VecSet(_maxdTVec,0.);

  // create scatter from body field to top boundary
  // indices to scatter from
  IS isf;
  ierr = ISCreateStride(PETSC_COMM_WORLD, _Ny, 0, _Nz, &isf); CHKERRQ(ierr);

  // indices to scatter to
  PetscInt *ti;
  ierr = PetscMalloc1(_Ny,&ti); CHKERRQ(ierr);
  for (PetscInt Ii=0; Ii<(_Ny); Ii++) { ti[Ii] = Ii; }
  IS ist;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny, ti, PETSC_COPY_VALUES, &ist); CHKERRQ(ierr);
  PetscFree(ti);

  // create scatter
  ierr = VecScatterCreate(*_y, isf, _bcT, ist, &_scatters["body2T"]); CHKERRQ(ierr);
  ISDestroy(&isf);
  ISDestroy(&ist);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}


// initialize values for all fields
PetscErrorCode HeatEquation::setFields()
{
PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setFields";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set each field using it's vals and depths vectors
  if (_isMMS) {
    mapToVec(_k,zzmms_k,*_y,*_z);
    mapToVec(_rho,zzmms_rho,*_y,*_z);
    mapToVec(_c,zzmms_c,*_y,*_z);
    mapToVec(_Qrad,zzmms_h,*_y,*_z);
    mapToVec(_Tamb,zzmms_T,*_y,*_z,_initTime);
    mapToVec(_dT,zzmms_dT,*_y,*_z,_initTime);
    setMMSBoundaryConditions(_initTime,"Dirichlet","Dirichlet","Dirichlet","Dirichlet");
  }
  else {
    ierr = setVec(_k,*_z,_kVals,_kDepths); CHKERRQ(ierr);
    ierr = setVec(_rho,*_z,_rhoVals,_rhoDepths); CHKERRQ(ierr);
    ierr = setVec(_c,*_z,_cVals,_cDepths); CHKERRQ(ierr);
    ierr = setVec(_T,*_z,_TVals,_TDepths); CHKERRQ(ierr);
    ierr = setVec(_Tamb,*_z,_TVals,_TDepths); CHKERRQ(ierr);
  }

  if (_wFrictionalHeating.compare("yes")==0) { constructMapV(); }

  // set up radioactive heat generation source term
  // Qrad = A0 * exp(-z/Lrad)
  if (_wRadioHeatGen.compare("yes") == 0) {
    Vec A0; VecDuplicate(_Qrad,&A0);
    ierr = setVec(A0,*_z,_A0Vals,_A0Depths); CHKERRQ(ierr);
    VecCopy(*_z,_Qrad);
    VecScale(_Qrad,-1.0/_Lrad);
    VecExp(_Qrad);
    VecPointwiseMult(_Qrad,A0,_Qrad);
    VecDestroy(&A0);
  }
  else {
    VecSet(_Qrad,0.);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode HeatEquation::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_heatEquationType.compare("transient")==0 ||
      _heatEquationType.compare("steadyState")==0 );

  assert(_bcRType_ss == "Dirichlet" || _bcRType_ss == "Neumann");
  assert(_bcTType_ss == "Dirichlet" || _bcTType_ss == "Neumann");
  assert(_bcLType_ss == "Dirichlet" || _bcLType_ss == "Neumann");
  assert(_bcBType_ss == "Dirichlet" || _bcBType_ss == "Neumann");
  assert(_bcRType_trans == "Dirichlet" || _bcRType_trans == "Neumann");
  assert(_bcTType_trans == "Dirichlet" || _bcTType_trans == "Neumann");
  assert(_bcLType_trans == "Dirichlet" || _bcLType_trans == "Neumann");
  assert(_bcBType_trans == "Dirichlet" || _bcBType_trans == "Neumann");

  assert(_kVals.size() == _kDepths.size() );
  assert(_rhoVals.size() == _rhoDepths.size() );
  assert(_cVals.size() == _cDepths.size() );
  assert(_TVals.size() == _TDepths.size() );
  assert(_wVals.size() == _wDepths.size() );

  assert(_TVals.size() == _TDepths.size() );
  assert(_Nz_lab <= _Nz);
  assert(_Lz_lab <= _Lz);

  if (_wRadioHeatGen.compare("yes") == 0) {
    assert(_TVals.size() >= 2 && _TVals.size() <= 4);
    assert(_A0Vals.size() == _A0Depths.size() );
    if (_A0Vals.size() == 0) {
      _A0Vals.push_back(0); _A0Depths.push_back(0);
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// create scatters for communication from full domain Vecs to lithosphere only Vecs
// SCATTER_FORWARD is from full domain -> lithosphere
PetscErrorCode HeatEquation::constructScatters(Vec& T, Vec& T_l)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::constructScatters()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  // create scatter from 2D full domain to 2D lithosphere only
  // indices to scatter from
  PetscInt *fi;
  ierr = PetscMalloc1(_Ny*_Nz_lab,&fi); CHKERRQ(ierr);
  PetscInt count = 0;

  for (PetscInt Ii=0; Ii<_Ny; Ii++) {
    for (PetscInt Jj=0; Jj<_Nz_lab; Jj++) {
      fi[count] = Ii*_Nz + Jj;
      count++;
    }
  }

  IS isf;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny*_Nz_lab, fi, PETSC_COPY_VALUES, &isf); CHKERRQ(ierr);
  PetscFree(fi);

  // indices to scatter to
  PetscInt *ti;
  ierr = PetscMalloc1(_Ny*_Nz_lab,&ti); CHKERRQ(ierr);
  for (PetscInt Ii=0; Ii<(_Ny*_Nz_lab); Ii++) {
    ti[Ii] = Ii;
  }

  IS ist;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, _Ny*_Nz_lab, ti, PETSC_COPY_VALUES, &ist); CHKERRQ(ierr);
  PetscFree(ti);

  // create scatter
  ierr = VecScatterCreate(_T, isf, T_l, ist, &_scatters["bodyFull2bodyLith"]); CHKERRQ(ierr);
  // free memory
  ISDestroy(&isf);
  ISDestroy(&ist);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// create matrix to map slip velocity, which lives on the fault, to a 2D body field
PetscErrorCode HeatEquation::constructMapV()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::constructMapV";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  MatCreate(PETSC_COMM_WORLD,&_MapV);
  MatSetSizes(_MapV,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Nz);
  MatSetFromOptions(_MapV);
  MatMPIAIJSetPreallocation(_MapV,1,NULL,1,NULL);
  MatSeqAIJSetPreallocation(_MapV,1,NULL);
  MatSetUp(_MapV);

  PetscScalar v=1.0;
  PetscInt Ii=0,Istart=0,Iend=0,Jj=0;
  MatGetOwnershipRange(_MapV,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    Jj = Ii % _Nz;
    MatSetValues(_MapV,1,&Ii,1,&Jj,&v,INSERT_VALUES);
  }

  MatAssemblyBegin(_MapV,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_MapV,MAT_FINAL_ASSEMBLY);

  // construct Gw = exp(-y^2/(2*w)) / sqrt(2*pi)/w
  VecDuplicate(_k,&_Gw); VecSet(_Gw,0.);        PetscObjectSetName((PetscObject) _Gw, "Gw");
  VecDuplicate(_k,&_w);        PetscObjectSetName((PetscObject) _w, "w");

  if (_wVals.size() > 0 ) {
    ierr = setVec(_w,*_z,_wVals,_wDepths); CHKERRQ(ierr);
    VecScale(_w,1e-3); // convert from m to km
  }
  else { VecSet(_w,0.); }
  VecMax(_w,NULL,&_wMax);

  if (_wVals.size() > 0 ) {
    PetscScalar const *y,*w;
    PetscScalar *g;
    VecGetOwnershipRange(_Gw,&Istart,&Iend);
    VecGetArrayRead(*_y,&y);
    VecGetArrayRead(_w,&w);
    VecGetArray(_Gw,&g);
    Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      g[Jj] = exp(-y[Jj]*y[Jj] / (2.*w[Jj]*w[Jj])) / sqrt(2. * M_PI) / w[Jj];
      assert(!std::isnan(g[Jj]));
      assert(!std::isinf(g[Jj]));
      Jj++;
    }
    VecRestoreArrayRead(*_y,&y);
    VecRestoreArrayRead(_w,&w);
    VecRestoreArray(_Gw,&g);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute 1D steady-state geotherm in thelithosphere, optionally
// including radioactive decay as a heat source
PetscErrorCode HeatEquation::computeInitialSteadyStateTemp()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeInitialSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // no need for linear solve step if Nz == 1
  if (_Nz == 1) {
    VecSet(_Tamb,_TVals[0]);
    VecSet(_T,_TVals[0]);
    VecSet(_dT,0.0);
    return 0;
  }

  // if no radioactive heat generation, just set Tamb from input TVals and TDepths
  if (_wRadioHeatGen == "no") {
    setVec(_Tamb,*_z,_TVals,_TDepths);
    // update _T, _dT
    VecSet(_dT,0.0);
    VecCopy(_Tamb,_T);
    PetscPrintf(PETSC_COMM_WORLD,"Creating Tamb purely through .in file inputs.\n");
    return 0;
  }

  // otherwise:
  // boundary conditions
  Vec bcT,bcB;
  ierr = VecCreate(PETSC_COMM_WORLD,&bcT); CHKERRQ(ierr);
  ierr = VecSetSizes(bcT,PETSC_DECIDE,_Ny); CHKERRQ(ierr);
  ierr = VecSetFromOptions(bcT); CHKERRQ(ierr);
  VecDuplicate(bcT,&bcB);
  PetscScalar bcTval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (0-_TDepths[0]) + _TVals[0];
  VecSet(bcT,bcTval);
  PetscScalar bcBval = (_TVals[1] - _TVals[0])/(_TDepths[1]-_TDepths[0]) * (_Lz_lab-_TDepths[0]) + _TVals[0];
  VecSet(bcB,bcBval);


  // fields that live only in the lithosphere
  Vec y,z,k,Qrad,Tamb_l;
  ierr = VecCreate(PETSC_COMM_WORLD,&y); CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,_Ny*_Nz_lab); CHKERRQ(ierr);
  ierr = VecSetFromOptions(y); CHKERRQ(ierr);
  VecSet(y,0.0);
  VecDuplicate(y,&z); VecSet(z,0.0);
  VecDuplicate(y,&k); VecSet(k,0.0);
  VecDuplicate(y,&Qrad); VecSet(Qrad,0.0);
  VecDuplicate(y,&Tamb_l); VecSet(Tamb_l,0.0);

  constructScatters(_T,Tamb_l);

  VecScatterBegin(_scatters["bodyFull2bodyLith"], *_y, y, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["bodyFull2bodyLith"], *_y, y, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(_scatters["bodyFull2bodyLith"], *_z, z, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["bodyFull2bodyLith"], *_z, z, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(_scatters["bodyFull2bodyLith"], _k, k, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["bodyFull2bodyLith"], _k, k, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterBegin(_scatters["bodyFull2bodyLith"], _Qrad, Qrad, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["bodyFull2bodyLith"], _Qrad, Qrad, INSERT_VALUES, SCATTER_FORWARD);

  // create SBP operators, 1D in z-direction only, only in lithosphere
  SbpOps* sbp;
  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    sbp = new SbpOps_m_constGrid(_order,_Ny,_Nz_lab,_Ly,_Lz,k);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    sbp = new SbpOps_m_varGrid(_order,_Ny,_Nz_lab,_Ly,_Lz,k);
    if (_Ny > 1 && _Nz_lab > 1) { sbp->setGrid(&y,&z); }
    else if (_Ny == 1 && _Nz_lab > 1) { sbp->setGrid(NULL,&z); }
    else if (_Ny > 1 && _Nz_lab == 1) { sbp->setGrid(&y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }

  sbp->setCompatibilityType(_D->_sbpCompatibilityType);
  sbp->setBCTypes("Dirichlet","Dirichlet","Dirichlet","Dirichlet");
  sbp->setMultiplyByH(1);
  sbp->setLaplaceType("z");
  sbp->setDeleteIntermediateFields(1);
  sbp->computeMatrices(); // actually create the matrices

  Mat A; sbp->getA(A);
  setupKSP_SS(A); // set up KSP for steady-state problem

  Vec rhs; VecDuplicate(k,&rhs); VecSet(rhs,0.);
  sbp->setRhs(rhs,_bcL,_bcR,bcT,bcB);

  // radioactive heat generation source term
  // Vec QradR,Qtemp;
  Vec Qtemp = NULL;
  if (_wRadioHeatGen.compare("yes") == 0) {
    VecDuplicate(Qrad,&Qtemp);
    if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
      Vec temp1; VecDuplicate(Qrad,&temp1);
      Mat J,Jinv,qy,rz,yq,zr;
      ierr = sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
      ierr = MatMult(J,Qrad,temp1);
      Mat H; sbp->getH(H);
      ierr = MatMult(H,temp1,Qtemp);
      VecDestroy(&temp1);
    }
    else{
      Mat H; sbp->getH(H);
      ierr = MatMult(H,Qrad,Qtemp); CHKERRQ(ierr);
    }

    VecAXPY(rhs,-1.0,Qtemp);
    VecDestroy(&Qtemp);
  }

  // solve for ambient temperature in the lithosphere
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspSS,rhs,Tamb_l);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  // scatter Tamb_l to Tamb and T
  VecScatterBegin(_scatters["bodyFull2bodyLith"], Tamb_l,_Tamb, INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(_scatters["bodyFull2bodyLith"], Tamb_l,_Tamb, INSERT_VALUES, SCATTER_REVERSE);

  KSPDestroy(&_kspSS); _kspSS = NULL;
  delete sbp; sbp = NULL;
  VecDestroy(&y);
  VecDestroy(&z);
  VecDestroy(&k);
  VecDestroy(&Qrad);
  VecDestroy(&rhs);
  VecDestroy(&Tamb_l);
  VecDestroy(&bcT);
  VecDestroy(&bcB);


  // now overwrite Tamb(z>=LAB) with mantle adiabat
  if (_Nz_lab < _Nz && _Lz_lab < _Lz && _TVals.size() > 3) {
    int len = _TVals.size();
    PetscScalar a = (_TVals[len-1] - _TVals[len-2])/(_TDepths[len-1]-_TDepths[len-2]); // adiabat slope
    PetscScalar const *zz;
    PetscScalar *Tamb;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_T,&Istart,&Iend);
    VecGetArrayRead(*_z,&zz);
    VecGetArray(_Tamb,&Tamb);
    PetscInt Jj = 0;
    for (Ii=Istart;Ii<Iend;Ii++) {
      if (zz[Jj] >= _Lz_lab) { Tamb[Jj] = a * (zz[Jj]-_TDepths[len-2]) + _TVals[len-2]; }
      Jj++;
    }
    VecRestoreArrayRead(*_z,&zz);
    VecRestoreArray(_Tamb,&Tamb);
  }

  // update _T, _dT
  VecSet(_dT,0.0);
  VecCopy(_Tamb,_T);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// Solve steady-state heat equation
PetscErrorCode HeatEquation::setupKSP_SS(Mat& A)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    string funcName = "HeatEquation::setupKSP_SS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_kspSS != NULL) { return ierr; }

  ierr = KSPCreate(PETSC_COMM_WORLD,&_kspSS); CHKERRQ(ierr);
  ierr = KSPSetOperators(_kspSS,A,A); CHKERRQ(ierr);
  PC pc;

  if (_linSolver == "AMG") { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(_kspSS,KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPSetOperators(_kspSS,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspSS,PETSC_TRUE); CHKERRQ(ierr); // necessary for solving steady state power law
    ierr = KSPGetPC(_kspSS,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE); CHKERRQ(ierr);
    //~ ierr = PCHYPRESetType(pc,"boomeramg"); CHKERRQ(ierr);
    ierr = KSPSetTolerances(_kspSS,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_kspSS,PETSC_TRUE); CHKERRQ(ierr);
  }
  else if (_linSolver == "MUMPSLU") { // direct LU from MUMPS
    // use direct LU from MUMPS
    ierr = KSPSetType(_kspSS,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetOperators(_kspSS,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspSS,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspSS,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU); CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }
  else if (_linSolver == "MUMPSCHOLESKY") { // direct Cholesky (RR^T) from MUMPS
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(_kspSS,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetOperators(_kspSS,A,A); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspSS,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspSS,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY); CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }
  else if (_linSolver == "CG_PCAMG") { // conjugate gradient
    ierr = KSPSetType(_kspSS,KSPCG);                                       CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_kspSS, PETSC_TRUE);                  CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspSS,PETSC_TRUE);                   CHKERRQ(ierr);
    ierr = KSPGetPC(_kspSS,&pc);                                           CHKERRQ(ierr);
    ierr = KSPSetTolerances(_kspSS,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);        CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0);
  }

  double startTime = MPI_Wtime();
  // finish setting up KSP context using options defined above
  ierr = KSPSetFromOptions(_kspSS);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(_kspSS);CHKERRQ(ierr);
  _factorTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// set up KSP for transient problem
PetscErrorCode HeatEquation::setupKSP(Mat& A)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setupKSP";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = KSPCreate(PETSC_COMM_WORLD,&_kspTrans); CHKERRQ(ierr);
  ierr = KSPSetOperators(_kspTrans,A,A); CHKERRQ(ierr);
  PC pc;

  if (_linSolver.compare("AMG")==0) { // algebraic multigrid from HYPRE
    // uses HYPRE's solver AMG (not HYPRE's preconditioners)
    ierr = KSPSetType(_kspTrans,KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspTrans,PETSC_FALSE); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspTrans,&_pc); CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCHYPRE); CHKERRQ(ierr);
    //~ ierr = PCHYPRESetType(_pc,"boomeramg"); CHKERRQ(ierr);
    ierr = KSPSetTolerances(_kspTrans,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(_pc,4); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_kspTrans,PETSC_TRUE); CHKERRQ(ierr);
  }
  else if (_linSolver.compare("MUMPSLU")==0) { // direct LU from MUMPS
    // use direct LU from MUMPS
    ierr = KSPSetType(_kspTrans,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspTrans,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspTrans,&_pc); CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCLU); CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(_pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(_pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(_pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(_pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }
  else if (_linSolver.compare("MUMPSCHOLESKY")==0) { // direct Cholesky (RR^T) from MUMPS
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(_kspTrans,KSPPREONLY); CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspTrans,PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspTrans,&_pc); CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCCHOLESKY); CHKERRQ(ierr);
    #if PETSC_VERSION_MINOR > 5
      ierr = PCFactorSetMatSolverType(_pc,MATSOLVERMUMPS);                 CHKERRQ(ierr); // new PETSc
      ierr = PCFactorSetUpMatSolverType(_pc);                              CHKERRQ(ierr); // new PETSc
    #endif
    #if PETSC_VERSION_MINOR < 5
      ierr = PCFactorSetMatSolverPackage(_pc,MATSOLVERMUMPS);              CHKERRQ(ierr); // old PETSc
      ierr = PCFactorSetUpMatSolverPackage(_pc);                           CHKERRQ(ierr); // old PETSc
    #endif
  }
  else if (_linSolver.compare("CG_PCAMG")==0) { // conjugate gradient
    ierr = KSPSetType(_kspTrans,KSPCG);                                 CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(_kspTrans,PETSC_TRUE);             CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(_kspTrans,PETSC_FALSE);            CHKERRQ(ierr);
    ierr = KSPGetPC(_kspTrans,&_pc);                                    CHKERRQ(ierr);
    ierr = KSPSetTolerances(_kspTrans,_kspTol,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPGetPC(_kspTrans,&pc);                                           CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);                                       CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");                              CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);        CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0);
  }

  // accept command line options
  ierr = KSPSetFromOptions(_kspTrans);CHKERRQ(ierr);
  //~ ierr = KSPSetUp(_kspTrans);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  double startTime = MPI_Wtime();
  ierr = KSPSetUp(_kspTrans);CHKERRQ(ierr);
  _factorTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated implicity into varIm
  if (varIm.find("Temp") != varIm.end() ) { VecCopy(_T,varIm["Temp"]); }
  else { Vec varTemp; VecDuplicate(_T,&varTemp); VecCopy(_T,varTemp); varIm["Temp"] = varTemp; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ Vec T;
  //~ VecCopy(varIm.find("Temp")->second,_dT);
  // not needed for implicit solve

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::setMMSBoundaryConditions(const double time,
  string bcRType,string bcTType,string bcLType,string bcBType)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setMMSBoundaryConditions";
    string fileName = "heatequation.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcL,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
    y = 0;
    if (!bcLType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
    else if (!bcLType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_y(y,z,time); }
    ierr = VecSetValues(_bcL,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!bcRType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
    else if (!bcRType.compare("Neumann")) { v = zzmms_k(y,z)*zzmms_T_y(y,z,time); }
    ierr = VecSetValues(_bcR,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcR);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcL);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcR);CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(*_y,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    if (Ii % _Nz == 0) {
      ierr = VecGetValues(*_y,1,&Ii,&y);CHKERRQ(ierr);
      PetscInt Jj = Ii / _Nz;

      z = 0;
      if (!bcTType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
      else if (!bcTType.compare("Neumann")) {
  v = zzmms_k(y,z)*zzmms_T_z(y,z,time);
      }
      ierr = VecSetValues(_bcT,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);

      z = _Lz;
      if (!bcBType.compare("Dirichlet")) { v = zzmms_T(y,z,time); }
      else if (!bcBType.compare("Neumann")) {
  v = zzmms_k(y,z)*zzmms_T_z(y,z,time);
      }
      ierr = VecSetValues(_bcB,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcT);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcB);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::measureMMSError(const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setMMSBoundaryConditions";
    string fileName = "heatequation.cpp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // measure error between analytical and numerical solution
  Vec dTA;
  VecDuplicate(_dT,&dTA);
  mapToVec(dTA,zzmms_T,*_y,*_z,time);
  writeVec(dTA,_outputDir+"mms_dTA");
  writeVec(_dT,_outputDir+"mms_dT");
  writeVec(_bcL,_outputDir+"mms_he_bcL");
  writeVec(_bcR,_outputDir+"mms_he_bcR");
  writeVec(_bcT,_outputDir+"mms_he_bcT");
  writeVec(_bcB,_outputDir+"mms_he_bcB");

  double err2u = computeNormDiff_2(_dT,dTA);

  PetscPrintf(PETSC_COMM_WORLD,"%i %3i %.4e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u));

  VecDestroy(&dTA);
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}


// for thermomechanical coupling with explicit time stepping
// Note: This actually returns d/dt (T - Tamb), where Tamb is the 1D steady-state geotherm
PetscErrorCode HeatEquation::d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau,const Vec& sdev, const Vec& dgdev, const Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::d_dt";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // update fields
  VecCopy(T,_T);
  VecWAXPY(_dT,-1.0,_Tamb,T);

  // set up boundary conditions and source terms

  // compute source term: Q = Qrad + Qfric + Qvisc
  // radioactive heat generation Qrad
  VecSet(_Q,0.0);

  // frictional heat generation: Qfric or bcL depending on shear zone width
  if (_wFrictionalHeating.compare("yes")==0) {
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,1.0,_Qfric);
  }

  // viscous shear heating: Qvisc
  if (_wViscShearHeating.compare("yes")==0 && dgdev!=NULL && sdev!=NULL) {
    computeViscousShearHeating(sdev, dgdev);
    VecAXPY(_Q,1.0,_Qvisc);
  }

  // rhs = -H*J*(SAT bc terms) + H*J*Q
  Vec rhs; VecDuplicate(_k,&rhs); VecSet(rhs,0.0);
  ierr = _sbp->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr); // put SAT terms in temp
  VecScale(rhs,-1.); // sign convention in setRhs is opposite of what's needed for explicit time stepping
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Vec temp1; VecDuplicate(_Q,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_Q,temp1);
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,temp1,rhs,rhs); CHKERRQ(ierr); // rhs = H*temp1 + temp
    VecDestroy(&temp1);
  }
  else {
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,_Q,rhs,rhs); CHKERRQ(ierr); // rhs = H*temp1 + temp
  }

  // add H*J*D2 * dTn
  Mat A; _sbp->getA(A);
  MatMultAdd(A,_dT,rhs,rhs); // rhs = A*dTn + rhs

  // dT = 1/(rho*c) * Hinv *Jinv * rhs
  VecPointwiseDivide(rhs,rhs,_rho);
  VecPointwiseDivide(rhs,rhs,_c);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Vec temp1; VecDuplicate(_Q,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(Jinv,rhs,temp1);
    _sbp->Hinv(temp1,dTdt);
    VecDestroy(&temp1);
  }
  else {
    _sbp->Hinv(rhs,dTdt);
  }

  VecDestroy(&rhs);

  computeHeatFlux();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// MMS test for thermomechanical coupling with explicity time stepping
PetscErrorCode HeatEquation::d_dt_mms(const PetscScalar time,const Vec& T, Vec& dTdt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::d_dt_mms";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecCopy(T,_dT); // so that the correct temperature is written out

  // update boundary conditions
  ierr = setMMSBoundaryConditions(time,"Dirichlet","Dirichlet","Neumann","Dirichlet"); CHKERRQ(ierr);

  Mat A;
  _sbp->getA(A);
  ierr = MatMult(A,T,dTdt); CHKERRQ(ierr);
  Vec rhs;
  VecDuplicate(T,&rhs);
  ierr = _sbp->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  ierr = VecAXPY(dTdt,-1.0,rhs);CHKERRQ(ierr);
  VecDestroy(&rhs);

  Vec temp;
  VecDuplicate(dTdt,&temp);
  _sbp->Hinv(dTdt,temp);
  VecCopy(temp,dTdt);
  VecDestroy(&temp);

  VecPointwiseDivide(dTdt,dTdt,_rho);
  VecPointwiseDivide(dTdt,dTdt,_c);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical coupling using backward Euler (implicit time stepping)
PetscErrorCode HeatEquation::be(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  double beStartTime = MPI_Wtime();

  if (_isMMS && _heatEquationType.compare("transient")==0) {
    assert(0);
  }
  else if (_isMMS && _heatEquationType.compare("steadyState")==0) {
    be_steadyStateMMS(time,slipVel,tau,sdev,dgdev,T,To,dt);
  }
  else if (!_isMMS && _heatEquationType.compare("transient")==0) {
    be_transient(time,slipVel,tau,sdev,dgdev,T,To,dt);
  }
  else if (!_isMMS && _heatEquationType.compare("steadyState")==0) {
    be_steadyState(time,slipVel,tau,sdev,dgdev,T,To,dt);
  }

  computeHeatFlux();

  _beTime += MPI_Wtime() - beStartTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical problem using implicit time stepping (backward Euler)
// Note: This function uses the KSP algorithm to solve for dT, where T = Tamb + dT
PetscErrorCode HeatEquation::be_transient(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& Tn,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_transient";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // update fields
  VecCopy(Tn,_T);

  // set up matrix
  MatCopy(_D2ath,_B,SAME_NONZERO_PATTERN);
  MatScale(_B,-dt);
  MatAXPY(_B,1.0,_I,SUBSET_NONZERO_PATTERN);
  if (_kspTrans == NULL) {
    KSPDestroy(&_kspSS);
    setupKSP(_B);
  }
  ierr = KSPSetOperators(_kspTrans,_B,_B);CHKERRQ(ierr);

  // set up boundary conditions and source terms: Q = Qfric + Qvisc
  // Note: there is no Qrad because radioactive heat generation is already included in Tamb
  Vec rhs,temp;
  VecDuplicate(_k,&rhs);
  VecDuplicate(_k,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);
  VecSet(_Q,0.); // radioactive heat generation is already included in Tamb

  // frictional heat generation: Qfric or bcL depending on shear zone width
  if (_wFrictionalHeating.compare("yes")==0) {
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,1.0,_Qfric);
  }

  // viscous shear heating: Qvisc
  if (_wViscShearHeating.compare("yes")==0 && dgdev!=NULL && sdev!=NULL) {
    computeViscousShearHeating(sdev, dgdev);
    VecAXPY(_Q,1.0,_Qvisc);
  }

  ierr = _sbp->setRhs(temp,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  Vec temp1; VecDuplicate(_Q,&temp1);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_Q,temp1);
  }

  Mat H; _sbp->getH(H);
  ierr = MatMultAdd(H,temp1,temp,temp); CHKERRQ(ierr);
  VecDestroy(&temp1);
  MatMult(_rcInv,temp,rhs);
  VecScale(rhs,dt);

  // solve in terms of dT
  // add H * dTn to rhs
  VecSet(temp,0.0);
  VecWAXPY(_dT,-1.0,_Tamb,Tn); // dTn = Tn - Tamb
  _sbp->H(_dT,temp);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    Vec temp1; VecDuplicate(temp,&temp1);
    MatMult(J,temp,temp1);
    VecCopy(temp1,temp);
    VecDestroy(&temp1);
  }
  VecAXPY(rhs,1.0,temp);
  VecDestroy(&temp);

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  KSPSolve(_kspTrans,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecDestroy(&rhs);

  // update total temperature: _T (internal variable) and T (output)
  VecWAXPY(_T,1.0,_Tamb,_dT); // T = dT + Tamb
  VecCopy(_T,T);
  computeHeatFlux();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// for thermomechanical problem when solving only the steady-state heat equation
// Note: This function uses the KSP algorithm to solve for dT, where T = Tamb + dT
PetscErrorCode HeatEquation::be_steadyState(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& Tn,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_steadyState";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // update fields
  VecCopy(Tn,_T);

  if (_kspSS == NULL) {
    KSPDestroy(&_kspTrans);
    Mat A; _sbp->getA(A);
    setupKSP_SS(A);
  }

  // set up boundary conditions and source terms: Q = Qrad + Qfric + Qvisc
  Vec rhs; VecDuplicate(_k,&rhs); VecSet(rhs,0.0);

  // compute heat source terms
  // Note: this does not include Qrad because that is included in the ambient geotherm
  VecSet(_Q,0.);

  // frictional heat generation: Qfric or bcL depending on shear zone width
  if (_wFrictionalHeating.compare("yes")==0) {
    // set bcL and/or Qfric depending on shear zone width
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,-1.0,_Qfric);
  }

  // viscous shear heating: Qvisc
  if (_wViscShearHeating.compare("yes")==0 && dgdev!=NULL && sdev!=NULL) {
    computeViscousShearHeating(sdev, dgdev);
    VecAXPY(_Q,-1.0,_Qvisc);
  }

  // rhs = J*H*Q + (SAT BC terms)
  ierr = _sbp->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Vec temp1; VecDuplicate(_Q,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_Q,temp1);
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,temp1,rhs,rhs);
    VecDestroy(&temp1);
  }
  else{
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,_Q,rhs,rhs); CHKERRQ(ierr);
  }

  // solve for dT and record run time required
  VecWAXPY(_dT,-1.0,_Tamb,Tn); // dT = Tn - Tamb
  double startTime = MPI_Wtime();
  KSPSolve(_kspSS,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecDestroy(&rhs);

  // update total temperature: _T (internal variable) and T (output)
  VecWAXPY(_T,1.0,_Tamb,_dT);
  VecCopy(_T,T);

  computeHeatFlux();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// for thermomechanical coupling solving only the steady-state heat equation with MMS test
PetscErrorCode HeatEquation::be_steadyStateMMS(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::be_steadyStateMMS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
/*
  // set up boundary conditions and source terms
  Vec rhs,temp;
  VecDuplicate(_dT,&rhs);
  VecDuplicate(_dT,&temp);
  VecSet(rhs,0.0);
  VecSet(temp,0.0);


  setMMSBoundaryConditions(time,"Dirichlet","Dirichlet","Neumann","Dirichlet");
  ierr = _sbp->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  Vec source,Hxsource;
  VecDuplicate(_dT,&source);
  VecDuplicate(_dT,&Hxsource);
  //~ mapToVec(source,zzmms_SSdTsource,*_y,*_z,time);
  mapToVec(source,zzmms_SSTsource,*_y,*_z,time);
  ierr = _sbp->H(source,Hxsource);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    multMatsVec(yq,zr,Hxsource);
  }
  //~ writeVec(source,_outputDir+"mms_SSdTsource");
  VecDestroy(&source);
  ierr = VecAXPY(rhs,1.0,Hxsource);CHKERRQ(ierr); // rhs = rhs + H*source
  VecDestroy(&Hxsource);


  //~ // compute shear heating component
  //~ if (_wViscShearHeating.compare("yes")==0 && dgxy!=NULL && dgxz!=NULL) {
    //~ Vec shearHeat;
    //~ computeViscousShearHeating(shearHeat,sdev, dgxy, dgxz);
    //~ VecSet(shearHeat,0.);
    //~ VecAXPY(temp,1.0,shearHeat);
    //~ VecDestroy(&shearHeat);
  //~ }

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  //~ VecCopy(To,_dT); // plausible guess
  KSPSolve(_ksp,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecCopy(_dT,T);
  //~ VecWAXPY(T,1.0,_dT,_Tamb); // T = dT + T0

  //~ mapToVec(_dT,zzmms_T,*_y,*_z,time);

  VecDestroy(&rhs);
  * */
  assert(0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::initiateVarSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // put variables to be integrated implicity into varIm
  Vec T;
  VecDuplicate(_Tamb,&T);
  VecWAXPY(T,1.0,_Tamb,_dT);
  varSS["Temp"] = T;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// compute steady-state temperature given boundary conditions and shear heating source terms (assuming these remain constant) Qfric and Qvisc
// Note: solves for dT, where dT = T - Tamb and Tamb also satisfies the steady-state heat equation
// and can includes radioactive heat generation.
PetscErrorCode HeatEquation::computeSteadyStateTemp(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sdev, const Vec& dgdev,Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeSteadyStateTemp";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  double beStartTime = MPI_Wtime();

  if (_kspSS == NULL) {
    KSPDestroy(&_kspTrans);
    Mat A; _sbp->getA(A);
    setupKSP_SS(A);
  }

  // set up boundary conditions and source terms
  VecSet(_Q,0.);

  // left boundary: heat generated by fault motion: bcL or Qfric depending on shear zone width
  if (_wFrictionalHeating == "yes") {
    computeFrictionalShearHeating(tau,slipVel);
    VecAXPY(_Q,-1.0,_Qfric);
    VecScale(_bcL,-1.);
  }

  // compute shear heating component
  if (_wViscShearHeating == "yes" && dgdev!=NULL && sdev!=NULL) {
    computeViscousShearHeating(sdev, dgdev);
    VecAXPY(_Q,-1.0,_Qvisc);
  }

  // rhs = J*H*Q + (SAT BC terms)
  Vec rhs; VecDuplicate(_k,&rhs); VecSet(rhs,0.0);
  ierr = _sbp->setRhs(rhs,_bcL,_bcR,_bcT,_bcB);CHKERRQ(ierr);
  if (_D->_gridSpacingType == "variableGridSpacing") {
    Vec temp1; VecDuplicate(_Q,&temp1);
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_Q,temp1);
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,temp1,rhs,rhs); // rhs = rhs + H*temp1
    VecDestroy(&temp1);
  }
  else{
    Mat H; _sbp->getH(H);
    ierr = MatMultAdd(H,_Q,rhs,rhs); CHKERRQ(ierr);
  }

  // solve for temperature and record run time required
  double startTime = MPI_Wtime();
  KSPSolve(_kspSS,rhs,_dT);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  VecDestroy(&rhs);

  // compute total temperature _T (internal variable) and T (output variable)
  VecWAXPY(_T,1.0,_dT,_Tamb);
  VecCopy(_T,T);

  computeHeatFlux();

  _beTime += MPI_Wtime() - beStartTime;

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute viscous shear heating term (uses temperature from previous time step)
PetscErrorCode HeatEquation::computeViscousShearHeating(const Vec& sdev, const Vec& dgdev)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeViscousShearHeating";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  VecSet(_Qvisc,0.0);
  VecPointwiseMult(_Qvisc,sdev,dgdev);

  // convert from engineering to geophysics convention for viscous strain rate
  VecScale(_Qvisc,sqrt(2.0));

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute frictional shear heating term (uses temperature from previous time step)
PetscErrorCode HeatEquation::computeFrictionalShearHeating(const Vec& tau, const Vec& slipVel)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeFrictionalShearHeating";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif

  // compute bcL = q = tau * slipVel
  VecPointwiseMult(_bcL,tau,slipVel);

  // if left boundary condition is heat flux: q = bcL = tau*slipVel/2
  if (_wMax == 0) {
    VecScale(_bcL,0.5); // was -0.5, changed to +0.5 July 13 2022
    VecSet(_Qfric,0.);
  }

  // if using finite width shear zone: Qfric = slipVel*tau * Gw
  else {
    ierr = MatMult(_MapV,_bcL,_Qfric); CHKERRQ(ierr); // Qfric = tau * slipVel (now a body field)
    VecPointwiseMult(_Qfric,_Qfric,_Gw); // Qfric = tau * slipVel * Gw
    VecSet(_bcL,0.); // q = 0, no flux
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),FILENAME,time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// set up KSP, matrices, boundary conditions for the steady state heat equation problem
PetscErrorCode HeatEquation::setUpSteadyStateProblem()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setUpSteadyStateProblem";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecSet(_bcR,0.0);
  VecSet(_bcT,0.0);
  VecSet(_bcB,0.0);

  delete _sbp; _sbp = NULL;

  // original version
  //~ string bcRType = "Dirichlet";
  //~ string bcTType = "Dirichlet";
  //~ string bcLType = "Neumann";
  //~ string bcBType = "Dirichlet";

  // construct matrices
  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    _sbp = new SbpOps_m_constGrid(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    _sbp = new SbpOps_m_varGrid(_order,_Ny,_Nz,_Ly,_Lz,_k);
    if (_Ny > 1 && _Nz > 1) { _sbp->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbp->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbp->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setCompatibilityType(_D->_sbpCompatibilityType);
  //~ _sbp->setBCTypes(bcRType,bcTType,bcLType,bcBType); // original
  _sbp->setBCTypes(_bcRType_ss,_bcTType_ss,_bcLType_ss,_bcBType_ss);
  _sbp->setMultiplyByH(1);
  _sbp->setLaplaceType("yz");
  _sbp->setDeleteIntermediateFields(1);
  _sbp->computeMatrices(); // actually create the matrices

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// set up KSP, matrices, boundary conditions for the transient heat equation problem
PetscErrorCode HeatEquation::setUpTransientProblem()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::setUpTransientProblem";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // ensure BCs are all 0
  VecSet(_bcR,0.);
  VecSet(_bcT,0.);
  VecSet(_bcL,0.);
  VecSet(_bcB,0.);

  delete _sbp; _sbp = NULL;
  // construct matrices
  // BC order: right,top, left, bottom
  if (_D->_gridSpacingType.compare("constantGridSpacing")==0) {
    _sbp = new SbpOps_m_constGrid(_order,_Ny,_Nz,_Ly,_Lz,_k);
  }
  else if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    _sbp = new SbpOps_m_varGrid(_order,_Ny,_Nz,_Ly,_Lz,_k);
    if (_Ny > 1 && _Nz > 1) { _sbp->setGrid(_y,_z); }
    else if (_Ny == 1 && _Nz > 1) { _sbp->setGrid(NULL,_z); }
    else if (_Ny > 1 && _Nz == 1) { _sbp->setGrid(_y,NULL); }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setCompatibilityType(_D->_sbpCompatibilityType);
  //~ _sbp->setBCTypes("Dirichlet","Dirichlet","Neumann","Dirichlet"); // original
  _sbp->setBCTypes(_bcRType_trans,_bcTType_trans,_bcLType_trans,_bcBType_trans); // original
  _sbp->setMultiplyByH(1);
  _sbp->setLaplaceType("yz");
  _sbp->computeMatrices(); // actually create the matrices

  // create identity matrix I (multiplied by H)
  Mat H;
  _sbp->getH(H);
  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatMatMult(J,H,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_I);
  }
  else {
    MatDuplicate(H,MAT_COPY_VALUES,&_I);
  }

  // create (rho*c)^-1 vector and matrix
  Vec rhocV;
  VecDuplicate(_rho,&rhocV);
  VecSet(rhocV,1.);
  VecPointwiseDivide(rhocV,rhocV,_rho);
  VecPointwiseDivide(rhocV,rhocV,_c);
  MatDuplicate(_I,MAT_DO_NOT_COPY_VALUES,&_rcInv);
  MatDiagonalSet(_rcInv,rhocV,INSERT_VALUES);

  // create _D2ath = (rho*c)^-1 H D2
  Mat D2;
  _sbp->getA(D2);
  MatMatMult(_rcInv,D2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&_D2ath);

  MatDuplicate(_D2ath,MAT_COPY_VALUES,&_B);

  // ensure diagonal of _D2ath has been allocated, even if 0
  PetscScalar v=0.0;
  PetscInt Ii,Istart,Iend=0;
  MatGetOwnershipRange(_D2ath,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    MatSetValues(_D2ath,1,&Ii,1,&Ii,&v,ADD_VALUES);
  }
  MatAssemblyBegin(_D2ath,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_D2ath,MAT_FINAL_ASSEMBLY);

  VecDestroy(&rhocV);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


// compute heat flux (full body field and surface heat flux) for output
PetscErrorCode HeatEquation::computeHeatFlux()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::computeHeatFlux";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // total heat flux in lithosphere
  ierr = _sbp->muxDz(_T,_kTz); CHKERRQ(ierr);
  VecScale(_kTz,1e9);

  // extract surface heat flux
  VecScatterBegin(_scatters["body2T"], _kTz, _kTz_z0, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_scatters["body2T"], _kTz, _kTz_z0, INSERT_VALUES, SCATTER_FORWARD);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::writeStep1D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep1D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  // compute max dT for output
  VecMax(_dT, NULL, &_maxdT);
  VecSet(_maxdTVec,_maxdT);

  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = VecView(_kTz_z0, viewer);                                      CHKERRQ(ierr);
  ierr = VecView(_bcR, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_bcT, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_bcL, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_bcB, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_maxdTVec, viewer);                                    CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;

#if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode HeatEquation::writeStep2D(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeStep2D";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = VecView(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_dT,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_kTz,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_Qfric,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_Qvisc,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_Q,viewer);                                            CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);                        CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::writeCheckpoint(PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");              CHKERRQ(ierr);
  ierr = VecView(_k, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Tamb, viewer);                                        CHKERRQ(ierr);
  if (_wFrictionalHeating.compare("yes")==0) {
    ierr = VecView(_Gw, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e3); // output w in m
    ierr = VecView(_w, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e-3); // convert w from m to km
  }
  if (_wRadioHeatGen.compare("yes")==0) {
    ierr = VecView(_Qrad, viewer);                                        CHKERRQ(ierr);
  }

  ierr = VecView(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_bcB,viewer);                                          CHKERRQ(ierr);

  ierr = VecView(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecView(_dT,viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_kTz,viewer);                                          CHKERRQ(ierr);
  ierr = VecView(_Qfric,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_Qvisc,viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_Q,viewer);                                            CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::loadCheckpoint";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = VecLoad(_k, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Tamb, viewer);                                        CHKERRQ(ierr);
  if (_wFrictionalHeating.compare("yes")==0) {
    ierr = VecLoad(_Gw, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e3); // output w in m
    ierr = VecLoad(_w, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e-3); // convert w from m to km
  }
  if (_wRadioHeatGen.compare("yes")==0) {
    ierr = VecLoad(_Qrad, viewer);                                        CHKERRQ(ierr);
  }

  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);

  ierr = VecLoad(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_dT,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_kTz,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_Qfric,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_Qvisc,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_Q,viewer);                                            CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode HeatEquation::loadCheckpointSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::loadCheckpointSS";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscViewer viewer;

  string fileName = _outputDir + "data_context.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = VecLoad(_k, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecLoad(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_Tamb, viewer);                                        CHKERRQ(ierr);
  if (_wFrictionalHeating.compare("yes")==0) {
    ierr = VecLoad(_Gw, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e3); // output w in m
    ierr = VecLoad(_w, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e-3); // convert w from m to km
  }
  if (_wRadioHeatGen.compare("yes")==0) {
    ierr = VecLoad(_Qrad, viewer);                                        CHKERRQ(ierr);
  }
  PetscViewerDestroy(&viewer);

  fileName = _outputDir + "data_steadyState.h5";
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);                       CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);

  ierr = VecLoad(_bcR,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcT,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcL,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_bcB,viewer);                                          CHKERRQ(ierr);

  ierr = VecLoad(_T,viewer);                                            CHKERRQ(ierr);
  ierr = VecLoad(_dT,viewer);                                           CHKERRQ(ierr);
  ierr = VecLoad(_kTz,viewer);                                          CHKERRQ(ierr);
  ierr = VecLoad(_Qfric,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_Qvisc,viewer);                                        CHKERRQ(ierr);
  ierr = VecLoad(_Q,viewer);                                            CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode HeatEquation::view()
{
  PetscErrorCode ierr = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Heat Equation Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   linear solver algorithm: %s\n",_linSolver.c_str()); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in be (s): %g\n",_beTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% be time spent solving linear system: %g\n",_linSolveTime/_beTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return ierr;
}


// Save all scalar fields to text file named he_domain.txt in output directory.
// Note that only the rank 0 processor's values will be saved.
PetscErrorCode HeatEquation::writeDomain(const string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeDomain";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // output scalar fields
  string str = outputDir + "heatEquation.txt";
  PetscViewer    viewer;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"heatEquationType = %s\n",_heatEquationType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withViscShearHeating = %s\n",_wViscShearHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withFrictionalHeating = %s\n",_wFrictionalHeating.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"withRadioHeatGeneration = %s\n",_wRadioHeatGen.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"linSolver_heateq = %s\n",_linSolver.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"kspTol_heateq = %.15e\n",_kspTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"bcRType_ss = %s\n",_bcRType_ss.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcTType_ss = %s\n",_bcTType_ss.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcLType_ss = %s\n",_bcLType_ss.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcBType_ss = %s\n",_bcBType_ss.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcRType_trans = %s\n",_bcRType_trans.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcTType_trans = %s\n",_bcTType_trans.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcLType_trans = %s\n",_bcLType_trans.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"bcBType_trans = %s\n",_bcBType_trans.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"Nz_lab = %i\n",_Nz_lab);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Lz_lab = %g\n",_Lz_lab);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"TVals = %s\n",vector2str(_TVals).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"TDepths = %s\n",vector2str(_TDepths).c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"w = %.5e\n",_w);CHKERRQ(ierr);


  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"numProcessors = %i\n",size);CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}


// write out material properties
PetscErrorCode HeatEquation::writeContext(const string outputDir, PetscViewer& viewer)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "HeatEquation::writeContext";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  writeDomain(outputDir);

  // write context variables
  ierr = PetscViewerHDF5PushGroup(viewer, "/heatEquation");             CHKERRQ(ierr);
  ierr = VecView(_k, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_rho, viewer);                                         CHKERRQ(ierr);
  ierr = VecView(_Qrad, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_c, viewer);                                           CHKERRQ(ierr);
  ierr = VecView(_Tamb, viewer);                                        CHKERRQ(ierr);
  ierr = VecView(_T, viewer);                                           CHKERRQ(ierr);
  if (_wFrictionalHeating.compare("yes")==0) {
    ierr = VecView(_Gw, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e3); // output w in m
    ierr = VecView(_w, viewer);                                         CHKERRQ(ierr);
    VecScale(_w,1e-3); // convert w from m to km
  }
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  //~ ierr = _sbp->writeOps(_outputDir + "ops_he_"); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


//======================================================================
// MMS  tests
double HeatEquation::zzmms_rho(const double y,const double z) { return 1.0; }
double HeatEquation::zzmms_c(const double y,const double z) { return 1.0; }
double HeatEquation::zzmms_h(const double y,const double z) { return 0.0; }
double HeatEquation::zzmms_k(const double y,const double z) { return sin(y)*sin(z) + 30.; }
double HeatEquation::zzmms_k_y(const double y,const double z) {return cos(y)*sin(z);}
double HeatEquation::zzmms_k_z(const double y,const double z) {return sin(y)*cos(z);}

double HeatEquation::zzmms_f(const double y,const double z) {return cos(y)*sin(z);}
double HeatEquation::zzmms_f_y(const double y,const double z) { return -sin(y)*sin(z); }
double HeatEquation::zzmms_f_yy(const double y,const double z) { return -cos(y)*sin(z); }
double HeatEquation::zzmms_f_z(const double y,const double z) { return cos(y)*cos(z); }
double HeatEquation::zzmms_f_zz(const double y,const double z) { return -cos(y)*sin(z); }
double HeatEquation::zzmms_g(const double t) { return exp(-2.*t); }
double HeatEquation::zzmms_g_t(const double t) { return -2.*exp(-2.*t); }

double HeatEquation::zzmms_T(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_y(const double y,const double z,const double t) { return zzmms_f_y(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_yy(const double y,const double z,const double t) { return zzmms_f_yy(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_z(const double y,const double z,const double t) { return zzmms_f_z(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_zz(const double y,const double z,const double t) { return zzmms_f_zz(y,z)*zzmms_g(t); }
double HeatEquation::zzmms_T_t(const double y,const double z,const double t) { return zzmms_f(y,z)*zzmms_g_t(t); }

double HeatEquation::zzmms_dT(const double y,const double z,const double t) { return zzmms_T(y,z,t) - zzmms_T(y,z,0.); }
double HeatEquation::zzmms_dT_y(const double y,const double z,const double t) { return zzmms_T_y(y,z,t) - zzmms_T_y(y,z,0.); }
double HeatEquation::zzmms_dT_yy(const double y,const double z,const double t) { return zzmms_T_yy(y,z,t) - zzmms_T_yy(y,z,0.); }
double HeatEquation::zzmms_dT_z(const double y,const double z,const double t) { return zzmms_T_z(y,z,t) - zzmms_T_z(y,z,0.); }
double HeatEquation::zzmms_dT_zz(const double y,const double z,const double t) { return zzmms_T_zz(y,z,t) - zzmms_T_zz(y,z,0.); }
double HeatEquation::zzmms_dT_t(const double y,const double z,const double t) { return zzmms_T_t(y,z,t) - zzmms_T_t(y,z,0.); }

double HeatEquation::zzmms_SSTsource(const double y,const double z,const double t)
{
  PetscScalar k = zzmms_k(y,z);
  PetscScalar k_y = zzmms_k_y(y,z);
  PetscScalar k_z = zzmms_k_z(y,z);
  PetscScalar T_y = zzmms_T_y(y,z,t);
  PetscScalar T_yy = zzmms_T_yy(y,z,t);
  PetscScalar T_z = zzmms_T_z(y,z,t);
  PetscScalar T_zz = zzmms_T_zz(y,z,t);
  return k*(T_yy + T_zz) + k_y*T_y + k_z*T_z;
}

double HeatEquation::zzmms_SSdTsource(const double y,const double z,const double t)
{
  PetscScalar k = zzmms_k(y,z);
  PetscScalar k_y = zzmms_k_y(y,z);
  PetscScalar k_z = zzmms_k_z(y,z);
  PetscScalar dT_y = zzmms_dT_y(y,z,t);
  PetscScalar dT_yy = zzmms_dT_yy(y,z,t);
  PetscScalar dT_z = zzmms_dT_z(y,z,t);
  PetscScalar dT_zz = zzmms_dT_zz(y,z,t);
  return k*(dT_yy + dT_zz) + k_y*dT_y + k_z*dT_z;
}
