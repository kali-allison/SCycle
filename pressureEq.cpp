#include "pressureEq.hpp"

#define FILENAME "pressureEq.cpp"

using namespace std;

PressureEq::PressureEq(Domain&D)
: _file(D._file),_delim(D._delim),_outputDir(D._outputDir),_isMMS(D._isMMS),
  _hydraulicTimeIntType("explicit"),
  _order(D._order),_N(D._Nz),_L(D._Lz),_h(D._dr),_z(NULL),
  _n_p(NULL),_beta_p(NULL),_k_p(NULL),_eta_p(NULL),_rho_f(NULL),_g(9.8),
  _linSolver("AMG"),_ksp(NULL),_kspTol(1e-10),_sbp(NULL),_sbpType("mfc"),_linSolveCount(0),
  _writeTime(0),_linSolveTime(0),_ptTime(0),_startTime(0),_miscTime(0),
  _p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "PressureEq::PressureEq";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set a, b, normal stress, and Dc
  loadSettings(_file);
  checkInput();
  setFields(D);
  computeInitialSteadyStatePressure(D);

  if (_isMMS) { // set initial condition
    // setMMSInitialConditions();
    mapToVec(_p, zzmms_pA1D, _z, D._initTime);
    VecSet(_bcL, 0);
    VecSet(_bcT, zzmms_pSource1D(0, D._initTime));
    VecSet(_bcB, zzmms_pSource1D(_L, D._initTime));
  }

  //~ if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PressureEq::~PressureEq()
{
  #if VERBOSE > 1
    std::string funcName = "PressureEq::~PressureEq";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // fields that exist on the fault
  VecDestroy(&_n_p);
  VecDestroy(&_beta_p);
  VecDestroy(&_k_p);
  VecDestroy(&_eta_p);
  VecDestroy(&_rho_f);

  VecDestroy(&_p);

  for (map<string,PetscViewer>::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first]);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// Check that required fields have been set by the input file
PetscErrorCode PressureEq::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::checkInput";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_n_pVals.size() == _n_pDepths.size() );
  assert(_beta_pVals.size() == _beta_pDepths.size() );
  assert(_k_pVals.size() == _k_pDepths.size() );
  assert(_eta_pVals.size() == _eta_pDepths.size() );
  assert(_rho_fVals.size() == _rho_fDepths.size() );
  assert(_pVals.size() == _pDepths.size() );

  assert(_n_pVals.size() != 0 );
  assert(_n_pDepths.size() != 0 );
  assert(_beta_pVals.size() != 0 );
  assert(_beta_pDepths.size() != 0 );
  assert(_k_pVals.size() != 0 );
  assert(_k_pDepths.size() != 0 );
  assert(_eta_pVals.size() != 0 );
  assert(_eta_pDepths.size() != 0 );
  assert(_rho_fVals.size() != 0 );
  assert(_rho_fDepths.size() != 0 );
  assert(_pVals.size() != 0 );
  assert(_pDepths.size() != 0 );

  assert(_g >= 0);

  assert(_hydraulicTimeIntType.compare("explicit")==0 || _hydraulicTimeIntType.compare("implicit")==0);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::loadSettings";
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

    // load Vec inputs
    if (var.compare("hydraulicTimeIntType")==0) {
      _hydraulicTimeIntType = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    else if (var.compare("n_pVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_n_pVals);
    }
    else if (var.compare("n_pDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_n_pDepths);
    }

    if (var.compare("beta_pVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_beta_pVals);
    }
    else if (var.compare("beta_pDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_beta_pDepths);
    }

    if (var.compare("k_pVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_k_pVals);
    }
    else if (var.compare("k_pDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_k_pDepths);
    }

    if (var.compare("eta_pVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_eta_pVals);
    }
    else if (var.compare("eta_pDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_eta_pDepths);
    }

    if (var.compare("rho_fVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rho_fVals);
    }
    else if (var.compare("rho_fDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_rho_fDepths);
    }

    else if (var.compare("g")==0) {
      _g = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    if (var.compare("pVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_pVals);
    }
    else if (var.compare("pDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_pDepths);
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::setFields(Domain&D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::setFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // allocate memory and partition accross processors
  VecCreate(PETSC_COMM_WORLD,&_n_p);
  VecSetSizes(_n_p,PETSC_DECIDE,_N);
  VecSetFromOptions(_n_p);
  PetscObjectSetName((PetscObject) _n_p, "n_p");
  VecSet(_n_p,0.0);

  VecDuplicate(_n_p,&_beta_p);  PetscObjectSetName((PetscObject) _beta_p, "beta_p");  VecSet(_beta_p,0.0);
  VecDuplicate(_n_p,&_k_p);  PetscObjectSetName((PetscObject) _k_p, "k_p");  VecSet(_k_p,0.0);
  VecDuplicate(_n_p,&_eta_p);  PetscObjectSetName((PetscObject) _eta_p, "eta_p");  VecSet(_eta_p,0.0);
  VecDuplicate(_n_p,&_rho_f);  PetscObjectSetName((PetscObject) _rho_f, "rho_f");  VecSet(_rho_f,0.0);

  VecDuplicate(_n_p,&_p);  PetscObjectSetName((PetscObject) _p, "p");  VecSet(_p,0.0);
  VecDuplicate(_n_p,&_p_t);  PetscObjectSetName((PetscObject) _p_t, "p_t");  VecSet(_p_t,0.0);

  // initialize values
  if (_N == 1) {
    VecSet(_n_p,_n_pVals[0]);
    VecSet(_beta_p,_beta_pVals[0]);
    VecSet(_k_p,_k_pVals[0]);
    VecSet(_eta_p,_eta_pVals[0]);
    VecSet(_rho_f,_rho_fVals[0]);
  }
  else {
    ierr = setVecFromVectors(_n_p,_n_pVals,_n_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_beta_p,_beta_pVals,_beta_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_k_p,_k_pVals,_k_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_eta_p,_eta_pVals,_eta_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_rho_f,_rho_fVals,_rho_fDepths);CHKERRQ(ierr);
  }


  // boundary conditions
  VecDuplicate(_n_p,&_bcL);
  PetscObjectSetName((PetscObject) _bcL, "bcL");
  VecSet(_bcL,0.0);

  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,1);
  VecSetFromOptions(_bcT);
  PetscObjectSetName((PetscObject) _bcT, "bcT");
  // VecSet(_bcT,_pVals[0]);
  VecSet(_bcT, 0);

  VecDuplicate(_bcT,&_bcB);
  // VecSet(_bcB,_g*_rho_fVals.back()*_k_pVals.back()/_eta_pVals.back());
  VecSet(_bcB, 0);
//  VecSet(_bcB,_g*_rho_fVals.back());

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::computeVariableCoefficient(const Vec& p,Vec& coeff)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::computeVariableCoefficient";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  VecDuplicate(p,&coeff);

  // coeff = rho_f * k_p / eta_p
  PetscScalar *rho_f,*k,*eta,*coeffA=0;
  PetscInt Ii,Istart,Iend;
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_k_p,&k);
  VecGetArray(_eta_p,&eta);
  VecGetArray(coeff,&coeffA);
  PetscInt Jj = 0;
  VecGetOwnershipRange(_p,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    coeffA[Jj] = rho_f[Jj]*k[Jj]/eta[Jj];

    assert(~isnan(coeffA[Jj]));
    assert(~isinf(coeffA[Jj]));
    Jj++;
  }
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_k_p,&k);
  VecRestoreArray(_eta_p,&eta);
  VecRestoreArray(coeff,&coeffA);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initial conditions
PetscErrorCode PressureEq::computeInitialSteadyStatePressure(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "PressureEq::computeInitialSteadyStatePressure";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up variable coefficient
  Vec coeff;
  VecSet(_p,0.);
  computeVariableCoefficient(_p,coeff);

  //~ VecView(_p,PETSC_VIEWER_STDOUT_WORLD);
  //~ PetscPrintf(PETSC_COMM_WORLD,"put words here");
  //~ PetscPrintf(PETSC_COMM_WORLD,"N = %i\n",,_N);
  //~ assert(0);

  // Set up linear system
    if (_sbpType.compare("mc")==0) {
    _sbp = new SbpOps_c(_order,1,_N,1,_L,coeff);
  }
  else if (_sbpType.compare("mfc")==0) {
    _sbp = new SbpOps_fc(_order,1,_N,1,_L,coeff);
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp = new SbpOps_fc_coordTrans(_order,1,_N,1,_L,coeff);
    _sbp->setGrid(NULL,&_z);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }
  _sbp->setBCTypes("Dirichlet","Dirichlet","Dirichlet","Dirichlet");
  _sbp->setLaplaceType("z");
  _sbp->computeMatrices(); // actually create the matrices


  // set up linear solver context
  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  Mat A;
  _sbp->getA(A);

  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
  ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

  // perform computation of preconditioners now, rather than on first use
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  // set up boundary conditions
  Vec rhs;
  VecDuplicate(_k_p,&rhs);
  _sbp->setRhs(rhs,_bcL,_bcL,_bcT,_bcB);

  // add source term from gravity
  Vec rhog,rhog_y,temp;
  VecDuplicate(_p,&rhog);
  VecDuplicate(_p,&temp);
  VecSet(rhog,_g);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_k_p);
  VecPointwiseDivide(rhog,rhog,_eta_p);
  VecDuplicate(_p,&rhog_y);
  _sbp->Dz(rhog,rhog_y);
  _sbp->H(rhog_y,temp);
  VecAXPY(rhs,-1.0,temp);

  // solve for temperature
  double startTime = MPI_Wtime();
  ierr = KSPSolve(ksp,rhs,_p);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&temp);
  VecDestroy(&rhs);
  KSPDestroy(&ksp);


  // force pressure to be hydrostatic
  // PetscScalar *rho_f,*p,*z=0;
  // PetscInt Ii,Istart,Iend;
  // VecGetOwnershipRange(_p,&Istart,&Iend);
  // VecGetArray(_rho_f,&rho_f);
  // VecGetArray(_p,&p);
  // VecGetArray(_z,&z);
  // PetscInt Jj = 0;
  // for (Ii=Istart;Ii<Iend;Ii++) {
  //   p[Jj] = rho_f[Jj]*_g*z[Jj];
  //   Jj++;
  // }
  // VecRestoreArray(_rho_f,&rho_f);
  // VecRestoreArray(_p,&p);
  // VecRestoreArray(_z,&z);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode PressureEq::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // make shallow copy of pressure
  Vec p;
  VecDuplicate(_p,&p);
  VecCopy(_p,p);

  // put variable to be integrated explicitly into varEx
  if (_hydraulicTimeIntType.compare("explicit")==0) {
    varEx["pressure"] = p;
  }
  else { // put variables to be integrated implicity into varIm
    varIm["pressure"] = p;
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("pressure")->second,_p);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_hydraulicTimeIntType.compare("explicit")==0) {
    VecCopy(varEx.find("pressure")->second,_p);
  }
  else { // put variables to be integrated implicity into varIm
    VecCopy(varIm.find("pressure")->second,_p);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// time stepping functions




// purely explicit time integration
PetscErrorCode PressureEq::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_isMMS) {
    ierr = d_dt_mms(time,varEx,dvarEx);CHKERRQ(ierr);
  }
  else {
    ierr = d_dt_main(time,varEx,dvarEx);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// implicit/explicit time integration
PetscErrorCode PressureEq::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_hydraulicTimeIntType.compare("explicit")==0) {
    ierr = d_dt(time,varEx,dvarEx); CHKERRQ(ierr);
  }
  else {
    if (_isMMS) {
      ierr = be_mms(time,varEx,dvarEx,varIm,varImo,dt);CHKERRQ(ierr);
    }
    else {
      ierr = be(time,varEx,dvarEx,varIm,varImo,dt);CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode PressureEq::d_dt_main(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::d_dt_main";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section


  // source term from gravity
  Vec rhog,rhog_y;
  VecDuplicate(_p,&rhog);
  VecSet(rhog,_g);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_k_p);
  VecPointwiseDivide(rhog,rhog,_eta_p);
  VecDuplicate(_p,&rhog_y);
  _sbp->Dz(rhog,rhog_y); // check that this works!

  Vec p_t;
  VecDuplicate(_k_p,&p_t);
  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2,_p,p_t); CHKERRQ(ierr);

  // set up boundary terms
  // VecSet(_bcB,2e-9); // add some fluid flow from depth
  Vec rhs;
  VecDuplicate(_k_p,&rhs);
  _sbp->setRhs(rhs,_bcL,_bcL,_bcT,_bcB);

  VecAXPY(p_t,-1.0,rhs);
  _sbp->Hinv(p_t,dvarEx["pressure"]);


  // compute rate for pressure
  PetscScalar *rhog_yA,*rho_f,*n,*beta,*p_tA,*rhsA=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(dvarEx["pressure"],&Istart,&Iend);
  VecGetArray(rhog_y,&rhog_yA);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_n_p,&n);
  VecGetArray(_beta_p,&beta);
  VecGetArray(rhs,&rhsA);

  VecGetArray(dvarEx["pressure"],&p_tA);
  PetscInt Jj = 0;

  for (Ii=Istart;Ii<Iend;Ii++) {
    p_tA[Jj] = p_tA[Jj] - rhog_yA[Jj];
    p_tA[Jj] = p_tA[Jj] / ( rho_f[Jj]*n[Jj]*beta[Jj] );
    assert(~isnan(p_tA[Jj]));
    assert(~isinf(p_tA[Jj]));
    Jj++;
  }

  VecRestoreArray(rhog_y,&rhog_yA);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_n_p,&n);
  VecRestoreArray(_beta_p,&beta);
  VecRestoreArray(rhs,&rhsA);
  VecRestoreArray(dvarEx["pressure"],&p_tA);

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&rhs);
  VecDestroy(&p_t);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode PressureEq::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::d_dt_mms";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  double startTime = MPI_Wtime(); // time this section

  // source term from gravity
  Vec rhog,rhog_y;
  VecDuplicate(_p,&rhog);
  VecSet(rhog,_g);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_k_p);
  VecPointwiseDivide(rhog,rhog,_eta_p);
  VecDuplicate(_p,&rhog_y);
  _sbp->Dz(rhog,rhog_y); // check that this works!

  // force p to be correct
  //~ mapToVec(_p, zzmms_pA1D, _z, time);

  Vec p_t;
  VecDuplicate(_k_p,&p_t);
  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2,_p,p_t); CHKERRQ(ierr);

  // set up boundary terms
  // VecSet(_bcB,2e-9); // add some fluid flow from depth
  Vec rhs;
  VecDuplicate(_k_p,&rhs);
  _sbp->setRhs(rhs,_bcL,_bcL,_bcT,_bcB);


  VecAXPY(p_t,-1.0,rhs);
  _sbp->Hinv(p_t,dvarEx["pressure"]); // remove extra H term that is applied to all SBP operators


  // compute source
  Vec source,Hxsource;
  VecDuplicate(_p,&source);
  VecDuplicate(_p,&Hxsource);
  mapToVec(source, zzmms_pSource1D, _z, time);
  writeVec(source,_outputDir + "mms_pSource");

  // compute rate for pressure
  PetscScalar *rhog_yA,*rho_f,*n,*beta,*p_tA,*rhsA, *sourceA;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(dvarEx["pressure"],&Istart,&Iend);
  VecGetArray(rhog_y,&rhog_yA);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_n_p,&n);
  VecGetArray(_beta_p,&beta);
  VecGetArray(rhs,&rhsA);
  VecGetArray(source, &sourceA);
  VecGetArray(dvarEx["pressure"],&p_tA);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    // p_tA[Jj] = p_tA[Jj] - rhog_yA[Jj] - rhsA[Jj] + sourceA[Jj];
    p_tA[Jj] = p_tA[Jj] + sourceA[Jj];
    p_tA[Jj] = p_tA[Jj] / ( rho_f[Jj]*n[Jj]*beta[Jj] );
    assert(~isnan(p_tA[Jj]));
    assert(~isinf(p_tA[Jj]));
    Jj++;
  }
  VecRestoreArray(rhog_y,&rhog_yA);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_n_p,&n);
  VecRestoreArray(_beta_p,&beta);
  VecRestoreArray(rhs,&rhsA);
  VecRestoreArray(dvarEx["pressure"],&p_tA);
  VecRestoreArray(source,&sourceA);

  //~ mapToVec(dvarEx["pressure"], zzmms_pt1D, _z, time);

  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&rhs);
  VecDestroy(&p_t);
  VecDestroy(&source);

  writeVec(dvarEx["pressure"],_outputDir + "mms_p_t");

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// backward Euler implicit solve
// new result goes in varIm
PetscErrorCode PressureEq::be(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
  map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::be";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// backward Euler implicit solve for MMS test
// new result goes in varIm
PetscErrorCode PressureEq::be_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::be_mms";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// =====================================================================
// IO commands

PetscErrorCode PressureEq::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"   pressure rate time (explicit) (s): %g\n",_ptTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent computing pressure rate: %g\n",_ptTime/totRunTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode PressureEq::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  ierr = _sbp->writeOps(_outputDir + "ops_p_"); CHKERRQ(ierr);

  PetscViewer    viewer;

  // write out scalar info
  std::string str = _outputDir + "p_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"g = %.15e\n",_g); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicTimeIntType = %s\n",_hydraulicTimeIntType.c_str()); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  //~ PetscErrorCode writeVec(Vec vec,std::string str)
  ierr = writeVec(_n_p,_outputDir + "p_n"); CHKERRQ(ierr);
  ierr = writeVec(_beta_p,_outputDir + "p_beta"); CHKERRQ(ierr);
  ierr = writeVec(_k_p,_outputDir + "p_k"); CHKERRQ(ierr);
  ierr = writeVec(_eta_p,_outputDir + "p_eta"); CHKERRQ(ierr);
  ierr = writeVec(_rho_f,_outputDir + "p_rho_f"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode PressureEq::writeStep(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::writeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif


    Vec pA;
    VecDuplicate(_p, &pA);
    mapToVec(pA, zzmms_pA1D, _z, time);



  if (stepCount==0) {
    _viewers["p"] = initiateViewer(_outputDir + "p_p");

    ierr = VecView(_p,_viewers["p"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["p"],_outputDir + "p_p");

    if (_isMMS) {
      _viewers["p_t"] = initiateViewer(_outputDir + "p_p_t");
      ierr = VecView(_p_t,_viewers["p_t"]); CHKERRQ(ierr);
      ierr = appendViewer(_viewers["p_t"],_outputDir + "p_p_t");

      _viewers["pA"] = initiateViewer(_outputDir + "p_pA");
      ierr = VecView(pA,_viewers["pA"]); CHKERRQ(ierr);
      ierr = appendViewer(_viewers["pA"],_outputDir + "p_pA");
    }

  }
  else {
    ierr = VecView(_p,_viewers["p"]); CHKERRQ(ierr);

    if (_isMMS) {
      ierr = VecView(_p_t,_viewers["p_t"]); CHKERRQ(ierr);
      ierr = VecView(pA,_viewers["pA"]); CHKERRQ(ierr);
    }
  }
  VecDestroy(&pA);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}



// Fills vec with the linear interpolation between the pairs of points (vals,depths).
PetscErrorCode PressureEq::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::setVecFromVectors";
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
PetscErrorCode PressureEq::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
  const PetscScalar maxVal)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "PressureEq::setVecFromVectors";
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







// MMS functions

PetscErrorCode  PressureEq::measureMMSError(const double totRunTime)
{
  Vec pA;
  VecDuplicate(_p, &pA);
  mapToVec(pA, zzmms_pA1D, _z, totRunTime);

  // writeVec(pA,_outputDir+"pA");

  double err2pA = computeNormDiff_2(_p,pA);

  PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e\n",
    _order,_N,_h,err2pA,log2(err2pA));

  return 0;
};


double PressureEq::zzmms_pt1D(const double z,const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;

  PetscScalar delta_p = 50;
  //~ PetscScalar delta_p = 5;
  PetscScalar omega = 2*PI / T0;

  PetscScalar kz = 2. * PI / 30.;
  //~ PetscScalar kz = 1.0;

  PetscScalar p_t = delta_p * sin(kz * z) * omega * cos(omega * t); // correct
  //~ PetscScalar p_t = 2.*cos(2.*t);
  return p_t;
}


double PressureEq::zzmms_pA1D(const double z,const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;

  //~ PetscScalar delta_p = 1e6;
  PetscScalar delta_p = 50;
  PetscScalar omega = 2.*PI / T0;

  PetscScalar kz = 2. * PI / 30.;
  //~ PetscScalar kz = 1.;

  PetscScalar p_src = delta_p * sin(kz * z) * sin(omega * t); // correct
  //~ PetscScalar p_src = sin(2.*t);
  // PetscScalar p_src = 5 * t;
  return p_src;
}

double PressureEq::zzmms_pSource1D(const double z, const double t)
{
  PetscScalar PI = 3.14159265359;
  PetscScalar T0 = 9e9;

  //~ PetscScalar delta_p = 1e6;
  PetscScalar delta_p = 50;
  PetscScalar omega = 2.*PI / T0;

  PetscScalar kz = 2. * PI / 30.;
  //~ PetscScalar kz = 1.;

  //~ PetscScalar beta0 = 1;
  //~ PetscScalar eta0 = 1;
  //~ PetscScalar n0 = 1;
  //~ PetscScalar k0 = 1;

  PetscScalar beta0 = 1e-2;
  PetscScalar eta0 = 1e-9;
  PetscScalar n0 = 0.1;
  PetscScalar k0 = 1e-19;

  PetscScalar p_src = delta_p*(beta0*n0*omega*cos(omega*t) + k0/eta0*kz*kz*sin(omega*t))*sin(kz*z); // correct
  //~ PetscScalar p_src = 0;
  return p_src;
}

