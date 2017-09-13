#include "fault_hydraulic.hpp"

#define FILENAME "fault_hydraulic.cpp"

using namespace std;

SymmFault_Hydr::SymmFault_Hydr(Domain&D, HeatEquation& He)
: SymmFault(D,He),
_hydraulicCoupling("yes"),_hydraulicTimeIntType("explicit"),
  _n_p(NULL),_beta_p(NULL),_k_p(NULL),_eta_p(NULL),_rho_f(NULL),_g(9.8),_sN(NULL),
  _linSolver("AMG"),_ksp(NULL),_kspTol(1e-10),_sbp(NULL),_sbpType("mfc"),_linSolveCount(0),
  _pViewer(NULL),_sNEffviewer(NULL),
  _writeTime(0),_linSolveTime(0),_ptTime(0),_startTime(0),_miscTime(0),
  _p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::SymmFault_Hydr";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set a, b, normal stress, and Dc
  loadSettings(_file);
  checkInput();
  setFields(D);
  computeInitialSteadyStatePressure(D);

  //~ if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

SymmFault_Hydr::~SymmFault_Hydr()
{
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::~SymmFault_Hydr";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // fields that exist on the fault
  VecDestroy(&_n_p);
  VecDestroy(&_beta_p);
  VecDestroy(&_k_p);
  VecDestroy(&_eta_p);
  VecDestroy(&_rho_f);

  VecDestroy(&_p);

  PetscViewerDestroy(&_pViewer);
  PetscViewerDestroy(&_sNEffviewer);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// Check that required fields have been set by the input file
PetscErrorCode SymmFault_Hydr::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::checkInput";
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

PetscErrorCode SymmFault_Hydr::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
      std::string funcName = "SymmFault_Hydr::loadSettings";
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
    if (var.compare("hydraulicCoupling")==0) {
      _hydraulicCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hydraulicTimeIntType")==0) {
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

PetscErrorCode SymmFault_Hydr::setFields(Domain&D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::setFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // allocate memory and partition accross processors
  VecDuplicate(_tauQSP,&_n_p);  PetscObjectSetName((PetscObject) _n_p, "n_p");  VecSet(_n_p,0.0);
  VecDuplicate(_tauQSP,&_beta_p);  PetscObjectSetName((PetscObject) _beta_p, "beta_p");  VecSet(_beta_p,0.0);
  VecDuplicate(_tauQSP,&_k_p);  PetscObjectSetName((PetscObject) _k_p, "k_p");  VecSet(_k_p,0.0);
  VecDuplicate(_tauQSP,&_eta_p);  PetscObjectSetName((PetscObject) _eta_p, "eta_p");  VecSet(_eta_p,0.0);
  VecDuplicate(_tauQSP,&_rho_f);  PetscObjectSetName((PetscObject) _rho_f, "rho_f");  VecSet(_rho_f,0.0);
  VecDuplicate(_sNEff,&_sN);  PetscObjectSetName((PetscObject) _sN, "sN");  VecCopy(_sNEff,_sN);

  VecDuplicate(_tauQSP,&_p);  PetscObjectSetName((PetscObject) _p, "p");  VecSet(_p,0.0);

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

  // initialize sN based on sNEff
  PetscScalar *sNEff,*sN,*p,*rho_f,*z=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_sNEff,&Istart,&Iend);
  VecGetArray(_sNEff,&sNEff);
  VecGetArray(_sN,&sN);
  VecGetArray(_p,&p);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_z,&z);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    sN[Jj] = sNEff[Jj] + p[Jj];

    assert(~isnan(sNEff[Jj]));
    assert(~isinf(sNEff[Jj]));
    Jj++;
  }
  VecRestoreArray(_sNEff,&sNEff);
  VecRestoreArray(_sN,&sN);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_z,&z);



  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcT);
  VecSetSizes(_bcT,PETSC_DECIDE,1);
  VecSetFromOptions(_bcT);
  PetscObjectSetName((PetscObject) _bcT, "bcT");
  VecSet(_bcT,_pVals[0]);

  VecDuplicate(_bcT,&_bcB);
  VecSet(_bcB,_g*_rho_fVals.back());

  VecCreate(PETSC_COMM_WORLD,&_bcL);
  VecSetSizes(_bcL,PETSC_DECIDE,_N);
  VecSetFromOptions(_bcL);
  PetscObjectSetName((PetscObject) _bcL, "bcL");
  VecSet(_bcL,0.0);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault_Hydr::computeVariableCoefficient(const Vec& p,Vec& coeff)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SymmFault_Hydr::computeVariableCoefficient";
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
PetscErrorCode SymmFault_Hydr::computeInitialSteadyStatePressure(Domain& D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "SymmFault_Hydr::computeInitialSteadyStatePressure";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // set up variable coefficient
  Vec coeff;
  VecSet(_p,0.);
  computeVariableCoefficient(_p,coeff);

  // Set up linear system
  if (_sbpType.compare("mfc")==0 || D._sbpType.compare("mc")==0) {
    _sbp = new SbpOps_fc(D,1,_N,coeff,"Dirichlet","Dirichlet","Neumann","Dirichlet","z");
  }
  else if (_sbpType.compare("mfc_coordTrans")==0) {
    _sbp = new SbpOps_fc_coordTrans(D,1,_N,coeff,"Dirichlet","Dirichlet","Neumann","Dirichlet","z");
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: SBP type type not understood\n");
    assert(0); // automatically fail
  }


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
  //~ PetscScalar *rho_f,*p,*z=0;
  //~ PetscInt Ii,Istart,Iend;
  //~ VecGetOwnershipRange(_p,&Istart,&Iend);
  //~ VecGetArray(_rho_f,&rho_f);
  //~ VecGetArray(_p,&p);
  //~ VecGetArray(_z,&z);
  //~ PetscInt Jj = 0;
  //~ for (Ii=Istart;Ii<Iend;Ii++) {
    //~ p[Jj] = rho_f[Jj]*_g*z[Jj];
    //~ Jj++;
  //~ }
  //~ VecRestoreArray(_rho_f,&rho_f);
  //~ VecRestoreArray(_p,&p);
  //~ VecRestoreArray(_z,&z);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmFault_Hydr::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // allow SymmFault to add slip and state to integrated variables
  SymmFault::initiateIntegrand(time,varEx,varIm);

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

PetscErrorCode SymmFault_Hydr::updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_hydraulicTimeIntType.compare("explicit")==0) {
    VecCopy(varEx.find("pressure")->second,_p);
  }
  else { // put variables to be integrated implicity into varIm
    //~ VecCopy(varIm.find("pressure")->second,_p); // not necessary
  }

  // !!! not sure this is the correct location for this
  if (_hydraulicCoupling.compare("coupled")==0) {
    setSNEff();
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// time stepping functions

// compute effective normal stress from total and pore pressure:
// sNEff = sN - rho*g*z - dp
PetscErrorCode SymmFault_Hydr::setSNEff()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::setSNEff";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  PetscScalar *sNEff,*sN,*p,*rho_f,*z=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_sNEff,&Istart,&Iend);
  VecGetArray(_sNEff,&sNEff);
  VecGetArray(_sN,&sN);
  VecGetArray(_p,&p);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_z,&z);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    sNEff[Jj] = sN[Jj] - p[Jj];

    assert(~isnan(sNEff[Jj]));
    assert(~isinf(sNEff[Jj]));
    Jj++;
  }
  VecRestoreArray(_p,&p);
  VecRestoreArray(_sNEff,&sNEff);
  VecRestoreArray(_sN,&sN);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_z,&z);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}


// explicit time integration
PetscErrorCode SymmFault_Hydr::d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::d_dt_eqCycle";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // call base class's rate function to update slip and state variable rate
  SymmFault::d_dt_eqCycle(time,varEx,dvarEx);


  double startTime = MPI_Wtime(); // time this section


  // source term from gravity
  Vec rhog,rhog_y,temp;
  VecDuplicate(_p,&rhog);
  VecSet(rhog,_g);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_rho_f);
  VecPointwiseMult(rhog,rhog,_k_p);
  VecPointwiseDivide(rhog,rhog,_eta_p);
  VecDuplicate(_p,&rhog_y);
  VecDuplicate(_p,&temp);
  _sbp->Dy(rhog,temp);
  _sbp->H(temp,rhog_y);

  Vec p_t;
  VecDuplicate(_k_p,&p_t);
  Mat D2;
  _sbp->getA(D2);
  ierr = MatMult(D2,_p,p_t); CHKERRQ(ierr);

  // set up boundary terms
  //~ VecSet(_bcB,20.); // add some fluid flow from depth
  Vec rhs;
  VecDuplicate(_k_p,&rhs);
  _sbp->setRhs(rhs,_bcL,_bcL,_bcT,_bcB);


  // compute rate for pressure
  PetscScalar *rhog_yA,*rho_f,*n,*beta,*p_tA,*rhsA=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_sNEff,&Istart,&Iend);
  VecGetArray(rhog_y,&rhog_yA);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_n_p,&n);
  VecGetArray(_beta_p,&beta);
  VecGetArray(rhs,&rhsA);
  VecGetArray(p_t,&p_tA);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    p_tA[Jj] = p_tA[Jj] - 1e3*rhog_yA[Jj] - rhsA[Jj];
    p_tA[Jj] = p_tA[Jj] / ( rho_f[Jj]*n[Jj]*beta[Jj] );

    p_tA[Jj] = 0.0;

    assert(~isnan(p_tA[Jj]));
    assert(~isinf(p_tA[Jj]));
    Jj++;
  }
  VecRestoreArray(rhog_y,&rhog_yA);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_n_p,&n);
  VecRestoreArray(_beta_p,&beta);
  VecRestoreArray(rhs,&rhsA);
  VecRestoreArray(p_t,&p_tA);

  _sbp->Hinv(p_t,dvarEx["pressure"]);



  VecDestroy(&rhog);
  VecDestroy(&rhog_y);
  VecDestroy(&temp);
  VecDestroy(&rhs);
  VecDestroy(&p_t);

  _ptTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault_Hydr::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::d_dt_mms";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// implicit time integration
PetscErrorCode SymmFault_Hydr::be(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::be";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault_Hydr::be_eqCycle(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::be_eqCycle";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(0);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault_Hydr::be_MMS(const PetscScalar time,const Vec slipVel,const Vec& tau,
  const Vec& sigmadev, const Vec& dgxy, const Vec& dgxz,Vec& T,const Vec& To,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::be_MMS";
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

PetscErrorCode SymmFault_Hydr::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  SymmFault::view(totRunTime);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"   pressure rate time (explicit) (s): %g\n",_ptTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent computing pressure rate: %g\n",_ptTime/totRunTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode SymmFault_Hydr::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  SymmFault::writeContext();

  PetscViewer    viewer;

  // write out scalar info
  std::string str = _outputDir + "fault_hydr_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"g = %.15e\n",_g); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str()); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicTimeIntType = %s\n",_hydraulicTimeIntType.c_str()); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  //~ PetscErrorCode writeVec(Vec vec,std::string str)
  ierr = writeVec(_n_p,_outputDir + "fault_hydr_n"); CHKERRQ(ierr);
  ierr = writeVec(_beta_p,_outputDir + "fault_hydr_beta"); CHKERRQ(ierr);
  ierr = writeVec(_k_p,_outputDir + "fault_hydr_k"); CHKERRQ(ierr);
  ierr = writeVec(_eta_p,_outputDir + "fault_hydr_eta"); CHKERRQ(ierr);
  ierr = writeVec(_rho_f,_outputDir + "fault_hydr_rho_f"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode SymmFault_Hydr::writeStep(const PetscInt step)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::writeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  SymmFault::writeStep(step);

  if (step==0) {
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"fault_hydr_p").c_str(),FILE_MODE_WRITE,&_pViewer);
    ierr = VecView(_p,_pViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_pViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"fault_hydr_p").c_str(),
                                 FILE_MODE_APPEND,&_pViewer);CHKERRQ(ierr);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"fault_hydr_sNEff").c_str(),FILE_MODE_WRITE,&_sNEffviewer);
    ierr = VecView(_sNEff,_sNEffviewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_sNEffviewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"fault_hydr_sNEff").c_str(),
                                 FILE_MODE_APPEND,&_sNEffviewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_p,_pViewer);CHKERRQ(ierr);
    ierr = VecView(_sNEff,_sNEffviewer);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}





