#include "fault_hydraulic.hpp"

#define FILENAME "fault_hydraulic.cpp"

using namespace std;


SymmFault_Hydr::SymmFault_Hydr(Domain&D, HeatEquation& He)
: SymmFault(D,He),
  _n_p(NULL),_beta_p(NULL),_k_p(NULL),_eta_p(NULL),_rho_f(NULL),_g(10.),
  _pViewer(NULL),_dpViewer(NULL),
  _p(NULL),_dp(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::SymmFault_Hydr";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set a, b, normal stress, and Dc
  loadSettings(_file);
  checkInput();
  setFields(D);

  //~ if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

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

  assert(_g >= 0);

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
    if (var.compare("n_pVals")==0) {
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

  VecDuplicate(_tauQSP,&_p);  PetscObjectSetName((PetscObject) _p, "p");  VecSet(_p,0.0);
  VecDuplicate(_tauQSP,&_dp);  PetscObjectSetName((PetscObject) _dp, "dp");  VecSet(_dp,0.0);

  // initialize values
  if (_N == 1) {
    VecSet(_n_p,_n_pVals[0]);
    VecSet(_beta_p,_beta_pVals[0]);
    VecSet(_k_p,_k_pVals[0]);
    VecSet(_eta_p,_eta_pVals[0]);
    VecSet(_rho_f,_rho_fVals[0]);
    VecSet(_p,_pVals[0]);
  }
  else {
    ierr = setVecFromVectors(_n_p,_n_pVals,_n_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_beta_p,_beta_pVals,_beta_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_k_p,_k_pVals,_k_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_eta_p,_eta_pVals,_eta_pDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_rho_f,_rho_fVals,_rho_fDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_p,_pVals,_pDepths);CHKERRQ(ierr);
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

    // compute effective viscosity
  PetscScalar *sNEff,*sN,*dp,*rho_f,*z=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_sNEff,&Istart,&Iend);
  VecGetArray(_sNEff,&sNEff);
  VecGetArray(_sN,&sN);
  VecGetArray(_dp,&dp);
  VecGetArray(_rho_f,&rho_f);
  VecGetArray(_z,&z);
  PetscInt Jj = 0;
  for (Ii=Istart;Ii<Iend;Ii++) {
    sNEff[Jj] = sN[Jj] - dp[Jj] - _g*rho_f[Jj]*z[Jj];

    assert(~isnan(sNEff[Jj]));
    assert(~isinf(sNEff[Jj]));
    Jj++;
  }
  VecRestoreArray(_sNEff,&sNEff);
  VecRestoreArray(_sN,&sN);
  VecRestoreArray(_rho_f,&rho_f);
  VecRestoreArray(_z,&z);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
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
  VecDestroy(&_dp);

  //~ PetscViewerDestroy(&_slipViewer);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// =====================================================================
// IO commands


// extends SymmFault's writeContext
PetscErrorCode SymmFault_Hydr::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  SymmFault::writeContext(outputDir);

  PetscViewer    viewer;

  // write out scalar info
  std::string str = outputDir + "fault_hydr_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"g = %.15e\n",_g); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


  //~ PetscErrorCode writeVec(Vec vec,std::string str)
  ierr = writeVec(_n_p,outputDir + "fault_hydr_n"); CHKERRQ(ierr);
  ierr = writeVec(_beta_p,outputDir + "fault_hydr_beta"); CHKERRQ(ierr);
  ierr = writeVec(_k_p,outputDir + "fault_hydr_k"); CHKERRQ(ierr);
  ierr = writeVec(_eta_p,outputDir + "fault_hydr_eta"); CHKERRQ(ierr);
  ierr = writeVec(_rho_f,outputDir + "fault_hydr_rho_f"); CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// extends SymmFault's writeContext
PetscErrorCode SymmFault_Hydr::writeStep(const string outputDir,const PetscInt step)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault_Hydr::writeStep";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  SymmFault::writeStep(outputDir,step);

  if (step==0) {
      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"fault_hydr_p").c_str(),FILE_MODE_WRITE,&_slipViewer);
      ierr = VecView(_p,_pViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_pViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"fault_hydr_p").c_str(),
                                   FILE_MODE_APPEND,&_pViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"fault_hydr_dp").c_str(),FILE_MODE_WRITE,&_slipViewer);
      ierr = VecView(_dp,_dpViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_dpViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"fault_hydr_dp").c_str(),
                                   FILE_MODE_APPEND,&_dpViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_p,_pViewer);CHKERRQ(ierr);
    ierr = VecView(_dp,_dpViewer);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}





