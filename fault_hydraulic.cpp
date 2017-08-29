#include "fault.hpp"

#define FILENAME "fault.cpp"

using namespace std;


Fault_p::Fault_p(Domain&D)
: _n_p(NULL),_beta_p(NULL),_k_p(NULL),_eta_p(NULL),_rho_f(NULL),_g(10.),
  _p(NULL),_dp(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Fault_p::Fault_p";
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
PetscErrorCode Fault_p::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault_p::checkInput";
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

PetscErrorCode Fault_p::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
      std::string funcName = "Fault_p::loadSettings";
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

PetscErrorCode Fault_p::setFields(Domain&D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault_p::setFields";
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

Fault_p::~Fault_p()
{
  #if VERBOSE > 1
    std::string funcName = "Fault_p::~Fault_p";
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
