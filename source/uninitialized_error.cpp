// constructor for ComputeVel_qd
ComputeVel_qd::ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0,const PetscScalar& vL,const PetscScalar* locked,const PetscScalar* Co)
: _a(a),_b(b),_sN(sN),_tauQS(tauQS),_eta(eta),_psi(psi),_locked(locked),_Co(Co),_N(N),_v0(v0),_vL(vL)
{ }


// compute slip velocity for quasidynamic setting
PetscErrorCode ComputeVel_qd::computeVel(PetscScalar *slipVelA, const PetscScalar rootTol, PetscInt &rootIts, const PetscInt maxNumIts)
{
  PetscErrorCode ierr = 0;
  PetscScalar left, right, out;
  PetscInt Jj;
  for (Jj = 0; Jj < _N; Jj++) {
    if (_locked[Jj] > 0.5) {
      slipVelA[Jj] = 0.;
    }

    else if (_locked[Jj] < -0.5) {
      slipVelA[Jj] = _vL;
    }
    else {
      left = 0.;
      right = _tauQS[Jj] / _eta[Jj];

      if (isnan(left) || isnan(right)) {
	assert(0);
      }
      out = slipVelA[Jj];

      if (abs(left-right)<1e-14) {
	out = left;
      }
      else {
	PetscScalar x0 = slipVelA[Jj];
	BracketedNewton rootFinder(maxNumIts,rootTol);
	ierr = rootFinder.setBounds(left,right);CHKERRQ(ierr);
	ierr = rootFinder.findRoot(this,Jj,x0,&out); CHKERRQ(ierr);
	rootIts += rootFinder.getNumIts();
      }
      slipVelA[Jj] = out;
    }
  }
  return ierr;
}


// calls the computeVel function above
/* all variables starting with _ are initialized to zero vectors in the Fault_qd constructor */
PetscErrorCode Fault_qd::computeVel()
{
  PetscErrorCode ierr = 0;

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

  ierr = VecRestoreArray(_slipVel,&slipVelA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_eta_rad,&etaA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_tauQSP,&tauQSA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_sNEff,&sNA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_psi,&psiA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_a,&aA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_b,&bA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_locked,&lockedA); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(_cohesion,&Co); CHKERRQ(ierr);

  return ierr;
}
