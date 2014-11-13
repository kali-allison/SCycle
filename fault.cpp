#include "fault.hpp"


//================= constructor and destructor ========================
Fault::Fault(Domain&D)
: _N(D._Nz),_sizeMuArr(D._Ny*D._Nz),_L(D._Lz),_h(_L/(_N-1.)),_Dc(D._Dc),
  _rootTol(D._rootTol),_rootIts(0),_maxNumIts(1e8),
  _depth(D._depth),_seisDepth(D._seisDepth),_cs(0),_f0(D._f0),_v0(D._v0),_vp(D._vp),
  _bAbove(D._bAbove),_bBelow(D._bBelow),
  _muArr(D._muArr),_rhoArr(D._rhoArr),_csArr(D._csArr),
  _sigma_N_val(D._sigma_N_val)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in fault.cpp.\n");
#endif

  //~_cs = std::max(sqrt(_muIn/_rhoIn),sqrt(_muOut/_rhoOut)); // shear wave speed (km/s)

  // fields that exist on the fault
  VecCreate(PETSC_COMM_WORLD,&_tau);
  VecSetSizes(_tau,PETSC_DECIDE,_N);
  VecSetFromOptions(_tau);     PetscObjectSetName((PetscObject) _tau, "tau");
  VecDuplicate(_tau,&_psi); PetscObjectSetName((PetscObject) _psi, "psi");
  VecDuplicate(_tau,&_tempPsi); PetscObjectSetName((PetscObject) _tempPsi, "tempPsi");
  VecDuplicate(_tau,&_dPsi); PetscObjectSetName((PetscObject) _dPsi, "dPsi");
  VecDuplicate(_tau,&_faultDisp); PetscObjectSetName((PetscObject) _faultDisp, "faultDisp");
  VecDuplicate(_tau,&_vel); PetscObjectSetName((PetscObject) _vel, "vel");

  _var[0] = _faultDisp;
  _var[1] = _psi;

  // frictional fields
  VecDuplicate(_tau,&_eta); PetscObjectSetName((PetscObject) _eta, "eta");
  VecDuplicate(_tau,&_sigma_N); PetscObjectSetName((PetscObject) _sigma_N, "sigma_N");
  VecDuplicate(_tau,&_a); PetscObjectSetName((PetscObject) _a, "_a");
  VecDuplicate(_tau,&_b); PetscObjectSetName((PetscObject) _b, "_b");

  VecDuplicate(_tau,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "_b");

  _rootAlg = new Bisect(_maxNumIts,_rootTol);

  setFields();


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in fault.cpp.\n");
#endif
}

Fault::~Fault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecDestroy(&_tau);
  VecDestroy(&_psi);
  VecDestroy(&_tempPsi);
  VecDestroy(&_dPsi);
  VecDestroy(&_faultDisp);
  VecDestroy(&_vel);

  //~VecDestroy(&_bcRShift);

  // frictional fields
  VecDestroy(&_eta);
  VecDestroy(&_sigma_N);
  VecDestroy(&_a);
  VecDestroy(&_b);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in fault.cpp.\n");
#endif
}


//==================== protected member functions ======================
PetscErrorCode Fault::computeVel()
{
  PetscErrorCode ierr = 0;
  Vec            left,right,out;
  PetscScalar    outVal,leftVal,rightVal;
  PetscInt       Ii,Istart,Iend;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecDuplicate(_tau,&right);CHKERRQ(ierr);
  ierr = VecCopy(_tau,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,_eta);CHKERRQ(ierr);
  ierr = VecAbs(right);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr); // assumes right-lateral fault

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);
    if (abs(leftVal-rightVal)<1e-14) { outVal = leftVal; }
    else {
      _rootAlg = new Bisect(_maxNumIts,_rootTol);
      ierr = _rootAlg->setBounds(leftVal,rightVal);CHKERRQ(ierr);
      ierr = _rootAlg->findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      _rootIts += _rootAlg->getNumIts();
      delete _rootAlg;
    }
    ierr = VecSetValue(_vel,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_vel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_vel);CHKERRQ(ierr);

  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode Fault::agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,vel;

  //~double startTime = MPI_Wtime();

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(_vel,1,&ind,&vel);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in agingLaw\n");
  }

  //~if (b==0) { *dPsi = 0; }
  if ( isinf(exp(1/b)) ) { *dPsi = 0; }
  else if ( b <= 1e-3 ) { *dPsi = 0; }
  else {
    *dPsi = (PetscScalar) (b*_v0/_Dc)*( exp((double) ( (_f0-psi)/b) ) - (vel/_v0) );
  }


  if (isnan(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,_f0,_Dc,_v0,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*_v0/_Dc));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (_f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/_v0));
    CHKERRQ(ierr);
  }
  else if (isinf(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,_f0,_Dc,_v0,vel);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*_v0/_Dc));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (_f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/_v0));
  }

  assert(!isnan(*dPsi));
  assert(!isinf(*dPsi));

  //~double endTime = MPI_Wtime();
  //~D->agingLawTime = D->agingLawTime + (endTime-startTime);

  return ierr;
}



//==================== set/get functions ===============================

PetscErrorCode Fault::setFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v,z;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFields in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecSet(_psi,_f0);CHKERRQ(ierr);
  ierr = VecCopy(_psi,_tempPsi);CHKERRQ(ierr);
  ierr = VecSet(_a,0.015);CHKERRQ(ierr);

  // Set b
  PetscScalar L2 = 1.5*_seisDepth;  //This is depth at which increase stops and fault is purely velocity strengthening
  PetscInt    N1 = _seisDepth/_h;
  PetscInt    N2 = L2/_h;
  ierr = VecGetOwnershipRange(_b,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < N1+1) {
      //~v=0.02;
      ierr = VecSetValues(_b,1,&Ii,&_bAbove,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (Ii>N1 && Ii<=N2) {
      v = (double) (Ii*_h-_seisDepth)*(_bAbove-_bBelow)/(_seisDepth-L2) + _bAbove;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      //~v = 0.0;
      ierr = VecSetValues(_b,1,&Ii,&_bBelow,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_b);CHKERRQ(ierr);
  //~ierr = VecSet(_b,0.02);CHKERRQ(ierr); // for spring-slider!!!!!!!!!!!!!!!!


  // tau, eta, gRShift, sigma_N
  PetscScalar a,b,eta,tau_inf,sigma_N,bcRShift;
  ierr = VecGetOwnershipRange(_tau,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    //~ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    z = ((double) Ii)*_h;

    if (_sigma_N_val!=0){ sigma_N = _sigma_N_val; }
    else { sigma_N = 9.8*_rhoArr[Ii]*z; }

    //~eta = 0.5*sqrt(_rhoArr[Ii]*_muArr[Ii]);
    eta = 0.5*_muArr[Ii]/_csArr[Ii];

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vp*exp(_f0/a)/_v0 );
    bcRShift = tau_inf*_L/_muArr[_sizeMuArr-_N+Ii]; // use last values of muArr

    ierr = VecSetValue(_tau,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_eta,Ii,eta,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_bcRShift,Ii,bcRShift,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_sigma_N,Ii,sigma_N,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_eta);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRShift);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sigma_N);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_eta);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sigma_N);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Fault::setFaultDisp(Vec const &bcF)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecCopy(bcF,_faultDisp);CHKERRQ(ierr);
  ierr = VecScale(_faultDisp,2.0);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFault in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode Fault::setTau(const Vec&sigma_xy)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setTau in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tau,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tau);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setTau in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}


const Vec& Fault::getBcRShift() const
{
  return _bcRShift;
}



PetscErrorCode Fault::getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,eta,tau;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
    ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
    ierr = VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);
    ierr = VecGetValues(_eta,1,&ind,&eta);CHKERRQ(ierr);
    ierr = VecGetValues(_tau,1,&ind,&tau);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in stressMstrength\n");
  }

   *out = (PetscScalar) a*sigma_N*asinh( (double) (vel/2/_v0)*exp(psi/a) ) + eta*vel - tau;
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,eta,tau,vel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,eta,tau,vel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,eta,tau,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",vel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"eta*vel=%.9e\n",eta*vel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending getResid in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;

}


PetscErrorCode Fault::d_dt(Vec const*var,Vec *dvar)
{
  PetscErrorCode ierr = 0;
  PetscScalar    val,psiVal;
  PetscInt       Ii,Istart,Iend;

  ierr = VecCopy(var[1],_tempPsi);CHKERRQ(ierr);
  ierr = computeVel();CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(_vel,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(_vel,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(dvar[0],Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(var[1],1,&Ii,&psiVal);
    ierr = agingLaw(Ii,psiVal,&val);CHKERRQ(ierr);
    ierr = VecSetValue(dvar[1],Ii,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(dvar[0]);CHKERRQ(ierr); ierr = VecAssemblyBegin(dvar[1]);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(dvar[0]);CHKERRQ(ierr);   ierr = VecAssemblyEnd(dvar[1]);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode Fault::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif

 //~ierr = PetscViewerHDF5PushGroup(viewer, "/frictionContext");CHKERRQ(ierr);
//~
  //~ierr = VecView(a, viewer);CHKERRQ(ierr);
  //~ierr = VecView(b, viewer);CHKERRQ(ierr);
  //~ierr = VecView(eta, viewer);CHKERRQ(ierr);
  //~ierr = VecView(sigma_N, viewer);CHKERRQ(ierr);
  //~ierr = MatView(mu, viewer);CHKERRQ(ierr);
//~
//~
  //~ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  PetscViewer    viewer;

  std::string str = outputDir + "a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "eta";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_eta,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode Fault::writeStep(const string outputDir,const PetscInt step)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"starting writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif

  if (step==0) {

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n in fault::writeStep, setting up viewers:\n");CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"faultDisp").c_str(),FILE_MODE_WRITE,&_faultDispViewer);
      ierr = VecView(_faultDisp,_faultDispViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_faultDispViewer);CHKERRQ(ierr);


      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"vel").c_str(),FILE_MODE_WRITE,&_velViewer);
      ierr = VecView(_vel,_velViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_velViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tau").c_str(),FILE_MODE_WRITE,&_tauViewer);
      ierr = VecView(_tau,_tauViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_tauViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),FILE_MODE_WRITE,&_psiViewer);
      ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_psiViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"faultDisp").c_str(),
                                   FILE_MODE_APPEND,&_faultDispViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"vel").c_str(),
                                   FILE_MODE_APPEND,&_velViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tau").c_str(),
                                   FILE_MODE_APPEND,&_tauViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),
                                   FILE_MODE_APPEND,&_psiViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_faultDisp,_faultDispViewer);CHKERRQ(ierr);
    ierr = VecView(_vel,_velViewer);CHKERRQ(ierr);
    ierr = VecView(_tau,_tauViewer);CHKERRQ(ierr);
    ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}

