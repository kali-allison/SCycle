#ifndef LINEARELASTIC_H_INCLUDED
#define LINEARELASTIC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "integratorContext.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "fault.hpp"



/* Base class for a linear elastic material
 */
class LinearElastic: public IntegratorContext
{
  private:
    // disable default copy constructor and assignment operator
    LinearElastic(const LinearElastic &that);
    LinearElastic& operator=(const LinearElastic &rhs);

  protected:

    // domain properties
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;
    const std::string    _problemType; // symmetric (only y>0) or full
    const bool _isMMS; // true if running mms test

    // output data
    std::string          _outputDir;

    const PetscScalar    _v0,_vL;

    // off-fault material fields: + side
    PetscScalar         *_muArrPlus;
    Mat                  _muP;
    Vec                  _bcRPShift,_surfDispPlus;
    Vec                  _rhsP,_uP,_stressxyP;
    Vec                  _uAnal; // for MMS tests

    // linear system data
    std::string          _linSolver;
    KSP                  _kspP;
    PC                   _pcP;
    PetscScalar          _kspTol;

    SbpOps               _sbpP;

    // time stepping data
    std::string          _timeIntegrator;
    PetscInt             _strideLength; // stride
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;

    // viewers
    PetscViewer          _timeViewer,_surfDispPlusViewer;

    // runtime data
    double               _integrateTime,_writeTime,_linSolveTime,_factorTime;
    PetscInt             _linSolveCount;

    PetscViewer          _bcRPlusV,_bcRMinusV,_bcRMinusShiftV,_bcRMlusShiftV,_bcLPlusV,_bcLMinusV,
                         _uPV,_uAnalV,_uMinusV,_rhsPlusV,_rhsMinusV,_sigma_xyPlusV,_sigma_xyMinusV;


    PetscErrorCode setupKSP(SbpOps& sbp,KSP& ksp,PC& pc);


  public:

    // boundary conditions
    string               _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction
    Vec                  _bcTP,_bcRP,_bcBP,_bcLP;

    OdeSolver           *_quadrature;
    //~Fault               *_fault;

    LinearElastic(Domain&D);
    ~LinearElastic();



    //~PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode virtual integrate() = 0; // will call OdeSolver method by same name

    PetscErrorCode virtual d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd) = 0;
    PetscErrorCode virtual debug(const PetscReal time,const PetscInt stepCount,
                         const_it_vec varBegin,const_it_vec varEnd,
                         const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage) = 0;
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    // IO commands
    PetscErrorCode view();
    PetscErrorCode virtual writeStep() = 0;

    PetscErrorCode virtual measureMMSError() = 0;
};





/* Contains all the fields and methods needed to model an elastic lithosphere
 * whose material properties are *symmetric* about the fault. The algorithm
 * is described in Brittany Erickson's paper on the earthquake cycle in
 * sedimentary basins.
 */
class SymmLinearElastic: public LinearElastic
{
  private:
    // disable default copy constructor and assignment operator
    SymmLinearElastic(const SymmLinearElastic &that);
    SymmLinearElastic& operator=(const SymmLinearElastic &rhs);

  protected:


    PetscErrorCode setShifts();
    PetscErrorCode setSurfDisp();
    PetscErrorCode computeShearStress();

    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSuSourceTerms(Vec& Hxsource,const PetscScalar time);
    PetscErrorCode setMMSBoundaryConditions(const double time);
    PetscErrorCode measureMMSError();


    // MMS functions (acting on scalars)
    double MMS_uA(double y,double z, double t);
    PetscErrorCode MMS_uA(Vec& vec,const double time);
    double MMS_uA_y(double y,double z, double t);
    double MMS_uA_yy(const double y,const double z,const double t);
    double MMS_uA_z(const double y,const double z,const double t);
    double MMS_uA_zz(const double y,const double z,const double t);
    double MMS_uA_t(const double y,const double z,const double t);
    double MMS_mu(const double y,const double z);
    double MMS_mu_y(const double y,const double z);
    double MMS_mu_z(const double y,const double z);
    double MMS_uSource(const double y,const double z,const double t);


  public:

    SymmFault       _fault;


    SymmLinearElastic(Domain&D);
    ~SymmLinearElastic();

    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                        it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                            it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                         const_it_vec varBegin,const_it_vec varEnd,
                         const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage);

    // IO commands
    PetscErrorCode writeStep();
};










/* Contains all the fields and methods needed to model an elastic lithosphere
 * whose material properties are *not* necessarily symmetric about the fault.
 * The algorithm is described is based on Brittany Erickson's paper on
 * the earthquake cycle in sedimentary basins, with modifications to the
 * boundary condition on the fault.
 */
class FullLinearElastic: public LinearElastic
{
  protected:

    // off-fault material fields: - side
    PetscScalar         *_muArrMinus;
    Mat                  _muM;
    Vec                  _bcLMShift,_surfDispMinus;
    Vec                  _rhsM,_uM,_sigma_xyMinus;

    PetscViewer          _surfDispMinusViewer;

    // linear system data
    KSP                  _kspM;
    PC                   _pcMinus;

    SbpOps               _sbpMinus;


    PetscErrorCode setShifts();
    PetscErrorCode setSurfDisp();

    PetscErrorCode computeShearStress();

  public:

    // boundary conditions
    Vec                  _bcTMinus,_bcRMinus,_bcBMinus,_bcLMinus;

    FullFault            _fault;


    FullLinearElastic(Domain&D);
    ~FullLinearElastic();

    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                         const_it_vec varBegin,const_it_vec varEnd,
                         const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage);
    // IO commands
    PetscErrorCode writeStep();

    // for debugging
    PetscErrorCode setU();
    PetscErrorCode setSigmaxy();

    PetscErrorCode measureMMSError();
};

#endif
