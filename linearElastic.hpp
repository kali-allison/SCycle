#ifndef LINEARELASTIC_H_INCLUDED
#define LINEARELASTIC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "odeSolver.hpp"
#include "odeSolverImex.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "fault.hpp"
#include "heatEquation.hpp"



/* Base class for a linear elastic material
 */
class LinearElastic: public IntegratorContextEx, public IntegratorContextImex
{
  private:
    // disable default copy constructor and assignment operator
    LinearElastic(const LinearElastic &that);
    LinearElastic& operator=(const LinearElastic &rhs);

  //~ protected:
  public:

    // domain properties
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load viscosity from
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;
    const Vec            *_y,*_z; // to handle variable grid spacing
    const std::string    _problemType; // symmetric (only y>0) or full
    const bool           _isMMS; // true if running mms test
    bool           _bcLTauQS; // true if spinning up Maxwell viscoelastic problem from constant stress on left boundary



    // output data
    std::string          _outputDir;

    const PetscScalar    _vL;

    // off-fault material fields: + side
    Vec                  _muVecP;
    Vec                  _bcRPShift,_surfDispPlus;
    Vec                  _rhsP,_uP,_sxyP;

    // linear system data
    std::string          _linSolver;
    KSP                  _kspP;
    PC                   _pcP;
    PetscScalar          _kspTol;

    SbpOps               *_sbpP;
    std::string           _sbpType;

    // time stepping data
    std::string          _timeIntegrator;
    PetscInt             _stride1D,_stride2D; // stride
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;
    std::vector<int>     _timeIntInds; // indices of variables to be used in time integration

    // thermomechanical coupling
    std::string _thermalCoupling,_heatEquationType;
    HeatEquation _he;
    Vec          _T; // temperature
    PetscViewer  _tempViewer;

    // viewers
    PetscViewer          _timeV1D,_timeV2D,_surfDispPlusViewer;

    // runtime data
    double               _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
    PetscInt             _linSolveCount;

    PetscViewer          _bcRPlusV,_bcRPShiftV,_bcLPlusV,
                         _uPV,_uAnalV,_rhsPlusV,_sxyPV;


    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput();
    PetscErrorCode setupKSP(SbpOps* sbp,KSP& ksp,PC& pc);


  //~ public:

    // boundary conditions
    string               _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction
    Vec                  _bcTP,_bcRP,_bcBP,_bcLP;

    OdeSolver           *_quadEx; // explicit time stepping
    OdeSolverImex       *_quadImex; // implicit time stepping

    PetscScalar _tLast; // time of last earthquake

    Vec _uPPrev;

    LinearElastic(Domain&D);
    ~LinearElastic();



    PetscErrorCode virtual integrate() = 0; // will call OdeSolver method by same name

    // explicit time-stepping methods
    PetscErrorCode virtual d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin) = 0;
    PetscErrorCode virtual debug(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin,const char *stage) = 0;
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin);

    // methods for implicit/explicit time stepping
    PetscErrorCode virtual d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin,
                      it_vec varBeginIm,const_it_vec varBeginImo,const PetscScalar dt) = 0; // IMEX backward Euler

    // IO commands
    PetscErrorCode virtual view() = 0;
    PetscErrorCode virtual writeStep1D() = 0;
    PetscErrorCode virtual writeStep2D() = 0;

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
  //~ public:

    // initialize data
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode loadFieldsFromFiles();
    PetscErrorCode setInitialConds(Domain& D);
    PetscErrorCode setUpSBPContext(Domain& D);

    PetscErrorCode setShifts();
    PetscErrorCode computeShearStress();

    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSBoundaryConditions(const double time);
    PetscErrorCode measureMMSError();

    PetscErrorCode computeEnergy(const PetscScalar time, Vec& out);
    PetscErrorCode computeEnergyRate(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);


  public:

    SymmFault           _fault;
    std::vector<Vec>    _var; // holds variables for explicit integration in time
    std::vector<Vec>    _varIm; // holds variables for implicit integration in time

    // for energy balance
    Vec _E;
    PetscViewer _eV,_intEV; // calculated energy, energy rate, and integrated energy


    SymmLinearElastic(Domain&D);
    ~SymmLinearElastic();

    PetscErrorCode integrate(); // will call OdeSolver method by same name

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                         const_it_vec varBegin,const_it_vec dvarBegin,const char *stage);

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin,
      it_vec varBeginIm,const_it_vec varBeginImo,const PetscScalar dt); // IMEX backward Euler

    // IO commands
    PetscErrorCode view();
    PetscErrorCode writeStep1D(); // write out 1D fields
    PetscErrorCode writeStep2D(); // write out 2D fields

    PetscErrorCode setSurfDisp();
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



    // linear system data
    KSP                  _kspM;
    PC                   _pcMinus;

    SbpOps               *_sbpM;

    PetscViewer          _surfDispMinusViewer;
    PetscViewer          __bcRMinusV,_bcRMShiftV,_bcLMinusV,
                         _uMV,_rhsMV_stressxyMV;


    PetscErrorCode setShifts();
    PetscErrorCode setSurfDisp();

  public:

    // boundary conditions
    Vec                  _bcTMinus,_bcRMinus,_bcBMinus,_bcLMinus;

    FullFault            _fault;
    std::vector<Vec>    _var; // holds variables to integrate


    FullLinearElastic(Domain&D);
    ~FullLinearElastic();

    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                         const_it_vec varBegin,const_it_vec dvarBegin,const char *stage);

    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin,
                      it_vec varBeginIm,const_it_vec varBeginImo,const PetscScalar dt); // IMEX backward Euler

    // IO commands
    PetscErrorCode view(){};
    PetscErrorCode writeStep1D();
    PetscErrorCode writeStep2D();

    // for debugging
    PetscErrorCode setU();
    PetscErrorCode setSigmaxy();

    PetscErrorCode measureMMSError();
};

#endif
