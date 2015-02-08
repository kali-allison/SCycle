#ifndef LITHOSPHERE_H_INCLUDED
#define LITHOSPHERE_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "integratorContext.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "fault.hpp"



/* TO DO:
 *   - Change dependence on fault so that integrate doesn't have to be
 *     defined separately for Full and Symm versions???
 *      -> will need to have check that it's been set to prevent seg faults
 *
 */


/* Base class for an elastic lithosphere
 */
class Lithosphere: public IntegratorContext
{
  private:
    // disable default copy constructor and assignment operator
    Lithosphere(const Lithosphere &that);
    Lithosphere& operator=(const Lithosphere &rhs);

  protected:

    // domain properties
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;
    const std::string    _problemType; // symmetric (only y>0) or full

    // output data
    std::string          _outputDir;

    const PetscScalar    _v0,_vp;

    // off-fault material fields: + side
    PetscScalar         *_muArrPlus;
    Mat                  _muPlus;
    Vec                  _bcRplusShift,_surfDispPlus;
    Vec                  _rhsPlus,_uhatPlus,_sigma_xyPlus;

    // linear system data
    std::string          _linSolver;
    KSP                  _kspPlus;
    PC                   _pcPlus;
    PetscScalar          _kspTol;

    SbpOps               _sbpPlus;

    // time stepping data
    std::string          _timeIntegrator;
    PetscInt             _strideLength; // stride
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;

    // viewers
    PetscViewer          _timeViewer,_surfDispPlusViewer;//,_uhatViewer;

    // runtime data
    double               _integrateTime,_writeTime,_linSolveTime,_factorTime;
    PetscInt             _linSolveCount;


    PetscErrorCode setupKSP(SbpOps& sbp,KSP& ksp,PC& pc);

  public:

    // boundary conditions
    Vec                  _bcTplus,_bcRplus,_bcBplus,_bcFplus;

    OdeSolver           *_quadrature;

    Lithosphere(Domain&D);
    ~Lithosphere();



    PetscErrorCode virtual integrate() = 0; // will call OdeSolver method by same name
    PetscErrorCode debug(const PetscReal time,const PetscInt steps,
                     const std::vector<Vec>& var,const std::vector<Vec>& dvar,const char *stage);


    PetscErrorCode virtual d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd) = 0;
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    // IO commands
    PetscErrorCode view();
    PetscErrorCode virtual writeStep() = 0;
};





/* Contains all the fields and methods needed to model an elastic lithosphere
 * whose material properties are *symmetric* about the fault. The algorithm
 * is described in Brittany Erickson's paper on the earthquake cycle in
 * sedimentary basins.
 */
class SymmLithosphere: public Lithosphere
{
  private:
    // disable default copy constructor and assignment operator
    SymmLithosphere(const SymmLithosphere &that);
    SymmLithosphere& operator=(const SymmLithosphere &rhs);

  protected:


    PetscErrorCode setShifts();
    PetscErrorCode setSurfDisp();

    PetscErrorCode computeShearStress();

  public:

    SymmFault       _fault;


    SymmLithosphere(Domain&D);
    ~SymmLithosphere();

    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);

    // IO commands
    PetscErrorCode writeStep();
};










/* Contains all the fields and methods needed to model an elastic lithosphere
 * whose material properties are *not* necessarily symmetric about the fault.
 * The algorithm is described is based on Brittany Erickson's paper on
 * the earthquake cycle in sedimentary basins, with modifications to the
 * boundary condition on the fault.
 */
class FullLithosphere: public Lithosphere
{
  protected:

    // off-fault material fields: - side
    PetscScalar         *_muArrMinus;
    Mat                  _muMinus;
    Vec                  _bcRminusShift,_surfDispMinus;
    Vec                  _rhsMinus,_uhatMinus,_sigma_xyMinus;

    PetscViewer          _surfDispMinusViewer;

    // linear system data
    KSP                  _kspMinus;
    PC                   _pcMinus;

    SbpOps               _sbpMinus;


    PetscErrorCode setShifts();
    PetscErrorCode setSurfDisp();

    PetscErrorCode computeShearStress();

  public:

    // boundary conditions
    Vec                  _bcTminus,_bcRminus,_bcBminus,_bcFminus;

    FullFault            _fault;


    FullLithosphere(Domain&D);
    ~FullLithosphere();


    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);

    // IO commands
    PetscErrorCode writeStep();
};

#endif
