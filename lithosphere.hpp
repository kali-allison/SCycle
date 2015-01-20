#ifndef LITHOSPHERE_H_INCLUDED
#define LITHOSPHERE_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "userContext.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "fault.hpp"


class Lithosphere: public UserContext
{
  private:
    // disable default copy constructor and assignment operator
    Lithosphere(const Lithosphere &that);
    Lithosphere& operator=(const Lithosphere &rhs);

  protected:

    // domain properties
    const PetscInt       _order,_Ny,_Nz;
    const PetscScalar    _Ly,_Lz,_dy,_dz;

    // debugging folder tree
    //~std::string _debugDir;

    // output data
    std::string          _outputDir;

    const PetscScalar    _v0,_vp;

    // boundary conditions
    PetscViewer          _bcFv,_bcSv,_bcRv,_bcDv,_rhsv;

    // off-fault material fields
    PetscScalar         *_muArr;
    Mat                  _mu;
    Vec                  _rhs,_uhat,_sigma_xy,_bcRShift,_surfDisp;


    // linear system data
    std::string          _linSolver;
    KSP                  _ksp;
    PC                   _pc;
    PetscScalar          _kspTol;

    SbpOps               _sbp;

    // time stepping data
    std::string          _timeIntegrator;
    PetscInt             _strideLength; // stride
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;

    // viewers
    PetscViewer          _timeViewer,_surfDispViewer;//,_uhatViewer;

    // runtime data
    double               _integrateTime,_writeTime,_linSolveTime,_factorTime;
    PetscInt             _linSolveCount;

    PetscErrorCode computeShearStress();
    PetscErrorCode setupKSP();
    PetscErrorCode setSurfDisp();

  public:

    //~typedef typename std::vector<Vec>::iterator it_vec;
    //~typedef typename std::vector<Vec>::const_iterator const_it_vec;

    // boundary conditions
    Vec                  _bcF,_bcS,_bcR,_bcD;

    Fault                _fault;

    OdeSolver           *_quadrature;

    Lithosphere(Domain&D);
    ~Lithosphere();
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    PetscErrorCode debug(const PetscReal time,const PetscInt steps,
                     const std::vector<Vec>& var,const std::vector<Vec>& dvar,const char *stage);

    // IO commands
    virtual PetscErrorCode view() = 0;
    PetscErrorCode writeStep();
    PetscErrorCode read();
};


// for models consisting solely of the lithosphere, uncoupled to anything else
// !!!TO DO: move Lithosphere's quadrature and integrate function here
class SymmLithosphere: public Lithosphere
{

  public:
    SymmLithosphere(Domain&D);
    // use Lithosphere's destructor

    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode view();
};


// for models consisting of coupled spring sliders, no damping
class AsymmLithosphere: public Lithosphere
{

  public:
    AsymmLithosphere(Domain&D);
    // use Lithosphere's destructor

    PetscErrorCode resetInitialConds();

    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd,Vec& tauMod); // if it's coupled to another spring-slider
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode view();
};



#endif
