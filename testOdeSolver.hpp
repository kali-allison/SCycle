#ifndef TESTODESOLVER_HPP_INCLUDED
#define TESTODESOLVER_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <cmath>
#include <assert.h>
#include <string>
#include "integratorContext.hpp"
#include "odeSolver.hpp"

/***********************************************************************
 *
 * Contains functions to test the odeSolver routines.
 *
 **********************************************************************/


class TestOdeSolver: public IntegratorContext
{

  private:
    // disable default copy constructor and assignment operator
    TestOdeSolver(const TestOdeSolver &that);
    TestOdeSolver& operator=(const TestOdeSolver &rhs);

  public:

    Vec _f;
    std::vector<Vec>    _var; // thing being integrated

    OdeSolver           *_quadrature;

    // viewers for file IO
    PetscViewer _timeViewer, _fViewer;

    // time integration fields
    PetscInt             _strideLength; // stride
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;

    TestOdeSolver();
    ~TestOdeSolver();

    PetscErrorCode integrate();

    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
        it_vec dvarBegin,it_vec dvarEnd);

    PetscErrorCode writeStep();

    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                           const_it_vec varBegin,const_it_vec varEnd,
                           const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                           const_it_vec varBegin,const_it_vec varEnd,
                           const_it_vec dvarBegin,const_it_vec dvarEnd, const char *stage);
};

#endif
