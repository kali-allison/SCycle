#ifndef EARTH_H_INCLUDED
#define EARTH_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "userContext.hpp"
#include "domain.hpp"
#include "lithosphere.hpp"

class OdeSolver;

class Earth : public UserContext
{

  protected:

    Domain                      _domain1, _domain2;
    CoupledLithosphere          _slider1,_slider2;

    // boundary conditions between each layer
    Vec                  _bcL, _bcR, bcShared;
    Vec                  _bcRShift;
    PetscScalar          _vL; // loading velocity
    Vec                  _tauMod; // holds the modification to force balance from the other slider(s)

    // fields that _quadrature needs to know about
    vector<Vec>         _var;

    // time stepping data
    std::string          _timeIntegrator;
    PetscInt             _strideLength; // stride between writes
    PetscInt             _maxStepCount; // largest number of time steps
    PetscReal            _initTime,_currTime,_maxTime,_minDeltaT,_maxDeltaT;
    int                  _stepCount;
    PetscScalar          _atol;
    PetscScalar          _initDeltaT;

    // disable default copy constructor and assignment operator
    Earth(const Earth &that);
    Earth& operator=(const Earth &rhs);

  public:

    OdeSolver           *_quadrature;

    Earth(const char* inputFile1, const char* inputFile2);
    ~Earth();

    //~PetscErrorCode d_dt(PetscScalar const time,const vector<Vec>& var,vector<Vec>& dvar);
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode integrate(); // will call OdeSolver method by same name
    //~PetscErrorCode timeMonitor(const PetscReal time, const PetscInt stepCount,
                     //~const vector<Vec>& var,const vector<Vec>& dvar);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    // IO commands
    PetscErrorCode view();
    PetscErrorCode writeStep();
    PetscErrorCode read();
};

#include "odeSolver.hpp"

#endif
