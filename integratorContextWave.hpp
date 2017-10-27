#ifndef IntegratorContextWave_HPP_INCLUDED
#define IntegratorContextWave_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"
#include "domain.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContextWave
{
  public:

    //~ typedef std::vector<Vec>::iterator it_vec;
    //~ typedef std::vector<Vec>::const_iterator const_it_vec;
    //~ typedef std::map<string,Vec>::iterator it_vec;
    //~ typedef std::map<string,Vec>::const_iterator const_it_vec;

    // Domain *_D;
    
    // PetscScalar    _vL;
    //     // time stepping data
    // std::map <string,Vec>  _varEx; // holds variables for explicit integration in time
    // std::string            _initialU; // gaussian
    // std::string            _timeIntegrator,_timeControlType;

    virtual Domain* getD() = 0;
    virtual PetscScalar getvL() = 0;
    virtual std::map <string,Vec> getvarEx() = 0;
    virtual std::string getinitialU() = 0;
    virtual std::string getTimeIntegrator() = 0;

    virtual PetscErrorCode d_dt_WaveEq(const PetscScalar time,const map<string,Vec>& var,map<string,Vec>& dvar, Vec& _ay) = 0;
    virtual PetscErrorCode integrate() = 0;
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar) = 0;
    virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar, const char *stage) = 0;
};

#include "odeSolver.hpp"

#endif
