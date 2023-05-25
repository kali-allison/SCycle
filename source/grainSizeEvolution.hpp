#ifndef GRAINSIZEEVOLUTION_H_INCLUDED
#define GRAINSIZEEVOLUTION_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"

using namespace std;

/*
 * Class to explore grain size evolution. Currently uses the grain evolution law
 * from Austin and Evans (2007).
 */

class GrainSizeEvolution
{
  private:
    // disable default copy constructor and assignment operator
    GrainSizeEvolution(const GrainSizeEvolution &that);
    GrainSizeEvolution& operator=(const GrainSizeEvolution &rhs);

  public:

    Domain              *_D;
    const char          *_file;
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // output data
    std::string          _grainSizeEvType; // transient, steadyState, or piezometer
    std::string          _grainSizeEvTypeSS; // transient, steadyState, or piezometer

    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz,_dy,_dz;
    Vec                 *_y,*_z;

    // material properties
    Vec                   _A,_QR,_p; // static grain growth parameters
    Vec                   _f; // fraction of mechanical work done by dislocation creep that reduces grain size
    Vec                   _gamma; // (GJ/m^2) specific surface energy
    PetscScalar           _c; // geometric constant
    std::vector<double>   _AVals,_ADepths,_QRVals,_QRDepths,_pVals,_pDepths,_fVals,_fDepths,_gammaVals,_gammaDepths;
    std::vector<double>   _dVals,_dDepths; // for initialization
    std::vector<double>   _piez_AVals,_piez_ADepths,_piez_nVals,_piez_nDepths; // optional piezometer parameters, d = A * sdev^n
    Vec                   _piez_A,_piez_n;

    Vec          _d; // grain size
    Vec          _d_t; // rate of grain size evolution


    // viewers
    // viewers:
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    //~ std::map <string,std::pair<PetscViewer,string> >  _viewers;
    PetscViewer _viewer;


    GrainSizeEvolution(Domain& D);
    ~GrainSizeEvolution();

    // initialize and set data
    PetscErrorCode loadSettings(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadFieldsFromFiles(); // load non-effective-viscosity parameters

    // for steady-state computation
    PetscErrorCode initiateVarSS(map<string,Vec>& varSS);
    PetscErrorCode computeSteadyStateGrainSize(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp);

    // methods for explicit time stepping
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& varIm);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode d_dt(Vec& grainSizeEv_t,const Vec& grainSize,const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp);
    PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep, const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp);

    // compute grain size based on piezometric relation
    PetscErrorCode computeGrainSizeFromPiez(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp);

    // file I/O
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeContext(const std::string outputDir, PetscViewer& viewer);
    PetscErrorCode writeStep(PetscViewer& viewer);
    PetscErrorCode writeCheckpoint(PetscViewer& viewer);
    PetscErrorCode loadCheckpoint();
    PetscErrorCode loadCheckpointSS();

};

#include "rootFinder.hpp"

#endif
