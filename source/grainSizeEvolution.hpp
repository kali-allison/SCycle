#ifndef GRAINSIZEEVOLUTION_H_INCLUDED
#define GRAINSIZEEVOLUTION_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "rootFinderContext.hpp"

class RootFinder;

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
    std::string          _timeIntegrationType;  // "explicit" or "implicit" time stepping, or "piez" to follow piezometer

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
    std::map <string,std::pair<PetscViewer,string> >  _viewers;

    // run time analysis
    double       _nonlinearSolveTime;



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

    // for implicit time stepping
    PetscErrorCode be(Vec& grainSizeNew,const Vec& grainSizePrev,const PetscScalar time,const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp,const PetscScalar dt);

    // compute grain size based on piezometric relation
    PetscErrorCode computeGrainSizeFromPiez(const Vec& sdev, const Vec& dgdev_disl, const Vec& Temp);

    // file I/O
    PetscErrorCode view(const double totRunTime);
    PetscErrorCode writeDomain(const std::string outputDir);
    PetscErrorCode writeContext(const std::string outputDir);
    PetscErrorCode writeStep(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);

};

// struct for root-finding pieces

// computing the slip velocity for the quasi-dynamic problem
struct AustinEvans2007 : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscInt        _N; // length of the arrays
  const PetscScalar     _deltaT; // (s) time step
  const PetscScalar     *_dprev; // previous grain size
  const PetscScalar     *_A, *_QR, *_p, *_T; // static grain growth parameters
  const PetscScalar     *_f, *_sdev, *_dgdev; // mechanical work done by dislocation creep that reduces grain size
  const PetscScalar     *_gamma; // (GJ/m^2) specific surface energy
  const PetscScalar     _c; // geometric constant


  // constructor and destructor
  AustinEvans2007(const PetscInt N,const PetscScalar deltaT,const PetscScalar* dprev,const PetscScalar* A,const PetscScalar* QR,const PetscScalar* p,const PetscScalar* T,const PetscScalar* f,const PetscScalar* sdev,const PetscScalar* dgdev,const PetscScalar* gamma,const PetscScalar& c);
  //~ ~AustinEvans2007(); // use default destructor, as this class consists entirely of shallow copies

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeGrainSize(PetscScalar* grainSize, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar dnew,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar dnew,PetscScalar *out,PetscScalar *J);
};

#include "rootFinder.hpp"

#endif
