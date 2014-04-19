#include <petscts.h>
#include <string>
#include "userContext.h"
#include "init.hpp"

PetscErrorCode setParameters(UserContext & ctx)
{
  //~ ctx->outFileRoot = ".data/";

  // domain geometry initialization
  ctx.Ly = 24.0; // (km) length of fault, y in [0,Ly]
  ctx.Lz = 24.0; // (km) z in [0,Lz]
  ctx.H = 12.0; // (km) This is the depth as which (a-b) begins to increase
  ctx.N  = ctx.Nz*ctx.Ny;
  ctx.dy = ctx.Ly/(ctx.Ny-1.0); // (km)
  ctx.dz = ctx.Lz/(ctx.Nz-1.0);// (km)

  //  frictional parameters
  ctx.f0 = 0.6;  //reference friction
  ctx.v0 = 1e-6; //(m/s) reference velocity
  ctx.vp = 1e-5; //(m/s) plate rate KLA: 1e-4, ORIG: 1e-9
  ctx.D_c = 8e-3; // (m) characteristic slip distance 0.008

  ctx.muIn = 18; // (GPa) shear modulus inside basin; sed cycle: 18
  ctx.muOut = 24; // (GPa) shear modulus outside basin; sed cycle: 24
  ctx.D = ctx.H/3; // (km) basin depth
  ctx.W = 24; // (km) basin width
  ctx.rhoIn = 2.6; // 1e3*(kg/m^3) density inside basin; sed cycle: 2.6
  ctx.rhoOut = 3; // 1e3*(kg/m^3) density outside basin; sed cycle: 3

  //  constitutive parameters
  ctx.cs = std::max(sqrt(ctx.muIn/ctx.rhoIn),sqrt(ctx.muOut/ctx.rhoOut)); // shear wave speed (km/s)
  //~ctx.G = 24; // shear modulus (GPa) ORIG:36
  //~ctx.rho = ctx.G/(ctx.cs*ctx.cs);// Material density

  // tolerances for linear and nonlinear solve (for V)
  ctx.kspTol = 1e-6;
  ctx.rootTol = 1e-12;

  // time monitering
  ctx.strideLength = 1;
  ctx.maxStepCount = 3e4; // 3e4
  //~ctx.maxStepCount = 20;
  ctx.initTime = 6.3e6; // spring-slider:5e5 eqCycle:5e7
  ctx.currTime = ctx.initTime;
  ctx.maxTime = ctx.initTime+2e6;// 5000.*3.1556926e7; spring-slider:5e+05
  ctx.atol = 1e-9;
  ctx.minDeltaT = std::min(0.5*std::min(ctx.Ly/ctx.Ny,ctx.Lz/ctx.Nz)/ctx.cs,1e-3);
  ctx.initDeltaT = ctx.minDeltaT;
  ctx.maxDeltaT = 1e3;


  return 0;
}

