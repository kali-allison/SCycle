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
  ctx.vp = 1e-4; //(m/s) plate rate KLA: 1e-4, ORIG: 1.e-9
  ctx.D_c = 8e-3; //characteristic slip distance (m) 0.008

  //  constitutive parameters
  ctx.cs = 3; // shear wave speed (km/s)
  ctx.G = 36; // shear modulus (GPa) ORIG:36
  ctx.rho = ctx.G/(ctx.cs*ctx.cs);// Material density

  // tolerances for linear and nonlinear solve (for V)
  ctx.kspTol = 1e-6;
  ctx.rootTol = 1e-12;

  // time monitering
  ctx.strideLength = 1;
  ctx.maxStepCount = 3e4; // 8e4
  ctx.initTime = 4.5e3; // 400*1e3
  ctx.currTime = ctx.initTime;
  ctx.maxTime = ctx.initTime+5.2e5;// 5000.*3.1556926e7; 5000.*3.155926e3; 4.376288924960000e+05
  ctx.atol = 1e-6;
  ctx.initDeltaT = 800;
  ctx.minDeltaT = std::min(0.5*std::min(ctx.Ly/ctx.Ny,ctx.Lz/ctx.Nz)/ctx.cs,0.01);
  ctx.maxDeltaT = 1e3;


  return 0;
}

