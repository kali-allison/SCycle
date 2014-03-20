#include <petscts.h>
#include <string>
#include "userContext.h"
//~#include "init.h"

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
  ctx.vp = 1.e-4; //(m/s) plate rate ORIG 1.e-9
  ctx.D_c = 8e-3; //characteristic slip distance (m) 0.008

  //  constitutive parameters
  ctx.cs = 3; // shear wave speed (km/s)
  ctx.G = 2000.0; // shear modulus (GPa) ORIG:36
  ctx.rho = ctx.G/(ctx.cs*ctx.cs);// Material density

  // tolerances for linear and nonlinear solve (for V)
  ctx.kspTol = 1e-6;
  ctx.rootTol = 1e-12;

  // time monitering
  ctx.strideLength = 1;
  ctx.maxStepCount = 5e3; // 3845
  ctx.initTime = 1e3;
  ctx.currTime = ctx.initTime;
  ctx.maxTime = 5000.*3.155926e1*3;// 5000.*3.1556926e7; 5000.*3.155926e1*3
  ctx.atol = 1e-9;
  ctx.initDeltaT = 5.0;
  ctx.minDeltaT = 0.5*std::min(ctx.Ly/ctx.Ny,ctx.Lz/ctx.Nz)/ctx.cs;
  ctx.maxDeltaT = 1e2;


  return 0;
}

