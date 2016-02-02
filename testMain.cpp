#include <petscts.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <petscviewerhdf5.h>
#include <string>

#include "genFuncs.hpp"
#include "domain.hpp"
#include "spmat.hpp"
#include "sbpOps.hpp"
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "testOdeSolver.hpp"
#include "odeSolver.hpp"
#include "testOdeSolver.hpp"



/*
 * Compare computation of d/dx with stencils, a matrix operators on DMDAs, and regular matrices.
 * Note that d/dx corresponds to z in my usual coordinate system.
 */
int createDxStencil()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting testMain::timeDx in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt Nx=500,Ny=600;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=6.0;
  PetscScalar dx=(xMax-xMin)/(Nx-1), dy=(yMax-yMin)/(Ny-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  PetscScalar x,y;


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);


  // Set the values for the global vector x based on the (x,y)
  // coordinates for each vertex
  Vec g=NULL,l=NULL; // g = global x, l = local x
  PetscScalar **lArr;
  DMCreateGlobalVector(da,&g); PetscObjectSetName((PetscObject) g, "global g");
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      x = i * dx;
      y = j * dy;
      lArr[j][i] = 2*x  + 3*y;
    }
  }
  DMDAVecRestoreArray(da,l,&lArr);
  ierr = DMLocalToGlobalBegin(da,l,INSERT_VALUES,g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,l,INSERT_VALUES,g);CHKERRQ(ierr);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending testMain::testDMDAWithMats in fault.cpp.\n");
#endif
  return ierr;
}


/*
 * Compare computation of d/dx with stencils, a matrix operators on DMDAs, and regular matrices.
 * Note that d/dx corresponds to z in my usual coordinate system.
 */
int timeDx()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting testMain::timeDx in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt Nx=500,Ny=600;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=6.0;
  PetscScalar dx=(xMax-xMin)/(Nx-1), dy=(yMax-yMin)/(Ny-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  PetscScalar x,y;


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);


  // Set the values for the global vector x based on the (x,y)
  // coordinates for each vertex
  Vec g=NULL,l=NULL; // g = global x, l = local x
  PetscScalar **lArr;
  DMCreateGlobalVector(da,&g); PetscObjectSetName((PetscObject) g, "global g");
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      x = i * dx;
      y = j * dy;
      lArr[j][i] = 2*x  + 3*y;
    }
  }
  DMDAVecRestoreArray(da,l,&lArr);
  ierr = DMLocalToGlobalBegin(da,l,INSERT_VALUES,g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,l,INSERT_VALUES,g);CHKERRQ(ierr);


double startTime = MPI_Wtime();
  // Compute 1st derivative in x-direction
  Vec gx=NULL,lx=NULL; // gx = global, lx = local
  PetscScalar **lxArr;
  VecDuplicate(g,&gx);
  VecSet(gx,0.0);
  PetscObjectSetName((PetscObject) gx, "global gx");
  ierr = DMCreateLocalVector(da,&lx);CHKERRQ(ierr);
  DMDAVecGetArray(da,lx,&lxArr);

  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);

  // only iterate over the local entries in f
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      if (i>0 && i<Nx-1) { lxArr[j][i] = (lArr[j][i+1] - lArr[j][i-1])/(2*dx); }
      else if (i==0) { lxArr[j][i] = (-1.0*lArr[j][0] + 1.0*lArr[j][1] )/dx; }
      else if (i==Nx-1) { lxArr[j][i] = (- 1.0*lArr[j][Nx-2] + 1.0*lArr[j][Nx-1])/dx; }
    }
  }
  DMDAVecRestoreArray(da,lx,&lxArr);
  DMDAVecRestoreArray(da,lx,&lArr);
  DMLocalToGlobalBegin(da,lx,INSERT_VALUES,gx);
  DMLocalToGlobalEnd(da,lx,INSERT_VALUES,gx);
double endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dx stencil: %.9e\n",endTime);



  // Compute 1st derivative in y-direction, store as gy.
  Vec gy=NULL,ly=NULL; // gx = global, lx = local
  PetscScalar **lyArr;
  VecDuplicate(g,&gy);
  VecSet(gy,0.0);
  PetscObjectSetName((PetscObject) gy, "global gy");
  ierr = DMCreateLocalVector(da,&ly);CHKERRQ(ierr);
  DMDAVecGetArray(da,ly,&lyArr);

  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);

  // only iterate over the local entries in f
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      if (j>0 && j<Ny-1) { lyArr[j][i] = (lArr[j+1][i] - lArr[j-1][i])/(2*dy); }
      else if (j==0) { lyArr[j][i] = (-1.0*lArr[0][i] + 1.0*lArr[1][i] )/dy; }
      else if (j==Ny-1) { lyArr[j][i] = (- 1.0*lArr[Ny-2][i] + 1.0*lArr[Ny-1][i])/dy; }
    }
  }
  DMDAVecRestoreArray(da,ly,&lyArr);
  DMDAVecRestoreArray(da,ly,&lArr);
  DMLocalToGlobalBegin(da,ly,INSERT_VALUES,gy);
  DMLocalToGlobalEnd(da,ly,INSERT_VALUES,gy);



  // try another way of making the matrix, this time with MatStencil
  ISLocalToGlobalMapping map,rmap;
  DMGetLocalToGlobalMapping(da,&map);

  PetscInt Istart,Iend;

  // allocate space for mat
  Mat mat;
  DMDALocalInfo lI;
  DMDAGetLocalInfo(da,&lI);


  MatCreate(PETSC_COMM_WORLD,&mat);
  MatSetSizes(mat,lI.xm*lI.ym,lI.xm*lI.ym,PETSC_DECIDE,PETSC_DECIDE);
  MatSetFromOptions(mat);
  MatMPIAIJSetPreallocation(mat,3,NULL,3,NULL);
  MatSeqAIJSetPreallocation(mat,3,NULL);
  MatSetLocalToGlobalMapping(mat,map,map);
  PetscInt dims[3] = {lI.gxm, lI.gym, lI.gzm};
  PetscInt starts[3] = {lI.gxs, lI.gys, lI.gzs};
  MatSetStencil(mat,2,dims,starts,1);
  MatSetUp(mat);


  MatStencil row;
  // closures
  MatStencil colC[2];
  colC[0].i = 0; colC[0].j = 0;
  colC[1].i = 1; colC[1].j = 0;
  PetscScalar vIC[2];
  vIC[0] = -1.0/dx; vIC[1] = 1.0/dx;

  PetscScalar vEC[2];
  vEC[0] = -1.0/dx; vEC[1] = 1.0/dx;

  // interior stencil
  MatStencil rowI,col[2];
  PetscScalar vI[2]; vI[0] = -0.5/dx; vI[1] = 0.5/dx;
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      row.i = i; row.j = j;
      col[0].j = j;
      col[1].j = j;
      if (i>0 && i<Nx-1) {
        col[0].i = i-1;
        col[1].i = i+1;
        ierr = MatSetValuesStencil(mat,1,&row,2,col,vI,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (i == 0) {
        colC[0].i = 0;
        colC[1].i = 1;
        ierr = MatSetValuesStencil(mat,1,&row,2,colC,vIC,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (i == Nx-1) {
        colC[0].i = Nx-2;
        colC[1].i = Nx-1;

        ierr = MatSetValuesStencil(mat,1,&row,2,colC,vEC,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


  Vec gx_ms;
  VecDuplicate(g,&gx_ms);
  PetscObjectSetName((PetscObject) gx_ms, "global gx_ms");

  VecSet(gx_ms,0.0);
startTime = MPI_Wtime();
  MatMult(mat,g,gx_ms);
endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dx stencil-matrix: %.9e\n",endTime);


// traditional matrix derivative
  Vec g_m;
  PetscInt Ii;
  PetscScalar v;
  VecCreate(PETSC_COMM_WORLD,&g_m);
  VecSetSizes(g_m,PETSC_DECIDE,Nx*Ny);
  VecSetFromOptions(g_m);
  PetscObjectSetName((PetscObject) g_m, "g_m");
  VecGetOwnershipRange(g_m,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    x = dy*(Ii-Nx*(Ii/Nx));
    y = dy*(Ii/Nx);
    v = 2*x + 3*y;
    VecSetValues(g_m,1,&Ii,&v,INSERT_VALUES);
  }

  Spmat Iy(Ny,Ny); Iy.eye();
  Spmat Ix(Nx,Nx); Ix.eye();
  Spmat Dx(Nx,Nx);
  Dx(0,0,-1.0/dx);Dx(0,1,1.0/dx); // first row
  for (Ii=1;Ii<Nx-1;Ii++) {
    Dx(Ii,Ii-1,-0.5/dx);
    Dx(Ii,Ii+1,0.5/dx);
  }
  Dx(Nx-1,Nx-1,1.0/dx);Dx(Nx-1,Nx-2,-1.0/dx); // last row

  Spmat Dy(Ny,Ny);
  Dy(0,0,-1.0/dy);Dy(0,1,1.0/dy); // first row
  for (Ii=1;Ii<Ny-1;Ii++) {
    Dy(Ii,Ii-1,-0.5/dy);
    Dy(Ii,Ii+1,0.5/dy);
  }
  Dy(Ny-1,Ny-1,1.0/dy);Dy(Ny-1,Ny-2,-1.0/dy); // last row

  Mat Dx_Iy;
  //~kronConvert(Dy,Ix,Dx_Iy,3,3);
  kronConvert(Iy,Dx,Dx_Iy,3,3);
  //~MatView(Dx_Iy,PETSC_VIEWER_STDOUT_WORLD);

  Vec gx_m;
  VecDuplicate(g_m,&gx_m);
startTime = MPI_Wtime();
  MatMult(Dx_Iy,g_m,gx_m);
endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dx matrix: %.9e\n",endTime);
  //~VecView(gx_m,PETSC_VIEWER_STDOUT_WORLD);



  VecDestroy(&l);
  VecDestroy(&g);
  VecDestroy(&lx);
  VecDestroy(&gx);
  VecDestroy(&ly);
  VecDestroy(&gy);
  DMDestroy(&da);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending testMain::testDMDAWithMats in fault.cpp.\n");
#endif
  return ierr;
}


/*
 * Compare computation of d/dy with stencils, a matrix operators on DMDAs, and regular matrices.
 */
int timeDy()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting testMain::timeDy in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt Nx=500,Ny=600;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=6.0;
  PetscScalar dx=(xMax-xMin)/(Nx-1), dy=(yMax-yMin)/(Ny-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  PetscScalar x,y;


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);


  // Set the values for the global vector g
  Vec g=NULL,l=NULL; // g = global, l = local
  PetscScalar **lArr;
  DMCreateGlobalVector(da,&g); PetscObjectSetName((PetscObject) g, "global g");
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      x = i * dx;
      y = j * dy;
      lArr[j][i] = 2*x  + 3*y ;
    }
  }
  DMDAVecRestoreArray(da,l,&lArr);
  ierr = DMLocalToGlobalBegin(da,l,INSERT_VALUES,g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,l,INSERT_VALUES,g);CHKERRQ(ierr);


double startTime = MPI_Wtime();
  // Compute 1st derivative in y-direction, store as gy.
  Vec gy=NULL,ly=NULL; // gx = global, lx = local
  PetscScalar **lyArr;
  VecDuplicate(g,&gy);
  VecSet(gy,0.0);
  PetscObjectSetName((PetscObject) gy, "global gy");
  ierr = DMCreateLocalVector(da,&ly);CHKERRQ(ierr);
  DMDAVecGetArray(da,ly,&lyArr);

  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);

  // only iterate over the local entries in g
  DMDAGetCorners(da,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      if (j>0 && j<Ny-1) { lyArr[j][i] = (lArr[j+1][i] - lArr[j-1][i])/(2*dy); }
      else if (j==0) { lyArr[j][i] = (-1.0*lArr[0][i] + 1.0*lArr[1][i] )/dy; }
      else if (j==Ny-1) { lyArr[j][i] = (- 1.0*lArr[Ny-2][i] + 1.0*lArr[Ny-1][i])/dy; }
    }
  }
  DMDAVecRestoreArray(da,ly,&lyArr);
  DMDAVecRestoreArray(da,ly,&lArr);
  DMLocalToGlobalBegin(da,ly,INSERT_VALUES,gy);
  DMLocalToGlobalEnd(da,ly,INSERT_VALUES,gy);

double endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dy stencil: %.9e\n",endTime);

//~VecView(gy,PETSC_VIEWER_STDOUT_WORLD);



  // try another way of making the matrix, this time with MatStencil
  ISLocalToGlobalMapping map,rmap;
  DMGetLocalToGlobalMapping(da,&map);

  PetscInt Istart,Iend;

  // allocate space for mat
  Mat mat;
  DMDALocalInfo lI;
  DMDAGetLocalInfo(da,&lI);

  MatCreate(PETSC_COMM_WORLD,&mat);
  MatSetSizes(mat,lI.xm*lI.ym,lI.xm*lI.ym,PETSC_DECIDE,PETSC_DECIDE);
  MatSetFromOptions(mat);
  MatMPIAIJSetPreallocation(mat,3,NULL,3,NULL);
  MatSeqAIJSetPreallocation(mat,3,NULL);
  MatSetLocalToGlobalMapping(mat,map,map);
  PetscInt dims[3] = {lI.gxm, lI.gym, lI.gzm};
  PetscInt starts[3] = {lI.gxs, lI.gys, lI.gzs};
  MatSetStencil(mat,2,dims,starts,1);
  MatSetUp(mat);


  MatStencil row;
  // closures
  PetscScalar vIC[2];
  vIC[0] = -1.0/dx; vIC[1] = 1.0/dy;

  PetscScalar vEC[2];
  vEC[0] = -1.0/dx; vEC[1] = 1.0/dy;

  // interior stencil
  MatStencil rowI,col[2];
  PetscScalar vI[2]; vI[0] = -0.5/dy; vI[1] = 0.5/dy;

  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      row.i = i; row.j = j;
      if (j>0 && j<Ny-1) {
        col[0].i = i; col[0].j = j-1;
        col[1].i = i; col[1].j = j+1;
        ierr = MatSetValuesStencil(mat,1,&row,2,col,vI,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (j == 0) {
        col[0].i = i; col[0].j = 0;
        col[1].i = i; col[1].j = 1;
        ierr = MatSetValuesStencil(mat,1,&row,2,col,vIC,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (j == Ny-1) {
        col[0].i = i; col[0].j = Ny-2;
        col[1].i = i; col[1].j = Ny-1;
        ierr = MatSetValuesStencil(mat,1,&row,2,col,vEC,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


  Vec gy_ms;
  VecDuplicate(g,&gy_ms);
  PetscObjectSetName((PetscObject) gy_ms, "global gy_ms");

  VecSet(gy_ms,0.0);
startTime = MPI_Wtime();
  MatMult(mat,g,gy_ms);
endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dy stencil-matrix: %.9e\n",endTime);
//~VecView(gy_ms,PETSC_VIEWER_STDOUT_WORLD);


// traditional matrix derivative
  Vec g_m;
  PetscInt Ii;
  PetscScalar v;
  VecCreate(PETSC_COMM_WORLD,&g_m);
  VecSetSizes(g_m,PETSC_DECIDE,Nx*Ny);
  VecSetFromOptions(g_m);
  PetscObjectSetName((PetscObject) g_m, "g_m");
  VecGetOwnershipRange(g_m,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    x = dy*(Ii-Nx*(Ii/Nx));
    y = dy*(Ii/Nx);
    v = 2*x + 3*y;
    VecSetValues(g_m,1,&Ii,&v,INSERT_VALUES);
  }

  Spmat Iy(Ny,Ny); Iy.eye();
  Spmat Ix(Nx,Nx); Ix.eye();
  Spmat Dx(Nx,Nx);
  Dx(0,0,-1.0/dx);Dx(0,1,1.0/dx); // first row
  for (Ii=1;Ii<Nx-1;Ii++) {
    Dx(Ii,Ii-1,-0.5/dx);
    Dx(Ii,Ii+1,0.5/dx);
  }
  Dx(Nx-1,Nx-1,1.0/dx);Dx(Nx-1,Nx-2,-1.0/dx); // last row

  Spmat Dy(Ny,Ny);
  Dy(0,0,-1.0/dy);Dy(0,1,1.0/dy); // first row
  for (Ii=1;Ii<Ny-1;Ii++) {
    Dy(Ii,Ii-1,-0.5/dy);
    Dy(Ii,Ii+1,0.5/dy);
  }
  Dy(Ny-1,Ny-1,1.0/dy);Dy(Ny-1,Ny-2,-1.0/dy); // last row

  Mat Dy_Ix;
  kronConvert(Dy,Ix,Dy_Ix,3,3);
  //~MatView(Dx_Iy,PETSC_VIEWER_STDOUT_WORLD);

  Vec gy_m;
  VecDuplicate(g_m,&gy_m);
startTime = MPI_Wtime();
  MatMult(Dy_Ix,g_m,gy_m);
endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dy matrix: %.9e\n",endTime);
//~VecView(gy_m,PETSC_VIEWER_STDOUT_WORLD);



  VecDestroy(&l);
  VecDestroy(&g);
  VecDestroy(&ly);
  VecDestroy(&gy);
  DMDestroy(&da);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending testMain::timeDy in fault.cpp.\n");
#endif
  return ierr;
}





/*
 * Test how to create matrix to perform derivatives on a DMDA
 */
int testDMDAWithMats()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting testMain::testDMDAWithMats in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt Nx=500,Ny=600;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=6.0;
  PetscScalar dx=(xMax-xMin)/(Nx-1), dy=(yMax-yMin)/(Ny-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below
  PetscScalar x,y;


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);

  // Set up uniform coordinate mesh.
  // Print to see distribution, organization on multiple processors.
  DM cda;
  Vec lcoords; // local vector containing coordinates
  DMDACoor2d **coords; // 2D array containing x and y data members
  DMDASetUniformCoordinates(da,yMin,yMax,xMin,xMax,0.0,1.0);
  DMGetCoordinateDM(da,&cda);
  DMGetCoordinatesLocal(da,&lcoords);
  DMDAVecGetArray(cda,lcoords,&coords);
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  //~for (j=nStart;j<nStart+n;j++) {
    //~for (i=mStart;i<mStart+m;i++) {
      //~PetscPrintf(PETSC_COMM_SELF,"%i: (coords[%i][%i].x,coords[%i][%i].y) = (%g,%g)\n",
        //~rank,j,i,j,i,coords[j][i].x,coords[j][i].y);
    //~}
  //~}

  // Set the values for the global vector x based on the (x,y)
  // coordinates for each vertex. Since this uses only local values, there
  // is no need to use the pair DMGlobalToLocalBegin/End to communicate
  // the ghost values.
  Vec g=NULL,l=NULL; // g = global x, l = local x
  PetscScalar **lArr;
  DMCreateGlobalVector(da,&g); PetscObjectSetName((PetscObject) g, "global g");
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      x = i * dx;
      y = j * dy;
      //~lArr[j][i] = (coords[j][i].x * coords[j][i].x) + (coords[j][i].y * coords[j][i].y);
      lArr[j][i] = 2*x  + 3*y ;
    }
  }
  DMDAVecRestoreArray(da,l,&lArr);
  ierr = DMLocalToGlobalBegin(da,l,INSERT_VALUES,g);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,l,INSERT_VALUES,g);CHKERRQ(ierr);
  //~VecView(g,PETSC_VIEWER_STDOUT_WORLD);
  //~printf_DM_2d(g,da);

double startTime = MPI_Wtime();
  // Compute 1st derivative in x-direction, store as gx.
  // This uses the ghost points in x, so it is necessary to use
  // DMGlobalToLocalBegin/End for g -> l.
  Vec gx=NULL,lx=NULL; // gx = global, lx = local
  PetscScalar **lxArr;
  VecDuplicate(g,&gx);
  VecSet(gx,0.0);
  PetscObjectSetName((PetscObject) gx, "global gx");
  ierr = DMCreateLocalVector(da,&lx);CHKERRQ(ierr);
  DMDAVecGetArray(da,lx,&lxArr);

  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);

  // only iterate over the local entries in f
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      if (i>0 && i<Nx-1) { lxArr[j][i] = (lArr[j][i+1] - lArr[j][i-1])/(2*dx); }
      else if (i==0) { lxArr[j][i] = (-1.0*lArr[j][0] + 1.0*lArr[j][1] )/dx; }
      else if (i==Nx-1) { lxArr[j][i] = (- 1.0*lArr[j][Nx-2] + 1.0*lArr[j][Nx-1])/dx; }
    }
  }
  DMDAVecRestoreArray(da,lx,&lxArr);
  DMDAVecRestoreArray(da,lx,&lArr);
  DMLocalToGlobalBegin(da,lx,INSERT_VALUES,gx);
  DMLocalToGlobalEnd(da,lx,INSERT_VALUES,gx);
  //~VecView(gx,PETSC_VIEWER_STDOUT_WORLD);
double endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dx stencil: %.9e\n",endTime);



  // Compute 1st derivative in y-direction, store as gy.
  // This uses the ghost points in x, so it is necessary to use
  // DMGlobalToLocalBegin/End for g -> l.
  Vec gy=NULL,ly=NULL; // gx = global, lx = local
  PetscScalar **lyArr;
  VecDuplicate(g,&gy);
  VecSet(gy,0.0);
  PetscObjectSetName((PetscObject) gy, "global gy");
  ierr = DMCreateLocalVector(da,&ly);CHKERRQ(ierr);
  DMDAVecGetArray(da,ly,&lyArr);

  ierr = DMGlobalToLocalBegin(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  DMDAVecGetArray(da,l,&lArr);

  // only iterate over the local entries in f
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      if (j>0 && j<Ny-1) { lyArr[j][i] = (lArr[j+1][i] - lArr[j-1][i])/(2*dy); }
      else if (j==0) { lyArr[j][i] = (-1.0*lArr[0][i] + 1.0*lArr[1][i] )/dy; }
      else if (j==Ny-1) { lyArr[j][i] = (- 1.0*lArr[Ny-2][i] + 1.0*lArr[Ny-1][i])/dy; }
    }
  }
  DMDAVecRestoreArray(da,ly,&lyArr);
  DMDAVecRestoreArray(da,ly,&lArr);
  DMLocalToGlobalBegin(da,ly,INSERT_VALUES,gy);
  DMLocalToGlobalEnd(da,ly,INSERT_VALUES,gy);
  //~VecView(gy,PETSC_VIEWER_STDOUT_WORLD);



  // output the global vectors so they can be loaded into MATLAB
  //~writeVec(g,"g");
  //~writeVec(gx,"gx");
  //~writeVec(gy,"gy");




  // try another way of making the matrix, this time with MatStencil
  ISLocalToGlobalMapping map,rmap;
  DMGetLocalToGlobalMapping(da,&map);
  AO ao;
  DMDAGetAO(da,&ao);

  PetscInt Istart,Iend;

  // allocate space for mat
  Mat mat;
  DMDALocalInfo lI;
  DMDAGetLocalInfo(da,&lI);

  //~Mat temp;
  //~DMSetMatType(da,MATAIJ);
  //~DMCreateMatrix(da,MATAIJ,&temp);
  //~PetscInt locM,locN;
  //~MatGetLocalSize(temp,&locM,&locN);
  //~PetscPrintf(PETSC_COMM_SELF,"[%i]: locM = %i, locN = %i | gxm = %i, gym = %i xm=%i, ym=%i\n",
    //~rank, locM,locN,lI.gxm,lI.gym,lI.xm,lI.ym);

  MatCreate(PETSC_COMM_WORLD,&mat);
  //~MatSetSizes(mat,locM,locN,PETSC_DECIDE,PETSC_DECIDE);
  MatSetSizes(mat,lI.xm*lI.ym,lI.xm*lI.ym,PETSC_DECIDE,PETSC_DECIDE);
  MatSetFromOptions(mat);
  MatMPIAIJSetPreallocation(mat,3,NULL,3,NULL);
  MatSeqAIJSetPreallocation(mat,3,NULL);
  MatSetLocalToGlobalMapping(mat,map,map);
  PetscInt dims[3] = {lI.gxm, lI.gym, lI.gzm};
  PetscInt starts[3] = {lI.gxs, lI.gys, lI.gzs};
  MatSetStencil(mat,2,dims,starts,1);
  MatSetUp(mat);


  MatStencil row;
  // closures
  MatStencil colIC[2];
  colIC[0].i = 0; colIC[0].j = 0;
  colIC[1].i = 1; colIC[1].j = 0;
  PetscScalar vIC[2];
  vIC[0] = -1.0/dx; vIC[1] = 1.0/dx;

  MatStencil colEC[2];
  colEC[0].i = Nx-2; colEC[0].j = Nx-1;
  colEC[1].i = Nx-1; colEC[1].j = Nx-1;
  PetscScalar vEC[2];
  vEC[0] = -1.0/dx; vEC[1] = 1.0/dx;

  // interior stencil
  MatStencil rowI,colI[2];
  PetscScalar vI[2]; vI[0] = -0.5/dx; vI[1] = 0.5/dx;
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      row.i = i; row.j = j;
      if (i>0 && i<Nx-1) {
        colI[0].i = i-1; colI[0].j = j;
        colI[1].i = i+1; colI[1].j = j;
        ierr = MatSetValuesStencil(mat,1,&row,2,colI,vI,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (i == 0) {
        colIC[0].i = 0; colIC[0].j = j;
        colIC[1].i = 1; colIC[1].j = j;
        ierr = MatSetValuesStencil(mat,1,&row,2,colIC,vIC,INSERT_VALUES); CHKERRQ(ierr);
      }
      else if (i == Nx-1) {
        colIC[0].i = Nx-2; colIC[0].j = j;
        colIC[1].i = Nx-1; colIC[1].j = j;

        ierr = MatSetValuesStencil(mat,1,&row,2,colIC,vEC,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  //~MatView(mat,PETSC_VIEWER_STDOUT_WORLD);


  Vec gx_ms;
  VecDuplicate(g,&gx_ms);
  PetscObjectSetName((PetscObject) gx_ms, "global gx_ms");

  VecSet(gx_ms,0.0);
startTime = MPI_Wtime();
  MatMult(mat,g,gx_ms);
endTime = MPI_Wtime() - startTime;
//~VecView(gx_ms,PETSC_VIEWER_STDOUT_WORLD);
PetscPrintf(PETSC_COMM_WORLD,"Dx stencil-matrix: %.9e\n",endTime);


// traditional matrix derivative
  Vec g_m;
  PetscInt Ii;
  PetscScalar v;
  VecCreate(PETSC_COMM_WORLD,&g_m);
  VecSetSizes(g_m,PETSC_DECIDE,Nx*Ny);
  VecSetFromOptions(g_m);
  PetscObjectSetName((PetscObject) g_m, "g_m");
  VecGetOwnershipRange(g_m,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    x = dy*(Ii-Nx*(Ii/Nx));
    y = dy*(Ii/Nx);
    v = 2*x + 3*y;
    VecSetValues(g_m,1,&Ii,&v,INSERT_VALUES);
  }
  //~VecView(g_m,PETSC_VIEWER_STDOUT_WORLD);

  Spmat Iy(Ny,Ny); Iy.eye();
  Spmat Ix(Nx,Nx); Ix.eye();
  Spmat Dx(Nx,Nx);
  Dx(0,0,-1.0/dx);Dx(0,1,1.0/dx); // first row
  for (Ii=1;Ii<Nx-1;Ii++) {
    Dx(Ii,Ii-1,-0.5/dx);
    Dx(Ii,Ii+1,0.5/dx);
  }
  Dx(Nx-1,Nx-1,1.0/dx);Dx(Nx-1,Nx-2,-1.0/dx); // last row

  Spmat Dy(Ny,Ny);
  Dy(0,0,-1.0/dy);Dy(0,1,1.0/dy); // first row
  for (Ii=1;Ii<Ny-1;Ii++) {
    Dy(Ii,Ii-1,-0.5/dy);
    Dy(Ii,Ii+1,0.5/dy);
  }
  Dy(Ny-1,Ny-1,1.0/dy);Dy(Ny-1,Ny-2,-1.0/dy); // last row

  Mat Dx_Iy;
  //~kronConvert(Dy,Ix,Dx_Iy,3,3);
  kronConvert(Iy,Dx,Dx_Iy,3,3);
  //~MatView(Dx_Iy,PETSC_VIEWER_STDOUT_WORLD);

  Vec gx_m;
  VecDuplicate(g_m,&gx_m);
startTime = MPI_Wtime();
  MatMult(Dx_Iy,g_m,gx_m);
endTime = MPI_Wtime() - startTime;
PetscPrintf(PETSC_COMM_WORLD,"Dx matrix: %.9e\n",endTime);
  //~VecView(gx_m,PETSC_VIEWER_STDOUT_WORLD);



  VecDestroy(&l);
  VecDestroy(&g);
  VecDestroy(&lx);
  VecDestroy(&gx);
  VecDestroy(&ly);
  VecDestroy(&gy);
  DMDestroy(&da);
  DMDestroy(&cda);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending testMain::testDMDAWithMats in fault.cpp.\n");
#endif
  return ierr;
}



// Test use of VecAXPY etc functions with global vectors from DMDA objects.
int testDMDA_ScatterToVec()
{
    PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::testDMDAMemory in fault.cpp.\n");
#endif

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt Nx=6,Ny=7;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=6.0;
  PetscScalar dx=(xMax-xMin)/(Nx-1), dy=(yMax-yMin)/(Ny-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below


  // create the 2D distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);


  DM cda;
  Vec lcoords; // local vector containing coordinates
  DMDACoor2d **coords; // 2D array containing x and y data members
  DMDASetUniformCoordinates(da,xMin,xMax,yMin,yMax,0.0,1.0);
  DMGetCoordinateDM(da,&cda);
  DMGetCoordinatesLocal(da,&lcoords);
  DMDAVecGetArray(cda,lcoords,&coords);
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);


  Vec gx=NULL; // gx = global x
  DMCreateGlobalVector(da,&gx);  PetscObjectSetName((PetscObject) gx, "global x");


  Vec lx=NULL; // lx = local x
  PetscScalar **lxArr;
  ierr = DMCreateLocalVector(da,&lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  DMDAVecGetArray(da,lx,&lxArr);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      lxArr[j][i] = 10*coords[j][i].y + coords[j][i].x;
    }
  }
  DMDAVecRestoreArray(da,lx,&lxArr);
  ierr = DMLocalToGlobalBegin(da,lx,INSERT_VALUES,gx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,lx,INSERT_VALUES,gx);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"gx:\n");CHKERRQ(ierr);
  printf_DM_2d(gx,da);


  // create vectors that represent only values on the fault
  Vec gy;
  VecCreate(PETSC_COMM_WORLD,&gy);
  VecSetSizes(gy,PETSC_DECIDE,Nx);
  VecSetFromOptions(gy);     PetscObjectSetName((PetscObject) gy, "gy");
  VecSet(gy,0.0);

  VecScatter scatter; // scatter context
  IS from,to; // index sets that define the scatter


  int idx_from[] = {0,6,12}, idx_to[]={0,1,2};
  AO ao;
  DMDAGetAO(da,&ao);
  //~AOApplicationToPetsc(ao,3,idx_from);


  ISCreateGeneral(PETSC_COMM_SELF,3,idx_from,PETSC_COPY_VALUES,&from);
  ISCreateGeneral(PETSC_COMM_SELF,3,idx_to,PETSC_COPY_VALUES,&to);
  VecScatterCreate(gx,from,gy,to,&scatter); // gx = source vector, gy = destination vector


  VecScatterBegin(scatter,gx,gy,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scatter,gx,gy,INSERT_VALUES,SCATTER_FORWARD);

  VecView(gy,PETSC_VIEWER_STDOUT_WORLD);

  ISDestroy(&from);
  ISDestroy(&to);
  VecScatterDestroy(&scatter);



  VecDestroy(&lx);
  VecDestroy(&gx);
  VecDestroy(&gy);
  DMDestroy(&da);
  DMDestroy(&cda);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::testDMDAMemory in fault.cpp.\n");
#endif
  return ierr;
}


int testDMDA_memory()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::testDMDA in fault.cpp.\n");
#endif
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt M=6,N=6;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=5.0;
  PetscScalar dx=(xMax-xMin)/(M-1), dy=(yMax-yMin)/(N-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);


  Vec gx=NULL,lx=NULL; // gx = global x, lx = local x
  PetscScalar **lxArr;
  DMCreateGlobalVector(da,&gx); PetscObjectSetName((PetscObject) gx, "global x");
  VecSet(gx,1.0);
  ierr = DMCreateLocalVector(da,&lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);



  VecDestroy(&lx);
  VecDestroy(&gx);
  DMDestroy(&da);



#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::testDMDA in fault.cpp.\n");
#endif
  return ierr;
}


int testDMDA_changeCoords()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::testDMDA in fault.cpp.\n");
#endif
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt M=11,N=11;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=5.0;
  PetscScalar dx=(xMax-xMin)/(M-1), dy=(yMax-yMin)/(N-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);

  // Set up uniform coordinate mesh.
  // Print to see distribution, organization on multiple processors.
  DM cda;
  Vec lcoords; // local vector containing coordinates
  DMDACoor2d **coords; // 2D array containing x and y data members
  DMDASetUniformCoordinates(da,xMin,xMax,yMin,yMax,0.0,1.0);
  DMGetCoordinateDM(da,&cda);
  DMGetCoordinatesLocal(da,&lcoords);
  DMDAVecGetArray(cda,lcoords,&coords);
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      PetscPrintf(PETSC_COMM_SELF,"%i: (coords[%i][%i].x,coords[%i][%i].y) = (%g,%g)\n",
        rank,j,i,j,i,coords[j][i].x,coords[j][i].y);
    }
  }


  DMDestroy(&da);
  DMDestroy(&cda);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::testDMDA in fault.cpp.\n");
#endif
  return ierr;
}

/* Demonstrates the use of PETSc's distributed memory distributed array
 * objects (DMDAs). This function demonstrates how to initialize values
 * in a global vector using built-in functions for creating a uniform
 * coordinate mesh, then demonstrated the use of the functions for
 * natural indexing and memory communication to compute the first
 * derivative of a 2D array.
 */
int testDMDA()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting main::testDMDA in fault.cpp.\n");
#endif
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // initialize size and range of grid
  PetscInt M=6,N=6;
  PetscScalar xMin=0.0,xMax=5.0,yMin=0.0,yMax=5.0;
  PetscScalar dx=(xMax-xMin)/(M-1), dy=(yMax-yMin)/(N-1); // grid spacing
  PetscInt i,j,mStart,m,nStart,n; // for for loops below


  // create the distributed array
  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
    DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL, &da); CHKERRQ(ierr);

  // Set up uniform coordinate mesh.
  // Print to see distribution, organization on multiple processors.
  DM cda;
  Vec lcoords; // local vector containing coordinates
  DMDACoor2d **coords; // 2D array containing x and y data members
  DMDASetUniformCoordinates(da,xMin,xMax,yMin,yMax,0.0,1.0);
  DMGetCoordinateDM(da,&cda);
  DMGetCoordinatesLocal(da,&lcoords);
  DMDAVecGetArray(cda,lcoords,&coords);
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      PetscPrintf(PETSC_COMM_SELF,"%i: (coords[%i][%i].x,coords[%i][%i].y) = (%g,%g)\n",
        rank,j,i,coords[j][i].x,coords[j][i].y);
    }
  }

  // Set the values for the global vector x based on the (x,y)
  // coordinates for each vertex. Since this uses only local values, there
  // is no need to use the pair DMGlobalToLocalBegin/End to communicate
  // the ghost values.
  Vec gx=NULL,lx=NULL; // gx = global x, lx = local x
  PetscScalar **lxArr;
  DMCreateGlobalVector(da,&gx); PetscObjectSetName((PetscObject) gx, "global x");
  ierr = DMCreateLocalVector(da,&lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  DMDAVecGetArray(da,lx,&lxArr);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      lxArr[j][i] = (coords[j][i].x * coords[j][i].x) + (coords[j][i].y * coords[j][i].y);
    }
  }
  DMDAVecRestoreArray(da,lx,&lxArr);
  ierr = DMLocalToGlobalBegin(da,lx,INSERT_VALUES,gx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,lx,INSERT_VALUES,gx);CHKERRQ(ierr);
  //~VecView(gx,PETSC_VIEWER_STDOUT_WORLD);

  // Careful: this way of initializing values does NOT work!!!
  //~PetscInt Ii,Istart,Iend;
  //~PetscScalar v;
  //~VecGetOwnershipRange(gx,&Istart,&Iend);
  //~for (Ii=Istart;Ii<Iend;Ii++) {
    //~v = (PetscScalar) Ii;
    //~VecSetValues(gx,1,&Ii,&v,INSERT_VALUES);
  //~}
  //~VecAssemblyBegin(gx);
  //~VecAssemblyEnd(gx);


  // Compute 1st derivative in x-direction, store as f.
  // This uses the ghost points in x, so it is necessary to use
  // DMGlobalToLocalBegin/End for gx -> lx.
  Vec gf=NULL,lf=NULL; // gf = global f, lf = local f
  PetscScalar **lfArr;
  VecDuplicate(gx,&gf);
  VecSet(gf,0.0);
  PetscObjectSetName((PetscObject) gf, "global f");
  ierr = DMCreateLocalVector(da,&lf);CHKERRQ(ierr);
  DMDAVecGetArray(da,lf,&lfArr);

  ierr = DMGlobalToLocalBegin(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,gx,INSERT_VALUES,lx);CHKERRQ(ierr);
  DMDAVecGetArray(da,lx,&lxArr);

  // only iterate over the local entries in f
  DMDAGetCorners(cda,&mStart,&nStart,0,&m,&n,0);
  for (j=nStart;j<nStart+n;j++) {
    for (i=mStart;i<mStart+m;i++) {
      //~PetscPrintf(PETSC_COMM_SELF,"%i: lxArr[%i][%i]  = %g \n",rank,j,i,lxArr[j][i]);
      //~if (i>0 && i<M-1) { lfArr[j][i] = (lxArr[j][i+1] - lxArr[j][i-1])/(2*dx); }
      //~else if (i==0) { lfArr[j][i] = (-1.5*lxArr[j][0] + 2.0*lxArr[j][1] - 0.5*lxArr[j][2])/dx; }
      //~else if (i==M-1) { lfArr[j][i] = (0.5*lxArr[j][M-3] - 2.0*lxArr[j][M-2] + 1.5*lxArr[j][M-1])/dx; }

      if (j>0 && j<N-1) { lfArr[j][i] = (lxArr[j+1][i] - lxArr[j-1][i])/(2*dy); }
      else if (j==0) { lfArr[j][i] = (-1.5*lxArr[0][i] + 2.0*lxArr[1][i] - 0.5*lxArr[2][i])/dy; }
      else if (j==N-1) { lfArr[j][i] = (0.5*lxArr[N-3][i] - 2.0*lxArr[N-2][i] + 1.5*lxArr[N-1][i])/dy; }
    }
  }
  DMDAVecRestoreArray(da,lf,&lfArr);
  DMDAVecRestoreArray(da,lf,&lxArr);
  DMLocalToGlobalBegin(da,lf,INSERT_VALUES,gf);
  DMLocalToGlobalEnd(da,lf,INSERT_VALUES,gf);
  VecView(gf,PETSC_VIEWER_STDOUT_WORLD);

  // output the global vectors so they can be loaded into MATLAB
  writeVec(gx,"gx");
  writeVec(gf,"gf");

  VecDestroy(&lx);
  VecDestroy(&gx);
  VecDestroy(&lf);
  VecDestroy(&gf);
  DMDestroy(&da);
  DMDestroy(&cda);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending main::testDMDA in fault.cpp.\n");
#endif
  return ierr;
}


// test MatCreateShell stuff
// NOTE: appears to work as expected right out of the box. Doesn't
// handle memory management for nonlocal memory for me :(
PetscErrorCode mult(Mat A, Vec x, Vec f)
{
  PetscErrorCode ierr = 0;

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  VecSet(f,1.0);

  return ierr;
}

PetscErrorCode testMatShell()
{
  PetscErrorCode ierr = 0;
  PetscInt   M=6,N=4;
  Vec x=NULL,f=NULL;

  // declare and initialize x and f (f = Dx(x) )
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,M*N);
  VecSetFromOptions(x); PetscObjectSetName((PetscObject) x, "x");
  PetscInt Ii,Istart,Iend;
  PetscScalar v=0;
  ierr = VecGetOwnershipRange(x,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = (double) Ii;
    ierr = VecSetValues(x,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  VecDuplicate(x,&f); PetscObjectSetName((PetscObject) f, "f");
  VecSet(f,0.0);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  // play around with MatCreateShell stuff
  PetscInt m,n;
  VecGetLocalSize(x,&m);
  Mat A;
  ierr = MatCreateShell(PETSC_COMM_WORLD,m,m,M*N,M*N,NULL,&A);
  MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);

  MatMult(A,x,f);

  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  return ierr;
}



/* Perform MMS in space:
 *  In order to test only the interior stencils and operators
 * use y,z = [0,2*pi]. This will make all boundary vectors 0, and so will
 * not test operators which map BC vectors to rhs and A.
 *
 * To test that BC vectors are being mapped correctly, use y,z =[L0,L]
 * where neither L0 nor L are multiples of pi.
 */
/*int mmsSpace(const char* inputFile,PetscInt Ny,PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii = 0;
  PetscScalar    y,z,v=0;

  Domain domain(inputFile,Ny,Nz);

  // set vectors containing analytical distribution for displacement and source
  Vec uAnal,source;
  ierr = VecCreate(PETSC_COMM_WORLD,&uAnal);CHKERRQ(ierr);
  ierr = VecSetSizes(uAnal,PETSC_DECIDE,Ny*Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(uAnal);CHKERRQ(ierr);
  ierr = VecDuplicate(uAnal,&source);CHKERRQ(ierr);

  PetscInt *inds;
  ierr = PetscMalloc(Ny*Nz*sizeof(PetscInt),&inds);CHKERRQ(ierr);

  PetscScalar *uAnalArr,*sourceArr;
  ierr = PetscMalloc(Ny*Nz*sizeof(PetscScalar),&uAnalArr);CHKERRQ(ierr);
  ierr = PetscMalloc(Ny*Nz*sizeof(PetscScalar),&sourceArr);CHKERRQ(ierr);

  // boundary conditions
  Vec bcF,bcR,bcS,bcD,rhs;
  VecCreate(PETSC_COMM_WORLD,&bcF);
  VecSetSizes(bcF,PETSC_DECIDE,Nz);
  VecSetFromOptions(bcF);
  VecDuplicate(bcF,&bcR);
  VecCreate(PETSC_COMM_WORLD,&bcS);
  VecSetSizes(bcS,PETSC_DECIDE,Ny);
  VecSetFromOptions(bcS);
  VecDuplicate(bcS,&bcD);

  // set values for boundaries, source, and analytic solution
  PetscInt indx = 0;
  for (Ii=0;Ii<Ny*Nz;Ii++)
  {
    y = domain._dy*(Ii/Nz);
    z = domain._dz*(Ii-Nz*(Ii/Nz));
    inds[Ii] = Ii;

    uAnalArr[Ii] = sin(y)*cos(z);
    sourceArr[Ii] = cos(y+z)*(-cos(y)*cos(z) + sin(y)*sin(z)) + 2*(sin(y+z)+2)*cos(z)*sin(y);

    // BCs
    if (y==0) {
      v = sin(y)*cos(z);
      ierr = VecSetValue(bcF,Ii,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (y==domain._Ly) {
      indx = z/domain._dz;
      v = sin(y)*cos(z);
      ierr = VecSetValue(bcR,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (z==0) {
      indx = (int) (y/domain._dy);
      v = -domain._muArrPlus[Ii]*sin(y)*sin(z);
      ierr = VecSetValue(bcS,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (z==domain._Lz) {
      indx = (int) (y/domain._dy);
      v = -domain._muArrPlus[Ii]*sin(y)*sin(z);
      ierr = VecSetValue(bcD,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecSetValues(uAnal,Ny*Nz,inds,uAnalArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uAnal);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uAnal);CHKERRQ(ierr);

  ierr = VecSetValues(source,Ny*Nz,inds,sourceArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(source);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(source);CHKERRQ(ierr);

  VecAssemblyBegin(bcF); VecAssemblyEnd(bcF);
  VecAssemblyBegin(bcR); VecAssemblyEnd(bcR);
  VecAssemblyBegin(bcS); VecAssemblyEnd(bcS);
  VecAssemblyBegin(bcD); VecAssemblyEnd(bcD);



  // set up linear system
  SbpOps_c sbp(domain,*domain._muArrPlus,domain._muP);
  TempMats tempFactors(domain._order,domain._Ny,domain._dy,domain._Nz,domain._dz,&domain._muP);

  VecCreate(PETSC_COMM_WORLD,&rhs);
  VecSetSizes(rhs,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(rhs);
  VecSet(rhs,0.0);
  ierr = sbp.setRhs(rhs,bcF,bcR,bcS,bcD);CHKERRQ(ierr);

  // without multiplying rhs by source
  //~ierr = VecAXPY(rhs,-1.0,source);CHKERRQ(ierr); // rhs = rhs - source

  // with multiplying rhs by source
  Vec temp;
  ierr = VecDuplicate(rhs,&temp);CHKERRQ(ierr);
  ierr = MatMult(tempFactors._H,source,temp);CHKERRQ(ierr);
  ierr = VecAXPY(rhs,-1.0,temp);CHKERRQ(ierr); // rhs = rhs - source



  KSP ksp;
  PC  pc;
  KSPCreate(PETSC_COMM_WORLD,&ksp);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
  PCFactorSetUpMatSolverPackage(pc);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);


  Vec uhat;
  ierr = VecDuplicate(rhs,&uhat);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,uhat);CHKERRQ(ierr);


  // output vectors for visualization with matlab
  ierr = domain.write();
  ierr = writeVec(uhat,"data/uhat");CHKERRQ(ierr);
  ierr = writeVec(uAnal,"data/uAnal");CHKERRQ(ierr);
  ierr = writeVec(source,"data/source");CHKERRQ(ierr);
  ierr = writeVec(rhs,"data/rhs");CHKERRQ(ierr);

  ierr = writeVec(bcF,"data/bcF");CHKERRQ(ierr);
  ierr = writeVec(bcR,"data/bcR");CHKERRQ(ierr);
  ierr = writeVec(bcS,"data/bcS");CHKERRQ(ierr);
  ierr = writeVec(bcD,"data/bcD");CHKERRQ(ierr);
 sbp.writeOps("data/");


  // MMS for shear stress on fault
  Vec tauHat, tauAnal, sigma_xy;
  ierr = VecDuplicate(rhs,&sigma_xy);CHKERRQ(ierr);
  ierr = MatMult(sbp._muxDy_Iz,uAnal,sigma_xy);CHKERRQ(ierr);

  ierr = VecDuplicate(bcF,&tauHat);CHKERRQ(ierr);
  ierr = VecDuplicate(bcF,&tauAnal);CHKERRQ(ierr);
  PetscInt Istart,Iend;
  v = 0.0;
  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<Nz) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(tauHat,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      z = domain._dz*(Ii-Nz*(Ii/Nz));
      y = domain._dy*(Ii/Nz);
      v = domain._muArrPlus[Ii]*cos(z)*cos(y);
      ierr = VecSetValues(tauAnal,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(tauHat);CHKERRQ(ierr); ierr = VecAssemblyBegin(tauAnal);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(tauHat);CHKERRQ(ierr); ierr = VecAssemblyEnd(tauAnal);CHKERRQ(ierr);
  //~ierr = VecView(tauHat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //~ierr = VecView(tauAnal,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  // measure error in L2 norm
  PetscScalar errU,errTau;
  ierr = VecAXPY(uAnal,-1.0,uhat);CHKERRQ(ierr); //overwrites 1st arg with sum
  ierr = VecNorm(uAnal,NORM_2,&errU);
  errU = errU/sqrt( (double) Ny*Nz );

  ierr = VecAXPY(tauAnal,-1.0,tauHat);CHKERRQ(ierr); //overwrites 1st arg with sum
  ierr = VecNorm(tauAnal,NORM_2,&errTau);
  errTau = errTau/sqrt( (double) Nz );


  ierr = PetscPrintf(PETSC_COMM_WORLD,"%5i %5i %5i %20.12e %20.12e\n",
                     domain._order,domain._Ny,domain._Nz,log2(errU),log2(errTau));CHKERRQ(ierr);


  // clean up
  VecDestroy(&uAnal);
  VecDestroy(&source);
  PetscFree(inds);
  PetscFree(uAnalArr);
  PetscFree(sourceArr);
  VecDestroy(&bcF);
  VecDestroy(&bcR);
  VecDestroy(&bcS);
  VecDestroy(&bcD);

  return ierr;
}*/



/*
 * Determine the critical grid spacing by imposing slip on the fault
 * similar to just prior to an eq and then measuring how shear stress
 * on the fault changes as a function of grid spacing.
 */
int critSpacing(const char * inputFile,PetscInt Ny, PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii = 0;
  PetscScalar    y,z,v=0;
  PetscScalar _seisDepth = 12.0;

  Domain domain(inputFile,Ny,Nz);

  // boundary conditions
  Vec bcF,bcR,bcS,bcD,rhs;
  VecCreate(PETSC_COMM_WORLD,&bcF);
  VecSetSizes(bcF,PETSC_DECIDE,Nz);
  VecSetFromOptions(bcF);
  VecDuplicate(bcF,&bcR);
  VecCreate(PETSC_COMM_WORLD,&bcS);
  VecSetSizes(bcS,PETSC_DECIDE,Ny);
  VecSetFromOptions(bcS);
  VecDuplicate(bcS,&bcD);

  // set values for boundaries, source, and analytic solution
  PetscInt indx = 0;
  for (Ii=0;Ii<Ny*Nz;Ii++)
  {
    y = domain._dy*(Ii/Nz);
    z = domain._dz*(Ii-Nz*(Ii/Nz));

    // BCs
    //~if (y==0) {
    if (Ii < domain._Nz ) {
      //~v = atan((z-domain._seisDepth)/2.0) - atan(-domain._seisDepth/2.0);
      v = atan((z-_seisDepth)/0.5) - atan(-_seisDepth/2.0);
      //~v = 0.0;
      ierr = VecSetValue(bcF,Ii,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    //~if (y==domain._Ly) {
    if (Ii >= domain._Ny*domain._Nz - domain._Nz) {
      indx = z/domain._dz;
      v = 5;
      ierr = VecSetValue(bcR,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (z==0) {
      indx = (int) (y/domain._dy);
      v = 0.0;
      ierr = VecSetValue(bcS,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (z==domain._Lz) {
      indx = (int) (y/domain._dy);
      v = 0.0;
      ierr = VecSetValue(bcD,indx,v,INSERT_VALUES);CHKERRQ(ierr);
    }

  }
  VecSet(bcD,0.0);
  VecSet(bcS,0.0);
  VecSet(bcR,50.0);
  VecAssemblyBegin(bcF); VecAssemblyEnd(bcF);
  //~VecAssemblyBegin(bcR); VecAssemblyEnd(bcR);
  //~VecAssemblyBegin(bcS); VecAssemblyEnd(bcS);
  //~VecAssemblyBegin(bcD); VecAssemblyEnd(bcD);



  // set up linear system
  SbpOps sbp(domain,*domain._muArrPlus,domain._muP);

  VecCreate(PETSC_COMM_WORLD,&rhs);
  VecSetSizes(rhs,PETSC_DECIDE,Ny*Nz);
  VecSetFromOptions(rhs);
  VecSet(rhs,0.0);
  ierr = sbp.setRhs(rhs,bcF,bcR,bcS,bcD);CHKERRQ(ierr);

  KSP ksp;
  PC  pc;
  KSPCreate(PETSC_COMM_WORLD,&ksp);
  //~ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  //~ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  //~ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  //~ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  //~PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
  //~PCFactorSetUpMatSolverPackage(pc);


  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
  ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,domain._kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);

  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);



  Vec uhat;
  ierr = VecDuplicate(rhs,&uhat);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,uhat);CHKERRQ(ierr);


  // output vectors for visualization with matlab
  ierr = domain.write();
  ierr = writeVec(bcF,"critGridSpacing/bcF");CHKERRQ(ierr);
  ierr = writeVec(bcR,"critGridSpacing/bcR");CHKERRQ(ierr);
  ierr = writeVec(bcS,"critGridSpacing/bcS");CHKERRQ(ierr);
  ierr = writeVec(bcD,"critGridSpacing/bcD");CHKERRQ(ierr);
  ierr = writeVec(uhat,"critGridSpacing/uhat");CHKERRQ(ierr);


  // MMS for shear stress on fault
  Vec tau, sigma_xy;
  ierr = VecDuplicate(rhs,&sigma_xy);CHKERRQ(ierr);
  //~ierr = MatMult(sbp._muxDy_Iz,uhat,sigma_xy);CHKERRQ(ierr);
  ierr = sbp.muxDy(uhat,sigma_xy);CHKERRQ(ierr);

  ierr = VecDuplicate(bcF,&tau);CHKERRQ(ierr);
  PetscInt Istart,Iend;
  v = 0.0;
  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<Nz) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(tau,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(tau);CHKERRQ(ierr);


  stringstream ss;
  ss << "critGridSpacing/tau_"<< domain._shearDistribution << "_order"
     << domain._order << "_Ny" << Ny << "_Nz" << Nz;
  std::string _debugFolder = ss.str();
  ierr = writeVec(tau,_debugFolder.c_str());CHKERRQ(ierr);

  // clean up
  VecDestroy(&bcF);
  VecDestroy(&bcR);
  VecDestroy(&bcS);
  VecDestroy(&bcD);
  VecDestroy(&tau);
  VecDestroy(&sigma_xy);
//~*/
  return ierr;
}

/* This has not been rewritten since the post-quals refactoring
int screwDislocation(PetscInt Ny,PetscInt Nz)
{
  PetscErrorCode ierr = 0;
  PetscInt       order=4;
  PetscBool      loadMat = PETSC_FALSE;
  PetscViewer    viewer;
  PetscScalar    u,z;


  // set up the problem context
  IntegratorContext D(order,Ny,Nz,"data/");
  ierr = setParameters(D);CHKERRQ(ierr);
  ierr = D.writeParameters();CHKERRQ(ierr);
  ierr = setRateAndState(D);CHKERRQ(ierr);
  ierr = D.writeRateAndState();CHKERRQ(ierr);
  ierr = setLinearSystem(D,loadMat);CHKERRQ(ierr);
  ierr = D.writeOperators();CHKERRQ(ierr);

  // set boundary conditions
  ierr = VecSet(D.gS,0.0);CHKERRQ(ierr); // surface
  ierr = VecSet(D.gD,0.0);CHKERRQ(ierr); // depth
  ierr = VecSet(D.gR,0.5);CHKERRQ(ierr); // remote

  // fault
  PetscInt Ii,Istart,Iend, N1 = D.H/D.dz;
  ierr = VecGetOwnershipRange(D.gF,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<N1) { ierr = VecSetValue(D.gF,Ii,0.0,INSERT_VALUES); }
    else { ierr = VecSetValue(D.gF,Ii,0.5,INSERT_VALUES); }
  }
  ierr = VecAssemblyBegin(D.gF);CHKERRQ(ierr);  ierr = VecAssemblyEnd(D.gF);CHKERRQ(ierr);

  // compute analytic surface displacement
  Vec anal;
  ierr = VecDuplicate(D.surfDisp,&anal);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(anal,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-D.Nz*(Ii/D.Nz);
    //~y = D.dy*(Ii/D.Nz);
    u = (1.0/PETSC_PI)*atan(D.dy*Ii/D.H);
    ierr = VecSetValues(anal,1,&Ii,&u,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(anal);CHKERRQ(ierr);  ierr = VecAssemblyEnd(anal);CHKERRQ(ierr);

  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/anal",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(anal,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/gR",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(D.gR,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/gF",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(D.gF,viewer);CHKERRQ(ierr);

  ierr = ComputeRHS(D);CHKERRQ(ierr); // assumes gS and gD are 0
  ierr = KSPSolve(D.ksp,D.rhs,D.uhat);CHKERRQ(ierr);

  // pull out surface displacement
  PetscInt ind;
  ierr = VecGetOwnershipRange(D.uhat,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-D.Nz*(Ii/D.Nz);
    ind = Ii/D.Nz;
    if (z == 0) {
      ierr = VecGetValues(D.uhat,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValues(D.surfDisp,1,&ind,&u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(D.surfDisp);CHKERRQ(ierr);  ierr = VecAssemblyEnd(D.surfDisp);CHKERRQ(ierr);

  std::ostringstream fileName;
  fileName << "data/surfDisp" << order << "Ny" << Ny;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileName.str().c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"data/surfDisp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(D.surfDisp,viewer);CHKERRQ(ierr);


  // compute error
  PetscScalar maxErr[2] = {0.0,0.0};
  Vec diff;
  VecDuplicate(D.surfDisp,&diff);CHKERRQ(ierr);

  ierr = VecWAXPY(diff,-1.0,D.surfDisp,anal);CHKERRQ(ierr);
  ierr = VecAbs(diff);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_2,&maxErr[0]);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_INFINITY,&maxErr[1]);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %i %g %g %g %g %.9g %.9g\n",
                     D.Ny,D.Nz,D.dy,D.dz,D.Ly,D.Lz,maxErr[0],maxErr[1]);CHKERRQ(ierr);


  return ierr;
}
*/


int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,NULL,NULL);

  PetscErrorCode ierr = 0;

  //~// test odeSolver
  //~TestOdeSolver *trial;
  //~trial = new TestOdeSolver();
  //~trial->writeStep();
  //~PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");
  //~trial->integrate();

  // test DMDA stuff
  //~testDMDA();
  //~testDMDA_changeCoords();
  //~testDMDA_ScatterToVec();
  //~testDMDA_memory();
  //~testDMDAWithMats();


  timeDx();
  //~PetscPrintf(PETSC_COMM_WORLD,"\n\n");
  //~timeDy();


  PetscFinalize();
  return ierr;
}
