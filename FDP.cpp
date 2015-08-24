/* FDP.cpp
 * -------
 * (currently in the process of adding actual functionality
 * as it is pretty much an MMS test of our methods at the moment)
 * Finite-Difference in Parallel
 * -----------------------------
 * This program allows users to calculate
 * the derivatives of every point on a one-dimensional vector or
 * two-dimensional grid (stored in a Vec) by using central
 * difference approximations on interior points. Boundary points
 * use second-order accurate approximations.
 */

#include <petscvec.h>
#include <petscdmda.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

//MMS Test Globals
#define MIN_GRID_POINT 0
#define MAX_GRID_POINT 3
#define STARTING_GRID_SPACING 201
#define NUM_GRID_SPACE_CALCULATIONS 0 // 8 // 14 FIXME what's the limit here?
//2-D
#define NUM_2D_GRID_SPACE_CALCULATIONS 9 //12  // FIXME is there a limit here? SLOWS DOWN A LOT ON 20k+ values in each direction?
#define NUM_2D_2ND_GRID_SPACE_CALCULATIONS 0
#define COEFFICIENT_SOLVE 1 // 0 for no coefficient solve (w mu), 1 (or any other numeric value) to solve w/ coefficient
#define NUM_X_PTS 11
#define NUM_Y_PTS 5
#define X_MIN 0.0
#define Y_MIN 0.0
#define X_MAX 0.05  //0.00000000000000000000000000000000000000000000000000005
#define Y_MAX 0.10 //0.00000000000000000000000000000000000000000000000000010

//Actual Globals
#define PTS_IN_X 6
#define PTS_IN_Y 5
#define X_MINIMUM 0.0
#define Y_MINIMUM 0.0
#define X_MAXIMUM 5.0
#define Y_MAXIMUM 10.0

typedef struct {
/* problem parameters */
    PetscInt       n_x, n_y;
    Vec            Bottom, Top, Left, Right; /* boundary values */

/* Working space */
    Vec         localX, localV;           /* ghosted local vector */
    DM          dm;                       /* distributed array data structure */
    Mat         H;
} AppCtx;

/* Function: writeVec
 * ------------------
 * This function allows users to pass in a vector and
 * a file location (including name) for a PETSc vector.
 * If the current directory is the desired location,
 * a string containing the name of the file
 * will suffice. The vector can then be imported to
 * MATLAB for error checking or other usage.
 */
PetscErrorCode writeVec(Vec vec,const char * loc)
{
    PetscErrorCode ierr = 0;
    PetscViewer    viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,loc,FILE_MODE_WRITE,&viewer);
    ierr = VecView(vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    return ierr;
}

/* Function: calculate_two_norm
 * ----------------------------
 * This function calculates the two-norm
 * of the difference of two vectors, multiplies
 * the calculated norm by 1/sqrt(length of vector)
 * and outputs this error value as well as the log (base 2)
 * of the error value. The Vec that will contain the difference is
 * the first argument (diff). The second and third arguments
 * are the two Vecs that we are taking the difference of. The last
 * argument is the length of the vectors, the number of entries
 * in the given Vecs.
 */
PetscErrorCode calculate_two_norm(Vec &diff, const Vec &x, const Vec &y, int size) {
    PetscErrorCode ierr = 0;
    PetscScalar a = -1; // set the multiple in the WAXPY operation to negative so it just subtracts
    VecWAXPY(diff,a,x,y); // diff = x - y
    PetscReal norm = 0; // zeroes out the norm value
    VecNorm(diff,NORM_2,&norm); // norm = NORM_2(diff)
    PetscReal answer = norm * (1/(sqrt(size))); // error equals norm times 1 over the sqrt of the Vec length
    PetscPrintf(PETSC_COMM_WORLD, "Error: % .5e   ", answer);
    answer = log2(answer); // take the log (base 2) for easier readability
    PetscPrintf(PETSC_COMM_WORLD, "Log of Error: % .5e\n", answer); // I USED TO HAVE %.15e
    return ierr;
}

/* Function: Dx_1d
 * ------------------------------------
 * This function takes in a derivative matrix (in the form of a vector)
 * that it will fill. It also takes in a grid spacing interval (PetscScalar)
 * and the values at each interval (which is a matrix placed in the form
 * of a vector). The vectors are passed in by reference. One
 * also needs to pass in the DMDA associated with the given Vecs.
 *
 * Vecs:
 * - u is the vector that one passes in to hold the calculated derivative.
 * - z is the vector that holds the sampled values of the function
 */
PetscErrorCode Dx_1d(Vec &u, Vec &z, PetscScalar spacing, DM &da) {
  PetscErrorCode ierr = 0;
  // Creating local Vecs y (function values) and calc_diff (derivative)
  Vec y, calc_diff;
  // Values that hold specific indices for iteration
  PetscInt yistart, yiend, fistart, fiend, yi_extra;
  // Creates local vectors using the DMDA to ensure correct partitioning
  DMCreateLocalVector(da,&y);
  DMCreateLocalVector(da,&calc_diff);
  // Make it such that the DMDA splits the global Vecs into local Vecs
  ierr = DMGlobalToLocalBegin(da,z,INSERT_VALUES,y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,z,INSERT_VALUES,y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,u,INSERT_VALUES,calc_diff);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,u,INSERT_VALUES,calc_diff);CHKERRQ(ierr);
  // Obtains the ownership range of both the function-value Vec and the derivative Vec
  ierr = VecGetOwnershipRange(y,&yistart,&yiend);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(calc_diff,&fistart,&fiend);CHKERRQ(ierr);
  // Here we set the current to the beginning and the previous to the beginning
  // We set the next to the one directly after the start.
  PetscInt yi_current = yistart;
  PetscInt yi_prev = yistart;
  PetscInt yi_next = yistart + 1;
  PetscScalar slope = 0; // This will contain the derivative on each calculation
  PetscScalar y_1, y_2, y_3; // These are used to keep the derivative values we pull from the function Vec
  PetscInt fi = fistart; // We start the current iteration on the first value in the function Vec
  while(yi_current < yiend) {
     yi_next = yi_current + 1;
     // If the current index is between the start and the end of the Vec
     // simply calculate the derivative using the central difference method
     if (yi_current > yistart && yi_current < yiend - 1) {
        yi_prev = yi_current - 1;
        ierr = VecGetValues(y,1,&yi_prev,&y_1);CHKERRQ(ierr);
        ierr = VecGetValues(y,1,&yi_next,&y_2);CHKERRQ(ierr);
        slope = ((y_2 - y_1) / (2 * spacing));
     // This is where I calculate the derivative at the beginning
     // of the Vec. I use coefficients and 3 points to obtain
     // second-order accuracy.
     } else if (yi_current == yistart) {
        ierr = VecGetValues(y,1,&yi_current,&y_1);CHKERRQ(ierr);
        ierr = VecGetValues(y,1,&yi_next,&y_2);CHKERRQ(ierr);
        yi_extra = yi_next + 1;
        ierr = VecGetValues(y,1,&yi_extra,&y_3);CHKERRQ(ierr);
        slope = (((-y_3) + (4 * y_2) + (-3 * y_1)) / (2 * spacing));
     // This is where I calculate the derivative at the end of the
     // Vec. I use coefficients and 3 points at the end of the Vec
     // to obtain second-order accuracy.
     } else if (yi_current == yiend - 1) {
        yi_prev = yi_current - 1;
        ierr = VecGetValues(y,1,&yi_prev,&y_1);CHKERRQ(ierr);
        ierr = VecGetValues(y,1,&yi_current,&y_2);CHKERRQ(ierr);
        yi_extra = yi_prev - 1;
        ierr = VecGetValues(y,1,&yi_extra,&y_3);CHKERRQ(ierr);
        slope = (((y_3) + (-4 * y_1) + (3 * y_2)) / (2 * spacing));
     }
     // This sets the Vec value to the calculated derivative
     ierr = VecSetValues(calc_diff,1,&fi,&slope,INSERT_VALUES);CHKERRQ(ierr);
     // Increment here
     fi++;
     yi_current++;
  }
  // Assemble the Vec - do we need this here? Not sure.
  ierr = VecAssemblyBegin(calc_diff);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(calc_diff);CHKERRQ(ierr);
  // Take the Vec values from the local Vecs and put them back in their respective global Vecs.
  ierr = DMLocalToGlobalBegin(da,y,INSERT_VALUES,z);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,y,INSERT_VALUES,z);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,calc_diff,INSERT_VALUES,u);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,calc_diff,INSERT_VALUES,u);CHKERRQ(ierr);
  // Dispose of the local Vecs
  VecDestroy(&y);
  VecDestroy(&calc_diff);
  return ierr;
}

/* Function: calculate_with_size
 * -----------------------------
 * This function calculates the first
 * derivative of a Vec using only
 * the number of entries (n), the minimum value (x_min),
 * and the maximum value (x_max). It calculates
 * all other need values.
 */
PetscErrorCode calculate_with_size(PetscInt n, PetscInt x_min, PetscInt x_max)
{
  PetscErrorCode    ierr = 0;
  PetscInt          i = 0, xistart, xiend, yistart, yiend, diff_start, diff_end; /* iteration values */
  PetscScalar       v = 0, mult = 0; /* v is my temp value I use to fill Vecs, mult is the spacing between each value of the Vec (h, dx) */
  Vec               w = NULL, x = NULL, y = NULL, calc_diff = NULL, diff = NULL; /* vectors */
  mult = ((PetscReal)(x_max - x_min))/(n - 1); // evenly partitioning the Vec to have equally sized spacing between values
  // MUST CREATE DMDA - STENCIL WIDTH 1 FOR BOUNDARIES
  // IMPORTANT: SINCE STENCIL WIDTH IS 1, BE SURE THAT THERE ARE AT LEAST 3 END VALUES ON THE END OF A PARTITIONED VEC
  DM            da;
  DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,n,1,1,NULL,&da);
  // Create first Vec, this one holds the values at specific points (at every spacing)
  ierr = DMCreateGlobalVector(da,&x); // Holds spacing values
  // Same Vec format, duplicated.
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr); // Will eventually be used to hold the difference between the actual derivative and the calculated one.
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr); // Holds function values
  ierr = VecDuplicate(x,&calc_diff);CHKERRQ(ierr); // Holds calculated derivative
  ierr = VecDuplicate(x,&diff);CHKERRQ(ierr); // Holds actual derivative

  ierr = PetscObjectSetName((PetscObject) x, "x (spacing values)");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) w, "w");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y, "y (func. values)");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) calc_diff, "calculated derivative");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) diff, "actual derivative");CHKERRQ(ierr);

/* This is where I fill in the x vector.
 * I put in multiples defined by the grid
 * spacing from the minimum value to the maximum.
 */
  VecGetOwnershipRange(x,&xistart,&xiend);
  for(i = xistart; i < xiend; i++) {
    v = (i*mult);
    VecSetValues(x,1,&i,&v,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

/* This is where I fill in the y vector.
 * Its values can be defined by a hard-coded
 * function below. Comment out a single
 * function to dictate what will be in the y Vec.
 */
  ierr = VecGetOwnershipRange(y,&yistart,&yiend);CHKERRQ(ierr);
  for(i = yistart; i < yiend; i++) {
    ierr = VecGetValues(x,1,&i,&v);CHKERRQ(ierr);
//    v *= v; // x^2
//    v = (v*v*v); // x^3
    v = sin(v); // sine function
//    v = cos(v); // cosine function
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

/* This is where I fill in the true
 * analytical derivative. It's hard-coded
 * as well.
 */
  ierr = VecGetOwnershipRange(diff,&diff_start,&diff_end);CHKERRQ(ierr);
  for(i = diff_start; i < diff_end; i++) {
    ierr = VecGetValues(x,1,&i,&v);CHKERRQ(ierr);
//    v = 2*v; // 2x
//    v = (3)*(v*v); //3x^2
    v = cos(v); // cosine function
//    v = -sin(v); // -sine function
    ierr = VecSetValues(diff,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(diff);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diff);CHKERRQ(ierr);

/* Here we pass in the calculated derivative Vec (to be filled),
 * our spacing, our Vec with function values (y), and the DMDA.
 * It will fill the calc_diff Vec with our calculated derivative.
 */
  ierr = Dx_1d(calc_diff, y, mult, da);CHKERRQ(ierr);

/* This is where we calculate norm using the difference between
 * the analytical (true) derivative and the calculated derivative and print
 * out the error. Below here, we print out information (number of entries,
 * the current spacing).
 */
  PetscPrintf(PETSC_COMM_WORLD, "Nx: %15i   Spacing: % .15e   ", n, mult);
  ierr = calculate_two_norm(w, calc_diff, diff, n);CHKERRQ(ierr);

/* To print out any of the Vecs, uncomment one of the lines below */
//  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr = VecView(calc_diff,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr = VecView(diff,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

// BE SURE TO DISPOSE OF ALL VECS AND DMDAS!
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&calc_diff);CHKERRQ(ierr);
  ierr = VecDestroy(&diff);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  return ierr;
}

/* Function: truncation_error
 * --------------------------
 *
 */
PetscErrorCode truncation_error(Vec &diff_x, PetscScalar &mult_x, DM &da) {
    PetscErrorCode ierr = 0;
    PetscInt m, n, mStart, nStart, j, i, M;
    DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

    Vec local_diff_x;
    PetscScalar** local_diff_x_arr;
    ierr = DMCreateLocalVector(da, &local_diff_x);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, diff_x, INSERT_VALUES, local_diff_x);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, diff_x, INSERT_VALUES, local_diff_x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);

    PetscInt x_index = (((int)M/5));

    PetscPrintf(PETSC_COMM_WORLD, "Spacing: %  .15e At index ([j][i]) [%i][%5i]F'(0.0001) = %  .15e\n", mult_x, 2, x_index, local_diff_x_arr[2][x_index]);


    return ierr;
}

/* Function: Dy_2d
 * ----------------------------------
 *
 */
PetscErrorCode Dy_2d(Vec &diff_y, Vec &grid, PetscScalar dy, DM &da) {
  PetscErrorCode ierr = 0;

  PetscInt m, n, mStart, nStart, j, i, N;
  DMDAGetInfo(da,NULL,NULL,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  Vec local_diff_y, local_grid;
  PetscScalar** local_diff_y_arr;
  PetscScalar** local_grid_arr;
  ierr = DMCreateLocalVector(da, &local_diff_y);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_y, &local_diff_y_arr);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);CHKERRQ(ierr);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (j > 0 && j < N - 1) { local_diff_y_arr[j][i] = (local_grid_arr[j+1][i] - local_grid_arr[j-1][i])/(2.0 * dy); }
          else if (j == 0) { local_diff_y_arr[j][i] = (-1.5 * local_grid_arr[0][i] + 2.0 * local_grid_arr[1][i] - 0.5 * local_grid_arr[2][i])/ dy; }
          else if (j == N - 1) { local_diff_y_arr[j][i] = (0.5 * local_grid_arr[N-3][i] - 2.0 *
                  local_grid_arr[N-2][i] + 1.5 * local_grid_arr[N-1][i]) / dy; }
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_y, &local_diff_y_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);

  VecDestroy(&local_diff_y);
  VecDestroy(&local_grid);
  return ierr;
}

/* Function: Dx_2d
 * ----------------------------------
 *
 */
PetscErrorCode Dx_2d(Vec &diff_x, Vec &grid, PetscScalar dx, DM &da) {
  PetscErrorCode ierr = 0;
  PetscInt m, n, mStart, nStart, j, i, M;
  DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  Vec local_diff_x, local_grid;
  PetscScalar** local_diff_x_arr;
  PetscScalar** local_grid_arr;
  ierr = DMCreateLocalVector(da, &local_diff_x);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr); CHKERRQ(ierr);

  ierr = DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);CHKERRQ(ierr);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (i > 0 && i < M - 1) {local_diff_x_arr[j][i] = (local_grid_arr[j][i+1] - local_grid_arr[j][i-1])/(2.0 * dx); }
          else if (i == 0) { local_diff_x_arr[j][i] = (-1.5 * local_grid_arr[j][0] + 2.0 * local_grid_arr[j][1] - 0.5 * local_grid_arr[j][2])/ dx; }
          else if (i == M - 1) { local_diff_x_arr[j][i] = (0.5 * local_grid_arr[j][M-3] - 2.0 *
                  local_grid_arr[j][M-2] + 1.5 * local_grid_arr[j][M-1]) / dx; }
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);

  VecDestroy(&local_diff_x);
  VecDestroy(&local_grid);
  return ierr;
}

/* Function: Dyy_2d
 * ----------------------------------
 *
 */
PetscErrorCode Dyy_2d(Vec &diff_y, const Vec &grid, const PetscScalar dy, const DM &da) {
  PetscErrorCode ierr = 0;

  PetscInt m, n, mStart, nStart, j, i, N;
  DMDAGetInfo(da,NULL,NULL,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  Vec local_diff_y, local_grid;
  PetscScalar** local_diff_y_arr;
  PetscScalar** local_grid_arr;
  ierr = DMCreateLocalVector(da, &local_diff_y);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_y, &local_diff_y_arr); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr); CHKERRQ(ierr);

  ierr = DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0); CHKERRQ(ierr);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (j > 0 && j < N - 1) { local_diff_y_arr[j][i] = (local_grid_arr[j+1][i] - (2*local_grid_arr[j][i]) + local_grid_arr[j-1][i])/(dy * dy); }
          else if (j == 0) { local_diff_y_arr[j][i] = (local_grid_arr[0][i] - (2 * local_grid_arr[1][i]) + local_grid_arr[2][i])/(dy * dy); }
          else if (j == N - 1) { local_diff_y_arr[j][i] = (local_grid_arr[N-1][i] - (2 * local_grid_arr[N-2][i]) + local_grid_arr[N-3][i])/(dy * dy); }
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_y, &local_diff_y_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);

  VecDestroy(&local_diff_y);
  VecDestroy(&local_grid);
  return ierr;
}

/* Function: Dxx_2d
 * --------------------------------------
 *
 */
PetscErrorCode Dxx_2d(Vec &diff_x, const Vec &grid, const PetscScalar dx, const DM &da) {
  PetscErrorCode ierr = 0;
  PetscInt m, n, mStart, nStart, j, i, M;
  DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  VecSet(diff_x,0.0);
  Vec local_diff_x, local_grid;
  PetscScalar** local_diff_x_arr;
  PetscScalar** local_grid_arr;
  ierr = DMCreateLocalVector(da, &local_diff_x);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);

  DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (i > 0 && i < M - 1) {local_diff_x_arr[j][i] = (local_grid_arr[j][i+1] - (2 * local_grid_arr[j][i]) + local_grid_arr[j][i-1])/(dx * dx); }
          else if (i == 0) { local_diff_x_arr[j][i] = (local_grid_arr[j][0] - (2 * local_grid_arr[j][1]) + local_grid_arr[j][2])/(dx * dx); }
          else if (i == M - 1) { local_diff_x_arr[j][i] = (local_grid_arr[j][M-1] - (2 * local_grid_arr[j][M-2]) + local_grid_arr[j][M-3])/(dx * dx); }
          //PetscPrintf(PETSC_COMM_SELF, "[%i][%i] = %g\n", j, i, local_diff_x_arr[j][i]);
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);

  VecDestroy(&local_diff_x);
  VecDestroy(&local_grid);
  return ierr;
}

/* Function: Dmuyy_2d
 * ------------------
 *
 */
PetscErrorCode Dmuyy_2d(Vec &diff_y, const Vec &grid, const Vec &mu, const PetscScalar dy, const DM &da) {
  PetscErrorCode ierr = 0;

  PetscInt m, n, mStart, nStart, j, i, N;
  DMDAGetInfo(da,NULL,NULL,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  Vec local_diff_y = NULL, local_grid = NULL, local_mu = NULL;
  PetscScalar** local_diff_y_arr;
  PetscScalar** local_grid_arr;
  PetscScalar** local_mu_arr;
  ierr = DMCreateLocalVector(da, &local_diff_y);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_mu);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_y, &local_diff_y_arr);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_mu, &local_mu_arr); CHKERRQ(ierr);

  ierr = DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);CHKERRQ(ierr);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (j > 0 && j < N - 1) { local_diff_y_arr[j][i] = (0.5 * (((local_mu_arr[j][i] + local_mu_arr[j-1][i]) * local_grid_arr[j-1][i])
                  - ((local_mu_arr[j+1][i] + 2 * local_mu_arr[j][i] + local_mu_arr[j-1][i]) * local_grid_arr[j][i])
                  + ((local_mu_arr[j][i] + local_mu_arr[j+1][i]) * local_grid_arr[j+1][i])
                  ))/(dy * dy); }
          else if (j == 0) { local_diff_y_arr[j][i] = (0.5 * (((local_mu_arr[1][i] + local_mu_arr[0][i]) * local_grid_arr[0][i])
                  - ((local_mu_arr[2][i] + 2 * local_mu_arr[1][i] + local_mu_arr[0][i]) * local_grid_arr[1][i])
                  + ((local_mu_arr[1][i] + local_mu_arr[2][i]) * local_grid_arr[2][i])
                  ))/(dy * dy); }
          else if (j == N - 1) { local_diff_y_arr[j][i] = (0.5 * (((local_mu_arr[N-2][i] + local_mu_arr[N-1][i]) * local_grid_arr[N-1][i])
                  - ((local_mu_arr[N-3][i] + 2 * local_mu_arr[N-2][i] + local_mu_arr[N-1][i]) * local_grid_arr[N-2][i])
                  + ((local_mu_arr[N-2][i] + local_mu_arr[N-3][i]) * local_grid_arr[N-3][i])
                  ))/(dy * dy); }
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_y, &local_diff_y_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_mu, &local_mu_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_y, INSERT_VALUES, diff_y);CHKERRQ(ierr);

  VecDestroy(&local_diff_y);
  VecDestroy(&local_grid);
  VecDestroy(&local_mu);
  return ierr;
}

/* Function: Dmuxx_2d
 * --------------------------------------
 *
 */
PetscErrorCode Dmuxx_2d(Vec &diff_x, const Vec &grid, const Vec &mu, const PetscScalar dx, const DM &da) {
  PetscErrorCode ierr = 0;
  PetscInt m, n, mStart, nStart, j, i, M;
  DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

  Vec local_diff_x = NULL, local_grid = NULL, local_mu = NULL;
  PetscScalar** local_diff_x_arr;
  PetscScalar** local_grid_arr;
  PetscScalar** local_mu_arr;
  ierr = DMCreateLocalVector(da, &local_diff_x);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local_mu);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, local_diff_x, &local_diff_x_arr); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_grid, &local_grid_arr); CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, local_mu, &local_mu_arr); CHKERRQ(ierr);

  ierr = DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0); CHKERRQ(ierr);
  for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          if (i > 0 && i < M - 1) {local_diff_x_arr[j][i] = (0.5 * (((local_mu_arr[j][i] + local_mu_arr[j][i-1]) * local_grid_arr[j][i-1])
                  - ((local_mu_arr[j][i+1] + 2 * local_mu_arr[j][i] + local_mu_arr[j][i-1]) * local_grid_arr[j][i])
                  + ((local_mu_arr[j][i] + local_mu_arr[j][i+1]) * local_grid_arr[j][i+1])
                  ))/(dx * dx); }
          else if (i == 0) { local_diff_x_arr[j][i] = (0.5 * (((local_mu_arr[j][1] + local_mu_arr[j][0]) * local_grid_arr[j][0])
                  - ((local_mu_arr[j][2] + 2 * local_mu_arr[j][1] + local_mu_arr[j][0]) * local_grid_arr[j][1])
                  + ((local_mu_arr[j][1] + local_mu_arr[j][2]) * local_grid_arr[j][2])
                  ))/(dx * dx); }
          else if (i == M - 1) { local_diff_x_arr[j][i] = (0.5 * (((local_mu_arr[j][M-2] + local_mu_arr[j][M-1]) * local_grid_arr[j][M-1])
                  - ((local_mu_arr[j][M-3] + 2 * local_mu_arr[j][M-2] + local_mu_arr[j][M-1]) * local_grid_arr[j][M-2])
                  + ((local_mu_arr[j][M-2] + local_mu_arr[j][M-3]) * local_grid_arr[j][M-3])
                  ))/(dx * dx); }
          //PetscPrintf(PETSC_COMM_SELF, "[%i][%i] = %g\n", j, i, local_diff_x_arr[j][i]);
      }
  }

  ierr = DMDAVecRestoreArray(da, local_diff_x, &local_diff_x_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_grid, &local_grid_arr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, local_mu, &local_mu_arr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_diff_x, INSERT_VALUES, diff_x);CHKERRQ(ierr);

  VecDestroy(&local_diff_x);
  VecDestroy(&local_grid);
  VecDestroy(&local_mu);
  return ierr;
}

/* Function: calculate_2D_grid_derivatives
 * ---------------------------------------
 *
 */
PetscErrorCode calculate_2D_grid_derivatives(PetscInt x_len, PetscInt y_len, PetscScalar x_max, PetscScalar x_min, PetscScalar y_max, PetscScalar y_min) {
  PetscErrorCode ierr = 0;
  double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
  t1 = MPI_Wtime();
  PetscInt i = 0, j = 0, mStart = 0, m = 0, nStart = 0, n = 0,
           grid_istart, grid_iend, diff_x_start, diff_x_end, diff_y_start, diff_y_end;
  PetscScalar v = 0, mult_x = 0, mult_y = 0;
  Vec grid = NULL, diff_x = NULL, diff_y = NULL, act_diff_x = NULL, act_diff_y = NULL, grid_error_diff = NULL, /* vectors */
      local_coords = NULL, local_grid = NULL, local_act_diff_x = NULL, local_act_diff_y = NULL;
      // first row, global vectors, second row, local vectors

  PetscInt num_grid_entries = (x_len) * (y_len); // total number of entries on the grid
  mult_x = ((PetscReal)(x_max - x_min))/(x_len - 1); // spacing in the x direction
  mult_y = ((PetscReal)(y_max - y_min))/(y_len - 1); // spacing in the y direction

  DM            da;
  DM            cda;
  DMDACoor2d **coords; // 2D array containing x and y data members

  // FIXME should I use DMDA_STENCIL_STAR && do we need a stencil width of 2?
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
          DMDA_STENCIL_BOX, x_len, y_len, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da); CHKERRQ(ierr);

  DMDASetUniformCoordinates(da, x_min, x_max, y_min, y_max, 0.0, 0.0);
  DMGetCoordinateDM(da, &cda);
  DMGetCoordinatesLocal(da, &local_coords);
  DMDAVecGetArray(cda, local_coords, &coords);
  DMDAGetCorners(cda, &mStart, &nStart, 0, &m, &n, 0);

  ierr = DMCreateGlobalVector(da,&grid);

  ierr = VecDuplicate(grid, &diff_x);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &diff_y);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &act_diff_x);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &act_diff_y);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &grid_error_diff);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) grid, "global grid");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) diff_x, "diff_x");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) diff_y, "diff_y");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) act_diff_x, "act_diff_x");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) act_diff_y, "act_diff_y");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) grid_error_diff, "grid_error_diff");CHKERRQ(ierr);

  // Set up GRID
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);

//FIXME
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

//FIXME
  PetscInt x_index = x_len/5;
  bool printed = false;

  PetscScalar **local_grid_arr;
  DMDAVecGetArray(da, local_grid, &local_grid_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
//        PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
//        local_grid_arr[j][i] = coords[j][i].y;
         // FIXME
         //if(i == x_index && printed == false) {
         //     PetscPrintf(PETSC_COMM_WORLD, "\nx_index: %i SHOULD BE 0.1 = %  .5e\n", x_index, coords[j][x_index].x);
         //     printed = true;
         //}

         //local_grid_arr[j][i] = (.01*(coords[j][i].x * coords[j][i].x * coords[j][i].x) + (coords[j][i].y * coords[j][i].y * coords[j][i].y));
        local_grid_arr[j][i] = sin(coords[j][i].x) + cos(coords[j][i].y);
    }
  }
  DMDAVecRestoreArray(da, local_grid, &local_grid_arr);

  ierr = DMLocalToGlobalBegin(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);

  // SET UP ACT_DIFF_X
  ierr = DMCreateLocalVector(da, &local_act_diff_x);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, act_diff_x, INSERT_VALUES, local_act_diff_x);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, act_diff_x, INSERT_VALUES, local_act_diff_x);CHKERRQ(ierr);

  PetscScalar **local_act_diff_x_arr;
  DMDAVecGetArray(da, local_act_diff_x, &local_act_diff_x_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
        // PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
//        local_act_diff_x_arr[j][i] = coords[j][i].x;
        local_act_diff_x_arr[j][i] = cos(coords[j][i].x);
        //local_act_diff_x_arr[j][i] = .01*(3 * coords[j][i].x * coords[j][i].x);
    }
  }
  DMDAVecRestoreArray(da, local_act_diff_x, &local_act_diff_x_arr);

  ierr = DMLocalToGlobalBegin(da, local_act_diff_x, INSERT_VALUES, act_diff_x);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_act_diff_x, INSERT_VALUES, act_diff_x);CHKERRQ(ierr);

  // SET UP ACT_DIFF_Y
  ierr = DMCreateLocalVector(da, &local_act_diff_y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, act_diff_y, INSERT_VALUES, local_act_diff_y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, act_diff_y, INSERT_VALUES, local_act_diff_y);CHKERRQ(ierr);

  PetscScalar **local_act_diff_y_arr;
  DMDAVecGetArray(da, local_act_diff_y, &local_act_diff_y_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
        // PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
        //local_act_diff_y_arr[j][i] = .01*(3 * coords[j][i].y * coords[j][i].y);
        local_act_diff_y_arr[j][i] = -sin(coords[j][i].y);
    }
  }
  DMDAVecRestoreArray(da, local_act_diff_y, &local_act_diff_y_arr);

  ierr = DMLocalToGlobalBegin(da, local_act_diff_y, INSERT_VALUES, act_diff_y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_act_diff_y, INSERT_VALUES, act_diff_y);CHKERRQ(ierr);

  t2 = MPI_Wtime();
  ierr = Dy_2d(diff_y, grid, mult_y, da);CHKERRQ(ierr);
  t3 = MPI_Wtime();
  ierr = Dx_2d(diff_x, grid, mult_x, da);CHKERRQ(ierr);
  t4 = MPI_Wtime();

//  PetscPrintf(PETSC_COMM_WORLD, "Ny: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", y_len, mult_y, log2(mult_y)); // I used to have %15!
//  calculate_two_norm(grid_error_diff, diff_y, act_diff_y, num_grid_entries);
  PetscPrintf(PETSC_COMM_WORLD, "                              Nx: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", x_len, mult_x, log2(mult_x));
  calculate_two_norm(grid_error_diff, diff_x, act_diff_x, num_grid_entries);

  t5 = MPI_Wtime();

//  writeVec(grid, "grid_file");
//  writeVec(diff_x, "calc_diff_x_file");
//  writeVec(diff_y, "calc_diff_y_file");
//  writeVec(act_diff_x, "actual_diff_x_file");
//  writeVec(act_diff_y, "actual_diff_y_file");

  // Truncation Error Test!!!
  //ierr =  truncation_error(diff_x, mult_x, da);CHKERRQ(ierr);

  // FIXME make sure all Vecs and DMs are being destroyed - go to other functions as well!
  ierr = VecDestroy(&grid);CHKERRQ(ierr);
  ierr = VecDestroy(&diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&diff_x);CHKERRQ(ierr);
  ierr = VecDestroy(&act_diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&act_diff_x);CHKERRQ(ierr);
  ierr = VecDestroy(&grid_error_diff);CHKERRQ(ierr);
  ierr = VecDestroy(&local_grid);CHKERRQ(ierr);
  ierr = VecDestroy(&local_act_diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&local_act_diff_x);CHKERRQ(ierr);
  //ierr = VecDestroy(&local_coords);CHKERRQ(ierr);

  //ierr = DMDestroy(&cda);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  t6 = MPI_Wtime();
//  PetscPrintf(PETSC_COMM_WORLD, "\nTime spent setting up and filling grid/actual derivative Vecs: % 8g\n", t2 - t1);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating 2Dy derivative: % 8g\n", t3 - t2);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating 2Dx derivative: % 8g\n", t4 - t3);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating norms: % 8g\n", t5 - t4);
//  PetscPrintf(PETSC_COMM_WORLD, "Total time for this iteration: % 8g\n\n", t6 - t1);
  return ierr;
}

/* Function: calculate_2D_2nd_grid_derivatives
 * -------------------------------------------
 *
 */
PetscErrorCode calculate_2D_2nd_grid_derivatives(PetscInt x_len, PetscInt y_len, PetscScalar x_max, PetscScalar x_min, PetscScalar y_max, PetscScalar y_min) {
  PetscErrorCode ierr = 0;
  double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
  t1 = MPI_Wtime();
  PetscInt i = 0, j = 0, mStart = 0, m = 0, nStart = 0, n = 0,
           grid_istart, grid_iend, diff_x_start, diff_x_end, diff_y_start, diff_y_end;
  PetscScalar v = 0, mult_x = 0, mult_y = 0;
  Vec grid = NULL, diff_x = NULL, diff_y = NULL, act_diff_x = NULL, act_diff_y = NULL, grid_error_diff = NULL, /* vectors */
      local_coords = NULL, local_grid = NULL, local_act_diff_x = NULL, local_act_diff_y = NULL,
      mu = NULL, local_mu = NULL, dxxmu = NULL, local_dxxmu = NULL, dyymu = NULL, local_dyymu = NULL;
      // first row, global vectors | second row, local vectors | third row, mu

  PetscInt num_grid_entries = (x_len) * (y_len); // total number of entries on the grid
  mult_x = ((PetscReal)(x_max - x_min))/(x_len - 1); // spacing in the x direction
  mult_y = ((PetscReal)(y_max - y_min))/(y_len - 1); // spacing in the y direction

  DM            da;
  DM            cda;
  DMDACoor2d **coords; // 2D array containing x and y data members

  // FIXME should I use DMDA_STENCIL_STAR?? DMDA Set Up
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
          DMDA_STENCIL_BOX, x_len, y_len, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da); CHKERRQ(ierr);
  DMDASetUniformCoordinates(da, x_min, x_max, y_min, y_max, 0.0, 0.0);
  DMGetCoordinateDM(da, &cda);
  DMGetCoordinatesLocal(da, &local_coords);
  DMDAVecGetArray(cda, local_coords, &coords);
  DMDAGetCorners(cda, &mStart, &nStart, 0, &m, &n, 0);

  ierr = DMCreateGlobalVector(da,&grid);

  ierr = VecDuplicate(grid, &diff_x);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &diff_y);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &act_diff_x);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &act_diff_y);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &grid_error_diff);CHKERRQ(ierr);

  ierr = VecDuplicate(grid, &mu);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &dxxmu);CHKERRQ(ierr);
  ierr = VecDuplicate(grid, &dyymu);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) grid, "global grid");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) diff_x, "diff_x");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) diff_y, "diff_y");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) act_diff_x, "act_diff_x");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) act_diff_y, "act_diff_y");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) grid_error_diff, "grid_error_diff");CHKERRQ(ierr);

  //FIXME
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Set up GRID
  ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);

  PetscScalar **local_grid_arr;
  DMDAVecGetArray(da, local_grid, &local_grid_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
//        PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
//        local_grid_arr[j][i] = coords[j][i].y;
//         local_grid_arr[j][i] = (coords[j][i].x * coords[j][i].x * coords[j][i].x) + (coords[j][i].y * coords[j][i].y * coords[j][i].y);
        local_grid_arr[j][i] = sin(coords[j][i].x) + cos(coords[j][i].y);
//        PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_grid_arr[j][i]);
    }
  }
  DMDAVecRestoreArray(da, local_grid, &local_grid_arr);

  ierr = DMLocalToGlobalBegin(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);

  // SET UP ACT_DIFF_X
  ierr = DMCreateLocalVector(da, &local_act_diff_x);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, act_diff_x, INSERT_VALUES, local_act_diff_x);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, act_diff_x, INSERT_VALUES, local_act_diff_x);CHKERRQ(ierr);

  PetscScalar **local_act_diff_x_arr;
  DMDAVecGetArray(da, local_act_diff_x, &local_act_diff_x_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
//        PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
//        local_act_diff_x_arr[j][i] = coords[j][i].x;
        local_act_diff_x_arr[j][i] = -sin(coords[j][i].x);
//        PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_act_diff_x_arr[j][i]);
//        local_act_diff_x_arr[j][i] = (6 * coords[j][i].x);
    }
  }
  DMDAVecRestoreArray(da, local_act_diff_x, &local_act_diff_x_arr);

  ierr = DMLocalToGlobalBegin(da, local_act_diff_x, INSERT_VALUES, act_diff_x);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_act_diff_x, INSERT_VALUES, act_diff_x);CHKERRQ(ierr);

  // SET UP ACT_DIFF_Y
  ierr = DMCreateLocalVector(da, &local_act_diff_y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, act_diff_y, INSERT_VALUES, local_act_diff_y);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, act_diff_y, INSERT_VALUES, local_act_diff_y);CHKERRQ(ierr);

  PetscScalar **local_act_diff_y_arr;
  DMDAVecGetArray(da, local_act_diff_y, &local_act_diff_y_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
//        PetscPrintf(PETSC_COMM_SELF, "%i: (coords[%i][%i].x, coords[%i][%i].y) = (%g, %g)\n)", rank, j, i, j, i, coords[j][i].x, coords[j][i].y);
//        local_act_diff_y_arr[j][i] = (6 * coords[j][i].y);
        local_act_diff_y_arr[j][i] = -cos(coords[j][i].y);
    }
  }
  DMDAVecRestoreArray(da, local_act_diff_y, &local_act_diff_y_arr);

  ierr = DMLocalToGlobalBegin(da, local_act_diff_y, INSERT_VALUES, act_diff_y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_act_diff_y, INSERT_VALUES, act_diff_y);CHKERRQ(ierr);

  // SET UP MU
  ierr = DMCreateLocalVector(da, &local_mu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, mu, INSERT_VALUES, local_mu);CHKERRQ(ierr);

  PetscScalar **local_mu_arr;
  DMDAVecGetArray(da, local_mu, &local_mu_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
        local_mu_arr[j][i] = cos(coords[j][i].x + coords[j][i].y) + 4;
//        local_mu_arr[j][i] = 1;
    }
  }
  DMDAVecRestoreArray(da, local_mu, &local_mu_arr);

  ierr = DMLocalToGlobalBegin(da, local_mu, INSERT_VALUES, mu);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_mu, INSERT_VALUES, mu);CHKERRQ(ierr);

  // SET UP DxxMU
  ierr = DMCreateLocalVector(da, &local_dxxmu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, dxxmu, INSERT_VALUES, local_dxxmu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, dxxmu, INSERT_VALUES, local_dxxmu);CHKERRQ(ierr);

  PetscScalar **local_dxxmu_arr;
  DMDAVecGetArray(da, local_dxxmu, &local_dxxmu_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
        local_dxxmu_arr[j][i] = (-sin(coords[j][i].x + coords[j][i].y) * cos(coords[j][i].x)) +
                ((cos(coords[j][i].x + coords[j][i].y) + 4) * (-sin(coords[j][i].x)));
//        local_dxxmu_arr[j][i] = -sin(coords[j][i].x);
    }
  }
  DMDAVecRestoreArray(da, local_dxxmu, &local_dxxmu_arr);

  ierr = DMLocalToGlobalBegin(da, local_dxxmu, INSERT_VALUES, dxxmu);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_dxxmu, INSERT_VALUES, dxxmu);CHKERRQ(ierr);

  // SET UP DyyMU
  ierr = DMCreateLocalVector(da, &local_dyymu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, dyymu, INSERT_VALUES, local_dyymu);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, dyymu, INSERT_VALUES, local_dyymu);CHKERRQ(ierr);

  PetscScalar **local_dyymu_arr;
  DMDAVecGetArray(da, local_dyymu, &local_dyymu_arr);
  for (j = nStart; j < nStart + n; j++) {
    for (i = mStart; i < mStart + m; i++) {
        local_dyymu_arr[j][i] = (-sin(coords[j][i].x + coords[j][i].y) * (-sin(coords[j][i].y))) +
                ((cos(coords[j][i].x + coords[j][i].y) + 4) * (-cos(coords[j][i].y)));
//        local_dyymu_arr[j][i] = -cos(coords[j][i].y);
    }
  }
  DMDAVecRestoreArray(da, local_dyymu, &local_dyymu_arr);

  ierr = DMLocalToGlobalBegin(da, local_dyymu, INSERT_VALUES, dyymu);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, local_dyymu, INSERT_VALUES, dyymu);CHKERRQ(ierr);

  // derivative calculations & error calculations
  if(COEFFICIENT_SOLVE) {
    t2 = MPI_Wtime();
    ierr = Dmuyy_2d(diff_y, grid, mu, mult_y, da);CHKERRQ(ierr);
    t3 = MPI_Wtime();
    ierr = Dmuxx_2d(diff_x, grid, mu, mult_x, da);CHKERRQ(ierr);
    t4 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Ny: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", y_len, mult_y, log2(mult_y)); // I used to have %15!
    calculate_two_norm(grid_error_diff, diff_y, dyymu, num_grid_entries);
//    PetscPrintf(PETSC_COMM_WORLD, "Nx: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", x_len, mult_x, log2(mult_x));
    PetscPrintf(PETSC_COMM_WORLD, "                              Nx: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", x_len, mult_x, log2(mult_x));
    calculate_two_norm(grid_error_diff, diff_x, dxxmu, num_grid_entries);
  } else {
    t2 = MPI_Wtime();
    ierr = Dyy_2d(diff_y, grid, mult_y, da);CHKERRQ(ierr);
    t3 = MPI_Wtime();
    ierr = Dxx_2d(diff_x, grid, mult_x, da);CHKERRQ(ierr);
    t4 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Ny: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", y_len, mult_y, log2(mult_y)); // I used to have %15!
    calculate_two_norm(grid_error_diff, diff_y, act_diff_y, num_grid_entries);
    //PetscPrintf(PETSC_COMM_WORLD, "Nx: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", x_len, mult_x, log2(mult_x));
    PetscPrintf(PETSC_COMM_WORLD, "                              Nx: %5i   Spacing: % .5e   Log of Spacing: % .5e   ", x_len, mult_x, log2(mult_x));
    calculate_two_norm(grid_error_diff, diff_x, act_diff_x, num_grid_entries);
  }
  t5 = MPI_Wtime();

//  writeVec(grid, "grid_file");
//  writeVec(diff_x, "calc_diff_x_file");
//  writeVec(diff_y, "calc_diff_y_file");
//  writeVec(act_diff_x, "actual_diff_x_file");
//  writeVec(act_diff_y, "actual_diff_y_file");

  // FIXME make sure all Vecs and DMs are being destroyed - go to other functions as well!
  ierr = VecDestroy(&grid);CHKERRQ(ierr);
  ierr = VecDestroy(&diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&diff_x);CHKERRQ(ierr);
  ierr = VecDestroy(&act_diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&act_diff_x);CHKERRQ(ierr);
  ierr = VecDestroy(&grid_error_diff);CHKERRQ(ierr);
  ierr = VecDestroy(&local_grid);CHKERRQ(ierr);
  ierr = VecDestroy(&local_act_diff_y);CHKERRQ(ierr);
  ierr = VecDestroy(&local_act_diff_x);CHKERRQ(ierr);
  ierr = VecDestroy(&mu);CHKERRQ(ierr);
  ierr = VecDestroy(&dxxmu);CHKERRQ(ierr);
  ierr = VecDestroy(&dyymu);CHKERRQ(ierr);
  ierr = VecDestroy(&local_mu);CHKERRQ(ierr);
  ierr = VecDestroy(&local_dxxmu);CHKERRQ(ierr);
  ierr = VecDestroy(&local_dyymu);CHKERRQ(ierr);

  //ierr = DMDestroy(&cda);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  t6 = MPI_Wtime();
//  PetscPrintf(PETSC_COMM_WORLD, "\nTime spent setting up and filling grid/actual derivative Vecs: % 8g\n", t2 - t1);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating 2Dy derivative: % 8g\n", t3 - t2);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating 2Dx derivative: % 8g\n", t4 - t3);
//  PetscPrintf(PETSC_COMM_WORLD, "Time spent calculating norms: % 8g\n", t5 - t4);
//  PetscPrintf(PETSC_COMM_WORLD, "Total time for this iteration: % 8g\n\n", t6 - t1);
  return ierr;
}

/* Function: MMSTest()
 * -------------------
 *
 */
PetscErrorCode MMSTest() {
  PetscErrorCode ierr = 0;
  PetscInt n = STARTING_GRID_SPACING, min = MIN_GRID_POINT, max = MAX_GRID_POINT, x_len = NUM_X_PTS, y_len = NUM_Y_PTS;
  PetscScalar x_min = X_MIN, x_max = X_MAX, y_max = Y_MAX, y_min = Y_MIN;
  assert(n > 2);

  PetscPrintf(PETSC_COMM_WORLD, "1-D 1ST DERIVATIVES (2ND DERIVATIVES TO BE IMPLEMENTED)\n");
  for(int i = 0; i < NUM_GRID_SPACE_CALCULATIONS; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "Iteration: %15i    ", i+1);
    ierr = calculate_with_size(n, min, max);CHKERRQ(ierr);
    n = ((n - 1) * 2) + 1;
  }

  PetscPrintf(PETSC_COMM_WORLD, "2-D 1ST DERIVATIVES\n");
  assert(x_len > 3);
  assert(y_len > 3);
  for(int i = 0; i < NUM_2D_GRID_SPACE_CALCULATIONS; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "Iteration: %15i    ", i+1);
    calculate_2D_grid_derivatives(x_len, y_len, x_max, x_min, y_max, y_min);
    x_len = ((x_len - 1) * 2) + 1;
    y_len = ((y_len - 1) * 2) + 1;
  }

  PetscPrintf(PETSC_COMM_WORLD, "2-D 2ND DERIVATIVES\n");
  x_len = NUM_X_PTS;
  y_len = NUM_Y_PTS;
  x_min = X_MIN;
  x_max = X_MAX;
  y_max = Y_MAX;
  y_min = Y_MIN;
  assert(x_len > 3);
  assert(y_len > 3);
  for(int i = 0; i < NUM_2D_2ND_GRID_SPACE_CALCULATIONS; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "Iteration: %15i    ", i+1);
    calculate_2D_2nd_grid_derivatives(x_len, y_len, x_max, x_min, y_max, y_min);
    x_len = ((x_len - 1) * 2) + 1;
    y_len = ((y_len - 1) * 2) + 1;
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n");
  return ierr;
}

/* Function: setRHS_R
 * ------------------
 *
 */
PetscErrorCode setRHS_R(Vec &RHS, const Vec &r, const PetscScalar &h_11,
        const PetscScalar alpha_R, const DM &da) {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscErrorCode ierr = 0;
    PetscInt m, n, mStart, nStart, j, i, M = NULL, N = NULL, ristart, riend;
    PetscScalar v;
    DMDAGetInfo(da, NULL, &M, &N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);

    ierr = VecGetOwnershipRange(r,&ristart,&riend);CHKERRQ(ierr);
    for(i = ristart; i < riend; i++) {
        ierr = VecGetValues(r,1,&i,&v);CHKERRQ(ierr);
        v = v * h_11 * alpha_R;
        ierr = VecSetValues(r,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(r);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(r);CHKERRQ(ierr);

    //ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // VecScatter routine! IMPORTANT!

    VecScatter scatter; // scatter context
    IS from,to; // index sets that define the scatter

    //int idx_from[] = {0,1,2,3,4}, idx_to[] = {0,11,22,33,44};

    int idx_from[M], idx_to[M];
    for(i = 0; i < M; i++) {
        idx_from[i] = i;
    }
    for(i = 0; i < M; i++) {
        idx_to[i] = i + ((N-1) * M);
    }


    AO ao;
    DMDAGetAO(da,&ao);
    AOApplicationToPetsc(ao,M,idx_to);

    ISCreateGeneral(PETSC_COMM_SELF,M,idx_from,PETSC_COPY_VALUES,&from);
    ISCreateGeneral(PETSC_COMM_SELF,M,idx_to,PETSC_COPY_VALUES,&to);
    VecScatterCreate(r,from,RHS,to,&scatter); // gx = source vector, gy = destination vector


    VecScatterBegin(scatter,r,RHS,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scatter,r,RHS,INSERT_VALUES,SCATTER_FORWARD);

    //VecView(RHS,PETSC_VIEWER_STDOUT_WORLD);

    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);


    // PRINT OUT RHS
   /*
    Vec local_RHS;
    ierr = DMCreateLocalVector(da, &local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);

    PetscScalar **local_RHS_arr;
    DMDAVecGetArray(da, local_RHS, &local_RHS_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS, &local_RHS_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);

    VecDestroy(&local_RHS);
    */


    return ierr;
}


/* Function: setRHS_L
 * ------------------
 *
 */
PetscErrorCode setRHS_L(Vec &RHS, const Vec &l, const PetscScalar &h_11,
        const PetscScalar alpha_L, const DM &da) {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscErrorCode ierr = 0;
    PetscInt m, n, mStart, nStart, j, i, M = NULL, N = NULL, listart, liend;
    PetscScalar v;
    DMDAGetInfo(da, NULL, &M, &N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);

    ierr = VecGetOwnershipRange(l,&listart,&liend);CHKERRQ(ierr);
    for(i = listart; i < liend; i++) {
        ierr = VecGetValues(l,1,&i,&v);CHKERRQ(ierr);
        v = v * h_11 * alpha_L;
        ierr = VecSetValues(l,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(l);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(l);CHKERRQ(ierr);

    //ierr = VecView(l,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // VecScatter routine! IMPORTANT!

    VecScatter scatter; // scatter context
    IS from,to; // index sets that define the scatter

    //int idx_from[] = {0,1,2,3,4}, idx_to[] = {0,11,22,33,44};

    int idx_from[M], idx_to[M];
    for(i = 0; i < M; i++) {
        idx_from[i] = i;
    }
    for(i = 0; i < M; i++) {
        idx_to[i] = i;
    }


    AO ao;
    DMDAGetAO(da,&ao);
    AOApplicationToPetsc(ao,M,idx_to);

    ISCreateGeneral(PETSC_COMM_SELF,M,idx_from,PETSC_COPY_VALUES,&from);
    ISCreateGeneral(PETSC_COMM_SELF,M,idx_to,PETSC_COPY_VALUES,&to);
    VecScatterCreate(l,from,RHS,to,&scatter); // gx = source vector, gy = destination vector


    VecScatterBegin(scatter,l,RHS,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scatter,l,RHS,INSERT_VALUES,SCATTER_FORWARD);

    //VecView(RHS,PETSC_VIEWER_STDOUT_WORLD);

    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);


    // PRINT OUT RHS
    /*
    Vec local_RHS;
    ierr = DMCreateLocalVector(da, &local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);

    PetscScalar **local_RHS_arr;
    DMDAVecGetArray(da, local_RHS, &local_RHS_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS, &local_RHS_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);

    VecDestroy(&local_RHS);
    */

    return ierr;
}


/* Function: setRHS_B
 * ------------------
 *
 */
PetscErrorCode setRHS_B(Vec &RHS, const Vec &b, const PetscScalar &h_11,
        const PetscScalar alpha_B, const DM &da) {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscErrorCode ierr = 0;
    PetscInt m, n, mStart, nStart, j, i, M = NULL, N = NULL, bistart, biend;
    PetscScalar v;
    DMDAGetInfo(da, NULL, &M, &N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);

    ierr = VecGetOwnershipRange(b,&bistart,&biend);CHKERRQ(ierr);
    for(i = bistart; i < biend; i++) {
        ierr = VecGetValues(b,1,&i,&v);CHKERRQ(ierr);
        v = v * h_11 * alpha_B;
        ierr = VecSetValues(b,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

    //ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // VecScatter routine! IMPORTANT!

    VecScatter scatter; // scatter context
    IS from,to; // index sets that define the scatter

    //int idx_from[] = {0,1,2,3,4}, idx_to[] = {0,11,22,33,44};

    int idx_from[N], idx_to[N];
    for(i = 0; i < N; i++) {
        idx_from[i] = i;
    }
    for(i = 0; i < N; i++) {
        idx_to[i] = i * M + (M-1);
    }


    AO ao;
    DMDAGetAO(da,&ao);
    AOApplicationToPetsc(ao,N,idx_to);

    ISCreateGeneral(PETSC_COMM_SELF,N,idx_from,PETSC_COPY_VALUES,&from);
    ISCreateGeneral(PETSC_COMM_SELF,N,idx_to,PETSC_COPY_VALUES,&to);
    VecScatterCreate(b,from,RHS,to,&scatter); // gx = source vector, gy = destination vector


    VecScatterBegin(scatter,b,RHS,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scatter,b,RHS,INSERT_VALUES,SCATTER_FORWARD);

    //VecView(RHS,PETSC_VIEWER_STDOUT_WORLD);

    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);


    // PRINT OUT RHS
    /*
    Vec local_RHS;
    ierr = DMCreateLocalVector(da, &local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);

    PetscScalar **local_RHS_arr;
    DMDAVecGetArray(da, local_RHS, &local_RHS_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS, &local_RHS_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);

    VecDestroy(&local_RHS);
    */

    return ierr;
}


/* Function: setRHS_T
 * --------------
 *
 */
PetscErrorCode setRHS_T(Vec &RHS, const Vec &g, const PetscScalar &h_11,
        const PetscScalar alpha_T, const DM &da) {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscErrorCode ierr = 0;
    PetscInt m, n, mStart, nStart, j, i, M = NULL, N = NULL, gistart, giend;
    PetscScalar v;
    DMDAGetInfo(da, NULL, &M, &N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    DMDAGetCorners(da, &mStart, &nStart, 0, &m, &n, 0);

    ierr = VecGetOwnershipRange(g,&gistart,&giend);CHKERRQ(ierr);
    for(i = gistart; i < giend; i++) {
        ierr = VecGetValues(g,1,&i,&v);CHKERRQ(ierr);
        v = (v + i) * (1/h_11) * alpha_T;
        //v = (double) i;
        ierr = VecSetValues(g,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(g);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(g);CHKERRQ(ierr);

    //ierr = VecView(g,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // VecScatter routine! IMPORTANT!

    VecScatter scatter; // scatter context
    IS from,to; // index sets that define the scatter

    //int idx_from[] = {0,1,2,3,4}, idx_to[] = {0,11,22,33,44};

    int idx_from[N], idx_to[N];
    for(i = 0; i < N; i++) {
        idx_from[i] = i;
    }
    for(i = 0; i < N; i++) {
        idx_to[i] = i * M;
    }


    AO ao;
    DMDAGetAO(da,&ao);
    AOApplicationToPetsc(ao,N,idx_to);

    ISCreateGeneral(PETSC_COMM_SELF,N,idx_from,PETSC_COPY_VALUES,&from);
    ISCreateGeneral(PETSC_COMM_SELF,N,idx_to,PETSC_COPY_VALUES,&to);
    VecScatterCreate(g,from,RHS,to,&scatter); // gx = source vector, gy = destination vector


    VecScatterBegin(scatter,g,RHS,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scatter,g,RHS,INSERT_VALUES,SCATTER_FORWARD);

    //VecView(RHS,PETSC_VIEWER_STDOUT_WORLD);

    ISDestroy(&from);
    ISDestroy(&to);
    VecScatterDestroy(&scatter);


    // PRINT OUT RHS
    /*
    Vec local_RHS;
    ierr = DMCreateLocalVector(da, &local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);

    PetscScalar **local_RHS_arr;
    DMDAVecGetArray(da, local_RHS, &local_RHS_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS, &local_RHS_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);

    VecDestroy(&local_RHS);
    */


    return ierr;
}

/* Function: MyMatMult
 * -------------------
 *
 *
 */
PetscErrorCode MyMatMult() {
    PetscErrorCode ierr = 0;
    

    return ierr;
}

/* Function: Solve_Linear_Equation
 * -------------------------------
 *
 */
PetscErrorCode Solve_Linear_Equation() {
    PetscErrorCode ierr = 0;

    PetscInt num_x_pts = NUM_X_PTS, num_y_pts = NUM_Y_PTS;
    PetscScalar x_min = X_MINIMUM, x_max = X_MAXIMUM, y_max = Y_MAXIMUM, y_min = Y_MINIMUM;
    PetscInt i = 0, j = 0, mStart = 0, m = 0, nStart = 0, n = 0;
    PetscScalar v = 0, mult_x = 0, mult_y = 0, alpha_T = -1, alpha_B = -2, alpha_L = -3, alpha_R = -4, h_11 = 2;

    Vec grid = NULL, RHS = NULL, g = NULL, b = NULL, l = NULL, r = NULL,
        local_grid = NULL, local_RHS = NULL, local_coords = NULL;

    PetscInt num_grid_entries = (num_x_pts) * (num_y_pts); // total number of entries on the grid
    mult_x = ((PetscReal)(x_max - x_min))/(num_x_pts - 1); // spacing in the x direction
    mult_y = ((PetscReal)(y_max - y_min))/(num_y_pts - 1); // spacing in the y direction

    DM            da;
    DM            cda;
    DMDACoor2d **coords; // 2D array containing x and y data members

    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
            DMDA_STENCIL_BOX, num_x_pts, num_y_pts, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da); CHKERRQ(ierr);
    DMDASetUniformCoordinates(da, x_min, x_max, y_min, y_max, 0.0, 0.0);
    DMGetCoordinateDM(da, &cda);
    DMGetCoordinatesLocal(da, &local_coords);
    DMDAVecGetArray(cda, local_coords, &coords);
    DMDAGetCorners(cda, &mStart, &nStart, 0, &m, &n, 0);

    ierr = DMCreateGlobalVector(da,&grid);
    ierr = VecDuplicate(grid, &RHS);CHKERRQ(ierr);

  //FIXME
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // FILL GRID
    ierr = DMCreateLocalVector(da, &local_grid);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, grid, INSERT_VALUES, local_grid);CHKERRQ(ierr);

    PetscScalar **local_grid_arr;
    DMDAVecGetArray(da, local_grid, &local_grid_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          local_grid_arr[j][i] = (coords[j][i].x);
      }
    }
    DMDAVecRestoreArray(da, local_grid, &local_grid_arr);

    ierr = DMLocalToGlobalBegin(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_grid, INSERT_VALUES, grid);CHKERRQ(ierr);

    //FILL RHS
    ierr = DMCreateLocalVector(da, &local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS);CHKERRQ(ierr);

    PetscScalar **local_RHS_arr;
    DMDAVecGetArray(da, local_RHS, &local_RHS_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
              local_RHS_arr[j][i] = 0;
  //        PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS, &local_RHS_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS, INSERT_VALUES, RHS);CHKERRQ(ierr);

    // FILL g
    VecCreate(PETSC_COMM_WORLD,&g);
    VecSetSizes(g,PETSC_DECIDE,num_y_pts);
    VecSetFromOptions(g);
    PetscObjectSetName((PetscObject) g, "g");
    VecSet(g,1.0);

    // FILL b
    VecCreate(PETSC_COMM_WORLD,&b);
    VecSetSizes(b,PETSC_DECIDE,num_y_pts);
    VecSetFromOptions(b);
    PetscObjectSetName((PetscObject) b, "b");
    VecSet(b,1.0);

    // FILL l
    VecCreate(PETSC_COMM_WORLD,&l);
    VecSetSizes(l,PETSC_DECIDE,num_x_pts);
    VecSetFromOptions(l);
    PetscObjectSetName((PetscObject) l, "l");
    VecSet(l,1.0);

    // FILL r
    VecCreate(PETSC_COMM_WORLD,&r);
    VecSetSizes(r,PETSC_DECIDE,num_x_pts);
    VecSetFromOptions(r);
    PetscObjectSetName((PetscObject) r, "r");
    VecSet(r,1.0);

    ierr = setRHS_T(RHS, g, h_11, alpha_T, da);CHKERRQ(ierr);
    ierr = setRHS_B(RHS, b, h_11, alpha_B, da);CHKERRQ(ierr);
    ierr = setRHS_L(RHS, l, h_11, alpha_L, da);CHKERRQ(ierr);
    ierr = setRHS_R(RHS, r, h_11, alpha_R, da);CHKERRQ(ierr);

    //PRINT RHS
    /*
    Vec local_RHS_print;
    ierr = DMCreateLocalVector(da, &local_RHS_print);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, RHS, INSERT_VALUES, local_RHS_print);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, RHS, INSERT_VALUES, local_RHS_print);CHKERRQ(ierr);

    PetscScalar **local_RHS_print_arr;
    DMDAVecGetArray(da, local_RHS_print, &local_RHS_print_arr);
    for (j = nStart; j < nStart + n; j++) {
      for (i = mStart; i < mStart + m; i++) {
          PetscPrintf(PETSC_COMM_SELF, "%i: [%i][%i] = %g\n", rank, j, i, local_RHS_print_arr[j][i]);
      }
    }
    DMDAVecRestoreArray(da, local_RHS_print, &local_RHS_print_arr);

    ierr = DMLocalToGlobalBegin(da, local_RHS_print, INSERT_VALUES, RHS);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da, local_RHS_print, INSERT_VALUES, RHS);CHKERRQ(ierr);

    VecDestroy(&local_RHS_print);
    // writeVec(RHS, "RHS");
    */

    VecDestroy(&grid);
    VecDestroy(&RHS);
    VecDestroy(&g);
    VecDestroy(&b);
    VecDestroy(&l);
    VecDestroy(&r);
    VecDestroy(&local_grid);
    VecDestroy(&local_RHS);
    DMDestroy(&da);
    DMDestroy(&cda);

    return ierr;
}

/* Function: main
 * --------------
 *
 */
int main(int argc,char **argv)
{
  PetscErrorCode ierr = 0;
  PetscInitialize(&argc,&argv,(char*)0,NULL);
  //ierr = MMSTest();CHKERRQ(ierr);
  ierr = Solve_Linear_Equation();CHKERRQ(ierr);
  PetscFinalize();
  return ierr;
}
