#include <petscsys.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
  PetscViewer viewer, viewer2;
  Vec vec, vec_read;
  PetscInt n = 5;
  PetscBool flg;
  PetscErrorCode ierr;
  
  PetscInitialize(&argc, &argv, NULL, NULL);
  ierr = VecCreate(PETSC_COMM_WORLD, &vec); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "vec");
  ierr = VecSetSizes(vec, PETSC_DECIDE, n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(vec);
  ierr = VecSet(vec, 1.0); CHKERRQ(ierr);
  
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "vecHDF5.h5", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);
  ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_FALSE); CHKERRQ(ierr);
  ierr = VecView(vec, viewer); CHKERRQ(ierr);

  // read vector
  ierr = VecCreate(PETSC_COMM_WORLD, &vec_read);
  ierr = PetscObjectSetName((PetscObject) vec_read, "vec");
  ierr = VecSetSizes(vec_read, PETSC_DECIDE, n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(vec_read);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "vecHDF5.h5", FILE_MODE_READ, &viewer2); CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer2);
  ierr = VecLoad(vec_read, viewer2); CHKERRQ(ierr);

  VecEqual(vec, vec_read, &flg);
  if (!flg) {
    PetscPrintf(PETSC_COMM_WORLD, "Error: vectors are not equal.");
    VecView(vec, PETSC_VIEWER_STDOUT_WORLD);
    VecView(vec_read, PETSC_VIEWER_STDOUT_WORLD);
  }

  PetscViewerDestroy(&viewer);
  PetscViewerDestroy(&viewer2);
  VecDestroy(&vec_read);
  VecDestroy(&vec);
  
  PetscFinalize();
  return ierr;
}
