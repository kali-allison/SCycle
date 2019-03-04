#include "domain.hpp"

using namespace std;

// creates a domain object
int main(int argc, char **argv) {

  PetscErrorCode ierr = 0;
  PetscInitialize(&argc, &argv, NULL, NULL);
  {
  Domain d;

  ierr = d.setFields(); CHKERRQ(ierr);
  ierr = d.setScatters(); CHKERRQ(ierr);
  }
  PetscFinalize();
  return ierr;
}
