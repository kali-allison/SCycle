#include "domain.hpp"

using namespace std;

// creates a domain object
int main(int argc, char **argv) {

  PetscErrorCode ierr = 0;
  PetscInitialize(&argc, &argv, NULL, NULL);

  const char* inputFile = args[1];
  Domain d(inputFile);

  PetscFinalize();
  return ierr;
}
