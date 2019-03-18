#ifndef DOMAIN_HPP_INCLUDED
#define DOMAIN_HPP_INCLUDED

#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

#include <petscts.h>
#include <petscdmda.h>
#include <petscdm.h>

using namespace std;

class Domain {

private:  
  // disable default copy constructor and assignment operator
  Domain(const Domain &that);
  Domain& operator=(const Domain &rhs);

public:
  string _sbpType;
  PetscInt _Ny,_Nz,_Ly,_Lz;
  Vec   _q,_r,_y,_z,_y0,_z0;
  PetscScalar _dq,_dr;
  PetscScalar _bCoordTrans;
  map<string, VecScatter> _scatters;
  
  Domain();
  ~Domain();
  PetscErrorCode setFields();
  PetscErrorCode setScatters();  
};

#endif
