function output = loadVecFromPetsc(sourceDir,fieldName,sourceSize)
% Loads a vector from a binary file created by PetSc.
%
% Inputs:
%          sourceDir = string containing source directory
%          fieldName = string containing name of file within source
%            directory
%
% Outputs:
%          out = matrix with 1 column per time step

if isdir('/Users/kallison/petsc-3.4.2-debug/bin/matlab/')
  addpath('/Users/kallison/petsc-3.4.2-debug/bin/matlab/');
elseif isdir('/usr/local/petsc-3.4.2_intel13_nodebug/bin/matlab');
    addpath('/usr/local/petsc-3.4.2_intel13_nodebug/bin/matlab');
else
  display('Cannot find directory containing PETSc loading functions!');
end
    

vec = PetscBinaryRead(strcat(sourceDir,fieldName),'cell',sourceSize);
output = cell2mat(vec);


end