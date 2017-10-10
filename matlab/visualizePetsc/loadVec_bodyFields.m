function output = loadVec_bodyFields(sourceDir,fieldName,strideT,sT,eT,sY,eY,Ny)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reads a set of vectors from a PETSc file, storing them in a matrix.
%
% This function assumes the file contains a 2d field stored columnwise, ie
% the original shape of the data can be recovered with reshape(data,Ny,Nx).
% Each instance of this field in the file, presumably corresponding to the
% field's value at a specific time step, is referred to below as a time
% step.
%
% The range of time steps can be selected with sT and eT (start and end
% indices for the time steps, respectively), and the number of steps to
% skip is set with strideT. The fields that are loaded correspond to
% sT:strideT:eT in MATLAB's usual syntax. Indices begin with 1.
%
% To read in set of transects in the y-direction, set sY and eY, the
% start and end indices within the field. Skipping indices within this
% range is not supported.
%
% Required inputs:
%   sourceDir = path to directory containing file, stored as string
%   fieldName = name of file in sourceDir, stored as string
%
% Optional inputs:  default value
%   strideT   = 1
%   sT        = 1
%   eT        = Inf (read to end of file)
%   sY        = 1
%   eY        = Inf (read to bottom of field)
%   Ny        = # of rows in file
%
%
%
% Written by Kali Allison, 09/17/2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output = []; % always return something

% process optional input arguments
if nargin < 3
  strideT = 1;
end
if nargin < 4
  sT = 1;
end
if nargin < 5
  eT = Inf;
end
if nargin < 6
  sY = 1;
end
if nargin < 7
  eY = Inf;
end
if nargin < 8
  Ny = Inf;
end


% check that input's are valid
if eT < sT
  display('Error: final time step index must be > than initial index!')
  return
elseif strideT < 1 || rem(strideT,1)~=0
  display('Error: stride must be a positive whole number!')
  return
elseif eY < sY
  display('Error: final time step index must be > than initial index!')
  return
end



headerPrecision = 'int32';
headerSize = 8; % in bytes, this is the size of 2 int32's
dataPrecision = 'float64';


% open file
inFile = strcat(sourceDir,fieldName);
fd = fopen(inFile,'r','ieee-be');

% process header information
header = double(fread(fd,1,headerPrecision)); % check that type is vector
if header ~= 1211214 % Petsc Vec Object
  display('Error: field is not a vector!')
  return
end


numRows = double(fread(fd,1,headerPrecision)); % number of rows in file

% check more inputs
if (Ny > numRows), Ny = numRows; end
if (mod(numRows,Ny) > 0)
  display('Error in loadVec_bodyFields: Ny must evenly divide into the total # of rows in the file.')
  return
end
if (Ny < eY), eY = Ny; end
if (Ny < sY), sY = Ny; end


m = eY-sY+1; % number of rows to grab
n = floor((eT-sT)/strideT) + 1; % number of columns to grab


% Note: the 8 in the following rows comes from the size of a float64 in
% bytes.
colSize = headerSize + numRows*8; % # of bytes in a column

% skip rows between desired values for a specific time step
skipSize1 = (Ny-m)*8; % bytes between floats to grab

% skip time steps
skipSize2 = headerSize + (strideT-1)*colSize; % bytes between floats to grab


offset = (sT-1)*colSize + (sY-1)*8 + headerSize;
fseek(fd,offset,'bof'); % go to start of data to be read

% read from file
M = (numRows/Ny)*m;
output = zeros(M,n);
for ind = 1:n
  tmp = fread(fd,[M,1],strcat(num2str(m),'*float64'),skipSize1);
  if (isempty(tmp)), break, end
  output(:,ind) = tmp;
  fseek(fd,skipSize2,'cof');
end


% close file
fclose(fd);

if n>ind
  output = output(:,1:ind-1);
end

end

