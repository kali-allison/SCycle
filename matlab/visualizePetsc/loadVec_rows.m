function output = loadVec_rows(sourceDir,fieldName,strideC,sC,eC,sR,eR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reads a set of vectors from a PETSc file, storing them in a matrix.
%
% This function thinks of the data as a 2d array stored columnwise. The
% range of columns can be selected with sC and eC (start and end indices
% for the columns, respectivey), and the number of columns to skip between
% is set with strideC.
%
% To read in only a subset of rows, set sR and eR. It is not possible to
% skip rows.
%
% Required inputs:
%   sourceDir = path to directory containing file, stored as string
%   fieldName = name of file in sourceDir, stored as string
%
% Optional inputs:  default value
%   strideC   = 1
%   sC       = 1
%   eC       = Inf (read to end of file)
%   sR       = 1
%   eR       = Inf (read to bottom of array)
%
%  For example, the data consists of 1 row, then the indices read from the
%  file are (sC:strideC:eC), with 1 as the first index, following Matlab
%  convention.
%   
%
% Written by Kali Allison, 09/12/2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output = []; % always return something

% process optional input arguments
if nargin < 3
  strideC = 1;
end
if nargin < 4
  sC = 1;
end
if nargin < 5
  eC = Inf;
end
if nargin < 6
  sR = 1;
end
if nargin < 7
  eR = Inf;
end



% check that input's are valid
if eC < sC
  display('Error: final index must be > than initial index!')
  return
elseif strideC < 1 || rem(strideC,1)~=0
  display('Error: stride must be a positive whole number!')
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




% new code
numRows = double(fread(fd,1,headerPrecision)); % number of rows in file
if (eR > (numRows-sR+1)), eR = numRows - sR + 2; end

m = eR-sR+1; % number of rows to grab
% m = 1;
n = floor((eC-sC+1)/strideC); % number of columns to grab


% Note: the 8 in the following rows comes from the size of a float64 in
% bytes.
colSize = headerSize + numRows*8; % # of bytes in a column
skipSize = headerSize + ((numRows-m)*8) + (strideC-1)*colSize; % bytes between floats to grab


offset = (sC-1)*colSize + (sR-1)*8 + headerSize;
fseek(fd,offset,'bof'); % go to start of data to be read


% read from file
output = fread(fd,[m,n],strcat(num2str(m),'*float64'),skipSize);


% close file
fclose(fd);

end

