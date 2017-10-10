function output = loadVec(sourceDir,fieldName,stride,startInd,endInd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reads a set of vectors from a PETSc file, storing them in a matrix.
% Required inputs:
%   sourceDir = path to directory containing file, stored as string
%   fieldName = name of file in sourceDir, stored as string
%
% Optional inputs:  default value
%   stride   = 1
%   startInd = 1
%   endInd   = Inf (read to end of file)
%
%   The indices read from the file are (startInd:stride:endInd), with
%   1 as the first index, following Matlab convention.
%   
%
% Written by Kali Allison, 03/24/2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output = []; % always return something

% process optional input arguments
if nargin < 3
  stride = 1;
end
if nargin < 4
  startInd = 1;
end
if nargin < 5
  endInd = Inf;
end

% check that input's are valid
if endInd < startInd
  display('Error: final index must be > than initial index!')
  return
elseif stride < 1 || rem(stride,1)~=0
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


m = double(fread(fd,1,headerPrecision)); % number of rows in file
dataSize = m*8; % in bytes, each float64 is 8 bytes
colSize = headerSize + dataSize; % # of bytes in a column
n = floor((endInd-startInd+1)/stride); % number of columns to grab
skipSize = headerSize + (headerSize+dataSize)*(stride-1); % bytes between columns to grab


% jump ahead to startInd
offset = (startInd-1)*colSize + headerSize;
fseek(fd,offset,'bof');


output = fread(fd,[m,n],strcat(num2str(m),'*float64'),skipSize);


% close file
fclose(fd);

end
