function output = loadData2D(output,sourceDir,stride,startInd,endInd)
% loads 2D fields

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

if startInd < 1, startInd = 1; end

% check that input's are valid
if endInd < startInd
  display('Error: final index must be > than initial index.')
  return
elseif stride < 1 || rem(stride,1)~=0
  display('Error: stride must be a positive whole number.')
  return
end


% add PETSc routines to MATLAB path
if isdir('/Users/kallison/petsc-3.7.3/share/petsc/matlab/')
  addpath('/Users/kallison/petsc-3.7.3/share/petsc/matlab/');
else
  display('Cannot find directory containing PETSc loading functions!');
end

output.loadStride2D = stride;
output.loadStartInd2D = startInd;
output.loadEndInd2D = endInd;

display(strcat('loading data 2D:',sourceDir))


if exist(strcat(sourceDir,'time2D.txt'),'file') == 2
  time2D = load(strcat(sourceDir,'time2D.txt'));
  output.time2D = time2D( startInd:stride:min(endInd,length(time2D)) );
  endInd = length(output.time2D) * stride + startInd - 1;
end

if exist(strcat(sourceDir,'u'),'file') == 2
  fprintf('    loading u')
  u = loadVec(sourceDir,'u',stride,startInd,endInd);
  output.u = squeeze(reshape(u,output.dom.Nz,output.dom.Ny,size(u,2)));
  fprintf('...finished.\n')
end

if exist(strcat(sourceDir,'gxy'),'file') == 2
  fprintf('    loading gxy')
  gxy = loadVec(sourceDir,'gxy',stride,startInd,endInd);
  output.Gxy = squeeze(reshape(gxy,output.dom.Nz,output.dom.Ny,size(gxy,2)));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'gxz'),'file') == 2
  fprintf('    loading gxz')
  gxz = loadVec(sourceDir,'gxz',stride,startInd,endInd);
  output.Gxz = squeeze(reshape(gxz,output.dom.Nz,output.dom.Ny,size(gxz,2)));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'gTxy'),'file') == 2
  fprintf('    loading gTxy')
  gTxy = loadVec(sourceDir,'gTxy',stride,startInd,endInd);
  output.GTxy = reshape(gTxy,output.dom.Nz,output.dom.Ny,size(gTxy,2));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'gTxz'),'file') == 2
  fprintf('    loading gTxy')
  gTxz = loadVec(sourceDir,'gTxz',stride,startInd,endInd);
  output.GTxz = reshape(gTxz,output.dom.Nz,output.dom.Ny,size(gTxz,2));
  fprintf('...finished.\n')
end

if exist(strcat(sourceDir,'sxy'),'file') == 2
  fprintf('    loading sxy')
  sxy = loadVec(sourceDir,'sxy',stride,startInd,endInd);
  output.Sxy = squeeze(reshape(sxy,output.dom.Nz,output.dom.Ny,size(sxy,2)));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'sxz'),'file') == 2
  fprintf('    loading sxz')
  sxz = loadVec(sourceDir,'sxz',stride,startInd,endInd);
  output.Sxz = squeeze(reshape(sxz,output.dom.Nz,output.dom.Ny,size(sxz,2)));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'effVisc'),'file') == 2
  fprintf('    loading effective viscosity')
  effVisc = loadVec(sourceDir,'effVisc',stride,startInd,endInd);
  output.EffVisc = squeeze(reshape(effVisc,output.dom.Nz,output.dom.Ny,size(effVisc,2)));
  fprintf('...finished.\n')
end

if exist(strcat(sourceDir,'T'),'file') == 2
  fprintf('    loading T0 (background)')
  T0 = loadVec(sourceDir,'T0',stride,startInd,endInd);
  output.T0 = reshape(T0,output.dom.Nz,output.dom.Ny,size(T0,2));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'dT'),'file') == 2
  fprintf('    loading dT (perturbation)')
  dT = loadVec(sourceDir,'dT',stride,startInd,endInd);
  output.dT = reshape(dT,output.dom.Nz,output.dom.Ny,size(dT,2));
  fprintf('...finished.\n')
end
if exist(strcat(sourceDir,'heatFlux'),'file') == 2
  fprintf('    loading heat flux dT/dy')
  heatFlux = loadVec(sourceDir,'heatFlux',stride,startInd,endInd);
  output.heatFlux = reshape(heatFlux,output.dom.Nz,output.dom.Ny,size(dT,2));
  fprintf('...finished.\n')
end
% if exist(strcat(sourceDir,'he_bcR'),'file') == 2
%   output.he_bcR = loadVec(sourceDir,'he_bcR',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'he_bcT'),'file') == 2
%   output.he_bcT = loadVec(sourceDir,'he_bcT',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'he_bcL'),'file') == 2
%   output.he_bcL = loadVec(sourceDir,'he_bcL',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'he_bcB'),'file') == 2
%   output.he_bcB = loadVec(sourceDir,'he_bcB',stride,startInd,endInd);
% end



end