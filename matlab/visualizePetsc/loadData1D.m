function output = loadData1D(output,sourceDir,stride,startInd,endInd)
% loads 1D fields

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

if startInd < 1, startInd = 1, end

% check that input's are valid
if endInd < startInd
  display('Error: final index must be > than initial index.')
  fprintf('initial indextInd = %i, final index = %i.\n',startInd,endInd)
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



display(strcat('loading 1D data:',sourceDir))

% output.sourceDir = sourceDir;

output.loadStride1D = stride;
output.loadStartInd1D = startInd;
output.loadEndInd1D = endInd;

time = load(strcat(sourceDir,'time.txt'));
fprintf('# of time steps: %i\n',length(time));


output.time = time( startInd:stride:min(endInd,length(time)) );
% if length(time) < endInd
%   keyboard
endInd = length(output.time) * stride + startInd - 1;
% end
output.dt = (output.time(2:end) - output.time(1:end-1))/output.dom.stride1D;


% load pressure
if exist(strcat(sourceDir,'fault_hydr_p'),'file') == 2
  output.p = loadVec(sourceDir,'fault_hydr_p',stride,startInd,endInd);
  display('   finished loading pressure')
end
if exist(strcat(sourceDir,'fault_hydr_sNEff'),'file') == 2
  output.sNEff = loadVec(sourceDir,'fault_hydr_sNEff',stride,startInd,endInd);
  display('   finished loading pressure')
end

% test
if exist(strcat(sourceDir,'SATL'),'file') == 2
  output.SATL = loadVec(sourceDir,'SATL',stride,startInd,endInd);
  display('   finished SATL')
end

if exist(strcat(sourceDir,'slipVel'),'file') == 2
  output.slipVel = loadVec(sourceDir,'slipVel',stride,startInd,endInd);
  display('   finished loading slip velocity')
elseif exist(strcat(sourceDir,'velPlus'),'file') == 2
  velPlus = loadVec(sourceDir,'velPlus',stride,startInd,endInd);
  output.slipVel = velPlus;
  display('   finished loading velocity: + side')
  if exist(strcat(sourceDir,'velMinus'),'file') == 2
    output.velPlus = velPlus;
    output.velMinus = loadVec(sourceDir,'velMinus',stride,startInd,endInd);
    display('   finished loading velocity: - side')
    output.slipVel = output.velPlus - output.velMinus;
  end
end

if exist(strcat(sourceDir,'slip'),'file') == 2
  output.slip = loadVec(sourceDir,'slip',stride,startInd,endInd);
  display('   finished loading slip on fault')
elseif exist(strcat(sourceDir,'uPlus'),'file') == 2
  uPlus = loadVec(sourceDir,'uPlus',stride,startInd,endInd);
  output.slip = uPlus;
  display('   finished loading particle displacement on + side of fault')
  if exist(strcat(sourceDir,'uMinus'),'file') == 2
    output.uPlus = uPlus;
    output.uMinus = loadVec(sourceDir,'uMinus',stride,startInd,endInd);
    display('   finished loading particle displacement on - side of fault')
    output.slip = output.uPlus - output.uMinus;
  end
end


output.tauQSP = loadVec(sourceDir,'tauQSP',stride,startInd,endInd);
display('   finished loading shear stress')


if (any(size(output.tauQSP) ~= size(output.slipVel)))
  display('Error: tauQS and slipVel are different sizes')
  output
end
output.tauP = output.tauQSP - bsxfun(@times,output.slipVel,(output.zPlus)/2);




% load state
if exist(strcat(sourceDir,'state'),'file') == 2
  output.state = loadVec(sourceDir,'state',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'psi'),'file') == 2
  output.psi = loadVec(sourceDir,'psi',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'theta'),'file') == 2
  output.theta = loadVec(sourceDir,'theta',stride,startInd,endInd);
end

% load surface displacement
if exist(strcat(sourceDir,'surfDispPlus'),'file') == 2
  output.surfU = loadVec(sourceDir,'surfDispPlus',stride,startInd,endInd);
  temp = (output.surfU(:,2:end) - output.surfU(:,1:end-1));
  
%   if length(output.time)>2
%   dt = output.time(2:end) - output.time(1:end-1);
%   output.surfVel = bsxfun(@rdivide,temp,dt');
%   end
end


if size(output.slipVel,1)>1
  output.maxVel = max(output.slipVel);
  output.faultSurfVel = output.slipVel(1,:);
  output.maxTauPlus = max(output.tauP);
  output.maxVel = max(output.slipVel);
else
  output.surfVelPlus = output.slipVel;
  output.maxTauP = max(abs(output.tauP));
  
  output.maxVel = output.slipVel;
end

% load boundary conditions for debugging
if exist(strcat(sourceDir,'bcR'),'file') == 2
  output.bcR = loadVec(sourceDir,'bcR',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'bcL'),'file') == 2
  output.bcL = loadVec(sourceDir,'bcL',stride,startInd,endInd);
end


% if using flash heating
if exist(strcat(sourceDir,'fault_T'),'file') == 2
  output.fault_T = loadVec(sourceDir,'fault_T',stride,startInd,endInd);
  display('   finished loading fault temperature')
end

if exist(strcat(sourceDir,'surfaceHeatFlux'),'file') == 2
  output.surfaceHeatFlux = loadVec(sourceDir,'surfaceHeatFlux',stride,startInd,endInd);
  display('   finished loading surface heat flux')
end


% % test spin up stuff
% if exist(strcat(sourceDir,'init1_bcL'),'file') == 2
%   output.init1_bcL = loadVec(sourceDir,'init1_bcL',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init1_bcR'),'file') == 2
%   output.init1_bcR = loadVec(sourceDir,'init1_bcR',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init1_bcT'),'file') == 2
%   output.init1_bcT = loadVec(sourceDir,'init1_bcT',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init1_bcB'),'file') == 2
%   output.init1_bcB = loadVec(sourceDir,'init1_bcB',stride,startInd,endInd);
% end
% 
% if exist(strcat(sourceDir,'init2_bcL'),'file') == 2
%   output.init2_bcL = loadVec(sourceDir,'init2_bcL',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init2_bcR'),'file') == 2
%   output.init2_bcR = loadVec(sourceDir,'init2_bcR',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init2_bcT'),'file') == 2
%   output.init2_bcT = loadVec(sourceDir,'init2_bcT',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init2_bcB'),'file') == 2
%   output.init2_bcB = loadVec(sourceDir,'init2_bcB',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'init2_u'),'file') == 2
%   output.init2_u = loadVec(sourceDir,'init2_u',stride,startInd,endInd);
%   output.init2_u = reshape(output.init2_u,output.dom.Nz,output.dom.Ny);
% end


% for MMS  tests
if exist(strcat(sourceDir,'mms_bcL'),'file') == 2
  output.mms_bcL = loadVec(sourceDir,'mms_bcL',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'mms_bcR'),'file') == 2
  output.mms_bcR = loadVec(sourceDir,'mms_bcR',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'mms_bcT'),'file') == 2
  output.mms_bcT = loadVec(sourceDir,'mms_bcT',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'mms_bcB'),'file') == 2
  output.mms_bcB = loadVec(sourceDir,'mms_bcB',stride,startInd,endInd);
end


end