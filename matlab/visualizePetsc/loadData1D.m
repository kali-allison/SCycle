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

if startInd < 1, startInd = 1; end

% check that input's are valid
if endInd < startInd
  display('Error: final index must be > than initial index.')
  fprintf('initial indextInd = %i, final index = %i.\n',startInd,endInd)
  return
elseif stride < 1 || rem(stride,1)~=0
  display('Error: stride must be a positive whole number.')
  return
end

display(strcat('loading 1D data:',sourceDir))

% output.sourceDir = sourceDir;

output.load.loadStride1D = stride;
output.load.loadStartInd1D = startInd;
output.load.loadEndInd1D = endInd;

time = load(strcat(sourceDir,'time.txt'));
fprintf('# of time steps: %i\n',length(time));


output.time = time( startInd:stride:min(endInd,length(time)) );
% if length(time) < endInd
%   keyboard
endInd = length(output.time) * stride + startInd - 1;
% end
% output.dt = (output.time(2:end) - output.time(1:end-1))/output.dom.stride1D;


% load pressure
if exist(strcat(sourceDir,'fault_hydr_p'),'file') == 2
  output.p = loadVec(sourceDir,'fault_hydr_p',stride,startInd,endInd);
  display('   finished loading pressure')
end
if exist(strcat(sourceDir,'fault_hydr_sNEff'),'file') == 2
  output.sNEff = loadVec(sourceDir,'fault_hydr_sNEff',stride,startInd,endInd);
  display('   finished loading pressure')
end

if exist(strcat(sourceDir,'slipVel'),'file') == 2
  output.slipVel = loadVec(sourceDir,'slipVel',stride,startInd,endInd);
  display('   finished loading slip velocity')
end

if exist(strcat(sourceDir,'slip'),'file') == 2
  output.slip = loadVec(sourceDir,'slip',stride,startInd,endInd);
  display('   finished loading slip on fault')
end

if exist(strcat(sourceDir,'tauQSP'),'file') == 2
  output.tauQSP = loadVec(sourceDir,'tauQSP',stride,startInd,endInd);
  display('   finished loading quasi-static shear stress')
end
if exist(strcat(sourceDir,'tauP'),'file') == 2
  output.tauP = loadVec(sourceDir,'tauP',stride,startInd,endInd);
  display('   finished loading shear stress')
end

% if (any(size(output.tauQSP) ~= size(output.slipVel)))
%   display('Error: tauQS and slipVel are different sizes')
%   output
% end
% output.tauP = output.tauQSP - bsxfun(@times,output.slipVel,(output.fault.imp)/2);
% output.tauP = output.tauQSP;



% load state variable
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
  output.maxTauP = max(output.tauP);
  output.maxVel = max(output.slipVel);
else
  output.maxTauP = max(abs(output.tauP));
  output.maxVel = output.slipVel;
end

% load boundary conditions for debugging
if exist(strcat(sourceDir,'bcR'),'file') == 2
  output.momBal.bcR = loadVec(sourceDir,'bcR',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'bcL'),'file') == 2
  output.momBal.bcL = loadVec(sourceDir,'bcL',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'bcT'),'file') == 2
  output.momBal.bcT = loadVec(sourceDir,'bcT',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'bcB'),'file') == 2
  output.momBal.bcB = loadVec(sourceDir,'bcB',stride,startInd,endInd);
end


% if using flash heating
if exist(strcat(sourceDir,'fault_T'),'file') == 2
  output.fault.T = loadVec(sourceDir,'fault_T',stride,startInd,endInd);
  display('   finished loading fault temperature')
end

if exist(strcat(sourceDir,'surfaceHeatFlux'),'file') == 2
  output.surfaceHeatFlux = loadVec(sourceDir,'surfaceHeatFlux',stride,startInd,endInd);
  display('   finished loading surface heat flux')
end


% test heat equation
if exist(strcat(sourceDir,'he_bcL'),'file') == 2
  output.he.bcL = loadVec(sourceDir,'he_bcL',stride,startInd,endInd);
end
if exist(strcat(sourceDir,'he_bcR'),'file') == 2
  output.he.bcR = loadVec(sourceDir,'he_bcR',stride,startInd,endInd);
end


% for MMS  tests
% if exist(strcat(sourceDir,'mms_bcL'),'file') == 2
%   output.mms_bcL = loadVec(sourceDir,'mms_bcL',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'mms_bcR'),'file') == 2
%   output.mms_bcR = loadVec(sourceDir,'mms_bcR',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'mms_bcT'),'file') == 2
%   output.mms_bcT = loadVec(sourceDir,'mms_bcT',stride,startInd,endInd);
% end
% if exist(strcat(sourceDir,'mms_bcB'),'file') == 2
%   output.mms_bcB = loadVec(sourceDir,'mms_bcB',stride,startInd,endInd);
% end

% if exist(strcat(sourceDir,'bcRShift'),'file') == 2
%   output.bcRShift = loadVec(sourceDir,'bcRShift',stride,startInd,endInd);
% end

end