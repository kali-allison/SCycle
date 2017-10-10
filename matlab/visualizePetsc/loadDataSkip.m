function output = loadDataSkip(sourceDir,stride,startInd,endInd)


% process optional input arguments
if nargin < 2
  stride = 1;
end
if nargin < 3
  startInd = 1;
end
if nargin < 4
  endInd = Inf;
end

% check that input's are valid
if endInd < startInd
  display('Error: final index must be > than initial index.')
  return
elseif stride < 1 || rem(stride,1)~=0
  display('Error: stride must be a positive whole number.')
  return
end




% add PETSc routines to MATLAB path
if isdir('/Users/kallison/petsc-3.4.2-debug/bin/matlab/')
  addpath('/Users/kallison/petsc-3.4.2-debug/bin/matlab/');
elseif isdir('/usr/local/petsc-3.4.2_intel13_nodebug/bin/matlab');
  addpath('/usr/local/petsc-3.4.2_intel13_nodebug/bin/matlab');
else
  display('Cannot find directory containing PETSc loading functions!');
end



display(strcat('loading data:',sourceDir))

output.sourceDir = sourceDir;

% scalar meta-data
% output.dom = load_struct(strcat(sourceDir,'parameters.txt'),' = ');
output.dom = load_struct(strcat(sourceDir,'domain.txt'),' = ');

output.yVec = 0:output.dom.dy:output.dom.Ly;
output.zVec = 0:output.dom.dz:output.dom.Lz;


% rate and state context
output.a = loadVec(sourceDir,'a');
output.b = loadVec(sourceDir,'b');


if exist(strcat(sourceDir,'Dc'),'file') == 2
  output.Dc = loadVec(sourceDir,'Dc');
end
output.zPlus = loadVec(sourceDir,'zPlus');
output.sigma_N = loadVec(sourceDir,'sigma_N');
output.muPlus = diag(PetscBinaryRead(strcat(sourceDir,'muPlus'),'cell'));
output.muPlusMat = diag(output.muPlus);
output.muPlus = reshape(output.muPlus,output.dom.Nz,output.dom.Ny);


if exist(strcat(sourceDir,'visc'),'file')==2
  output.visc = loadVec(sourceDir,'visc');
end

display('   finished loading rate and state context')

% heat equation parameters
if exist(strcat(sourceDir,'h'),'file') == 2
  output.h = loadVec(sourceDir,'h');
end
if exist(strcat(sourceDir,'c'),'file') == 2
  output.c = loadVec(sourceDir,'c');
end
if exist(strcat(sourceDir,'rho'),'file') == 2
  output.rho = loadVec(sourceDir,'rho');
end

% maxwell paramaters
output.visc = loadVec(sourceDir,'visc',1,1,1);

% power law parameters
if exist(strcat(sourceDir,'powerLawA'),'file') == 2
  output.powerLawA = loadVec(sourceDir,'powerLawA');
end
if exist(strcat(sourceDir,'powerLawB'),'file') == 2
  output.powerLawB = loadVec(sourceDir,'powerLawB');
end
if exist(strcat(sourceDir,'n'),'file') == 2
  output.n = loadVec(sourceDir,'n');
end


% time integration results
time = load(strcat(sourceDir,'time.txt'));
% if (length(time)>100), time=time(1:end-5); end
output.time = time( startInd:stride:min(endInd,length(time)) );
endInd = length(output.time) * stride + startInd - 1;


if exist(strcat(sourceDir,'bcR'),'file') == 2
  output.bcR = loadVec(sourceDir,'bcR',stride,startInd,endInd);
  display('   finished loading bcR')
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


output.tauQSPlus = loadVec(sourceDir,'tauQSPlus',stride,startInd,endInd);
display('   finished loading shear stress: + side')


% output.tau = output.tau - output.eta.*output.vel;
if (any(size(output.tauQSPlus) ~= size(output.slipVel)))
  display('Error: tauQSPlus and slipVel are different sizes')
  keyboard
end
output.tauPlus = output.tauQSPlus - bsxfun(@times,output.slipVel,(output.zPlus)/2);


if strcmp(output.dom.problemType,'full')
  output.yVec = [-output.dom.Ly:output.dom.dy:0, 0:output.dom.dy:output.dom.Ly];
  
  loadVec(sourceDir,'zMinus');
  output.zMinus = loadVec(sourceDir,'zMinus');
  output.muMinus = diag(PetscBinaryRead(strcat(sourceDir,'muMinus'),'cell'));
  output.muMinusMat = diag(output.muMinus);
  output.muMinus = reshape(output.muMinus,output.dom.Nz,output.dom.Ny);
  
  %   output.velMinus = loadVec(sourceDir,'velMinus',stride,startInd,endInd);
  %   display('   finished loading velocity: - side')
  %   output.slipVel = output.velPlus - output.velMinus;
  %
  %   output.uMinus = loadVec(sourceDir,'uMinus',stride,startInd,endInd);
  %   display('   finished loading slip on fault: - side')
  %   output.slip = output.uPlus - output.uMinus;
  
  output.tauQSMinus = loadVec(sourceDir,'tauQSMinus',stride,startInd,endInd);
  display('   finished loading shear stress: - side')
  
  if (all(size(output.tauQSMinus) ~= size(output.velMinus)))
    keyboard
  end
  output.tauMinus = -output.tauQSMinus + bsxfun(@times,output.velMinus,output.zMinus);
end

% load surface displacement
if exist(strcat(sourceDir,'surfDispPlus'),'file') == 2
  output.surfDispPlus = loadVec(sourceDir,'surfDispPlus',stride,startInd,endInd);
end

% load bcR
if exist(strcat(sourceDir,'bcR'),'file') == 2
  output.bcR = loadVec(sourceDir,'bcR',stride,startInd,endInd);
end

% body fields
if exist(strcat(sourceDir,'stressxyP'),'file') == 2
%   output.time2D = load(strcat(sourceDir,'time2D.txt'));
%   output.effVisc = loadVec(sourceDir,'effVisc');
  
  % load extra body waves
% output.epsxy = loadVec(sourceDir,'epsVxyP',stride,startInd,endInd);
% output.epsxz = loadVec(sourceDir,'epsVxzP',stride,startInd,endInd);
% output.uN = loadVec(sourceDir,'uBodyP',stride,startInd,endInd);
% output.uA = loadVec(sourceDir,'uAnal',stride,startInd,endInd);
% output.time2D = load(strcat(sourceDir,'time2D.txt'));
end
% temperature
if exist(strcat(sourceDir,'T'),'file') == 2
  output.time2D = load(strcat(sourceDir,'time2D.txt'));
  output.T = loadVec(sourceDir,'T');
%   output.hebcL = loadVec(sourceDir,'he_bcL');
end

if size(output.slipVel,1)>1
  output.maxVel = max(output.slipVel);
  output.faultSurfVel = output.slipVel(1,:);
  output.maxTauPlus = max(output.tauPlus);
  output.maxVel = max(output.slipVel);
  %   if strcmp(output.dom.problemType,'full')
  %     output.maxVel = max(output.velPlus - output.velMinus);
  %     output.faultSurfVelMinus = output.velMinus(1,:);
  %     output.maxTauMinus = output.tauMinus;
  %   end
else
  output.surfVelPlus = output.slipVel;
  output.maxTauPlus = max(output.tauPlus);
  
  output.maxVel = output.slipVel;
end




% % create Hinv matrix for computation of potency
% switch output.dom.order
%   case(2)
%     Hinvy=[2 ones(1,output.dom.Ny-2) 2]./output.dom.dy;
%     Hinvz=[2 ones(1,output.dom.Nz-2) 2]./output.dom.dz;
%   case(4)
%     Hinvy=[48/17 48/59 48/43 48/49 ones(1,output.dom.Ny-8) 48/49 48/43 48/59 48/17]./output.dom.dy;
%     Hinvz=[48/17 48/59 48/43 48/49 ones(1,output.dom.Nz-8) 48/49 48/43 48/59 48/17]./output.dom.dz;
% end
%
% output.dom.Hinv = (1./Hinvz').*1/Hinvy(1);

end