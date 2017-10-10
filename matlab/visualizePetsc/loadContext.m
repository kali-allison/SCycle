function output = loadContext(sourceDir)
% loads all context fields


% add PETSc routines to MATLAB path
if isdir('/Users/kallison/petsc-3.7.3/share/petsc/matlab/')
  addpath('/Users/kallison/petsc-3.7.3/share/petsc/matlab/');
else
  display('Cannot find directory containing PETSc loading functions!');
end

output.sourceDir = sourceDir;

% scalar meta-data
% output.dom = load_struct(strcat(sourceDir,'domain.txt'),' = '); % old call
output.dom = loadStruct(strcat(sourceDir,'domain.txt'),' = ');

if exist(strcat(sourceDir,'he_context.txt'),'file') == 2
  output.dom_he = loadStruct(strcat(sourceDir,'he_context.txt'),' = ');
end
if exist(strcat(sourceDir,'fault_context.txt'),'file') == 2
  output.dom_fault = loadStruct(strcat(sourceDir,'fault_context.txt'),' = ');
end
if exist(strcat(sourceDir,'linEl_context.txt'),'file') == 2
  output.dom_linEl = loadStruct(strcat(sourceDir,'linEl_context.txt'),' = ');
end
if exist(strcat(sourceDir,'pl_context.txt'),'file') == 2
  output.dom_pl = loadStruct(strcat(sourceDir,'pl_context.txt'),' = ');
  output.dom_pl.numSSIts = load(strcat(sourceDir,'pl_numSSIts.txt'));
end
% output.yVec = 0:output.dom.dy:output.dom.Ly;
% output.zVec = 0:output.dom.dz:output.dom.Lz;


% rate and state context
if exist(strcat(sourceDir,'a'),'file') == 2
  output.a = loadVec(sourceDir,'a');
end
if exist(strcat(sourceDir,'b'),'file') == 2
  output.b = loadVec(sourceDir,'b');
end


% coordinate transform
if exist(strcat(sourceDir,'q'),'file') == 2
  output.q = loadVec(sourceDir,'q');
  output.q = reshape(output.q,output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'r'),'file') == 2
  output.r = loadVec(sourceDir,'r');
  output.r = reshape(output.r,output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'y'),'file') == 2
  output.y = loadVec(sourceDir,'y');
  output.y = reshape(output.y,output.dom.Nz,output.dom.Ny);
  output.dy = (output.y(1,2:end) - output.y(1,1:end-1)).*1e3; % (m)
  
  % output min and max grid spacing
  output.dyV = (output.y(1,2:end) - output.y(1,1:end-1))*1e3; % (m)
  output.dy_min = min(output.dyV);
  output.dy_max = max(output.dyV);
end
if exist(strcat(sourceDir,'z'),'file') == 2
  output.z = loadVec(sourceDir,'z');
  output.z = reshape(output.z,output.dom.Nz,output.dom.Ny);
  output.dz = (output.z(2:end,1) - output.z(1:end-1,1)).*1e3; % (m)
  
  % output min and max grid spacing
  output.dzV = (output.z(2:end,1) - output.z(1:end-1,1))*1e3; % (m)
  output.dz_min = min(output.dzV);
  output.dz_max = max(output.dzV);
end


if exist(strcat(sourceDir,'Dc'),'file') == 2
  output.Dc = loadVec(sourceDir,'Dc');
end
if exist(strcat(sourceDir,'zPlus'),'file') == 2
  output.zPlus = loadVec(sourceDir,'zPlus');
end
if exist(strcat(sourceDir,'sigma_N'),'file') == 2
  output.sigma_N = loadVec(sourceDir,'sigma_N');
end
if exist(strcat(sourceDir,'muPlus'),'file') == 2
  output.muPlus = PetscBinaryRead(strcat(sourceDir,'muPlus'),'cell');
  % output.muPlusMat = diag(output.muPlus);
  % output.muPlus = reshape(output.muPlus,output.dom.Nz,output.dom.Ny);
end

% load pressure equation material properties
if exist(strcat(sourceDir,'fault_hydr_rho_f'),'file') == 2
  output.rho_f = loadVec(sourceDir,'fault_hydr_rho_f');
end
if exist(strcat(sourceDir,'fault_hydr_k'),'file') == 2
  output.k_p = loadVec(sourceDir,'fault_hydr_k');
end
if exist(strcat(sourceDir,'fault_hydr_n'),'file') == 2
  output.n_p = loadVec(sourceDir,'fault_hydr_n');
end
if exist(strcat(sourceDir,'fault_hydr_beta'),'file') == 2
  output.beta_p = loadVec(sourceDir,'fault_hydr_beta');
end
if exist(strcat(sourceDir,'fault_hydr_eta'),'file') == 2
  output.eta_p = loadVec(sourceDir,'fault_hydr_eta');
end


if exist(strcat(sourceDir,'visc'),'file')==2
  visc = loadVec(sourceDir,'visc',1,1,1);
  output.Visc = squeeze(reshape(visc,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'effVisc'),'file')==2
  time2D = load(strcat(sourceDir,'time2D.txt'));
  visc = loadVec(sourceDir,'effVisc',1,length(time2D),length(time2D));
  output.Visc = squeeze(reshape(visc,output.dom.Nz,output.dom.Ny));
  
  visc0 = loadVec(sourceDir,'effVisc',1,1,1);
  output.Visc0 = squeeze(reshape(visc0,output.dom.Nz,output.dom.Ny));
end

if exist(strcat(sourceDir,'T0'),'file')==2
  tt = loadVec(sourceDir,'T0',1,1,1);
  output.T0 = squeeze(reshape(tt,output.dom.Nz,output.dom.Ny));
end
% if exist(strcat(sourceDir,'dT'),'file')==2
%   time2D = load(strcat(sourceDir,'time2D.txt'));
%   %   tt = loadVec(sourceDir,'T',1,length(time2D),length(time2D));
%   tt = loadVec(sourceDir,'dT',1,length(time2D),length(time2D));
%   output.dT = squeeze(reshape(tt,output.dom.Nz,output.dom.Ny));
% end

% heat equation parameters
if exist(strcat(sourceDir,'h'),'file') == 2
  output.h = reshape(loadVec(sourceDir,'h'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'c'),'file') == 2
  output.c = reshape(loadVec(sourceDir,'c'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'rho'),'file') == 2
  output.rho = reshape(loadVec(sourceDir,'rho'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'k'),'file') == 2
  output.k = reshape(loadVec(sourceDir,'k'),output.dom.Nz,output.dom.Ny);
end

% power law parameters
if exist(strcat(sourceDir,'powerLawA'),'file') == 2
  powerLawA = loadVec(sourceDir,'powerLawA');
  output.powerLawA = squeeze(reshape(powerLawA,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'powerLawB'),'file') == 2
  powerLawB = loadVec(sourceDir,'powerLawB');
  output.powerLawB = squeeze(reshape(powerLawB,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'n'),'file') == 2
  n = loadVec(sourceDir,'n');
  output.n = squeeze(reshape(n,output.dom.Nz,output.dom.Ny));
end

% heat equation fields
if exist(strcat(sourceDir,'T0'),'file') == 2
  T0 = loadVec(sourceDir,'T0');
  output.T0 = squeeze(reshape(T0,output.dom.Nz,output.dom.Ny));
end

end