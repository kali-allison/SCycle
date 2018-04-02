function output = loadContext(sourceDir)
% loads all context fields

output.sourceDir = sourceDir;

% scalar meta-data
% output.dom = load_struct(strcat(sourceDir,'domain.txt'),' = '); % old call
output.dom = loadStruct(strcat(sourceDir,'domain.txt'),' = ');

if exist(strcat(sourceDir,'mediator_context.txt'),'file') == 2
  output.med = loadStruct(strcat(sourceDir,'mediator_context.txt'),' = ');
end
if exist(strcat(sourceDir,'he_context.txt'),'file') == 2
  output.he = loadStruct(strcat(sourceDir,'he_context.txt'),' = ');
end
if exist(strcat(sourceDir,'fault_context.txt'),'file') == 2
  output.fault = loadStruct(strcat(sourceDir,'fault_context.txt'),' = ');
end
if exist(strcat(sourceDir,'linEl_context.txt'),'file') == 2
  output.momBal = loadStruct(strcat(sourceDir,'linEl_context.txt'),' = ');
end
if exist(strcat(sourceDir,'pl_context.txt'),'file') == 2
  output.momBal = loadStruct(strcat(sourceDir,'pl_context.txt'),' = ');
end


% coordinate transform
if exist(strcat(sourceDir,'q'),'file') == 2
  output.dom.q = loadVec(sourceDir,'q');
  output.dom.q = reshape(output.dom.q,output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'r'),'file') == 2
  output.dom.r = loadVec(sourceDir,'r');
  output.dom.r = reshape(output.dom.r,output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'y'),'file') == 2
  output.y = loadVec(sourceDir,'y');
  output.y = reshape(output.y,output.dom.Nz,output.dom.Ny);
  output.dom.dy = (output.y(1,2:end) - output.y(1,1:end-1)).*1e3; % (m)
  
  % output min and max grid spacing
  output.dom.dyV = (output.y(1,2:end) - output.y(1,1:end-1))*1e3; % (m)
  output.dom.dy_min = min(output.dom.dyV);
  output.dom.dy_max = max(output.dom.dyV);
end
if exist(strcat(sourceDir,'z'),'file') == 2
  output.z = loadVec(sourceDir,'z');
  output.z = reshape(output.z,output.dom.Nz,output.dom.Ny);
  output.dom.dz = (output.z(2:end,1) - output.z(1:end-1,1)).*1e3; % (m)
  
  % output min and max grid spacing
  output.dom.dzV = (output.z(2:end,1) - output.z(1:end-1,1))*1e3; % (m)
  output.dom.dz_min = min(output.dom.dzV);
  output.dom.dz_max = max(output.dom.dzV);
end

% rate and state context
clear fault
if exist(strcat(sourceDir,'fault_a'),'file') == 2
  output.fault.a = loadVec(sourceDir,'fault_a');
end
if exist(strcat(sourceDir,'fault_b'),'file') == 2
  output.fault.b = loadVec(sourceDir,'fault_b');
end
if exist(strcat(sourceDir,'fault_sNEff'),'file') == 2
  output.fault.sNEff = loadVec(sourceDir,'fault_sNEff');
end
if exist(strcat(sourceDir,'fault_cohesion'),'file') == 2
  output.fault.Co = loadVec(sourceDir,'fault_cohesion');
end
if exist(strcat(sourceDir,'fault_Dc'),'file') == 2
  output.fault.Dc = loadVec(sourceDir,'fault_Dc');
end
if exist(strcat(sourceDir,'fault_imp'),'file') == 2
  output.fault.imp = loadVec(sourceDir,'fault_imp');
end
if exist(strcat(sourceDir,'fault_eta_rad'),'file') == 2
  output.fault.imp = loadVec(sourceDir,'fault_eta_rad');
end
if exist(strcat(sourceDir,'fault_locked'),'file') == 2
  output.fault.locked = loadVec(sourceDir,'fault_locked');
end

% dynamic wave equation
if exist(strcat(sourceDir,'ay'),'file') == 2
  output.ay = loadVec(sourceDir,'ay');
end
if exist(strcat(sourceDir,'cs'),'file') == 2
  output.cs = loadVec(sourceDir,'cs');
end


if exist(strcat(sourceDir,'momBal_mu'),'file') == 2
  output.momBal.mu = PetscBinaryRead(strcat(sourceDir,'momBal_mu'),'cell');
end

% load pressure equation material properties
if exist(strcat(sourceDir,'p_rho_f'),'file') == 2
  output.rho_f = loadVec(sourceDir,'p_rho_f');
end
if exist(strcat(sourceDir,'p_k'),'file') == 2
  output.k_p = loadVec(sourceDir,'p_k');
end
if exist(strcat(sourceDir,'p_n'),'file') == 2
  output.n_p = loadVec(sourceDir,'p_n');
end
if exist(strcat(sourceDir,'p_beta'),'file') == 2
  output.beta_p = loadVec(sourceDir,'p_beta');
end
if exist(strcat(sourceDir,'p_eta'),'file') == 2
  output.eta_p = loadVec(sourceDir,'p_eta');
end


% heat equation parameters
clear he
if exist(strcat(sourceDir,'he_T0'),'file')==2
  tt = loadVec(sourceDir,'he_T0',1,1,1);
  output.he.T0 = squeeze(reshape(tt,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'he_h'),'file') == 2
  output.he.h = reshape(loadVec(sourceDir,'he_h'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'he_c'),'file') == 2
  output.he.c = reshape(loadVec(sourceDir,'he_c'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'he_rho'),'file') == 2
  output.he.rho = reshape(loadVec(sourceDir,'he_rho'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'he_k'),'file') == 2
  output.he.k = reshape(loadVec(sourceDir,'he_k'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'he_w'),'file') == 2
  output.he.w = reshape(loadVec(sourceDir,'he_w'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'he_Gw'),'file') == 2
  output.he.Gw = reshape(loadVec(sourceDir,'he_Gw'),output.dom.Nz,output.dom.Ny);
end

% power law parameters
if exist(strcat(sourceDir,'effVisc'),'file')==2
  time2D = load(strcat(sourceDir,'time2D.txt'));
  visc = loadVec(sourceDir,'effVisc',1,length(time2D),length(time2D));
  output.Visc = squeeze(reshape(visc,output.dom.Nz,output.dom.Ny));
  
  visc0 = loadVec(sourceDir,'effVisc',1,1,1);
  output.Visc0 = squeeze(reshape(visc0,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'momBal_mu'),'file') == 2
  output.momBal.mu = reshape(loadVec(sourceDir,'momBal_mu'),output.dom.Nz,output.dom.Ny);
end
if exist(strcat(sourceDir,'momBal_A'),'file') == 2
  powerLawA = loadVec(sourceDir,'momBal_A');
  output.momBal.A = squeeze(reshape(powerLawA,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'momBal_QR'),'file') == 2
  powerLawB = loadVec(sourceDir,'momBal_QR');
  output.momBal.QR = squeeze(reshape(powerLawB,output.dom.Nz,output.dom.Ny));
end
if exist(strcat(sourceDir,'momBal_n'),'file') == 2
  n = loadVec(sourceDir,'momBal_n');
  output.momBal.n = squeeze(reshape(n,output.dom.Nz,output.dom.Ny));
end

end