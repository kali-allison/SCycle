function [Ny,Nz, Lz,hask, Lb] = gridSpacing(mu,ord,Ly,Lz,sigma_N)

if nargin < 4
  Ly = 24; % km
  Lz = 24; % km
end
if nargin < 2
  ord = 2;
end

if nargin < 4
  sigma_N = 50; % MPa
end


% mu = 36; % GPa
a = 0.015;
b = 0.02;
Dc = 8e-3;
% sigma_N = 250; % MPa: 50


  


% % for Kato 2002 simulation
% a = 0.012;
% b = 0.017;
% Dc = 5e-2;
% sigma_N = 200;

H = 12; % km (seismogenic depth)
% Ly = 2*H; % km
% Lz = fact*H; % km

hask = pi*mu*Dc./(sigma_N.*(b-a)); % km
Lb = mu*Dc./(sigma_N.*b);

% Ny = Ly*50/hask;
% Nz = Lz*50/hask;

if ord == 4
  % for order 4
  Ny = Ly*4./Lb;
  Nz = Lz*4./Lb;
elseif ord == 2
  % for order 2
  Ny = Ly*5./Lb;
  Nz = Lz*5./Lb;
end

end
   

% sourceDir = '/Users/kallison/rcf/sedEqCycle/data/';
% D = load_struct(strcat(sourceDir,'parameters.txt'),' = ');
% uhat = loadVecFromPetsc('/Users/kallison/rcf/sedEqCycle/data/','uhat',1);
% % uhat = reshape(uhat,D.Nz,D.Ny);
% uhat = uhat(1:D.Nz:end);
% yVec = 0:D.dy:D.Ly;
% 
% anal = ((uhat(end)-uhat(1))/D.Ly).*yVec + uhat(1);
%  
% figure,clf
% plot(yVec,anal,'LineWidth',3,'color', [1 1 1]*.75)
% hold on
% plot(yVec,uhat)
