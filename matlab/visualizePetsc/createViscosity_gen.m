%% creates smooth viscosity profile for n vertical layers
% assuming constant strain for mixing
%
% input:
%   z          (m)      vector of coordinate z, variable grid spacing allowed
%   Ny                  number of points in the y-direction
%   geotherm   (C/km)   slope for temperature as a function of depth
%   lc         string   material for lower crust: 'wet feldspar' or 'dry felspar'
%   plotStuff  bool     whether or not to plot: 1 or 0 respectively
%
% output:
%   effVisc_pet  (GPa s)     viscosity 2D array formatted to be written to PETSc
%   effVisc_ustress (GPa s)  viscosity vector
%   z_c                      constant grid spacing vector for depth

function [effVisc_pet,effVisc,z,Ac_pet,Bc_pet,nc_pet,s_ustrain] = createViscosity_gen(zq,Ny,geotherm,vN,sR,plotStuff)
Nz = length(zq);
Lz = zq(end);

if nargin < 3, geotherm = 30; end
if nargin < 5, sR = 1e-14; end % strain rate
if nargin < 6, plotStuff = 1; end

T = linspace(293,geotherm*Lz+293,Nz); % geotherm 20
z = linspace(0,Lz,Nz);



% set power law material parameters
if vN == 1 % wet quartz, wet feldspar, wet olivine
  nL = 3; % number of layers
  zL = [0 15 30 Lz]; % depth of layers
  transRate = [1 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(1) = 1e-9; b(1) = 1.62e4; n(1) = 4; % wet quartz
  a(2) = 398; b(2) = 4.28e4; n(2) = 3; % wet felspar
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 1.5 % wet quartz, wet feldspar, wet olivine
  nL = 4; % number of layers
  zL = [0 3 15 30 Lz]; % depth of layers
  transRate = [3 1 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1e-9; b(2) = 1.62e4; n(2) = 4; % wet quartz
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=3,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  b(1) = 0; n(1) = 1; % wet quartz
  a(3) = 398; b(3) = 4.28e4; n(3) = 3; % wet felspar
  a(4) = 3.6e5; b(4) = 5.77e4; n(4) = 3.5; % wet olivine
end
if vN == 2 % wet feldspar, wet olivine
  nL = 2; % number of layers
  zL = [0 30 Lz]; % depth of layers
  transRate = 1/3; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(1) = 398; b(1) = 4.28e4; n(1) = 3; % wet feldspar
  a(2) = 3.6e5; b(2) = 5.77e4; n(2) = 3.5; % wet olivine
end
if vN == 2.5 % wet feldspar, wet olivine
  nL = 3; % number of layers
  zL = [0 6 30 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 398; b(2) = 4.28e4; n(2) = 3; % wet feldspar
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  b(1) = 0; n(1) = 1;
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 3
  nL = 3; % number of layers
  zL = [0 10 30 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 2.2e-4; b(2) = 260/8.314e-3; n(2) = 3.4; % diabase
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=10,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  b(1) = 0; n(1) = 1; % diabase with no temp dependence
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 4
  nL = 2; % number of layers
  zL = [0 30 Lz]; % depth of layers
  transRate = [1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(1) = 1.3e-3; b(1) = 219/8.314e-3; n(1) = 2.4; % quarz-diorite
  a(2) = 3.6e5; b(2) = 5.77e4; n(2) = 3.5; % wet olivine
end
if vN == 5 % quartz-diorite, wet olivine
  nL = 2; % number of layers
  zL = [0 30 Lz]; % depth of layers
  transRate = 1/3; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(1) = 1.3e-3; b(1) = 219/8.314e-3; n(1) = 2.4; % quartz-diorite
  a(2) = 3.6e5; b(2) = 5.77e4; n(2) = 3.5; % wet olivine
end
if vN == 5.5 % quartz-diorite, wet olivine, capped at 6 km
  nL = 3; % number of layers
  zL = [0 6 30 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1.3e-3; b(2) = 219/8.314e-3; n(2) = 2.4; % quarz-diorite
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  b(1) = 0; n(1) = 1; % quarz-diorite with no temp dependence
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 5.6 % quartz-diorite, wet olivine, capped at 6 km, for power law geotherm 30
  nL = 3; % number of layers
  zL = [0 6 30 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1.3e-3; b(2) = 219/8.314e-3; n(2) = 2.4; % quarz-diorite
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  if geotherm == 30, a(1) = a(1)*1e-4; end
  if geotherm == 20, a(1) = a(1)*1e-6; end
  b(1) = 0; n(1) = 1; % quarz-diorite with no temp dependence
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 5.7 % quartz-diorite, wet olivine, capped at 2 km
  nL = 3; % number of layers
  zL = [0 2 30 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1.3e-3; b(2) = 219/8.314e-3; n(2) = 2.4; % quarz-diorite
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  if geotherm == 30, a(1) = a(1)*1e-12; end
  if geotherm == 25, a(1) = a(1)*1e-6; end
  if geotherm == 20, a(1) = a(1)*1e-6; end
  b(1) = 0; n(1) = 1; % quarz-diorite with no temp dependence
  a(3) = 3.6e5; b(3) = 5.77e4; n(3) = 3.5; % wet olivine
end
if vN == 6 % quartz-diorite, capped at 6 km
  nL = 2; % number of layers
  zL = [0 6 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1.3e-3; b(2) = 219/8.314e-3; n(2) = 2.4; % quarz-diorite
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  if geotherm == 30, a(1) = a(1)*1e-4; end
  if geotherm == 20, a(1) = a(1)*1e-6; end
  b(1) = 0; n(1) = 1; % quarz-diorite with no temp dependence
end
if vN == 6.5 % quartz-diorite, capped at 2 km
  nL = 2; % number of layers
  zL = [0 2 Lz]; % depth of layers
  transRate = [2 1/3]; % rate of transition between layers
  a = zeros(nL,1); b = zeros(nL,1); n = zeros(nL,1);
  a(2) = 1.3e-3; b(2) = 219/8.314e-3; n(2) = 2.4; % quarz-diorite
  a(1) = ( a(2).*exp(-b(2)./T(find(z>=6,1))) ).^(1/n(2)) .* sR^(-1/n(2) + 1);
  if geotherm == 30, a(1) = a(1)*1e-12; end
  if geotherm == 25, a(1) = a(1)*1e-6; end
  if geotherm == 20, a(1) = a(1)*1e-6; end
  b(1) = 0; n(1) = 1; % quarz-diorite with no temp dependence
end
zLI = findIndices(z,zL);


% % effective viscosity and strength with no mixing
A = T.*0; B = T.*0; N = T.*0;
for ind = 1:length(zLI)-1
  sI = zLI(ind); eI = zLI(ind+1);
  if sI > 1, sI = sI + 1; end
  A(sI:eI) = A(sI:eI) + a(ind);
  B(sI:eI) = B(sI:eI) + b(ind);
  N(sI:eI) = N(sI:eI) + n(ind);
end
s_noMix = (1e-14./(A.*exp(-B./T))).^(1./n); % strength
effVisc_noMix = s_noMix./1e-14 .*1e-3; % effective viscosity



% compute volume fractions for mixing
phi = zeros(Nz,nL);
for ind = 1:(nL-1)
  zI = zLI(ind+1);
  c = transRate(ind);
  phi(:,ind) = -tanh( (z-z(zI)).*c ).*0.5 + 0.5;
  temp = sum(phi(:,1:ind-1),2);
  phi(:,ind) = phi(:,ind) - temp;
end
ind = nL; temp = sum(phi(:,1:ind-1),2); phi(:,ind) = ones(size(phi(:,ind))) - temp;
% figure(21),clf,plot(phi,-z,'o-')




% composite material: uniform stress
% Ac = (A1.^phi1).*(A2.^phi2).*(A3.^phi3);
% Bc = phi1.*B1 + phi2.*B2 + phi3.*B3;
% nc = phi1.*n1 + phi2.*n2 + phi3.*n3;
% sc = (1e-14./(Ac.*exp(-Bc./T))).^(1./(nc)); % strength
% effVisc_ustress = sc./1e-14 .*1e-3; % effective viscosity


% composite material: uniform strain
nc = 1./sum(phi./n',2); % sum { 1./(phi1./n1 + phi2./n2 +...) }
Bc = nc .* sum(phi.*b'./n',2); % nc * sum( phi1*B1/n1 + phi2*B2/n2 +...)
Ac = prod(a'.^(nc.*phi./n'),2); % A1.^(nc.*phi1./n1)).*(A2.^(nc.*phi2./n2)).*...

s_ustrain = (sR./(Ac.*exp(-Bc./T'))).^(1./nc); % strength
effVisc_ustrain = s_ustrain./sR .*1e-3; % effective viscosity



% decide which mixing law to output
% effVisc = effVisc_ustress;
effVisc = effVisc_ustrain;


% interpolate fields to output at desired z values
Ac_interp = interp1(z,Ac,zq);
Bc_interp = interp1(z,Bc,zq);
nc_interp = interp1(z,nc,zq);
viscF_interp = interp1(z,effVisc,zq);

% create 2D versions for output to PETSc
Ac_pet = repmat(Ac_interp',1,Ny);
Bc_pet = repmat(Bc_interp',1,Ny);
nc_pet = repmat(nc_interp',1,Ny);
effVisc_pet = repmat(viscF_interp',1,Ny);
z_pet = repmat(zq,1,Ny);

% return
if plotStuff
  figure(21),clf
  semilogx(effVisc.*1e9,z,'b.-','Linewidth',1)
  set(gca,'YDir','reverse')
end

% fprintf('min Tmax = %.4e, max Tmax = %.4e\n',min(effVisc_pet(:))/30,max(effVisc_pet(:))/30)