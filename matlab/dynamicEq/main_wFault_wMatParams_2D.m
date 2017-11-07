% 2D wave eq with variable material parameters
% and a fault with rate-and-state
%
% Boundary Conditions: (outgoing characteristics)
%  y=0,1: u_t -+ u_y = 0
%  z=0,1: u_t -+ u_z = 0

% Initial Conditions:
%   u(y,0) = 2*exp( -(y-0.5)^2/0.01
%   u_t(y,0) = 0

% Notes:
% first index corresponds to z, second index to y

clear all

% domain
order = 2;
Ny = 101;
Nz = 81;
Ly = 30;
Lz = 25;
dy = Ly / (Ny - 1);
dz = Lz / (Nz - 1);
y = linspace(0,Ly,Ny);
z = linspace(0,Lz,Nz);
[Y, Z] = meshgrid(y,z);


% material parameters
G = Y.*0 + 30; % (GPa) shear modulus
rho = Y.*0 + 2.6702; % (g/cm^3) density
cs = sqrt(G./rho); % (km/s) shear wave speed

% fault material parameters (all stored in p)
p.a = 0.015;
p.b = 0.0;
p.sNEff = 50; % MPa, effective normal stress
p.v0 = 1e-6;
p.f0 = 0.6;
p.Dc = 8e-3;
p.rho = rho(:,1);
p.tau0 = 0;

% time
tmax = 8;
cfl = 0.25;
dt1 = 0.05*cfl/max(cs(:)) * dy;
dt = cfl/max(cs(:)) * dy;
t = [0 dt1:dt:tmax];

% create SBP operators

% SBP
By = zeros(Ny,Ny); By(1,1)=-1; By(end,end)=1;
Bz = zeros(Nz,Nz); Bz(1,1)=-1; Bz(end,end)=1;
if order==2
  h11y = 0.5*dy;
  h11z = 0.5*dz;
else
  h11y = 0.4567e4/0.14400e5 * dy;
  h11z = 0.4567e4/0.14400e5 * dz;
end

% initial conditions
amp = 10;
u =  amp*exp(-(Y-0.5*Ly).^2./5).*exp(-(Z-0.5*Lz).^2./5);
u0_t =  0*exp(-(Y-0.5*Ly).^2./5).*exp((Z-0.5*Lz).^2./5);


% first time step
% intermediate fields
uy = Dy(u,dy,order);
uz = Dz(u,dz,order);
uLap = Dyy(u,dy,order) + Dzz(u,dz,order);

% apply part of boundary conditions to intermediate fields
uLap(:,1) = uLap(:,1) + (1/h11y)*G(:,1) .* uy(:,1);
uLap(:,end) = uLap(:,end) - (1/h11y)*G(:,end) .* uy(:,end);
uLap(1,:) = uLap(1,:) + (1/h11z)*G(1,:) .* uz(1,:);
uLap(end,:) = uLap(end,:) - (1/h11z)*G(end,:) .* uz(end,:);

uPrev = u; % n-1
u = u + uLap.*0.5*dt1^2/2./rho; % n
uNew = 0.*u; % n+1

psi = zeros(Nz,1) + 0.6;
velPrev = zeros(Nz,1);

ay = cs .* 0.5*dt/h11y;
az = cs .* 0.5*dt/h11z;
%%
for tInd = 2:length(t)
  
  % intermediate fields
  uy = Dy(u,dy,order);
  uz = Dz(u,dz,order);
  uLap = Dyy_mu(u,G,dy,order) + Dzz_mu(u,G,dz,order);
  
  % apply part of boundary conditions to intermediate fields
  uLap(:,1) = uLap(:,1) + (1/h11y)*G(:,1) .* uy(:,1);
  uLap(:,end) = uLap(:,end) - (1/h11y)*G(:,end) .* uy(:,end);
  uLap(1,:) = uLap(1,:) + (1/h11z)*G(1,:) .* uz(1,:);
  uLap(end,:) = uLap(end,:) - (1/h11z)*G(end,:) .* uz(end,:);
  
  % update interior
  uNew(2:end-1,2:end-1) = uLap(2:end-1,2:end-1).*dt^2./rho(2:end-1,2:end-1) + 2*u(2:end-1,2:end-1) - uPrev(2:end-1,2:end-1);
  
  
  % update boundary conditions
  
  % y = 0: cs u_t - mu u_y = 0
%   uNew(:,1) = dt^2*uLap(:,1)./rho(:,1)  + 2*u(:,1) + (ay(:,1)-1).*uPrev(:,1);
%   uNew(:,1) = uNew(:,1)./(1+ay(:,1));
  
  % y = Ly: cs u_t + mu u_y = 0
  uNew(:,end) = dt^2*uLap(:,end)./rho(:,end)  + 2*u(:,end) + (ay(:,end)-1).*uPrev(:,end);
  uNew(:,end) = uNew(:,end)./(1+ay(:,end));
  
  % z = 0: cs u_t - mu u_z = 0
  uNew(1,:) = dt^2*uLap(1,:)./rho(1,:)  + 2*u(1,:) + (az(1,:)-1).*uPrev(1,:);
  uNew(1,:) = uNew(1,:)./(1+az(1,:));
  
  % z = Lz: cs u_t + mu u_z = 0
  uNew(end,:) = dt^2*uLap(end,:)./rho(end,:)  + 2*u(end,:) + (az(end,:)-1).*uPrev(end,:);
  uNew(end,:) = uNew(end,:)./(1+az(end,:));
 
  
  pen = h11y;
  [out1, psi, vel] = fault_2d(dt,pen,uLap(:,1),u(:,1),uPrev(:,1),psi,velPrev,p);
  uNew(:,1) = out1;
  
  
  % update which is the n+1, n, and n-1 steps
  uPrev = u;
  u = uNew;
  velPrev = vel;
  
  % plot displacement as simulation runs
  if mod(tInd-1,1) == 0
    figure(1),clf
    surf(uNew)
    colorbar
    xlabel('y'),ylabel('z')
    xlim([0 Ny]),ylim([0 Nz])
    title(sprintf('t = %g s',t(tInd)))
    drawnow
  end
  
end





