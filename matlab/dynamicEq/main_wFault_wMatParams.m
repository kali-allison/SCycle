% 1D wave eq
% - with fault
% - has nontrivial material parameters
%
% Boundary Conditions: (outgoing characteristics)
%  y=0,1: u_t -+ u_y = 0

% Initial Conditions:
%   u(y,0) = 2*exp( -(y-0.5)^2/0.01
%   u_t(y,0) = 0

clear all

% domain
order = 2;
Ny = 203;
Ly = 30;
dy = Ly / (Ny - 1);
y = linspace(0,Ly,Ny);

% material parameters
% G = y.*0 + 1; % (GPa) shear modulus
% rho = y.*0 + 1; % (g/cm^3) density
% cs = sqrt(G./rho); % (km/s) shear wave speed

G = y.*0 + 30; % (GPa) shear modulus
rho = y.*0 + 2.6702; % (g/cm^3) density
cs = sqrt(G./rho); % (km/s) shear wave speed

% fault material parameters (all stored in p)
p.a = 0.015;
p.b = 0.0;
p.sNEff = 50; % MPa, effective normal stress
p.v0 = 1e-6;
p.f0 = 0.6;
p.Dc = 8e-3;
p.rho = rho(1);
p.tau0 = 0;

% time
tmax = 10;
cfl = 0.25;
dt1 = 0.05*cfl/max(cs) * dy;
dt = cfl/max(cs) * dy;
t = [0 dt1:dt:tmax];


% create SBP operators

% SBP
By = zeros(Ny,Ny); By(1,1)=-1; By(end,end)=1;
if order==2
  h11y = 0.5*dy;
else
  h11y = 0.4567e4/0.14400e5 * dy;
end


% initial conditions
amp = 10;
u =  amp*exp(-(y-0.5*Ly).^2./5);
u0_t =  0*exp(-(y-0.5*Ly).^2./5);

% first time step
% intermediate fields
uy = Dy(u,dy,order);
uLap = Dyy_mu(u,G,dy,order);

% apply part of boundary conditions to intermediate fields
uLap(1) = uLap(1) + (1/h11y)*G(1) .* uy(1);
uLap(end) = uLap(end) - (1/h11y)*G(end) .* uy(end);

uPrev = u; % n-1
u = u + uLap.*0.5*dt1^2/2./rho; % n
uNew = 0.*u; % n+1

psi = 0.6;
velPrev = 0;

% figure(1),clf,plot(u),xlabel('y')
% return

%% time integration

ay = cs .* 0.5*dt/h11y;
for tInd = 2:length(t)-1
  
  % intermediate fields
  uy = Dy(u,dy,order);
  uLap = Dyy_mu(u,G,dy,order);
  
  % apply part of boundary conditions to intermediate fields
  uLap(1) = uLap(1) + (1/h11y)*G(1) .* uy(1);
  uLap(end) = uLap(end) - (1/h11y)*G(end) .* uy(end);
  
  % update interior
  uNew(2:end-1) = uLap(2:end-1).*dt^2./rho(2:end-1) + 2*u(2:end-1) - uPrev(2:end-1);

  
  
  % update boundary conditions

  % y = 0: cs u_t - mu u_y = 0
  uNew(1) = dt^2*uLap(1)./rho(1)  + 2*u(1) + (ay(1)-1)*uPrev(1);
  uNew(1) = uNew(1)./(1+ay(1));
  
  % y = Lz: cs u_t + mu u_y = 0
  uNew(end) = dt^2*uLap(end)./rho(end)  + 2*u(end) + (ay(end)-1)*uPrev(end);
  uNew(end) = uNew(end)./(1+ay(end));
  
  
  % fault
%   
%   [out1, psi, vel] = fault_2d(dt,pen,uLap(:,1),u(:,1),uPrev(:,1),psi,velPrev,p);
%   
  
  pen = h11y;
  [out1, psi, vel] = fault_1d(dt,pen,uLap(1),u(1),uPrev(1),psi,velPrev,p);
  uNew(1) = out1;
  
  
  % update which is the n+1, n, and n-1 steps
  uPrev = u;
  u = uNew;
  
  % plot displacement as simulation runs
  if mod(tInd-1,1) == 0
    figure(1),clf
    plot(uNew)
    xlabel('y')
    title(sprintf('t = %g s',t(tInd)))
    drawnow
  end
  
end





