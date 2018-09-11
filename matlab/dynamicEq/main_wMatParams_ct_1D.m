% 1D wave eq
% - with or without fault
% - has nontrivial material parameters
% - includes variable grid spacing
%
% Boundary Conditions: (outgoing characteristics)
%  y=0,1: u_t -+ u_y = 0

% Initial Conditions:
%   u(y,0) = 2*exp( -(y-0.5)^2/0.01
%   u_t(y,0) = 0

clear all

% domain
order = 4;
Ny = 201;
Ly = 30;
dq = 1 / (Ny - 1);
y = linspace(0,Ly,Ny);

% % material parameters
% p.cs = y.*0 + 1; % wave speed
% p.rho = y.*0 + 1; % (g/cm^3) density
% p.G = p.rho.*p.cs.^2; % (GPa) shear modulus

p.cs = y.*0 + 3.4641;
p.rho = y.*0 + 2.7; % (g/cm^3) density
p.G = p.rho.*p.cs.^2; % (GPa) shear modulus


% fault material parameters (all stored in p)
p.a = 0.015;
p.b = 0.02;
p.sNEff = 50; % MPa, effective normal stress
p.v0 = 1e-6;
p.f0 = 0.6;
p.Dc = 0.05;
p.rho_1D = p.rho(1);
p.tau0 = 0;


% time
tmax = 10;
cfl = 0.25;
dt1 = 0.05*cfl/max(p.cs) * dq*Ly;
dt = cfl/max(p.cs) * dq*Ly;
t = [0 dt1:dt:tmax];
if t(end) < tmax
  t = [t, tmax];
end

% t = linspace(0,tmax,3000);
% dt = t(end)-t(end-1);

% create SBP operators

% curvilinear coordinates: partial derivs and Jacobian
y_q = Dy(y,dq,order);
q_y = 1./y_q;
J = y_q;
Jinv = q_y;
D2muP =@(u,G) Dyy_mu(u,q_y.*G,dq,order) .* Jinv;


By = zeros(Ny,Ny); By(1,1)=-1; By(end,end)=1;
if order==2
  h11y = 0.5*dq;
elseif order == 4
  h11y = 17/48 * dq;
elseif order == 6
  h11y = 13649/43200*dq;
end
% pen = q_y./h11y;
pen = 1./h11y;


% initial conditions
amp = 5;
yC = Ly/2;
ss = Ly/30;
u =  amp*exp(-(y-yC).^2./ss^2);
u0_t =  0*exp(-(y-yC).^2./ss^2);



% first time step
% intermediate fields
uy = q_y .* Dy(u,dq,order);
uLap = D2muP(u,p.G);

% apply part of boundary conditions to intermediate fields
uLap(1) = uLap(1) + pen(1)*p.G(1) .* uy(1);
uLap(end) = uLap(end) - pen(end)*p.G(end) .* uy(end);

uPrev = u; % n-1
u = u + uLap.*0.5*dt^2/2./p.rho; % n
uNew = 0.*u; % n+1

psi = 0.6;
psiPrev = 0.6;
psiNew = 0.6;
vel = 0;

% figure(1),clf,plot(u),xlabel('y')
% return

%% time integration

U = zeros(Ny,length(t));
V = zeros(1,length(t));
Tau = zeros(1,length(t));
Psi = zeros(1,length(t));

U(:,1) = u';
V(:,1) = vel;
Tau(:,1) = 0;
Psi(:,1) = psi;

ay = p.cs .* pen .* 0.5*dt;
for tInd = 2:length(t)
  
  % intermediate fields
  uy = q_y .* Dy(u,dq,order);
  uLap = D2muP(u,p.G);
  
  % apply part of boundary conditions to intermediate fields
  uLap(1) = uLap(1) + pen(1)*p.G(1) .* uy(1);
  uLap(end) = uLap(end) - pen(end)*p.G(end) .* uy(end);
  
  % update interior
  uNew(2:end-1) = uLap(2:end-1).*dt^2./p.rho(2:end-1) + 2*u(2:end-1) - uPrev(2:end-1);

  
  % update boundary conditions
  
  % fault
%   [uNew0, psiNew,vel,strength] = fault_1d(dt,pen(1),uLap(1),u(1),uPrev(1),psi,psiPrev,vel,p);
%   uNew(1) = uNew0;
  
    % y = 0: cs u_t - mu u_y = 0
  uNew(1) = dt^2*uLap(1)./p.rho(1)  + 2*u(1) + (ay(1)-1)*uPrev(1);
  uNew(1) = uNew(1) ./ (1 + ay(1));
  
    % y = Ly: cs u_t + mu u_y = 0
  uNew(end) = dt^2*uLap(end)./p.rho(end)  + 2*u(end) + (ay(end)-1)*uPrev(end);
  uNew(end) = uNew(end) ./ (1 + ay(end));
  

  
  % update which is the n+1, n, and n-1 steps
  uPrev = u;
  u = uNew;
  psiPrev = psi;
  psi = psiNew;
  
  U(:,tInd) = u';
  Psi(tInd) = psi;
  V(tInd) = vel;
%   Tau = strength;
  
  
  % plot displacement as simulation runs
  if mod(tInd-1,10) == 0 || tInd == length(t)
    figure(1),clf
    plot(y,uNew)
    xlabel('y')
    title(sprintf('t = %g s',t(tInd)))
    drawnow
  end
  
end

% clear D2muP
% fileName = sprintf('/Users/kallison/testInertia/wf_Scycle_1D_o%i_N%i.mat',order,Ny);
% save(fileName);


