% 2D wave eq with variable material parameters
% with curvilinear coordinates
% - no fault
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
order = 4;
Ny = 101;
Nz = 81;
Ly = 30;
Lz = 20;
dy = Ly / (Ny - 1);
dz = Lz / (Nz - 1);
dq = 1 / (Ny - 1);
dr = 1 / (Nz - 1);
q = linspace(0,1,Ny);
r = linspace(0,1,Nz);
y = linspace(0,Ly,Ny);
z = linspace(0,Lz,Nz);
[Q, R] = meshgrid(q,r);
[Y, Z] = meshgrid(y,z);


% material parameters
G = Y.*0 + 30; % (GPa) shear modulus
rho = Y.*0 + 2.6702; % (g/cm^3) density
cs = sqrt(G./rho); % (km/s) shear wave speed

% time
tmax = 8;
cfl = 0.25;
dt1 = 0.05*cfl/max(cs(:)) * dy;
dt = cfl/max(cs(:)) * dy;
t = [0 dt1:dt:tmax];

% initial conditions
amp = 1;
u =  amp*exp(-(Y-0.5*Ly).^2./5).*exp(-(Z-0.5*Lz).^2./5);
u0_t =  0*exp(-(Y-0.5*Ly).^2./5).*exp((Z-0.5*Lz).^2./5);

% create SBP operators

% construct coordinate transform terms
yq = Dq(Y,dq,order);
zr = Dr(Z,dr,order);
qy = 1./yq;
rz = 1./zr;
J = yq .* zr;
Jinv = qy .* rz;

% SBP
By = zeros(Ny,Ny); By(1,1)=-1; By(end,end)=1;
Bz = zeros(Nz,Nz); Bz(1,1)=-1; Bz(end,end)=1;
if order==2
  h11y = 0.5*dq;
  h11z = 0.5*dr;
else
  h11y = 0.4567e4/0.14400e5 * dq;
  h11z = 0.4567e4/0.14400e5 * dr;
end



% first time step
% intermediate fields
uy = qy .* Dq(u,dq,order);
uz = rz .* Dr(u,dr,order);
uLap = zr.*Dqq_mu(u,qy.*G,dq,order) + yq.*Drr_mu(u,rz.*G,dr,order);


% apply part of boundary conditions to intermediate fields
uLap(:,1) = uLap(:,1) + (zr(:,1)./h11y).*G(:,1) .* uy(:,1);
uLap(:,end) = uLap(:,end) - (zr(:,end)./h11y).*G(:,end) .* uy(:,end);
uLap(1,:) = uLap(1,:) + (yq(1,:)./h11z).*G(1,:) .* uz(1,:);
uLap(end,:) = uLap(end,:) - (yq(end,:)./h11z).*G(end,:) .* uz(end,:);
uLap = uLap ./ J;

uPrev = u; % n-1
u = u + uLap.*0.5*dt1^2/2./rho; % n
uNew = 0.*u; % n+1

ay = cs .* 0.5*dt./h11y;
az = cs .* 0.5*dt./h11z;

%%
for tInd = 2:length(t)
  
  % intermediate fields
  uy = qy .* Dq(u,dq,order);
  uz = rz .* Dr(u,dr,order);
  uLap = zr.*Dqq_mu(u,qy.*G,dq,order) + yq.*Drr_mu(u,rz.*G,dr,order);
  
  
  % apply part of boundary conditions to intermediate fields
  uLap(:,1) = uLap(:,1) + (zr(:,1)./h11y).*G(:,1) .* uy(:,1);
  uLap(:,end) = uLap(:,end) - (zr(:,end)./h11y).*G(:,end) .* uy(:,end);
  uLap(1,:) = uLap(1,:) + (yq(1,:)./h11z).*G(1,:) .* uz(1,:);
  uLap(end,:) = uLap(end,:) - (yq(end,:)./h11z).*G(end,:) .* uz(end,:);
  
  uLap = uLap ./ J;
  
  % update interior
  uNew(2:end-1,2:end-1) = uLap(2:end-1,2:end-1).*dt^2./rho(2:end-1,2:end-1) + 2*u(2:end-1,2:end-1) - uPrev(2:end-1,2:end-1);
  
  
  % update boundary conditions
  
  % y = 0: cs u_t - mu u_y = 0
  uNew(:,1) = dt^2*uLap(:,1)./rho(:,1)  + 2*u(:,1) + (ay(:,1)-1).*uPrev(:,1);
  uNew(:,1) = uNew(:,1)./(1+ay(:,1));
  
  % y = Ly: cs u_t + mu u_y = 0
  uNew(:,end) = dt^2*uLap(:,end)./rho(:,end)  + 2*u(:,end) + (ay(:,end)-1).*uPrev(:,end);
  uNew(:,end) = uNew(:,end)./(1+ay(:,end));
  
  % z = 0: cs u_t - mu u_z = 0
  uNew(1,:) = dt^2*uLap(1,:)./rho(1,:)  + 2*u(1,:) + (az(1,:)-1).*uPrev(1,:);
  uNew(1,:) = uNew(1,:)./(1+az(1,:));
  
  % z = Lz: cs u_t + mu u_z = 0
  uNew(end,:) = dt^2*uLap(end,:)./rho(end,:)  + 2*u(end,:) + (az(end,:)-1).*uPrev(end,:);
  uNew(end,:) = uNew(end,:)./(1+az(end,:));
  
  
  % update which is the n+1, n, and n-1 steps
  uPrev = u;
  u = uNew;
  
  % plot displacement as simulation runs
  if mod(tInd-1,1) == 0
    figure(1),clf
    surf(uNew)
    colorbar
    xlabel('y'),ylabel('z')
    %xlim([0 Ny]),ylim([0 Nz])
    title(sprintf('t = %g s',t(tInd)))
    drawnow
  end
  
end





