% Homogeneous 1D wave eq
% - no fault
% - assumes material parameters are all 1
%
% Boundary Conditions: (outgoing characteristics)
%  y=0,1: u_t -+ u_y = 0

% Initial Conditions:
%   u(y,0) = 2*exp( -(y-0.5)^2/0.01
%   u_t(y,0) = 0

clear all

% domain
order = 2;
Ny = 53;
Ly = 1;
dy = Ly / (Ny - 1);
y = linspace(0,Ly,Ny);


tmax = 1;
dt = 0.2*dy;
t = 0:dt:tmax;


% create SBP operators

% SBP
By = zeros(Ny,Ny); By(1,1)=-1; By(end,end)=1;
if order==2
  h11y = 0.5*dy;
else
  h11y = 0.4567e4/0.14400e5 * dy;
end


% initial conditions
amp = 20;
u =  amp*exp(-(y-0.5*Ly).^2./0.01);
u0_t =  0*exp(-(y-0.5*Ly).^2./0.01);


% first time step
% intermediate fields
uy = Dy(u,dy,order);
uLap = Dyy(u,dy,order);

% apply part of boundary conditions to intermediate fields
uLap(1) = uLap(1) + 1/h11y * uy(1);
uLap(end) = uLap(end) + 1/h11y * uy(end);

uPrev = u; % n-1
u = u + uLap.*dt^2/2; % n
uNew = 0.*u; % n+1


%%
for tInd = 2:length(t)-1
  
  % intermediate fields
  uy = Dy(u,dy,order);
  uLap = Dyy(u,dy,order);
  
  % apply part of boundary conditions to intermediate fields
  uLap(1) = uLap(1) + 1/h11y * uy(1);
  uLap(end) = uLap(end) - 1/h11y * uy(end);
  
  % update interior
  uNew(2:end-1) = uLap(2:end-1).*dt^2 + 2*u(2:end-1) - uPrev(2:end-1);
  
  
  % update boundary conditions
    
  % y = 0: u_t - u_y = 0
  uNew(1) = dt^2*uLap(1)  + 2*u(1) + (0.5*dt/h11y-1)*uPrev(1);
  uNew(1) = uNew(1)./(1+0.5*dt*1/h11y);
  
  % y = Ly: u_t + u_y = 0
  uNew(end) = dt^2*uLap(end)  + 2*u(end) + (0.5*dt/h11y-1)*uPrev(end);
  uNew(end) = uNew(end)./(1+0.5*dt*1/h11y);
  
  
  % update which is the n+1, n, and n-1 steps
  uPrev = u;
  u = uNew;
  
  % plot displacement as simulation runs
  if mod(tInd-1,1) == 0
    figure(1),clf
    plot(uNew)
    xlabel('y')
    drawnow
  end
  
end




