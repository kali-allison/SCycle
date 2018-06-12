function [uNew, psiNew, vel,strength] = fault_1D_wPrestress(dt,pen,DmuU,u,uPrev,psi,psiPrev,velPrev,p,t)
% Enforce friction on a scalar fault interface, for the 1d wave equation with
% inertia.
% Inputs:
%  dt = time step
%  h11 = penalty weight for SAT boundaries
%  DmuU = D^(mu) u = d/dy (G d/dy u) in 1D
%  props = material properties, containing rhoP/M (density) and GP/M (shear mod)


% bounded nonlinear solve for abs(slip velocity)
an = DmuU + (1/pen).*p.tau0; 
Phi = 2/dt *(u-uPrev) + dt/p.rho * an;
func = @(absVel) absVel - abs(Phi) + (dt./(pen*p.rho)) * fricStrength(psi,absVel,p);

if p.locked < 1
lowBound = 0;
upBound = abs(Phi);
guess = velPrev;
if lowBound>guess || upBound<guess % if velPrev is not inside bounds, use midpoint
 guess = 0.5*(lowBound+upBound);
end
% [absVel,resid,vCnvgd] = regulaFalsi(func,lowBound,upBound,guess,opt);
% process options
opt = struct('aTolX',1e-11,'aTolF',1e-11);
% [absVel,resid,vCnvgd] = bisect(func,lowBound,upBound,opt);
[absVel,~,vCnvgd] = regulaFalsi(func,lowBound,upBound,guess);
if isnan(absVel), keyboard, end
if vCnvgd~=1, keyboard, end
else
  absVel = 0;
end

% convert abs(slip vel) to slip vel
if (absVel <= 1e-14)
  uNew = uPrev;
  vel = 0;
  strength = p.tau0;
else
  fric = fricStrength(psi,absVel,p);
  alpha = (1./(pen*p.rho)) * fricStrength(psi,absVel,p)/absVel;
  A = 1 + alpha * dt;
  
  uNew = 2*u + (dt.*alpha-1).*uPrev + dt.^2/p.rho.*an;
  uNew = uNew ./ A;
  denom = 1 + dt./(p.rho*pen) .* fric./absVel;
  vel = Phi./denom;
  strength = fric * vel ./ absVel;
end
if isnan(vel), keyboard, end

% update psi using slipLaw2
func = @(psiNew) stateEvolution(dt,psiNew,psi,psiPrev,vel,velPrev, p, t);
[psiNew,~,stateCnvgd] = newtonRaphson(func,psi);
if ~stateCnvgd,keyboard,end

end

% see SymmFault::getResid for an example computation of this function, but
% note that this is only the "strength" part of that
function out = fricStrength(psi,vel,p)
% rate-and-state frictional strength
% f = force from regularized rate and state law
% df = derivative of f with respect to uNew

A = p.a*p.sNEff;
B = exp(psi/p.a)/(2*p.v0);
out = A*asinh(B*vel);
% df = A*B/sqrt(1+(B*vel)^2);
end

function [g, dg] = stateEvolution(dt,psiNew,psi,psiPrev,velNew,vel, p, t)

[gNew, dgNew] = agingLaw(0.5*(psiNew+psiPrev), velNew, p);
g = psiNew - psiPrev - 2*dt*(gNew);
dg = 1 - dt*dgNew;

if (t == 0)
  %[gNew, dgNew] = agingLaw(psiNew, velNew, p.b,p.Dc,p.v0,p.f0);
  [g, dg] = agingLaw(psi,velNew, p);
  g = psiNew - psi - dt*g;
  dg = 1;
end

end

function [g, dg] = agingLaw(psi,vel,p)
if p.b ==0
  g = 0;
  dg = 0;
else
g = (p.b*p.v0/p.Dc) * ( exp((p.f0-psi)/p.b) - vel/p.v0 );
dg = -(p.v0/p.Dc)*exp((p.f0-psi)/p.b);
end
end

