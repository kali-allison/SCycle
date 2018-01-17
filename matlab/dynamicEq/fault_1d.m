function [uNew, psiNew, vel] = fault_1d(dt,pen,DmuU,u,uPrev,psi,velPrev,p)
% Enforce friction on a scalar fault interface, for the 1d wave equation with
% inertia.
% Inputs:
%  dt = time step
%  h11 = penalty weight for SAT boundaries
%  DmuU = D^(mu) u = d/dy (G d/dy u) in 1D
%  props = material properties, containing rhoP/M (density) and GP/M (shear mod)


% bounded nonlinear solve for abs(slip velocity)
an = DmuU + (1/pen).*p.tau0; % check sign on tau0 here
Phi = 2/dt *(u-uPrev) + dt/p.rho * an;
func = @(absVel) absVel - abs(Phi) + (dt./(pen*p.rho)) * fricStrength(psi,absVel,p);

lowBound = 0;
upBound = abs(Phi);
guess = velPrev;
if lowBound>guess || upBound<guess % if velPrev is not inside bounds, use midpoint
 guess = 0.5*(lowBound+upBound);
end
% [absVel,resid,vCnvgd] = regulaFalsi(func,lowBound,upBound,guess,opt);
[absVel,resid,vCnvgd] = bisect(func,lowBound,upBound);
if isnan(absVel), keyboard, end
if vCnvgd~=1, keyboard, end


% convert abs(slip vel) to slip vel
if (absVel <= 1e-14)
  uNew = 2*u - uPrev + dt^2./p.rho * an;
  vel = 0;
else
  fric = fricStrength(psi,absVel,p);
  alpha = (1./(pen*p.rho)) * fricStrength(psi,absVel,p)/absVel;
  A = 1 + alpha * dt;

  uNew = (2*u + dt^2./p.rho * an + (dt*alpha-1)*uPrev)./A;
  denom = 1 + dt*pen./p.rho .* fric./absVel;
  vel = Phi./denom;
end
if isnan(vel), keyboard, end

% update psi using slipLaw2
% func = @(stateNew) -stateNew + psi + 0.5*dt*(agingLaw(stateNew,vel,p) + agingLaw(psi,vel,p));
func = @(stateNew) -stateNew + psi + dt*agingLaw((stateNew+psi)/2,(vel+velPrev)/2,p);
[psiNew,~,stateCnvgd] = regulaFalsi(func,-10,10,psi);
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

function g = agingLaw(psi,vel,p)

if isinf(exp(p.f0./p.b))
  g = 0;
else
  g = (p.b*p.v0/p.Dc) * (exp((p.f0-psi)/p.b) - vel/p.v0);
end
end

