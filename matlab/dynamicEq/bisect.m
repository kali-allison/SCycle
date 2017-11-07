function [mid,fMid,converged,numIts] = bisect(func,xL,xR,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Finds the root of the scalar func between [xL,xR] using the
% bisection method.
%
% See http://en.wikipedia.org/wiki/Bisection_method.
%
% Written by Kali Allison, 3/16/2015
%
% Convergence criteria: absolute tolerance on current guess vs previous
% guess for the root, and on the distance between func(guess) and 0.
%
% Outputs:
%    mid            the root
%    converged      1 if the root meets at least 1 convergence criteria
%                   0 otherwise
%    numIts         total number of iterations used
%    
%    
% Options:
%    maxNumIts      maximum allowed iterations (default 1e5)
%    aTolX          absolute tolerance on change in root: |root - prev root| < aTolX
%    aTolF          absolute tolerance on residual: |func(root)| < aTolF 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% process options
defaultOptions = struct('maxNumIts',1e5,'aTolX',1e-14,'aTolF',1e-14);
if (nargin<4), options = []; end
maxNumIts = optimget(options,'maxNumIts',defaultOptions,'fast');
aTolX = optimget(options,'aTolX',defaultOptions,'fast');
aTolF = optimget(options,'aTolF',defaultOptions,'fast');


converged = 0;
numIts = 0;
fL = func(xL);
fR = func(xR);


% check that root is bracketed
if (fL*fR >= 0)
  display('bracket invalid: func(xL) and func(xR) must have opposite signs.')
  return
end
if (xL>xR)
  display('bracket invalid: left bound must be less than right bound.')
  return
end

% check if endpoints are root
if (sqrt(fL*fL) <= aTolF)
  mid = xL;
  fMid = fL;
  converged = 1;
  return
elseif (sqrt(fR*fR) <= aTolF)
  mid = xR;
  fMid = fR;
  converged = 1;
  return
end


mid = 0.5*(xL + xR);
fMid = func(mid);

tolX = 3*aTolX; % ensure the while loop is entered
tolF = abs(fMid);
while  numIts <= maxNumIts && tolF >= aTolF && tolX >= aTolX

  if (fL*fMid <= 0)
    xR = mid;
    fR = fMid;
  else
    xL = mid;
    fL = fMid;
  end
  
  prev = mid;
  mid = 0.5*(xL + xR);
  fMid = func(mid);

  tolX = abs((mid-prev)/mid);
  tolF = abs(fMid);
  numIts = numIts + 1;
end



if tolX < aTolX || tolF < aTolF
  converged = 1;
else
  converged = 0;
end


end