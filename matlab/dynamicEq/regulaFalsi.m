function [x,f,converged,numIts] = regulaFalsi(func,xL,xR,x0,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Finds the root of the scalar function func between [xL,xR] using a combination of the
% bisection and secant methods starting with the initial guess x0.
%
% See http://mathworld.wolfram.com/MethodofFalsePosition.html.
%
% Written by Kali Allison 3/16/2015
%
% Convergence criteria: absolute tolerance on current guess vs previous
% guess for the root, and on the distance between func(guess) and 0.
%
% Outputs:
%    x              the root
%    converged      1 if the root meets at least 1 convergence criteria
%                   0 otherwise
%    numIts         total number of iterations used
%
%
% Options:
%    maxNumIts      maximum allowed iterations (default 1e5)
%    aTolX          absolute tolerance on change in root: |(root - prev root)/root| < aTolX
%    aTolF          absolute tolerance on residual: |func(root)| < aTolF
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% process options
defaultOptions = struct('maxNumIts',1e5,'aTolX',1e-14,'aTolF',1e-14);
if (nargin<5), options = []; end
maxNumIts = optimget(options,'maxNumIts',defaultOptions,'fast');
aTolX = optimget(options,'aTolX',defaultOptions,'fast');
aTolF = optimget(options,'aTolF',defaultOptions,'fast');



converged = 0;
numIts = 0;

x = x0;
f = func(x);
% check if guess is root
if (sqrt(f*f) <= aTolF)
  x = x0;
  converged = 1;
  return
end


fL = func(xL);
fR = func(xR);

% check that root is bracketed
if (fL*fR > 0)
  display('bracket invalid: func(xL) and func(xR) must have opposite signs.')
  return
end
if (xL>xR)
  display('bracket invalid: left bound must be less than right bound.')
  return
end
if (xL>x0 || x0>xR)
  display('Initial guess is not inside bracket.')
  keyboard
  return
end

% check if endpoints are root
if (sqrt(fL*fL) <= aTolF)
  x = xL; f = fL;
  converged = 1;
  return
elseif (sqrt(fR*fR) <= aTolF)
  x = xR; f = fR;
  converged = 1;
  return
end



tolX = 3*aTolX; % ensure the while loop is entered
tolF = abs(f);
while  numIts <= maxNumIts  &&  tolX  >= aTolX  &&  tolF >= aTolF
  prev = x;

  if f*fL > aTolF
    xL = x;
    x = xR - (xR - xL)*fR/(fR-fL);
    fL = f;
  elseif f*fR > aTolF
    xR = x;
    x = xR - (xR - xL)*fR/(fR-fL);
    fR = f;
  end
  
  f = func(x);
  
  tolX = abs((x-prev)/x);
  tolF = abs(f);
  numIts = numIts + 1;
  
end


if tolX <= aTolX || tolF <= aTolF
  converged = 1;
else
  converged = 0;
end


end