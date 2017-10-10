function xx=simil(x,ds,rx,ry,rz,t,cs)
% SIMIL  Computes similarity coordinate transformation. Transforms
%   a single point in a either a left-handed LG (default) or right-
%   handed CT Cartesian coordinate system using the model
%     xx=sRx+t
%   where
%     x  = coordinate vector to transform from
%     xx = coordinate vector to transform to
%     s  = scale = 1+ds, ds=scale change
%     R  = total rotation matrix (RxRyRz)
%     t  = translation vector
%   Non-vectorized for multiple stations.
% Version: 2016-04-22
% Usage:   xx=simil(x,ds,rx,ry,rz,t,cs)
% Input:   x  - coordinate vector to transform
%          ds - scale change (unitless)
%          rx - rotation about x axis (rad)
%          ry - rotation about y axis (rad)
%          rz - rotation about z axis (rad)
%          t  - translation vector
%          cs - optional coordinate system identifier (char)
%               'lg' = left-handed LG coordinate system (default)
%               'ct' = right-handed CT coordinate system
% Output:  xx - transformed coordinate vector
%
% NOTE: rotations are clockwise for left-handed LG coordinate
%       system (default) and anti-clockwise for right-handed
%       CT coordinate system.

% Copyright (c) 2016, Michael R. Craymer
% All rights reserved.
% Email: mike@craymer.com

if nargin ~= 6 & nargin ~= 7
  warning('Incorrect number of input arguments');
  return
end
if nargin == 6
  cs='lg';
end

crx=cos(rx);
srx=sin(rx);
cry=cos(ry);
sry=sin(ry);
crz=cos(rz);
srz=sin(rz);
if cs == 'lg'
  Rx=[1 0 0; 0 crx -srx; 0 srx crx];
  Ry=[cry 0 sry; 0 1 0; -sry 0 cry];
  Rz=[crz -srz 0; srz crz 0; 0 0 1];
elseif cs == 'ct'
  Rx=[1 0 0; 0 crx srx; 0 -srx crx];
  Ry=[cry 0 -sry; 0 1 0; sry 0 cry];
  Rz=[crz srz 0; -srz crz 0; 0 0 1];
end
R=Rx*Ry*Rz;
xx=(1+ds)*R*x+t;
