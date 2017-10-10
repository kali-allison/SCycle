function [rx,ry,rz]=euler2cart(lat,lon,r)
% EULER2CART  Converts Euler pole rotation to Cartesian coordinate rotations.
%   Used for modelling tectonic plate motion with spherical approximation.
% Version: 2013-11-22
% Usage:   [x,y,z]=euler2cart(lat,lon,w)
% Input:   lat - latitude of Euler pole (radians)
%          lon - longitude of Euler pole (radians)
%          r   - rotation about Euler pole (radians; counterclockwise)
% Output:  rx  - rotation about Cartesian X axis (radians; counterclockwise)
%          ry  - rotation about Cartesian Y axis (radians; counterclockwise)
%          rz  - rotation about Cartesian Z axis (radians; counterclockwise)

% Created 2012-07-11
% Revised 2013-11-22 Corrected comments/help info.

% Copyright (c) 2013, Michael R. Craymer
% All rights reserved.
% Email: mike@craymer.com

if nargin ~= 3
  warning('Incorrect number of input arguments');
  return
end

[rx,ry,rz]=sph2cart(lon,lat,r);
