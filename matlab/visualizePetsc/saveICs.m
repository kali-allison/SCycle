% write output to C++ code to spin up simulations
% Note that this assumes the 1D and 2D data were loaded at the same times.
function out = saveICs(outDir,d,bcLIsTau,Ny,Nz,zq,interpZ,tI_1D,tI_2D)

% get indices for 1D and 2D data fields
if nargin < 8, tI_1D = length(d.time); end
if nargin < 9, tI_2D = length(d.time2D); end




% set up for interpolation
interpType = 'pchip';

% pick out 2D fields and perform 1D interpolation
y = d.y;
z = d.z;
bcL = d.bcL(:,tI_1D);
bcR = d.bcR(:,tI_1D);

% state = d.state(:,tI_1D);
psi = d.psi(:,tI_1D);
theta = d.theta(:,tI_1D);

slip = d.slip(:,tI_1D);
tauQS = d.tauQSPlus(:,tI_1D);
u = d.u(:,:,tI_2D);
sxy = d.Sxy(:,:,tI_2D);
if isfield(d,'Sxz'), sxz = d.Sxz(:,:,tI_2D); else sxz = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'Gxy'), gxy = d.Gxy(:,:,tI_2D); else gxy = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'Gxz'), gxz = d.Gxz(:,:,tI_2D); else gxz = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'powerLawA'), A = d.powerLawA; else A = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'powerLawB'), B = d.powerLawB; else B = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'n'), n = d.n; else n = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'T0'), T0 = d.T0; else T0 = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'dT'), dT = d.dT(:,:,tI_2D); else dT = zeros(d.dom.Nz,d.dom.Ny); end
if isfield(d,'EffVisc'),
  effVisc = d.EffVisc(:,:,tI_2D);
elseif isfield(d,'Visc')
  effVisc = d.Visc;
else
  effVisc = 1e30+zeros(d.dom.Nz,d.dom.Ny);
end

if Nz ~= d.dom.Nz && interpZ==1
  r = 0:1/(d.dom.Nz-1):1; rr = 0:1/(Nz-1):1;
  if (nargin < 6),
    zq = interp1(r,d.z(:,1),rr,interpType);
    z = interp1(d.z(:,1),d.z,zq,interpType);
  else
    z = repmat(zq,1,Ny);
  end

  y = interp1(d.z(:,1),d.y,zq,interpType);
  
  gxy = interp1(d.z(:,1),gxy,zq,interpType);
  gxz = interp1(d.z(:,1),gxz,zq,interpType);
  sxy = interp1(d.z(:,1),sxy,zq,interpType);
  sxz = interp1(d.z(:,1),sxz,zq,interpType);
  effVisc = interp1(d.z(:,1),effVisc,zq,interpType);
  u = interp1(d.z(:,1),u,zq,interpType);
  bcL = interp1(d.z(:,1),bcL,zq,interpType);
  bcR = interp1(d.z(:,1),bcR,zq,interpType);
  tauQS = interp1(d.z(:,1),tauQS,zq,interpType);
  state = interp1(d.z(:,1),state,zq,interpType);
  'interpolating'
  
  if isfield(d,'powerLawA'),A = interp1(d.z(:,1),A,zq,interpType); end
  if isfield(d,'powerLawB'),B = interp1(d.z(:,1),B,zq,interpType); end
  if isfield(d,'n'),n = interp1(d.z(:,1),n,zq,interpType); end
  if isfield(d,'T0'),T0 = interp1(d.z(:,1),T0,zq,interpType); end
  if isfield(d,'dT'),dT = interp1(d.z(:,1),dT,zq,interpType); end
end

% build 1D fields from interpolated 2D
slip = u(:,1).*2;
tauQS = sxy(:,1);
bcR = u(:,end);
% set boundary conditions
% slip = d.slip(:,tI_1D);
if bcLIsTau == 0
  bcL = u(:,1);
%   bcL = slip./2;
else
  bcL = tauQS;
end

% force state to be 0.6
% psi = 0.*psi + 0.6;
% theta = 0.*theta + 1e9;






% write ICs to file in binary format
PetscBinaryWrite(strcat(outDir,'slip'),slip);

% PetscBinaryWrite(strcat(outDir,'state'),state);
PetscBinaryWrite(strcat(outDir,'psi'),psi);
PetscBinaryWrite(strcat(outDir,'theta'),theta);

PetscBinaryWrite(strcat(outDir,'tauQS'),tauQS);
PetscBinaryWrite(strcat(outDir,'bcL'),bcL);
PetscBinaryWrite(strcat(outDir,'bcR'),bcR);
PetscBinaryWrite(strcat(outDir,'Gxy'),gxy(:));
PetscBinaryWrite(strcat(outDir,'Gxz'),gxz(:));
PetscBinaryWrite(strcat(outDir,'Sxy'),sxy(:));
PetscBinaryWrite(strcat(outDir,'Sxz'),sxz(:));
PetscBinaryWrite(strcat(outDir,'EffVisc'),effVisc(:));
PetscBinaryWrite(strcat(outDir,'u'),u(:));
PetscBinaryWrite(strcat(outDir,'y'),y(:));
PetscBinaryWrite(strcat(outDir,'z'),z(:));


if isfield(d,'powerLawA'), PetscBinaryWrite(strcat(outDir,'A'),A(:)); end
if isfield(d,'powerLawB'), PetscBinaryWrite(strcat(outDir,'B'),B(:)); end
if isfield(d,'n'), PetscBinaryWrite(strcat(outDir,'n'),n(:)); end

if isfield(d,'T0'), PetscBinaryWrite(strcat(outDir,'T0'),T0(:)); end
if isfield(d,'dT'), PetscBinaryWrite(strcat(outDir,'dT'),dT(:)); end
if isfield(d,'dT'), PetscBinaryWrite(strcat(outDir,'T'),T0(:)+dT(:)); end

if (nargout > 0)
  out.tI_1D = tI_1D;
  out.tI_2D = tI_2D;
  
  out.z = z;
  out.y = y;
  
  out.slip = slip;
  %out.state = state;
  out.psi = psi;
  out.theta = theta;
  out.tauQS = tauQS;
  out.bcL = bcL;
  out.bcR = bcR;
  
  out.gxy = gxy;
  out.gxz = gxz;
  out.sxy = sxy;
  out.sxz = sxz;
  out.u = u;
  out.visc = effVisc;
  if isfield(d,'powerLawA'),out.powerLawA = A; end
  if isfield(d,'powerLawB'),out.powerLawB = B; end
  if isfield(d,'n'),out.n = n; end
  if isfield(d,'T0'),out.T0 = T0; end
  if isfield(d,'dT'),out.dT = dT; end
end
end