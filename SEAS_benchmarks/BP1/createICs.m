% Create initial condition files for SEAS benchmark BP1

clear all


% domain size
Ly = 500;
Lz = 500;

% distance with constant spacing in y and z
Hy = 5;
Hz = 55;

% determine Nz
% require: 25 m grid spacing to depth Hz
dz1 = 25 .* 1e-3;
Nz1 = Hz./dz1 + 1; % # grid points in 1st portion
Nz2 = 201; % # grid points in 2nd portion
Nz = Nz1 + Nz2 - 1; % total # grid points

% material parameters
p = 2670; % kg/m^3
cs = 3.464; % km/s
sN = 50; % MPa
G = cs^2*p *1e-3; 
nrad = 0.5*sqrt(G*p);

% rate-and-state parameters
H = 15; % km
h = 3; % km
v0 = 1e-6; % m/s
f0 = 0.6;
Dc = 0.008; % m
Vp = 1e-9; % m/s
Vinit = 1e-9; % m/s
a0 = 0.01;
amax = 0.025;
b0 = 0.015;


% create y vector
Ny1 = Hy/dz1 + 1;  % # grid points in 1st portion
Ny2 = 201; % # grid points in 2nd portion
Ny = Ny1 + Ny2 - 1; % total # grid points
y = constructCoord_constL1(Ly,Hy,Ny1,Ny2)';

% create z vector
z = constructCoord_constL1(Lz,Hz,Nz1,Nz2);

% rate-and-state parameters
aVals = [a0 a0 amax amax];
aDepths = [0 H H+h Lz];
a = interp1(aDepths,aVals,z);

% prestress
f = amax * asinh((0.5*Vinit/v0) * exp((f0 + b0*log(v0/Vinit))/amax));
tau0 = sN*f + nrad*Vinit + z.*0; % tauQS
tau = sN*f + z.*0; % strength

% theta from BP1
theta = (Dc/v0).*exp( (a/b0) .* log( (2*v0/Vinit) .* sinh((tau0 - nrad*Vinit)./(a*sN)) ) - f0/b0);

% compute psi from theta
psi = f0 + b0.*log(theta.*v0./Dc);

%% write files to binary
outDir = '/Users/kallison/mazama/scycle/SEAS_benchmarks/BP1/ICs/';

Y_pet = repmat(y,Nz,1);
Z_pet = repmat(z,1,Ny);
PetscBinaryWrite(strcat(outDir,'y'),Y_pet(:));
PetscBinaryWrite(strcat(outDir,'z'),Z_pet(:));

PetscBinaryWrite(strcat(outDir,'psi'),psi(:));
PetscBinaryWrite(strcat(outDir,'prestress'),tau0(:));
PetscBinaryWrite(strcat(outDir,'tau'),tau(:));
PetscBinaryWrite(strcat(outDir,'tauQS'),tau0(:));

