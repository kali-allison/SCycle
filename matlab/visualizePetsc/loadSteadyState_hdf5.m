function out = loadSteadyState_hdf5(out,sourceDir,stride,sI,eI)
% loads steady-state iterative method solutions
%   stride = spacing between loaded time steps
%   sI = starting index
%   eI = ending index


% process optional input arguments
if nargin < 3
  stride = 1;
end
if nargin < 4
  sI = 1;
end
if nargin < 5
  eI = Inf;
end

if sI < 1, sI = 1; end

% check that input's are valid
if eI < sI
  disp('Error: final index must be > than initial index.')
  fprintf('initial indextInd = %i, final index = %i.\n',startInd,endInd)
  return
elseif stride < 1 || rem(stride,1)~=0
  disp('Error: stride must be a positive whole number.')
  return
end

disp(strcat('loading steady state data:',sourceDir))

fileName = strcat(sourceDir,'data_steadyState.h5');

% get list of names of all datasets for testing if dataset exists before
% loading
info = h5info(fileName);
[~, allDataSetNames] = recursiveGroupFinder(info, info.Groups,{});

% for ICs: struct containing Ny, Nz and start, count, stride data
a.Ny = out.dom.Ny;
a.Nz = out.dom.Nz;
a.stride = 1;
a.sI = 1;
a.eI = Inf;

% for post linear solve: struct containing Ny, Nz and start, count, stride data
a.Ny = out.dom.Ny;
a.Nz = out.dom.Nz;
a.stride = stride;
% update eI to reflect actual maximum size
info = h5info(fileName,'/steadyState/SS_index');
Nt = info.Dataspace.Size(end);
a.eI = min(eI,Nt);
a.sI = min(sI,Nt);

%==========================================================================
% fault
if isDatasetPresent(allDataSetNames,'/fault/psi'), out.psi = h5read_reshape(fileName,'/fault/psi',a); end
if isDatasetPresent(allDataSetNames,'/fault/slip'), out.slip = h5read_reshape(fileName,'/fault/slip',a); end
if isDatasetPresent(allDataSetNames,'/fault/slipVel'), out.slipVel = h5read_reshape(fileName,'/fault/slipVel',a); end
if isDatasetPresent(allDataSetNames,'/fault/strength'), out.strength = h5read_reshape(fileName,'/fault/strength',a); end
if isDatasetPresent(allDataSetNames,'/fault/tau'), out.tau = h5read_reshape(fileName,'/fault/tau',a); end
if isDatasetPresent(allDataSetNames,'/fault/tauQS'), out.tauQS = h5read_reshape(fileName,'/fault/tauQS',a); end
if isDatasetPresent(allDataSetNames,'/fault/T'), out.fault.T = h5read_reshape(fileName,'/fault/T',a); end
if isDatasetPresent(allDataSetNames,'/fault/Tw'), out.fault.Tw = h5read_reshape(fileName,'/fault/Tw',a); end
if isDatasetPresent(allDataSetNames,'/fault/Vw'), out.fault.Vw = h5read_reshape(fileName,'/fault/Vw',a); end

%==========================================================================
% momentum balance equation
% displacement
if isDatasetPresent(allDataSetNames,'/momBal/bcRShift'), out.momBal.bcRShift = h5read_reshape(fileName,'/momBal/bcRShift',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcR'), out.momBal.bcR = h5read_reshape(fileName,'/momBal/bcR',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcL'), out.momBal.bcL = h5read_reshape(fileName,'/momBal/bcL',a); end

if isDatasetPresent(allDataSetNames,'/momBal/u'), out.momBal.U = h5read_reshape(fileName,'/momBal/u',a); end

% stresses
if isDatasetPresent(allDataSetNames,'/momBal/sxy'), out.momBal.Sxy = h5read_reshape(fileName,'/momBal/sxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/sxz'), out.momBal.Sxz = h5read_reshape(fileName,'/momBal/sxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/sdev'), out.momBal.Sdev = h5read_reshape(fileName,'/momBal/sdev',a); end

% total and viscous strains and strain rates
if isDatasetPresent(allDataSetNames,'/momBal/gTxy'), out.momBal.GTxy = h5read_reshape(fileName,'/momBal/gTxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gTxy'), out.momBal.GTxz = h5read_reshape(fileName,'/momBal/gTxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gVxy'), out.momBal.GVxy = h5read_reshape(fileName,'/momBal/gVxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gVxy'), out.momBal.GVxz = h5read_reshape(fileName,'/momBal/gVxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dgVxy'), out.momBal.dGVxy = h5read_reshape(fileName,'/momBal/dgVxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dgVxy'), out.momBal.dGVxz = h5read_reshape(fileName,'/momBal/dgVxz',a); end

% effective viscosity
if isDatasetPresent(allDataSetNames,'/momBal/effVisc'), out.momBal.EffVisc = h5read_reshape(fileName,'/momBal/effVisc',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/invEffVisc'),
  temp = h5read_reshape(fileName,'/momBal/dislocationCreep/invEffVisc',a);
  out.momBal.disl.EffVisc = 1./temp;
end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/invEffVisc'),
  temp = h5read_reshape(fileName,'/momBal/diffusionCreep/invEffVisc',a);
  out.momBal.diff.EffVisc = 1./temp;
end
%==========================================================================
% heat equation
if isDatasetPresent(allDataSetNames,'/heatEquation/Tamb'), out.he.Tamb = h5read_reshape(fileName,'/heatEquation/Tamb',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/T'), out.he.T = h5read_reshape(fileName,'/heatEquation/T',a); end


if isDatasetPresent(allDataSetNames,'/grainSizeEv/d'), out.grainSizeEv.GrainSize = h5read_reshape(fileName,'/grainSizeEv/d',a); end



end