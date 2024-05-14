function out = loadData1D_hdf5(out,sourceDir,stride,sI,eI)
% loads 1D fields
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
  fprintf('initial indextInd = %i, final index = %i.\n',sI,eI)
  return
elseif stride < 1 || rem(stride,1)~=0
  disp('Error: stride must be a positive whole number.')
  return
end

disp(strcat('loading 1D data:',sourceDir))

fileName = strcat(sourceDir,'data_1D.h5');

% get list of names of all datasets for testing if dataset exists before
% loading
info = h5info(fileName);
[~, allDataSetNames] = recursiveGroupFinder(info, info.Groups,{});

% struct containing Ny, Nz and start, count, stride data
a.Ny = out.dom.Ny;
a.Nz = out.dom.Nz;
a.stride = stride;
% update eI to reflect actual maximum size
info = h5info(fileName,'/time/time1D');
Nt = info.Dataspace.Size(3);
a.eI = min(eI,Nt);
a.sI = min(sI,Nt);


time1D = h5read_reshape(fileName,'/time/time1D',a);
out = catField(out,'time',1,time1D);
out = catField(out,'time1D',1,time1D);

dt1D = h5read_reshape(fileName,'/time/dtime1D',a); out = catField(out,'dt1D',1,dt1D);

if isDatasetPresent(allDataSetNames,'/time/regime1D')
  out.regime1D = h5read_reshape(fileName,'/time/regime1D',a);
  eventInds = findEvents_dyn(out.regime1D);
  out.eventInds = eventInds;
end

if isDatasetPresent(allDataSetNames,'/time/SS_index'), out.SS_index = h5read_reshape(fileName,'/time/SS_index',a); end

% load fault fields
if isDatasetPresent(allDataSetNames,'/fault/psi'), out.psi = h5read_reshape(fileName,'/fault/psi',a); end
if isDatasetPresent(allDataSetNames,'/fault/slip'), out.slip = h5read_reshape(fileName,'/fault/slip',a); end
if isDatasetPresent(allDataSetNames,'/fault/slipVel'), out.slipVel = h5read_reshape(fileName,'/fault/slipVel',a); end
if isDatasetPresent(allDataSetNames,'/fault/strength'), out.strength = h5read_reshape(fileName,'/fault/strength',a); end
if isDatasetPresent(allDataSetNames,'/fault/tau'), out.tau = h5read_reshape(fileName,'/fault/tau',a); end
if isDatasetPresent(allDataSetNames,'/fault/tauQS'), out.tauQS = h5read_reshape(fileName,'/fault/tauQS',a); end
if isDatasetPresent(allDataSetNames,'/fault/T'), out.fault.T = h5read_reshape(fileName,'/fault/T',a); end
if isDatasetPresent(allDataSetNames,'/fault/Tw'), out.fault.Tw = h5read_reshape(fileName,'/fault/Tw',a); end
if isDatasetPresent(allDataSetNames,'/fault/Vw'), out.fault.Vw = h5read_reshape(fileName,'/fault/Vw',a); end

% compute max slip velocity and shear stress
out.maxTauP = max(out.strength);
out.maxVel = max(out.slipVel);


% surface heat flux
if isDatasetPresent(allDataSetNames,'/heatEquation/kTz_y0'), out.he.kTz_y0 = h5read_reshape(fileName,'/heatEquation/kTz_y0',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/maxdT'), out.he.maxdT = h5read_reshape(fileName,'/heatEquation/maxdT',a); end



% for debugging purposes: boundary conditions

% momentum balance equation
if isDatasetPresent(allDataSetNames,'/momBal/bcR'), out.momBal.bcR = h5read_reshape(fileName,'/momBal/bcR',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcT'), out.momBal.bcT = h5read_reshape(fileName,'/momBal/bcT',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcL'), out.momBal.bcL = h5read_reshape(fileName,'/momBal/bcL',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcB'), out.momBal.bcB = h5read_reshape(fileName,'/momBal/bcB',a); end
if isDatasetPresent(allDataSetNames,'/momBal/bcRShift'), out.momBal.bcRShift = h5read_reshape(fileName,'/momBal/bcRShift',a); end

% heat equation
if isDatasetPresent(allDataSetNames,'/heatEquation/bcR'), out.he.bcR = h5read_reshape(fileName,'/heatEquation/bcR',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/bcT'), out.he.bcT = h5read_reshape(fileName,'/heatEquation/bcT',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/bcL'), out.he.bcL = h5read_reshape(fileName,'/heatEquation/bcL',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/bcB'), out.he.bcB = h5read_reshape(fileName,'/heatEquation/bcB',a); end







