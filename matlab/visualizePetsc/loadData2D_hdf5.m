function out = loadData2D_hdf5(out,sourceDir,stride,sI,eI,loadOnlySmallFiles)
% loads 2D fields
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
if nargin < 6
  loadOnlySmallFiles = 0;
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

disp(strcat('loading 2D data:',sourceDir))

fileName = strcat(sourceDir,'data_2D.h5');


% get list of names of all datasets for testing if dataset exists before
% loading
info = h5info(fileName);
[~, allDataSetNames] = recursiveGroupFinder(info, info.Groups,{});

% struct containing Ny, Nz and start, count, stride data
a.Ny = out.dom.Ny;
a.Nz = out.dom.Nz;
a.stride = stride;
% update eI to reflect actual maximum size
info = h5info(fileName,'/time/time2D');
Nt = info.Dataspace.Size(3);
a.eI = min(eI,Nt);
a.sI = min(sI,Nt);

if isDatasetPresent(allDataSetNames,'/time/time2D')
time2D = h5read_reshape(fileName,'/time/time2D',a);
% out = catField(out,'time2D',1,time2D);
out.time2D = time2D;
end

if isDatasetPresent(allDataSetNames,'/time/dtime2D')
dt2D = h5read_reshape(fileName,'/time/dtime2D',a);
% out = catField(out,'dt2D',1,dt2D);
out.dt2D = dt2D;
end

if isDatasetPresent(allDataSetNames,'/time/regime2D')
  out.regime2D = h5read_reshape(fileName,'/time/regime2D',a);
end

if loadOnlySmallFiles
  return
end

%==========================================================================
% momentum balance equation
% displacement
if isDatasetPresent(allDataSetNames,'/momBal/u'), out.momBal.U = h5read_reshape(fileName,'/momBal/u',a); end


% stresses
if isDatasetPresent(allDataSetNames,'/momBal/sxy'), out.momBal.Sxy = h5read_reshape(fileName,'/momBal/sxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/sxz'), out.momBal.Sxz = h5read_reshape(fileName,'/momBal/sxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/sdev'), out.momBal.Sdev = h5read_reshape(fileName,'/momBal/sdev',a); end

% total strains, viscous strains, viscous strain rates
if isDatasetPresent(allDataSetNames,'/momBal/gTxy'), out.momBal.GTxy = h5read_reshape(fileName,'/momBal/gTxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gTxz'), out.momBal.GTxz = h5read_reshape(fileName,'/momBal/gTxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gVxy'), out.momBal.GVxy = h5read_reshape(fileName,'/momBal/gVxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/gVxz'), out.momBal.GVxz = h5read_reshape(fileName,'/momBal/gVxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dgVxy'), out.momBal.dGVxy = h5read_reshape(fileName,'/momBal/dgVxy',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dgVxz'), out.momBal.dGVxz = h5read_reshape(fileName,'/momBal/dgVxz',a); end
if isDatasetPresent(allDataSetNames,'/momBal/effVisc'), out.momBal.EffVisc = h5read_reshape(fileName,'/momBal/effVisc',a); end

if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/invEffVisc')
  out.momBal.diff.invEffVisc = h5read_reshape(fileName,'/momBal/diffusionCreep/invEffVisc',a);
  out.momBal.diff.EffVisc = 1./out.momBal.diff.invEffVisc;
end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/invEffVisc')
  out.momBal.disl.invEffVisc = h5read_reshape(fileName,'/momBal/dislocationCreep/invEffVisc',a);
  out.momBal.disl.EffVisc = 1./out.momBal.disl.invEffVisc;
end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/dgVdev_disl'), out.momBal.disl.dGVdev = h5read_reshape(fileName,'/momBal/dislocationCreep/dgVdev_disl',a); end


% %==========================================================================
% % heat equation
% % temperature and thermal anomaly
% if isDatasetPresent(allDataSetNames,'/heatEquation/T'), out.T = h5read_reshape(fileName,'/heatEquation/T',a); end
% if isDatasetPresent(allDataSetNames,'/heatEquation/dT'), out.dT = h5read_reshape(fileName,'/heatEquation/dT',a); end
% 
% % heat sources
% if isDatasetPresent(allDataSetNames,'/heatEquation/Q'), out.he.Qfric = h5read_reshape(fileName,'/heatEquation/Q',a); end
% if isDatasetPresent(allDataSetNames,'/heatEquation/Qfric'), out.he.Qfric = h5read_reshape(fileName,'/heatEquation/Qfric',a); end
% if isDatasetPresent(allDataSetNames,'/heatEquation/Qvisc'), out.he.Qfric = h5read_reshape(fileName,'/heatEquation/Qvisc',a); end
% 
% % heat flux
% if isDatasetPresent(allDataSetNames,'/heatEquation/kTz'), out.he.kTz = h5read_reshape(fileName,'/heatEquation/kTz',a); end
% 
% 
%==========================================================================
% grain size evolution
if isDatasetPresent(allDataSetNames,'/grainSizeEv/d'), out.grainSizeEv.GrainSize = h5read_reshape(fileName,'/grainSizeEv/d',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/d_t'), out.grainSizeEv.dGrainSize = h5read_reshape(fileName,'/grainSizeEv/d_t',a); end








