function out = loadContext_hdf5(sourceDir)
% loads all context fields
sourceDir

out.sourceDir = sourceDir;

% scalar meta-data
out.dom = loadStruct(strcat(sourceDir,'domain.txt'),' = ');

if exist(strcat(sourceDir,'mediator.txt'),'file') == 2
  out.med = loadStruct(strcat(sourceDir,'mediator.txt'),' = ');
end
if exist(strcat(sourceDir,'heatEquation.txt'),'file') == 2
  out.he = loadStruct(strcat(sourceDir,'heatEquation.txt'),' = ');
end
if exist(strcat(sourceDir,'grainSizeEv.txt'),'file') == 2
  out.grainSizeEv = loadStruct(strcat(sourceDir,'grainSizeEv.txt'),' = ');
end
if exist(strcat(sourceDir,'fault.txt'),'file') == 2
  out.fault = loadStruct(strcat(sourceDir,'fault.txt'),' = ');
end
if exist(strcat(sourceDir,'momBal.txt'),'file') == 2
  out.momBal = loadStruct(strcat(sourceDir,'momBal.txt'),' = ');
end

fprintf('loaded context files\n')

% struct containing Ny, Nz and start, count, stride data
a.Nz = out.dom.Nz;
a.Ny = out.dom.Ny;
a.stride = 1;
a.sI = 1;
a.eI = 1;


% coordinate system
fileName = strcat(sourceDir,'data_context.h5');

% get list of names of all datasets for testing if dataset exists before
% loading
info = h5info(fileName);
[~, allDataSetNames] = recursiveGroupFinder(info, info.Groups,{});


% out.q = h5read_reshape(a,fileName,'/static/domain/q');
% out.r = h5read_reshape(a,fileName,'/static/domain/r');
out.y = h5read_reshape(fileName,'/domain/y',a);
out.z = h5read_reshape(fileName,'/domain/z',a);

% output grid spacing, and min and max grid spacing
out.dom.dy = diff(out.y(1,:));
out.dom.dz = diff(out.z(:,1));
% output min and max grid spacing
out.dom.dy_min = min(out.dom.dy);
out.dom.dy_max = max(out.dom.dy);
out.dom.dz_min = min(out.dom.dz);
out.dom.dz_max = max(out.dom.dz);

fprintf('loaded coordinates\n')

% load rate and state parameters
out.fault.a = h5read(fileName,'/fault/a');
out.fault.b = h5read(fileName,'/fault/b');
out.fault.Dc = h5read(fileName,'/fault/Dc');
out.fault.sNEff = h5read(fileName,'/fault/sNEff');
if isDatasetPresent(allDataSetNames,'/fault/Tw'), out.fault.Tw = h5read(fileName,'/fault/Tw'); end
if isDatasetPresent(allDataSetNames,'/fault_qd/eta_rad'), out.fault.eta_rad = h5read(fileName,'/fault_qd/eta_rad'); end
if isDatasetPresent(allDataSetNames,'/fault/cs'), out.fault.cs = h5read(fileName,'/fault/cs'); end
if isDatasetPresent(allDataSetNames,'/fault_qd/rho'), out.fault.rho = h5read(fileName,'/fault_qd/rho'); end
if isDatasetPresent(allDataSetNames,'/fault/locked'), out.fault.locked = h5read(fileName,'/fault/locked'); end
if isDatasetPresent(allDataSetNames,'/fault/prestress'), out.fault.prestress = h5read(fileName,'/fault/prestress'); end

% load momentum balance equation parameters
out.momBal.mu = h5read_reshape(fileName,'/momBal/mu',a);
if isDatasetPresent(allDataSetNames,'/momBal/cs'), out.momBal.cs = h5read_reshape(fileName,'/momBal/cs',a); end
if isDatasetPresent(allDataSetNames,'/momBal/T'), out.momBal.T = h5read_reshape(fileName,'/momBal/T',a); end
if isDatasetPresent(allDataSetNames,'/momBal/fh2o'), out.momBal.fh2o = h5read_reshape(fileName,'/momBal/fh2o',a); end

if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/A'), out.momBal.diff.A = h5read_reshape(fileName,'/momBal/diffusionCreep/A',a); end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/QR'), out.momBal.diff.QR = h5read_reshape(fileName,'/momBal/diffusionCreep/QR',a); end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/n'), out.momBal.diff.n = h5read_reshape(fileName,'/momBal/diffusionCreep/n',a); end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/m'), out.momBal.diff.m = h5read_reshape(fileName,'/momBal/diffusionCreep/m',a); end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/r'), out.momBal.diff.r = h5read_reshape(fileName,'/momBal/diffusionCreep/r',a); end
if isDatasetPresent(allDataSetNames,'/momBal/diffusionCreep/grainSize') out.momBal.diff.GrainSize = h5read_reshape(fileName,'/momBal/diffusionCreep/grainSize',a); end
if isfield(out.momBal,'diff'), fprintf('loaded diffusion creep files\n'), end

if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/A'), out.momBal.disl.A = h5read_reshape(fileName,'/momBal/dislocationCreep/A',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/QR'), out.momBal.disl.QR = h5read_reshape(fileName,'/momBal/dislocationCreep/QR',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/n'), out.momBal.disl.n = h5read_reshape(fileName,'/momBal/dislocationCreep/n',a); end
if isDatasetPresent(allDataSetNames,'/momBal/dislocationCreep/r'), out.momBal.disl.r = h5read_reshape(fileName,'/momBal/dislocationCreep/r',a); end
if isfield(out.momBal,'disl'), fprintf('loaded dislocation creep files\n'); end

if isDatasetPresent(allDataSetNames,'/momBal/DissolutionPrecipitationCreep/B'), out.momBal.dp.B = h5read_reshape(fileName,'/momBal/DissolutionPrecipitationCreep/B',a); end
if isDatasetPresent(allDataSetNames,'/momBal/DissolutionPrecipitationCreep/Vs'), out.momBal.dp.Vs = h5read_reshape(fileName,'/momBal/DissolutionPrecipitationCreep/Vs',a); end
if isDatasetPresent(allDataSetNames,'/momBal/DissolutionPrecipitationCreep/m'), out.momBal.dp.m = h5read_reshape(fileName,'/momBal/DissolutionPrecipitationCreep/m',a); end
if isDatasetPresent(allDataSetNames,'/momBal/DissolutionPrecipitationCreep/r'), out.momBal.dp.r = h5read_reshape(fileName,'/momBal/DissolutionPrecipitationCreep/r',a); end
if isfield(out.momBal,'disl'), fprintf('loaded dissolution-precipitation creep files\n'); end

if isDatasetPresent(allDataSetNames,'/heatEquation/k'), out.he.k = h5read_reshape(fileName,'/heatEquation/k',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/rho'), out.he.rho = h5read_reshape(fileName,'/heatEquation/rho',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/Qrad'), out.he.Qrad = h5read_reshape(fileName,'/heatEquation/Qrad',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/c'), out.he.c = h5read_reshape(fileName,'/heatEquation/c',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/Tamb'), out.he.Tamb = h5read_reshape(fileName,'/heatEquation/Tamb',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/T'), out.he.T = h5read_reshape(fileName,'/heatEquation/T',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/Gw'), out.he.Gw = h5read_reshape(fileName,'/heatEquation/Gw',a); end
if isDatasetPresent(allDataSetNames,'/heatEquation/w'), out.he.w = h5read_reshape(fileName,'/heatEquation/w',a); end
if isfield(out,'he'), fprintf('loaded heat equation files\n'); end

if isDatasetPresent(allDataSetNames,'/grainSizeEv/wattmeter/A'), out.grainSizeEv.A = h5read_reshape(fileName,'/grainSizeEv/wattmeter/A',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/wattmeter/QR'), out.grainSizeEv.QR = h5read_reshape(fileName,'/grainSizeEv/wattmeter/QR',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/wattmeter/p'), out.grainSizeEv.p = h5read_reshape(fileName,'/grainSizeEv/wattmeter/p',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/f'), out.grainSizeEv.f = h5read_reshape(fileName,'/grainSizeEv/f',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/wattmeter/gamma'), out.grainSizeEv.gamma = h5read_reshape(fileName,'/grainSizeEv/wattmeter/gamma',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/piezometer/A'), out.grainSizeEv.piez_A = h5read_reshape(fileName,'/grainSizeEv/piezometer/A',a); end
if isDatasetPresent(allDataSetNames,'/grainSizeEv/piezometer/n'), out.grainSizeEv.piez_n = h5read_reshape(fileName,'/grainSizeEv/piezometer/n',a); end
if isfield(out,'grainSizeEv'), fprintf('loaded grain size evolution files\n'); end

















end