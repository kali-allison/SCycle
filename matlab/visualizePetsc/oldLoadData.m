function output = loadData(sourceDir)

display('loading data:')

output.sourceDir = sourceDir;

% scalar meta-data
% output.dom = load_struct(strcat(sourceDir,'parameters.txt'),' = ');
output.dom = load_struct(strcat(sourceDir,'domain.txt'),' = ');

% rate and state context
output.a = loadVecFromPetsc(sourceDir,'a',1);
output.b = loadVecFromPetsc(sourceDir,'b',1);
output.eta = loadVecFromPetsc(sourceDir,'eta',1);
output.sigma_N = loadVecFromPetsc(sourceDir,'sigma_N',1);
output.mu = diag(PetscBinaryRead(strcat(sourceDir,'mu'),'cell'));
output.muMat = diag(output.mu);
output.mu = reshape(output.mu,output.dom.Nz,output.dom.Ny);

% time integration results
time = load(strcat(sourceDir,'time.txt'));
% time=time(1:end-20);
output.time = time;

output.vel = loadVecFromPetsc(sourceDir,'vel',length(time));
display('   finished loading velocity')
output.faultDisp = loadVecFromPetsc(sourceDir,'faultDisp',length(time));
display('   finished loading slip on fault')
output.tauQS = loadVecFromPetsc(sourceDir,'tau',length(time));
display('   finished loading shear stress')
output.tau = output.tauQS - output.eta(1,1).*output.vel;


output.maxVel = max(output.vel);
output.surfVel = output.vel(1,:);
output.maxTau = max(output.tau);

% create Hinv matrix for computation of potency
switch output.dom.order
  case(2)
    Hinvy=[2 ones(1,output.dom.Ny-2) 2]./output.dom.dy;
    Hinvz=[2 ones(1,output.dom.Nz-2) 2]./output.dom.dz;
  case(4)
    Hinvy=[48/17 48/59 48/43 48/49 ones(1,output.dom.Ny-8) 48/49 48/43 48/59 48/17]./output.dom.dy;
    Hinvz=[48/17 48/59 48/43 48/49 ones(1,output.dom.Nz-8) 48/49 48/43 48/59 48/17]./output.dom.dz;
end

output.dom.Hinv = (1./Hinvz').*1/Hinvy(1);

end