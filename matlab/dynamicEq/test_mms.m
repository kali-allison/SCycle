% clear all
clc

%% load data

sourceDir = '/Users/kallison/eqcycle/data/mms_';
% sourceDir = '/Users/kallison/mazama/eqcycle/data/test_';

time = load(strcat(sourceDir,'time.txt'));
time2D = load(strcat(sourceDir,'time2D.txt'));
d = loadContext(sourceDir);
d = loadData1D(d,sourceDir,1,1,length(time));
d = loadData2D(d,sourceDir,1,1,length(time));


clear mms
mms.gxyA = loadVec(sourceDir,'mms_gxyA');
mms.GxyA = reshape(mms.gxyA,d.dom.Nz,d.dom.Ny);
mms.gxzA = loadVec(sourceDir,'mms_gxzA');
mms.GxzA = reshape(mms.gxzA,d.dom.Nz,d.dom.Ny);

fprintf('min dy = %.4e m, max dy = %.4e m\n',min(d.dy),max(d.dy))
fprintf('min dz = %.4e m, max dz = %.4e m\n',min(d.dz),max(d.dz))
fprintf('dc = %.1e m\n',d.Dc(1))
fprintf('final time = %.9e s\n',d.time(end))
if isfield(d,'Visc'), fprintf('min Tmax = %.4e, max Tmax = %.4e\n',min(d.Visc(:,1))/30,max(d.Visc(:,1))/30),end
%% main plotting routines
fI = 1;

figure(fI),clf,fI = fI + 1;
subplot(2,1,1),plot(d.z(:,1)),ylabel('z')
subplot(2,1,2),plot(d.y(1,:)),ylabel('y')


% plot viscosity
if isfield(d,'Visc')
  figure(fI),clf,fI=fI+1;
  subplot(1,3,1)
  semilogx(d.Visc(:,1),d.z(:,1),'b.')
  hold on
  semilogx(d.Visc(:,end),d.z(:,1),'r.')
  set(gca,'YDir','reverse')
  % xlim([1e15 1e35])
  xlabel('viscosity (Pa s)'),ylabel('depth (km)')
  subplot(1,3,[2 3])
  pcolored(d.y',d.z',log10(d.Visc)')
  colorbar
  set(gca,'YDir','reverse')
else
  fI = fI + 1;
end


% figure(fI),clf,fI = fI + 1;
% plot(ss.tau)
% hold on
% plot(ss.Sxy(:,1),'--')
% title('bcL')



figure(fI),clf,fI = fI + 1;
subplot(1,2,1)
pcolored(d.y',d.z',mms.GxyA')
set(gca,'YDir','reverse'),colorbar,title('A Gxy')
subplot(1,2,2)
pcolored(d.y',d.z',mms.GxzA')
set(gca,'YDir','reverse'),colorbar,title('A Gxz')

figure(fI),clf,fI = fI + 1;
subplot(1,2,1)
pcolored(d.y',d.z',d.Gxy(:,:,end)')
set(gca,'YDir','reverse'),colorbar,title('N Gxy')
subplot(1,2,2)
pcolored(d.y',d.z',d.Gxz(:,:,end)')
set(gca,'YDir','reverse'),colorbar,title('N Gxz')