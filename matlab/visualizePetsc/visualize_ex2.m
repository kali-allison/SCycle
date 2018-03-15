% clear all
clc

%% load data


sourceDir = '/Users/kallison/mazama/eqcycle/data/ex2_';
time = load(strcat(sourceDir,'time.txt'));
time2D = load(strcat(sourceDir,'time2D.txt'));
d = loadContext(sourceDir);
d = loadData1D(d,sourceDir,1);
d = loadData2D(d,sourceDir,1,length(time2D),length(time2D));


fprintf('min dy = %.4e m, max dy = %.4e m\n',min(d.dom.dy),max(d.dom.dy))
fprintf('min dz = %.4e m, max dz = %.4e m\n',min(d.dom.dz),max(d.dom.dz))
fprintf('dc = %.1f mm\n',d.fault.Dc(1).*1e3)
fprintf('final time = %.9e s\n',d.time(end))
fprintf('sNEff(Lz) = %g\n',d.fault.sNEff(end))
%% main plotting routines
fI = 1;

figure(fI),clf,fI = fI + 1;
subplot(2,1,1),plot(d.z(:,1)),ylabel('z')
subplot(2,1,2),plot(d.y(1,:)),ylabel('y')

% plot friction parameters
figure(fI),clf,fI=fI+1;
subplot(1,2,1)
plot(d.fault.sNEff,d.z(:,1))
set(gca,'YDir','reverse')
xlabel('\sigma_N (MPa)')

subplot(1,2,2)
plot(d.fault.a',d.z(:,1),'k-'),hold on
plot(d.fault.a'-d.fault.b',d.z(:,1),'r--')
plot(d.fault.b,d.z(:,1),'b-')
plot([0 0],[0 d.z(end,1)],'k--')
set(gca,'YDir','reverse')
legend('a','a-b','b')


figure(fI),clf,fI = fI + 1;
tauStride = ceil(length(d.time)/100);
h1 = plot(d.tauP(:,1:tauStride:end),d.z(:,1),'c-');
hold on
h2 = plot(d.tauP(:,1),d.z(:,1),'.-','Color',[0,128,0]./255,'Linewidth',1);
h3 = plot(d.tauP(:,end),d.z(:,1),'r.-','Linewidth',1);
set(gca,'Ydir','reverse')
ylabel('depth (km)'),xlabel('\tau (MPa)')
legend([h2 h3 h1(1)],'\tau(0)','\tau(end)','\tau(intermediate)','Location','Southeast')
legend boxoff

figure(fI),clf,fI = fI + 1;
subplot(2,1,1),plot(d.time,d.maxTauP),xlabel('time (s)'),ylabel('max \tau (MPa)')
subplot(2,1,2),semilogy(d.time,d.maxVel)
ylabel('max V (m/s)'),xlabel('time (s)')
hold on;yPos = 1e-3; semilogy(get(gca,'xlim'), [yPos yPos],'k','Linewidth',1);
yPos = 1e-9; semilogy(get(gca,'xlim'), [yPos yPos],'k--','Linewidth',1);
ylim([1e-14 20])

figure(fI),clf,fI = fI + 1;
subplot(2,1,1),plot(d.maxTauP),xlabel('model step'),ylabel('max \tau (MPa)')
subplot(2,1,2),semilogy(d.maxVel)
ylabel('max V (m/s)'),xlabel('model step')
hold on;yPos = 1e-3; semilogy(get(gca,'xlim'), [yPos yPos],'k','Linewidth',1);
yPos = 1e-9; semilogy(get(gca,'xlim'), [yPos yPos],'k--','Linewidth',1);
ylim([1e-14 20])



[map,hotMap] = createDivColormap(-14,-9,1,100); % create colormap highlighting loading vel

figure(fI),clf,fI = fI + 1;
pcolored(1:length(d.time),d.z(:,1)',log10(abs(d.slipVel)))
colormap(map)
hcb = colorbar;
set(hcb,'YTick',-14:1:1)
caxis([-14 1])
hold on
contour(1:length(d.time),d.z(:,1)',log10(abs(d.slipVel)),[-3 -3],'k-','Linewidth',1)
set(gca,'YDir','reverse')
title('log_{10}(slip velocity)'),xlabel('model step'),ylabel('depth (km)')


figure(fI),clf,fI = fI + 1;
subplot(1,2,1)
pcolored(d.y',d.z',d.Sxy(:,:,end)')
set(gca,'YDir','reverse')
ylabel('depth (km)'),xlabel('distance from fault (km)')
colorbar
title('Sxy')
subplot(1,2,2)
pcolored(d.y',d.z',d.Sxz(:,:,end)')
set(gca,'YDir','reverse'),colorbar,title('Sxz')
ylabel('depth (km)'),xlabel('distance from fault (km)')
return
%% contour plot

fI = 18;

dd = d;
% dd = truncateData(d,1e3);
eventInds = findEvents(dd);


sI = eventInds.startInds(2) - 25;
[wwCo wwInt] = plotEvents(dd.slip,dd.time,dd.z(:,1),dd.slip(:,sI),eventInds.inds,eventInds);
figure(fI),clf,fI = fI + 1;
plot(wwCo,dd.z(:,1),'r-','Linewidth',1)
hold on
plot(wwInt,dd.z(:,1),'b-','Linewidth',1)
set(gca,'Ydir','reverse')
xlim([0 15])
set(gca,'FontSize',20)