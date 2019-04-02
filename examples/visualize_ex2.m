% Script demonstrating how to load and visualize example 2.
% Example 2 is a quasidynamic earthquake cycle simulation with a vertical
% strike-slip fault, and linear elastic off-fault material.
%
% Required matlab functions are located in matlab/visualizePetsc.

% define directory for output (can be relative or absolute path)
sourceDir = '../data/ex2_';

% load context information, such as size of domain, number of points, etc
d.dom = loadStruct(strcat(sourceDir,'domain.txt'),' = ');

% load fields that are either size Ny or size Nz
d.time = load(strcat(sourceDir,'med_time1D.txt'));
d.tau = loadVec(sourceDir,'tauP');
d.slipVel = loadVec(sourceDir,'slipVel');

%% plot results

% plot friction parameters
figure(1),clf
subplot(1,2,1)
plot(d.fault.sNEff,d.z(:,1))
set(gca,'YDir','reverse')
xlabel('\sigma_N (MPa)'),ylabel('depth (km)'),title('effective normal stress')

subplot(1,2,2)
plot(d.fault.a',d.z(:,1),'k-'),hold on
plot(d.fault.a'-d.fault.b',d.z(:,1),'r-')
plot(d.fault.b,d.z(:,1),'b-')
plot([0 0],[0 d.z(end,1)],'-','Color',[1 1 1].*0.6)
set(gca,'YDir','reverse')
legend('a','a-b','b'),ylabel('depth (km)'),title('rate-and-state parameters')


% shear stress
figure(2),clf
tauStride = ceil(length(d.time)/100);
h3 = plot(d.tau(:,1:tauStride:end),d.z(:,1),'c-');
hold on
h1 = plot(d.tau(:,1),d.z(:,1),'.-','Color',[0,128,0]./255,'Linewidth',1);
h2 = plot(d.tau(:,end),d.z(:,1),'r.-','Linewidth',1);
set(gca,'YDir','reverse')
title(sprintf('%i: %.9e',length(d.time),d.time(end)))
ylabel('depth (km)'),xlabel('\tau (MPa)')
grid on, grid minor
legend([h1 h2 h3(1)],{'initial \tau','final \tau','intermediate values of \tau'},'Location','Southeast')


% slip velocity
[map,hotMap] = createDivColormap(-14,-9,1,100); % create colormap highlighting loading vel
figure(3),clf
pcolored(1:size(d.slipVel,2),d.z(:,1)',log10(abs(d.slipVel)))
colormap(map)
hcb = colorbar; ylabel(hcb,'log V (m/s)')
set(hcb,'YTick',-14:1:1)
caxis([-14 1]),ylim([0 d.dom.Lz])
set(gca,'YDir','reverse')
title('slip velocity'),ylabel('depth (km)'),xlabel('model step count')
grid on, grid minor



% plot phase plot: integrated slip velocity vs integrated shear stress
p = trapz(d.z(:,1),d.slipVel);
t = trapz(d.z(:,1),d.tau);

figure(4),clf
semilogx(p,t) % phase plot
hold on
semilogx(p(1),t(1),'g*') % indicate initial condition
semilogx(p(end),t(end),'r*') % indicate final condition
xlabel('integrated slip velocity (m/s km)'),ylabel('integrated shear stress (MPa km)')
















