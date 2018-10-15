% Example script demonstrating how to load and visualize example 1.

% define directory for output (can be relative or absolute path)
sourceDir = '../data/ex1_';

% load context information, such as size of domain, number of points, etc
d = loadContext(sourceDir);

% load fields that are either size Ny or size Nz
d = loadData1D(d,sourceDir);


%% make plots

figure(1),clf

% plot shear stress
subplot(2,1,1)
plot(d.time./3.14e7,d.tauP)
xlabel('time (years)'),ylabel('\tau (MPa)')
title('Spring Slider')

% plot slip velocity
subplot(2,1,2)
semilogy(d.time./3.14e7,d.slipVel)
xlabel('time (years)'),ylabel('V (m/s)'),ylim([1e-14 10])
