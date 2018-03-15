% load and visualize data from ex1
clear all
clc


sourceDir = '/Users/kallison/mazama/eqcycle/data/ex1_';
d = loadContext(sourceDir);
d = loadData1D(d,sourceDir,1);
d = loadData2D(d,sourceDir,1);
fprintf('min dy = %.3e m, max dy = %.3e m\n',min(d.dom.dy),max(d.dom.dy))

%%
fI = 1;

% plot simulation
figure(fI),clf,fI=fI+1;
subplot(2,1,1)
plot(d.time,d.tauP,'b.-','Linewidth',1)
ylabel('tau (MPa)')

subplot(2,1,2)
semilogy(d.time,d.slipVel,'b.-','Linewidth',1)
ylabel('slip velocity')

figure(fI),clf,fI=fI+1;
subplot(2,1,1)
plot(d.tauP,'b.-','Linewidth',1)
ylabel('tau (MPa)')

subplot(2,1,2)
semilogy(d.slipVel,'b.-','Linewidth',1)
ylabel('slip velocity')






