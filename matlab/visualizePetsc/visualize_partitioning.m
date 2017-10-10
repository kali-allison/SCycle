% Computes and plots relative partitioning between slip, strains

%% integrate strains horizontally



% remove offset
% % eventInds = findEvents(d);
tI = 1;%eventInds.finishInds(1);
gVxy = bsxfun(@minus,d.Gxy(:,:,tI:end),d.Gxy(:,:,tI));
% gVxz = bsxfun(@minus,d.Gxz(:,:,tI:end),d.Gxz(:,:,tI));
gTxy = bsxfun(@minus,d.GTxy(:,:,tI:end),d.GTxy(:,:,tI));
gExy = bsxfun(@minus,gTxy,gVxy);
gExy = bsxfun(@minus,gExy,gExy(:,:,1));




% % don't remove offset
% gVxy = d.Gxy(:,:,tI:end);
% gTxy = d.GTxy(:,:,tI:end);
% gExy = bsxfun(@minus,gTxy,gVxy);

intgVxy = squeeze(trapz( d.y(1,:),permute(gVxy,[2 1 3]) ));
% intgVxz = squeeze(trapz( d.y(1,:),permute(gVxz,[2 1 3]) ));
% intgTxy = squeeze(trapz( d.y(1,:),permute(gTxy,[2 1 3]) ));
intgExy = squeeze(trapz( d.y(1,:),permute(gExy,[2 1 3]) ));


%% plots of partitioning between slip, strains
if ~exist('fI'), fI = 31; end
figure(fI),clf,fI = fI + 1;

% % elastic problem
% gExy = d.Sxy(:,:,:)./30;
% gExy = bsxfun(@minus,gExy,gExy(:,:,1));
% intgExy = squeeze(trapz( d.y(1,:),permute(gExy,[2 1 3]) ));

% zL = [0 10 20 30 40 49];
zL = [0 2 4 6 8 10];
% zL = [0 10 12 15 20 30];
for ind = 1:length(zL)
  subplot(2,3,ind)
  zI = find(d.z(:,1)>=zL(ind),1);
  time = (d.time(tI:end) - d.time(tI))/3.15e7;
  slip = d.slip(zI,tI:end)./2-d.slip(zI,tI)./2;
  bcR = d.bcR(zI,tI:end) - d.bcR(zI,tI);
  
%   plot([time(1) time(end)+100],[0 0],'-','Color',0.8.*ones(1,3),'Linewidth',2)
%   hold on
%   h1 = plot(time,bcR,'-','Color',0.8.*ones(1,3),'Linewidth',4);
  h2 = plot(time,intgExy(zI,:),'r');
%   h3 = plot(time,slip,'b');
%   h4 = plot(time,intgVxy(zI,:),'c');
  
  
%   % add plots of Maxwell time at each depth
%   maxTime = d.Visc(zI,1)/d.muPlus(1) /3e7;
%   plot([maxTime maxTime],ylim,'k-')
%   plot([maxTime maxTime].*10,ylim,'k--')
%   plot([maxTime maxTime].*100,ylim,'k:')
  
  xlim([time(1) time(end)])
  title(sprintf('z = %2.1f',d.z(zI,1)))
  xlabel('time (years)')
end
% legend([h1 h2 h3 h4],'bcR','elastic strain','displacement','viscous strain','Location','Northwest')

return
%% plot partitioning at specific depth
fI = 20;
figure(fI),clf,fI = fI + 1;

% tI = 1;
% zI = 1;
% zI = find(d.z(:,1)>=2.5,1);
% zI = find(d.z(:,1)>=5,1);
% zI = find(d.z(:,1)>=12,1);
% zI = find(d.z(:,1)>=15,1);
% zI = find(d.z(:,1)>=20,1);
% zI = find(d.z(:,1)>=25,1);
zI = find(d.z(:,1)>=40,1);


time = d.time(tI:end) - d.time(tI);
slip = d.slip(zI,tI:end)./2-d.slip(zI,tI)./2;
bcR = d.bcR(zI,tI:end) - d.bcR(zI,tI);
plot(time,bcR,'-','Color',0.8.*ones(1,3),'Linewidth',4)
hold on
plot(time,intgExy(zI,:),'r')
plot(time,slip,'b')
plot(time,intgVxy(zI,:),'c')
% plot(time,intgVxy(zI,:) + intgExy(zI,:) + slip,'k')
% plot(time,intgTxy(zI,:) + slip,'k')


