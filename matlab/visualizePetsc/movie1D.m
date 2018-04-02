function movie1D(d,fI,stride,iI,eI)
% plots a movie of various fields on the fault

if nargin < 2, fI = 1; end
if nargin < 3, stride = 1; end
if nargin < 4, iI = 1; end
if nargin < 5, eI = length(d.time); end


% to save avi version (1/3)
% fileName = '/Users/kallison/data/linEl_psi_401_dots.avi';
% writerObj = VideoWriter(fileName);
% writerObj.FrameRate = 1;
% open(writerObj);
% frames = eI;
% FigHandle = figure('Position', [1500, 100, 1280, 540],'Color',[1.0 1.0 1.0]);
% mov(1:frames) = struct('cdata',[], 'colormap',[]);

figure(fI),clf,fI = fI + 1;
for ind = iI:stride:eI
  
  % plot of shear stress
  subplot(1,4,1)
  plot(d.tauP(:,ind),d.z(:,1),'b.-')
  xlabel('tau (MPa)')
  %xlim([0 400])
  set(gca,'YDir','reverse')
  drawnow
  
  % plot slip velocity
  subplot(1,4,2)
  semilogx(d.slipVel(:,ind),d.z(:,1),'b.')
  hold on
  semilogx([1e-9 1e-9],[min(d.z(:,1)) max(d.z(:,1))],'k-')
  hold off
  xlim([1e-14 10])
  set(gca,'XTick',10.^(-14:2:1))
  ylim([min(d.z(:,1)) max(d.z(:,1))])
  xlabel('slip velocity (m/s)')
  title(sprintf('%s: %i -> t = %.4e s',d.id,ind,d.time(ind))) 
  set(gca,'YDir','reverse')
  drawnow
  
  % plot state
  subplot(1,4,3)
  plot(d.slip(:,ind),d.z(:,1),'b.')
  xlabel('slip (m)')
  set(gca,'YDir','reverse')
  drawnow
  
  % plot of slip
  subplot(1,4,4)
  plot(d.psi(:,ind),d.z(:,1),'b.')
  %   plot(d.he_bcL(:,ind),-d.z(:,1),'b.')
  xlabel('state')
  set(gca,'YDir','reverse')
  drawnow
  
  %pause
  
  % to save avi version (2/3)
  %   figureHandle = gcf;
  %   set(gcf,'renderer','zbuffer')
  %   mov(ind) = getframe(gcf);
  %   writeVideo(writerObj,mov(ind));
end

% to save avi version (3/3)
% close(writerObj);