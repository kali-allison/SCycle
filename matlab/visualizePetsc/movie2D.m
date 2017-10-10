function movie2D(d,C,fI,stride,iI,eI,map,clim)
% plots a movie of C (assumed to have 3 dimensions, 2 spatial)

if nargin < 3, fI = 1; end
if nargin < 4, stride = 1; end
if nargin < 5, iI = 1; end
if nargin < 6, eI = length(d.time); end
if nargin < 7, map = hot; end

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
  
  pcolored(d.y',d.z',C(:,:,ind)')
  colormap(map)
  colorbar
  set(gca,'YDir','reverse')
  if nargin == 8, caxis(clim), end
  xlim([0 5])
  ylim([0 d.dom.Lz])
%   ylim([13 18])
  

  
  t = (d.time(ind) - d.time(iI))./3.14e7;
  title(sprintf('%s: %i -> t = %.4e years',d.id,ind,t))
  
  set(gca,'layer','top')
  pause

  drawnow
  
  % to save avi version (2/3)
  %   figureHandle = gcf;
  %   set(gcf,'renderer','zbuffer')
  %   mov(ind) = getframe(gcf);
  %   writeVideo(writerObj,mov(ind));
end

% to save avi version (3/3)
% close(writerObj);