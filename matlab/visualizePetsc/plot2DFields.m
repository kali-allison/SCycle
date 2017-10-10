function plot2DFields(d,fI)

if nargin < 2, fI = 1; end

if isfield(d,'Gxy') && isfield(d,'Gxz')
  figure(fI),clf,fI=fI+1;
  subplot(1,2,1)
  pcolored(d.y',d.z',d.Gxy(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Gxy')
  
  subplot(1,2,2)
  pcolored(d.y',d.z',d.Gxz(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Gxz')
elseif isfield(d,'Gxy')
  figure(fI),clf,fI=fI+1;
  pcolored(d.y',d.z',d.Gxy(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Gxy')
elseif isfield(d,'Gxz')
  figure(fI),clf,fI=fI+1;
  pcolored(d.y',d.z',d.Gxz(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Gxz')
end

if isfield(d,'Sxy') && isfield(d,'Sxz')
  figure(fI),clf,fI=fI+1;
  subplot(1,2,1)
  pcolored(d.y',d.z',d.Sxy(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Sxy')
  
  subplot(1,2,2)
  pcolored(d.y',d.z',d.Sxz(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Sxz')
elseif isfield(d,'Sxy')
  figure(fI),clf,fI=fI+1;
  pcolored(d.y',d.z',d.Sxy(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Sxy')
elseif isfield(d,'Sxz')
  figure(fI),clf,fI=fI+1;
  pcolored(d.y',d.z',d.Sxz(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('Sxz')
end

if isfield(d,'T')
  figure(fI),clf,fI=fI+1;
  subplot(1,2,1)
  pcolored(d.y',d.z',d.T(:,:,1)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('T')
  
  subplot(1,2,2)
  pcolored(d.y',d.z',d.T(:,:,end)')
  colormap(hot)
  set(gca,'YDir','reverse')
  xlabel('distance from fault (km)'),ylabel('depth')
  title('T')

end

end