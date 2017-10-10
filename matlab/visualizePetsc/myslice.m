function hv = myslice(X,Y,Z,V,sx,sy,sz)
% plots specified slices of v
% grid is assumed to be indexed as y,x,z

hv = cell(numel(sx) + numel(sy) + numel(sz));

if ~isempty(sx)
for i = 1:numel(sx)
  xi = sx(i);
  xt = squeeze(X(:,xi,:));
  yt = squeeze(Y(:,xi,:));
  zt = squeeze(Z(:,xi,:));
  vt = squeeze(V(:,xi,:));
  h = contourf(xt,yt,zt,vt);
  hold on
end
end

if ~isempty(sy)
for i = 1:numel(sy)
  yi = sy(i);
  xt = squeeze(X(yi,:,:));
  yt = squeeze(Y(yi,:,:));
  zt = squeeze(Z(yi,:,:));
  vt = squeeze(V(yi,:,:));
  surf(xt,yt,zt,vt)
  shading flat
% contourfm(xt,yt,zt,vt,20,'LineStyle','none')
  hold on
end
end

if ~isempty(sz)
for i = 1:numel(sz)
  zi = sz(i);
  xt = squeeze(X(:,:,zi));
  yt = squeeze(Y(:,:,zi));
  zt = squeeze(Z(:,:,zi));
  vt = squeeze(V(:,:,zi));
  contourf(xt,yt,zt,vt)
  hold on
end
end

hold off



