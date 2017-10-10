function pred = predictSurfDisp(time,slip,Ny,G,tL,vL)
% predicts surface displacement for a given slip on the fault, right
% boundary displacement, and Greens function matrix (G) at the time indices
% listed in tL.

Nz = size(slip,1);

if nargin < 6, vL = 0.5e-9; end


bcR = repmat(time.*vL,1,Nz)';
bcL = bsxfun(@minus,slip./2,bcR);
pred = zeros(Ny,length(tL));
for ind = 1:length(tL)
  tI = tL(ind);
  pred(:,ind) = G * bcL(:,tI);
  pred(:,ind) = pred(:,ind) + bcR(1,tI);
end