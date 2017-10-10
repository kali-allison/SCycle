function [eventInds,fInd] = findEvents(D,fInd)
% Input:
%     - D: struct containing maximum velocity (maxVel), velocity at the
%     surface of the fault (surfVel), time, and a struct called dom that
%     holds the loading rate vp.
%     - fInd: number for the figure that may optionally be produced
%
% Output: 
% eventInds contains:
%     - full: indices of the start (row 1) and finish (row 2) of all events
%     - surf: indices of the start (row 1) and finish (row 2) of the surface-rupturing events
%     - sub: indices of the start (row 1) and finish (row 2) of sub-basin events
%     - refInd = index at which maximum velocity (maxVel) first approaches
%       the loading rate



domain = D.dom;
maxInd = length(D.maxVel);

threshold = 1e-3;
loadingRate = domain.vL;

% first onset of interseismic interval
eventInds.refInd = find(D.maxVel<loadingRate,1);

% find start and finish of all events
above = D.maxVel>threshold;
aboveInds = find(above);
aboveInds = aboveInds(aboveInds<maxInd & aboveInds>1);
startInds = aboveInds(above(aboveInds-1) == 0);
finishInds = aboveInds(above(aboveInds+1) == 0);

eventInds.startInds = startInds;
eventInds.finishInds = finishInds;

% ensure start and finish inds are paired
if isempty(eventInds.finishInds) || isempty(eventInds.startInds)
  eventInds.inds = [];
  return
end
if eventInds.finishInds(1)<eventInds.startInds(1),
  temp2 = eventInds.finishInds(2:end);
else
  temp2 = eventInds.finishInds;
end
if length(eventInds.startInds) > length(temp2),
  temp1 = eventInds.startInds(1:length(temp2));
else
  temp1 = eventInds.startInds;
end

eventInds.inds(1,:) = temp1;
eventInds.inds(2,:) = temp2;






end
