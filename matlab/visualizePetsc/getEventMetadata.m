function metadata = getEventMetadata(D,eventInds)

yVec = 0:D.dom.dy:D.dom.Ly;
zVec = 0:D.dom.dz:D.dom.Lz;

faultDisp = D.slip;
vel = D.slipVel;
% faultDisp = D.uPlus - D.uMinus;
% vel = D.velPlus - D.velMinus;

% surface slip all events
metadata.surfaceSlip = faultDisp(1,eventInds.inds(2,:))-faultDisp(1,eventInds.inds(1,:));
metadata.meanSurfaceSlip = mean(faultDisp(1,eventInds.inds(2,:))-faultDisp(1,eventInds.inds(1,:)));

% avg slip and slip bounds
metadata.avgSlip = zeros(size(eventInds.inds,2),1);
metadata.slipExtent = zeros(2,size(eventInds.inds,2));
for eventInd = 1:size(eventInds.inds,2)
  startInd = eventInds.inds(1,eventInd);
  finishInd = eventInds.inds(2,eventInd);
  eventSlip = faultDisp(:,finishInd)-faultDisp(:,startInd);
  topInd = find(eventSlip>1e-3,1);
  bottomInd = find(eventSlip>1e-3,1,'last');
  
  metadata.slipExtent(1,eventInd) = zVec(topInd);
  metadata.slipExtent(2,eventInd) = zVec(bottomInd);
  
  metadata.avgSlip(eventInd) = mean(faultDisp(topInd:bottomInd,finishInd)-faultDisp(topInd:bottomInd,startInd));
end
metadata.meanAvgSlip = mean(metadata.avgSlip);
metadata.meanSlipExtent = mean(metadata.slipExtent')';



% nucleation depth
metadata.nucDepth = zeros(size(eventInds.inds,2),1);
for eventInd = 1:size(eventInds.inds,2)
  startInd = eventInds.inds(1,eventInd);
  [maxVal maxInd] = max(vel(:,startInd));
  metadata.nucDepth(eventInd) = zVec(maxInd);
end
metadata.meanNucDepth = mean(metadata.nucDepth);

end