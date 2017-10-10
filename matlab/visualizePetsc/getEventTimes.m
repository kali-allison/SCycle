function timeInfo = getEventTimes(D,eventInds)
% computes the timing of events using interp1 to achieve more accurate times


threshold = 1e-3;


for ind = 1:length(eventInds.inds(1,:))
  
  startInd = max(1,eventInds.inds(1,ind)-5);
  finishInd = min(length(D.time),eventInds.inds(2,ind)+5);
  
%   x = linspace(D.time(startInd),D.time(finishInd),(finishInd-startInd)*5e2);
  xq = 1:(finishInd-startInd)/1e3:(finishInd-startInd);
  x = interp1(1:length(D.time(startInd:finishInd)),D.time(startInd:finishInd),xq);
  y = interp1(D.time(startInd:finishInd),D.maxVel(startInd:finishInd),x);
  
  
  above = y>=threshold;
  aboveInds = find(above);
  aboveInds = aboveInds(aboveInds<=length(y) & aboveInds>=1);
  startInds = aboveInds(above(aboveInds-1) == 0);
  finishInds = aboveInds(above(aboveInds+1) == 0);
 
  
  if length(startInds)>1,startInds = startInds(1);end
  if length(finishInds)>1,finishInds = finishInds(end);end
  if length(startInds)==0,keyboard,end
  
  timeInfo.inds(1,ind) = x(startInds);
  timeInfo.inds(2,ind) = x(finishInds);
 
end


numEvents = size(eventInds.inds,2);
% timeInfo.full(1,1:numEvents) = sort([timeInfo.surf(1,:),timeInfo.sub(1,:)]);
% timeInfo.full(2,1:numEvents) = sort([timeInfo.surf(2,:),timeInfo.sub(2,:)]);


% system recurrance interval (s), aka period
if size(timeInfo.inds,2)>0
  timeInfo.period = timeInfo.inds(1,2:end)-timeInfo.inds(1,1:end-1);
  timeInfo.meanPeriod = mean(timeInfo.period);
else
  timeInfo.period = NaN;
  timeInfo.meanPeriod = NaN;
end

% interval between each event
timeInfo.spacing = timeInfo.inds(1,2:end) - timeInfo.inds(1,1:end-1);

% duration of each event (s)
if size(timeInfo.inds,2)>0
  timeInfo.duration = timeInfo.inds(2,:) - timeInfo.inds(1,:);
else
  timeInfo.duration = NaN;
end

timeInfo.meanDuration = mean(timeInfo.duration);

end
