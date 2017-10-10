function [wwCo wwInt] = plotEvents(slip,time,z,refSlip,eventInds,temp)


intCo = 1; % (s) coseismic time interval
intInt = 3.14e7*5; % (s) interseismic time interval

% intCo = 5; % coseismic time interval
% intInt = 3.14e7*10; % interseismic time interval

% construct time list for interseismic period
timeInt = [];
timeInt = [timeInt time(1):intInt:time(eventInds(1,1));];
for ind = 1:( length(eventInds(1,:)) - 1 )
  sI = eventInds(2,ind);
  eI = eventInds(2,ind+1);
  timeInt = [timeInt time(sI):intInt:time(eI)];
end
timeInt = [timeInt time(eI):intInt:time(end)];

% construct time list for coseismic period
timeCo = [];
if temp.finishInds(1) < eventInds(1,1),
  timeCo = [timeCo time(1):intCo:time(temp.finishInds(1))];
end

for ind = 1:( length(eventInds(1,:)))
  sI = eventInds(1,ind);
  eI = eventInds(2,ind);
  timeCo = [timeCo time(sI):intCo:time(eI)];
end
if temp.startInds(end) > eventInds(end,1) ...
    && time(end)-temp.startInds(end)<300 ,
  timeCo = [timeCo time(temp.startInds(end)):intCo:time(end)];
end

% interpolate slip for interseismic period
[T,Z] = meshgrid(time,z);
[TqInt,ZqInt] = meshgrid(timeInt,z);
wInt = interp2(T,Z,slip,TqInt,ZqInt);
wwInt = bsxfun(@minus,wInt,refSlip);

% interpolate slip for coseismic period
[TqCo,ZqCo] = meshgrid(timeCo,z);
wCo = interp2(T,Z,slip,TqCo,ZqCo);
wwCo = bsxfun(@minus,wCo,refSlip);
% wwCo = [];