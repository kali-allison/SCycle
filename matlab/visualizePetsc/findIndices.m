function [tL, times] = findIndices(time,targetTimes,ref)
% tL = list of indices in "time" that approximate the list
% of desired times "targetTimes".
% ref is an option argument for if targetTimes is supposed to be the time
% past a particular reference time.

if nargin < 3, ref = 0; end

tL = zeros(size(targetTimes)) + NaN;
for ind = 1:length(targetTimes)
  tL(ind) = find(time-ref>=targetTimes(ind),1);
end
times = time(tL) - ref;