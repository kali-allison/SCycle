function [outDisplay,out,outUnits] = humanUnits(in,units)
% convert time to human-readable strings
%
% input:
%     in = input vector
%     units = string containing units for in, accepts: seconds, s, years, yrs
%
% output:
%     out = vector of floats
%     outUnits = cell array of corresponding units
%     outDisplay = cell array of numbers + units in human-readable form

% clc

out = in.*0 + NaN;
outUnits = cell(size(in,1),size(in,2));
outUnits = repmat(cellstr(''),size(in,1),1);

% sample input
% in = [0 1e-5 30 60 100 120 3600*30 3600*36 3600*24*13 3600*24*30*1.5 3600*24*30*5 3600*24*30*24]';
% units = 's';

% in = [0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2]';
% units = 'yrs';

% convert input to seconds
if strcmp(units,'s') || strcmp(units,'seconds'), t = in; end % do nothing
if strcmp(units,'yrs') || strcmp(units,'years'), t = in.*3.154e7; end


% return in seconds
tI = t <= 60;
out(tI) = t(tI);
outUnits(tI) = repmat(cellstr('s'),sum(tI),1);

% return in minutes
tI = t > 60 & t < 3600;
out(tI) = t(tI)./60;
outUnits(tI) = repmat(cellstr('min'),sum(tI),1);

% return in hours
tI = t > 3600 & t < 24*3600;
out(tI) = t(tI)./3600;
outUnits(tI) = repmat(cellstr('hours'),sum(tI),1);

% return in days
tI = t > 24*3600 & t < 30*24*3600;
out(tI) = t(tI)./(24*3600);
outUnits(tI) = repmat(cellstr('days'),sum(tI),1);

% return in months
tI = t > 30*24*3600 & t < 12*30*24*3600;
out(tI) = t(tI)./(30*24*3600);
outUnits(tI) = repmat(cellstr('months'),sum(tI),1);

% return in years
tI = t > 12*30*24*3600 & t < 12*30*24*3600 * 1e3;
out(tI) = t(tI)./(12*30*24*3600);
outUnits(tI) = repmat(cellstr('years'),sum(tI),1);

% return in ka (thousands of years)
tI = t > 12*30*24*3600 * 1e3 & t < 12*30*24*3600 * 1e6;
out(tI) = t(tI)./(12*30*24*3600 * 1e3);
outUnits(tI) = repmat(cellstr('ka'),sum(tI),1);

% return in ka (thousands of years)
tI = t > 12*30*24*3600 * 1e6;
out(tI) = t(tI)./(12*30*24*3600 * 1e6);
outUnits(tI) = repmat(cellstr('ma'),sum(tI),1);

% display output
outDisplay = repmat(cellstr(''),size(in,1),1);
for ii = 1:length(out)
%   fprintf('%g %s\n',out(ii),outUnits{ii})
outDisplay{ii} = sprintf('%.4g %s',out(ii),outUnits{ii});
end





