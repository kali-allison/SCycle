function [map,hotMap,coldMap] = createDivColormap(minVal,divVal,maxVal,n)
% creates a diverging colormap with blue at one end, red at the other, and
% white located at divVal rather than the center.
% minVal = minimum value for colormap (will be dark blue)
% maxVal = maximum value for colormap (will be dark red)
% divVal = value that will be white
% n = # of shades to output

% original
coldMap = [256 256 256; 209 229 240; 146 197 222;67 147 195;33 102 172;5 48 97]./256;
hotMap = [103 0 31; 178 24 43; 214 24 43; 214 96 77; 244 165 130;256 256 256]./256;
% coldMap = [209 229 240; 146 197 222;67 147 195;33 102 172;5 48 97]./256;
% hotMap = [103 0 31; 178 24 43; 214 24 43; 214 96 77; 244 165 130]./256;


dd = (maxVal - minVal) / 1e3;
nHot = length(divVal:dd:maxVal);
nCold = length(minVal:dd:divVal);
nWhite = 5;
mymap = [interp1(linspace(0,1,size(hotMap,1)),hotMap,linspace(0,1,nHot));...
  interp1(linspace(0,1,size(coldMap,1)),coldMap,linspace(0,1,nCold))];
% mymap = [interp1(linspace(0,1,size(hotMap,1)),hotMap,linspace(0,1,nHot));...
%   repmat([1 1 1],nWhite,1);...
%   interp1(linspace(0,1,size(coldMap,1)),coldMap,linspace(0,1,nCold))];

mymap = flipud(mymap);
map = mymap;
% map = interp1(linspace(0,1,size(mymap,1)),mymap,linspace(0,1,n));

if nargout == 2
%   hotMap = [hotMap;1 1 1];
  hotMap = interp1(linspace(0,1,size(hotMap,1)),hotMap,linspace(0,1,nHot));
  hotMap = flipud(hotMap);
  hotMap = interp1(linspace(0,1,size(hotMap,1)),hotMap,linspace(0,1,n));
end
  