function coldMapOut = createColdColormap(n)
% creates a diverging colormap with blue at one end, red at the other, and
% white located at divVal rather than the center.
% minVal = minimum value for colormap (will be dark blue)
% maxVal = maximum value for colormap (will be dark red)
% divVal = value that will be white
% n = # of shades to output

coldMap = [256 256 256;209 229 240; 146 197 222;67 147 195;33 102 172;5 48 97]./256;


coldMapOut = interp1(linspace(0,1,size(coldMap,1)),coldMap,linspace(0,1,n));
coldMapOut = flipud(coldMapOut);
  