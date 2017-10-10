function hotMapOut = createHotColormap(n)
% creates a diverging colormap with blue at one end, red at the other, and
% white located at divVal rather than the center.
% minVal = minimum value for colormap (will be dark blue)
% maxVal = maximum value for colormap (will be dark red)
% divVal = value that will be white
% n = # of shades to output


hotMap = [0 0 0;103 0 31; 178 24 43; 214 24 43; 214 96 77; 244 165 130; 253 219 199; 256 256 256]./256;


hotMapOut = [interp1(linspace(0,1,size(hotMap,1)),hotMap,linspace(0,1,n))];
hotMapOut = flipud(hotMapOut);

  