function imagescCompare_2(D,a,b,titleStrA,titleStrB,fI)
if nargin < 3, titleStrA = 'A'; end
if nargin < 4, titleStrA = 'B'; end
if nargin < 5, fI = 1; end

A = reshape(a,D.Nz,D.Ny);
B = reshape(b,D.Nz,D.Ny);

minVal = min(min(A(:)),min(B(:)));
maxVal = max(max(A(:)),max(B(:)));

figure(fI),clf
subplot(1,2,1),imagesc(A),colorbar,title(titleStrA),caxis([minVal maxVal])
subplot(1,2,2),imagesc(B),colorbar,title(titleStrB),caxis([minVal maxVal])

end