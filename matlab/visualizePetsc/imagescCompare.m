function imagescCompare(A,B,titleStrA,titleStrB,fI)
if nargin < 3, titleStrA = 'A'; end
if nargin < 4, titleStrA = 'B'; end
if nargin < 5, fI = 1; end


minVal = min(min(A(:)),min(B(:)));
maxVal = max(max(A(:)),max(B(:)));

figure(fI),clf
subplot(1,2,1),imagesc(A),colorbar,title(titleStrA),caxis([minVal maxVal])
subplot(1,2,2),imagesc(B),colorbar,title(titleStrB),caxis([minVal maxVal])

end







