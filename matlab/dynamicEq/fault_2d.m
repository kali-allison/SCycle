function [uNew, psiNew, velNew] = fault_2d(dt,pen,DmuU,u,uPrev,psi,velPrev,p2D)
% loops over 1d interface function
% pen = penalty parameter for fault interface (for rectangular coordinates,
% this is 1/h11).


% allocate space for output
psiNew = 0.*psi;
velNew = 0.*psi;
uNew = 0.*psi;
for zI = 1:length(psi)

  p1D = p2D;
  p1D.rho = p2D.rho(zI);
  
  [out1, out2, out3] = fault_1d(dt,pen,DmuU(zI),u(zI),uPrev(zI),psi(zI),velPrev(zI),p1D);  
   
  uNew(zI) = out1;
  psiNew(zI) = out2;
  velNew(zI) = out3;
  
end



end