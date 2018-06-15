function [uNew, psiNew, velNew,strength] = fault_2d_wPrestress(dt,pen,DmuU,u,uPrev,psi,psiPrev,velPrev,p2D,t)
% loops over 1d interface function
% pen = penalty parameter for fault interface (for rectangular coordinates,
% this is 1/h11).


% allocate space for output
psiNew = 0.*psi;
velNew = 0.*psi;
uNew = 0.*psi;
strength = 0.*psi;
for zI = 1:length(psi)

  p1D = p2D;
  p1D.a = p2D.a(zI);
  p1D.b = p2D.b(zI);
  p1D.rho = p2D.rho(zI);
  p1D.tau0 = p2D.tau0(zI);
  p1D.locked = p2D.locked(zI);
  
  [out1, out2, out3, out4] = fault_1D_wPrestress(dt,pen,DmuU(zI),u(zI),uPrev(zI),psi(zI),psiPrev(zI),velPrev(zI),p1D,t);  
   
  uNew(zI) = out1;
  psiNew(zI) = out2;
  velNew(zI) = out3;
  strength(zI) = out4;
  
end



end