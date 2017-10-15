function u_zz = Dzz_mu(u,G,h,order,bcType)
% 2nd spatial derivative for variable material properties
% G = shear modulus

if nargin < 5
  bcType = 'not periodic';
end


u_zz = 0*u;
[m,n] = size(u);


if(order == 2)
  if strcmp(bcType,'periodic')
    u_zz(1,:) = u(end-1, :) - 2*u(1,:) + u(2,:);
    u_zz(end,:) =u(end-1,:) - 2*u(end,:) + u(2,:);
  end
  for ind = 2:m-1
    %u_yy(i,:) = u(i+1,:) - 2*u(i,:) + u(i-1,:);
    
    u_zz(ind,:) = 0.5*(G(ind,:)+G(ind-1,:)).*u(ind-1,:) ...
      - 0.5*(G(ind+1,:)+2*G(ind,:)+G(ind-1,:)).*u(ind,:) ...
      + 0.5*(G(ind,:)+G(ind+1,:)).*u(ind+1,:);
  end
elseif(order == 4)
  'Error: 4th order accuracy not yet supported!!'
end

u_zz = (1/h^2)*u_zz;
end