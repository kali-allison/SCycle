function u_yy = Dyy_mu(u,G,h,order,bcType)

if nargin < 5
  bcType = 'not periodic';
end

u_yy = 0*u;
[m,n] = size(u);

if(order == 2)
  if strcmp(bcType,'periodic')
    u_yy(:,1) = u(:,n-1) - 2*u(:,1) + u(:,2);
    u_yy(:,end) =u(:,n-1) - 2*u(:,end) + u(:,2);
  end
  for ind = 2:n-1
     u_yy(:,ind) = u(:,ind+1) - 2*u(:,ind) + u(:,ind-1);
     
    u_yy(:,ind) = 0.5*(G(:,ind)+G(:,ind-1)).*u(:,ind-1) ...
      - 0.5*(G(:,ind+1)+2*G(:,ind)+G(:,ind-1)).*u(:,ind) ...
      + 0.5*(G(:,ind)+G(:,ind+1)).*u(:,ind+1);
  end
elseif(order == 4)
  'Error: 4th order accuracy not yet supported!!'
end

u_yy = (1/h^2)*u_yy;
end