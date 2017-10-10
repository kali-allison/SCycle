% returns a,a-b parameters at all depths by linear interpolation between
% input pairs of points.

function [a_fit,ab_fit,out] = computefricParams_gen(z,T,geotherm,a,T_a,ab,T_ab)
% inputs:
%   z = (km) depths, vector
%   T = temperature at each depth in z, vector
%   a = direct effect friction parameter a
%   T_a = temperature at each depth in a
%   ab = friction parameter a-b
%   T_ab = temperature at each depth in ab
%
% outputs: vector profiles for a, a-b corresponding to z

% z = linspace(-5,60,100);
surfTemp = 293; % (K) temperature of surface of Earth
% T = 25.*z + surfTemp;
% 
% % construct line for wet granite a-b:
% ab = [0.9 -1.5 11].*1e-2;
% T_ab = [0 300 600]; % (C)
% a = [0.008 0.035 0.135];
% T_a = [0 480 600]; % (C)


% compute depths corresponding to temperatures T_a and T_ab
z_a = interp1(T,z,T_a);
z_ab = interp1(T,z,T_ab);

% compute a and ab at all depths
a_fit = interp1(z_a,a,z,'linear','extrap');
ab_fit = interp1(z_ab,ab,z,'linear','extrap');
b_fit = a_fit - ab_fit;


% compute a at specific depths for input into PETSC .in file
T_a_out = [surfTemp T_a(2:end)];
z_a_out = [interp1(T,z,T_a_out) 60];
a_out = interp1(z,a_fit,z_a_out,'linear','extrap');

% compute a-b at specific depths for input into PETSC .in file
T_ab_out = [surfTemp T_ab(2:end)];
z_ab_out = [interp1(T,z,T_ab_out) 60];
ab_out = interp1(z,ab_fit,z_ab_out,'linear','extrap');

% compute b at specific depths for input into PETSC .in file
z_b_out = sort(unique([z_a_out, z_ab_out]));
b_out = interp1(z,b_fit,z_b_out,'linear','extrap');



% output data for petsc
out.a = a_out;
out.z_a = z_a_out;
out.b = b_out;
out.z_b = z_b_out;

% % plot results
% figure(21),clf
% plot(a,z_a,'ko')
% hold on
% plot(a_fit,z,'k.')
% plot(a_out,z_a_out,'k*')
% 
% plot(ab,z_ab,'ro')
% plot(ab_fit,z,'r.')
% plot(ab_out,z_ab_out,'r*')
% 
% plot(b_fit,z,'b.')
% plot(b_out,z_b_out,'b*')
% 
% plot(d.a,d.z,'k-','Linewidth',2)
% plot(d.b,d.z,'b-','Linewidth',2)
% plot(d.a - d.b,d.z,'r--','Linewidth',2)
% 
% set(gca,'YDir','reverse')
% % ylim([0 30]),xlim([-0.05 0.4])

