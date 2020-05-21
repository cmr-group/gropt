%% 
% This is the same G_ss from demo_moments.m, except for this example we dont
% need to know the moments because the slice select will be included in the
% optimization

dt = 10e-6;

g_ss = .0070;
N_ss = round(1.375/(g_ss*dt)*1e-6);
G_ss = ones(1, N_ss) .* g_ss;

figure()
plot(G_ss)
%% Compute flow comped refocuser
params = struct;
params.mode = 'free';
params.gmax = 0.05;
params.smax = 100.0;
params.moment_params = [];
params.moment_params(:,end+1) = [0, 0, 0, -1, -1, 0, 1.0e-4];
params.moment_params(:,end+1) = [0, 1, 0, -1, -1, 0, 1.0e-4];
params.dt = dt;

TE = 1.0e-3;
N = TE/dt;
gfix = ones(1,N) * -999999;
gfix(1) = 0.0;
gfix(end) = 0.0;
gfix(1:N_ss) = G_ss;

params.TE = 1.0;
params.gfix = gfix;

[G, T_min] = get_min_TE_gfix(params, 1.0);

figure();
plot(G);

%% Same code as demo_moments.m, but once again we dont need to append G_ss, its already in there
tvec = (0:numel(G)-1)*dt*1e3; 
tMat = zeros( Nm, numel(G) );
for mm=1:Nm,
    tMat( mm, : ) = tvec.^(mm-1);
end

moments = (1e3*1e3*dt*tMat*(G'));
fprintf('Final moments = %.03f  %.03f\n', moments(1), moments(2));