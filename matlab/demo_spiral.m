%%
% You need to download Brian Hargreaves spiral code:
% https://mrsrl.stanford.edu/~brian/vdspiral/
% and add it to path
addpath('./vdspiral');

%%
% We start by making a spiral (similar to vdspiral/vdsexample.m)
% Note this code doesn't handle big dt (T) well, so keep it small, and
% resample later if desired.

smax = 15000;	 % 150 G/cm/s
gmax = 4;	 % G/cm
T = 4e-6;	 % Seconds
N = 32;		 % Interleaves
Fcoeff = [24 -12]; 	% FOV decreases linearly from 24 to 12cm.
res = 1;
rmax = 5/res;		% cm^(-1), corresponds to 1mm resolution.

disp('Calculating Gradient');
[k,g,s,time,r,theta] = vds(smax,gmax,T,N,Fcoeff,rmax);

disp('Plotting Gradient');
g = [real(g(:)) imag(g(:))];
plotgradinfo(g,T);

%%
% Now we design a M1 comped refocuser for Gx

% Change dt to 40e-6 seconds for compute speed (10x the dt above)
down_sample = 10;
dt = T * down_sample;
Gx_spiral = g(1:down_sample:end, 1) ./ 100; % downsample and fix units

% Gropt params
params = struct;
params.mode = 'free';
params.gmax = gmax/100; % same gmax as spiral, fix units
params.smax = smax/100; % same smax as spiral, fix units

% Full TR should be M0 and M1 comped
params.moment_params = [];
params.moment_params(:,end+1) = [0, 0, 0, -1, -1, 0, 1.0e-4];
params.moment_params(:,end+1) = [0, 1, 0, -1, -1, 0, 1.0e-4];
params.dt = dt;

% gfix is the fixed gradient values, with blocks of big negative values
% designating regions free to optimize
N_fix = 100; % This doesnt matter if we use the TE finder, it will get resized
gfix = [Gx_spiral.' ones(1, N_fix).*-999999 0];

params.gfix = gfix;

% Run the TE finder (the second argument is the upper bound on T);
[G, T_min] = get_min_TE_gfix(params, 10.0);

%% Plots
% TODO: Fix units and labels

figure()
plot(G);
hold on;
plot([min(xlim()),max(xlim())],[0,0], 'k--');
title('Gx');

figure()
plot(diff(G)./dt);
hold on;
plot([min(xlim()),max(xlim())],[0,0], 'k--');
title('Slew Gx');

% M0 with lazy (no) units
figure()
plot(cumsum(G));
hold on;
plot([min(xlim()),max(xlim())],[0,0], 'k--');
title('M0 Gx [AU]');

% M1 with lazy (no) units
figure()
plot(cumsum(G .* [1:numel(G)]));
hold on;
plot([min(xlim()),max(xlim())],[0,0], 'k--');
title('M1 Gx [AU]');

