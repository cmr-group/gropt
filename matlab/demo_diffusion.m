%% Simple diffusion solver
params = struct;
params.mode = 'diff_bval';  % This tells the optimization to maximize b-value
params.gmax = 0.04;  % Gmax in T/m
params.smax = 200.0;  % Slewmax in T/m/s
params.MMT = 1;  % Highest moment order to null
params.TE = 60.0;  % The total TE in ms (optimization length will be smaller based on T_readout
params.T_readout = 12.0;  % Duration after wavefore before TE in ms
params.T_90 = 3.0;  % Duration of the RF after excitation in ms
params.T_180 = 6.0;  % Duration of the 180 pulse in ms
params.dt = 500e-6;  % Temporal resolution of the waveform to optimize
params.dt_out = 10e-6;  % Linearly interpolate the output waveform to this resolution

% Run the optimization
[G, lim_break] = gropt(params);

plot_waveform(G, params)

%% Diffusion min_TE finder

% This finds the shortest possible waveform with bval = 250 given the above
% params
target_bval = 250.0;
min_TE = 30.0;
max_TE = 200.0;
G_min = get_min_TE_diff(target_bval, 30.0, 200.0, params);

plot_waveform(G_min, params)
