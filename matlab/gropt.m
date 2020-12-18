function [ G, lim_break, params ] = gropt( params )
%GROPT calls the gropt mex functions and returns waveforms. 
%   
%    This does a lot of checks and conversions on the params struct, and
%    then calls the right gropt mex function.
%         
%    Parameters
%    ----------
%    params
%        struct containing all of the GrOpt constraint parameters
%
%   Returns
%   -------
%   G
%       This is the output gradient waveform vector
%   lim_break
%       This will return 0 if the waveform is feasible, 1 if not
%   params
%       Returns a copy of the params struct, which is changed by this
%       function in certain scenarios (this is mostly in TE searches)

    if isfield(params, 'mode')
        mode = params.mode;
    else
        fprintf('ERROR: params.mode does not exist.\n');
        return
    end
    
    if isfield(params, 'gmax')
        gmax = params.gmax;
        if gmax > 1
            gmax = gmax * 1e-3;
        end
    else
        fprintf('ERROR: params.gmax does not exist.\n');
        return
    end
    
    if isfield(params, 'smax')
        smax = params.smax;
    else
        fprintf('ERROR: params.smax does not exist.\n');
        return
    end
    
    % TODO: make TE a better distinction between diffusion and free modes
    if isfield(params, 'TE')
        TE = params.TE; 
    elseif isfield(params, 'T')
        TE = params.T;
    else
        fprintf('ERROR: params.TE does not exist.\n');
        return
    end
    
    if TE < 1
        TE = TE * 1e3;
    end
    
    if strcmp(mode, 'diff_bval') || strcmp(mode, 'diff_beta')
        if isfield(params, 'T_90')
            T_90 = params.T_90;
            if T_90 < .1
                T_90 = T_90 * 1e3;
            end
        else
            fprintf('ERROR: params.T_90 does not exist.\n');
            return
        end
        if isfield(params, 'T_180')
            T_180 = params.T_180;
            if T_180 < .1
                T_180 = T_180 * 1e3;
            end
        else
            fprintf('ERROR: params.T_180 does not exist.\n');
            return
        end
        if isfield(params, 'T_readout')
            T_readout = params.T_readout;
            if T_readout < .1
                T_readout = T_readout * 1e3;
            end
        else
            fprintf('ERROR: params.T_readout does not exist.\n');
            return
        end
        if isfield(params, 'MMT')
            MMT = params.MMT;
        else
            fprintf('ERROR: params.MMT does not exist.\n');
            return
        end
        moment_params = [];
        moment_params(:,end+1) = [0, 0, 0, -1, -1, 0, 1.0e-3];
        if MMT > 0
            moment_params(:,end+1) = [0, 1, 0, -1, -1, 0, 1.0e-3];
        end
        if MMT > 1
            moment_params(:,end+1) = [0, 2, 0, -1, -1, 0, 1.0e-3];
        end
    elseif strcmp(mode, 'free')
        T_readout = 0.0;
        T_90 = 0.0;
        T_180 = 0.0;
        if isfield(params, 'moment_params')
            moment_params = params.moment_params;
        elseif isfield(params, 'M0')
            moment_params = [];
            moment_params(:,end+1) = [0, 0, 0, -1, -1, params.M0, 1.0e-4];
            if isfield(params, 'M1')
                moment_params(:,end+1) = [0, 1, 0, -1, -1, params.M1, 1.0e-4];
            end
            
            
        else
            fprintf('ERROR: params.moment_params does not exist.\n');
            return
        end
    else
        fprintf('ERROR: params.mode = %s is not valid.\n', mode);
        return
    end
    
    if isfield(params, 'N0')
        N0 = params.N0;
        if isfield(params, 'dt')
%             fprintf('WARNING: dt and N0 entered, ignoring dt.\n');
        end
        dt = (TE-T_readout) * 1.0e-3 / N0;
        params.dt = dt;
    elseif isfield(params, 'dt')
        dt = params.dt;
        N0 = -1.0;
    else
        fprintf('ERROR: need params.dt or params.N0.\n');
        return
    end
    
    if isfield(params, 'dt_out')
        dt_out = params.dt_out;
    else
        dt_out = -1.0; % No output interpolation is default
    end
    
    
    if isfield(params, 'eddy_params')
        eddy_params = params.eddy_params;
    else
        eddy_params = []; % No eddy currents is default
    end
    
    if isfield(params, 'pns_thresh')
        pns_thresh = params.pns_thresh;
    else
        pns_thresh = -1.0; % No PNS threhold is default
    end
    
    if strcmp(mode, 'diff_bval')
        diffmode = 2;
    elseif strcmp(mode, 'diff_beta')
        diffmode = 1;
    else
        diffmode = 0;
    end
    
    if isfield(params, 'slew_reg')
        slew_reg = params.slew_reg;
    else
        slew_reg = 1.0;
    end
    
    if isfield(params, 'gfix')
        gfix = params.gfix;
    else
        gfix = [];
    end
    
    
    
    if N0 > 0
        [G, lim_break] = mex_gropt_diff_fixN(gmax, smax, moment_params, TE, T_readout, T_90, T_180, N0, diffmode, ...
        dt_out, pns_thresh, eddy_params, slew_reg);
    elseif dt > 0
        if numel(gfix) > 0
            [G, lim_break] = mex_gropt_diff_fixdt_fixG(gmax, smax, moment_params, TE, T_readout, T_90, T_180, dt, diffmode, ...
            dt_out, pns_thresh, eddy_params, slew_reg, gfix);
        else
            [G, lim_break] = mex_gropt_diff_fixdt(gmax, smax, moment_params, TE, T_readout, T_90, T_180, dt, diffmode, ...
            dt_out, pns_thresh, eddy_params, slew_reg);
        end
    end

    

end

