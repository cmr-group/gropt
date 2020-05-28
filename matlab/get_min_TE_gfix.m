function [G_out, T_out] = get_min_TE_gfix( params, T_hi )
%GET_MIN_TE Summary of this function goes here
%   Detailed explanation goes here

dt = params.dt;
    
gfix_init = params.gfix;
free_vals = gfix_init<-9999;
dd = diff(free_vals);
block_check = sum(abs(dd(:)));
if block_check > 2
    fprintf('ERROR: Found more than one consecutive free block in gfix, TE finder not supported yet\n');
end
N_free = sum(free_vals(:));
N_fix = numel(free_vals) - N_free;
T_fix = N_fix * params.dt;
fprintf('Fixed region of waveform = %f ms\n', T_fix*1e3);
block_start = find(dd==1)+1;
block_stop = find(dd==-1);

gfix_p0 = gfix_init(1:block_start-1);
gfix_p2 = gfix_init(block_stop+1:end);


T_lo = (N_fix+1) * params.dt * 1e3;
T_range = T_hi-T_lo;

fprintf('Searching between %f and %f ms\n', T_lo, T_hi);

best_time = 999999.9;

fprintf('Testing TE =');
while ((T_range*1e-3) > (dt/4.0))
    TE = T_lo + (T_range)/2.0;
    params.TE = TE;
    NN = floor(TE*1e-3/params.dt) + 1;
    gfix_p0 = gfix_init(1:block_start-1);
    gfix_p2 = gfix_init(block_stop+1:end);
    gfix_TE = [gfix_p0 ones(1,NN-N_fix)*-99999 gfix_p2];
    params.gfix = gfix_TE;
    
    fprintf(' %.3f', params.TE);
    [G, lim_break, params] = gropt(params);
    if lim_break == 0
        T_hi = params.TE;
        if T_hi < best_time
            G_out = G;
            T_out = T_hi;
            best_time = T_hi;
        end
    else
        T_lo = params.TE;
    end
    T_range = T_hi-T_lo;
end

fprintf(' Final TE = %.3f ms\n', T_out);

end

