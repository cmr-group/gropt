function [G1, G2] = split_diff(G, params)
% Splits a gropt diffusion waveform into the individual components

dt = params.dt_out;
if dt < 0
    dt = params.dt;
end

TE = params.TE;
if TE > 1
    TE = TE * 1e-3;
end

idx_mid = round(TE/2/dt);

G1 = G(1:idx_mid);
G2 = G(idx_mid:end);

G1 = G1(find(G1,1,'first')-1:find(G1,1,'last')+1);
G2 = G2(find(G2,1,'first')-1:find(G2,1,'last')+1);
end

