function [all_lam, all_e] = get_eddy(lam_max, G, dt)
%GET_EDDY Summary of this function goes here
%   Detailed explanation goes here

N_lam = 200;
all_lam = linspace(1e-4,lam_max,N_lam);
all_e = zeros(1, N_lam);
ii = 1;
for lam = all_lam
    lam_s = lam * 1.0e-3;
    r = diff(exp(-[1:numel(G)+1].*dt./lam_s));
    r = r(end:-1:1);
    e = dot(r,G);
    all_e(ii) = e;
    ii = ii + 1;
end

if all_e(2) < 0
    all_e = -all_e;
end

end

