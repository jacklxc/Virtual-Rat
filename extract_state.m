function out = extract_state(peh, state, first)
% Copied from https://gitlab.erlichlab.org/erlichlab/elutils.git/stats
if nargin<3
    first = true;
end

out = nan(size(peh));
if first
    for tx = 1:numel(peh)
        try
            out(tx) = peh(tx).states.(state)(1);
        catch
            
        end
    end
else
    for tx = 1:numel(peh)
        try
            out(tx) = peh(tx).states.(state)(end);
        catch
            
        end
    end
end
