function ys = qfind(x,ts)
% Copied from https://gitlab.erlichlab.org/erlichlab/elutils.git/stats

% function y= stats.qfind(x,ts)
% x is a vector , t is the target (can be one or many targets),
% y is same length as ts
% does a binary search: assumes that x is sorted low to high and unique.
ys=zeros(size(ts));

for i=1:length(ts)
    t=ts(i);
    if isnan(t)
        y=nan;
    else
    high = length(x);
    low = 0;
    if t>=x(end)
        y=length(x);
    else
        try
            while (high - low > 1) 
                probe = ceil((high + low) / 2);
                if (x(probe) > t)
                    high = probe;
                else
                    low = probe;
                end
            end
            
            y=low;
        catch 
            y=low;
        end
    end
    end
    
    ys(i)=y;
end

