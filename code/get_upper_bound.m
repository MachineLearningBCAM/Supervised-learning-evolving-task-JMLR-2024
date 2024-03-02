function R = get_upper_bound(model, parameters, k)
%{
﻿    Get upper bound
     This function obtains the bound for the error probability of each task
 
     Input
     -----
     model: model
     k: step
     parameters: model parameters

     Output
     ------
     R: bound for the error probability
%}
if strcmp(parameters.paradigm, 'CL')
    if k > 1
        b = model{3};
        R = b.R_Ut;
    end
    f = model{2};
    R(k) = f.R_Ut(end);
elseif strcmp(parameters.paradigm, 'MTL') || strcmp(parameters.paradigm, 'MDA')
    b = model;
    R = b.R_Ut;
elseif strcmp(parameters.paradigm, 'SCD')
    b = model{2};
    R = b.R_Ut(end);
end
end