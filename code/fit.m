function [model] = fit(X, y, model, k, parameters)
%{
﻿    Fit
     This function fits the model

     Input
     -----
     X and y: sample sets
     model: model at previous time in CL and SCD, and empty in MTL, MDA
     k: step
     parameters: model parameters

     Output
     ------
     model: model
%}
n_tasks = length(X);
paradigm = parameters.paradigm;
if strcmp(paradigm, 'CL')

    if k == 1
        %%% Initialization of mean and confidence vectors
        [stl, f, b] = initialize();
    else
        stl = model{1};
        f = model{2};
        b = model{3};
    end
    %%% Single task learning
    stl = single_task(k, X, y, stl, parameters);

    %%% Forward learning
    f = forward(k, stl, f, parameters);

    %%% Forward and backward learning
    b = backward(k, stl, f, b, parameters);

    model{1} = stl;
    model{2} = f;
    model{3} = b;

elseif strcmp(paradigm, 'SCD')
    if k == 1
        [stl, f, ~] = initialize();
    else
        stl = model{1};
        f = model{2};
    end
    %%% Single task learning
    stl = single_task(k, X, y, stl, parameters);
    %%% Forward learning
    f = forward(k, stl, f, parameters);

    model{1} = stl;
    model{2} = f;

elseif strcmp(paradigm, 'MTL')
    [stl, f, b] = initialize();
    for k = 1:n_tasks
        stl = single_task(k, X{k}, y{k}, stl, parameters);
        f = forward(k, stl, f, parameters);
    end
    b = backward(k, stl, f, b, parameters);
    model = b;
elseif strcmp(paradigm, 'MDA')
    [stl, f, ~] = initialize();
    for k = 1:n_tasks
        stl = single_task(k, X{k}, y{k}, stl, parameters);
        f = forward(k, stl, f, parameters);
    end
    opt{1} = f;
    opt{2} = stl.x{n_tasks};
    prmt = optimization('f', n_tasks, parameters, f, opt);
    model = prmt;
end
end