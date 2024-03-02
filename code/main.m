%%% Load dataset
load('data.mat')

%%%%% Input parameters
parameters.batch_size = 10; % Samples per task, -1 if we use all the samples
parameters.n_classes = 2; % Number of classes
parameters.b_steps = 3; % Number of backward steps
parameters.w = 2; % Window size
parameters.max_iter = 2000; % Maximum number of iterations in the optimization
parameters.lambda0 = 0.7;
parameters.m = parameters.n_classes*(length(X_train{1}(1, :))+1); % Length of the feature vector
parameters.paradigm = 'CL';
%%%%%

%%%%% Feature mapping
d = length(X_train{1}(1, :));
D = 200;
parameters.feature_mapping = 'linear';
if strcmp(parameters.feature_mapping,'linear')
    parameters.feature_parameters = [];
    parameters.m = parameters.n_classes*(d+1);
elseif strcmp(parameters.feature_mapping,'RFF')
    parameters.feature_parameters{1} = D;% random Fourier components D
    parameters.feature_parameters{2} = (1/20)*randn(d, feature_parameters{1});
    parameters.m = parameters.n_classes*2*feature_parameters{1};
end
%%%%%

%%% Number of tasks
n_tasks = length(X_train);
error_backward = zeros(n_tasks, n_tasks);

%%% Initialization of mean and confidence vectors
model = [];

%%% Continual learning

if strcmp(parameters.paradigm, 'CL')
    %%% Start running
    for k = 1:n_tasks
        model = fit(X_train{k}, Y_train{k}, model, k, parameters);
        %%% Prediction
        [error_deter{k}, error_random{k}] = prediction(model, X_test(1, 1:k), Y_test(1, 1:k), k, parameters);
        bound{k} = get_upper_bound(model, parameters, k);
    end
elseif strcmp(parameters.paradigm, 'MTL')
    model = fit(X_train, Y_train, model, n_tasks, parameters);
    [error_deter, error_random] = prediction(model, X_test, Y_test, n_tasks, parameters);
    bound = get_upper_bound(model, parameters, n_tasks);
elseif strcmp(parameters.paradigm, 'MDA')
    model = fit(X_train, Y_train, model, n_tasks, parameters);
    [error_deter, error_random] = prediction(model, X_test, Y_test, n_tasks, parameters);
    bound = get_upper_bound(model, parameters, n_tasks);
elseif strcmp(parameters.paradigm, 'SCD')
    for k = 1:n_tasks
        model = fit(X_train{k}, Y_train{k}, model, k, parameters);
        %%% Prediction
        [error_deter(k), error_random(k)] = prediction(model,  X_test{k}, Y_test{k}, k, parameters);
        bound(k) = get_upper_bound(model, parameters, k);

    end
end
