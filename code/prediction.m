function [error_deter, error_random] = prediction(model, X, y, n_tasks, parameters)
if strcmp(parameters.paradigm,'CL')
    b = model{3};
    if n_tasks>1
        for k = 1:n_tasks-1
            x_test = X{k};
            y_test = y{k};
            [error_deter(k), error_random(k)] = prediction_x(x_test, y_test, b.mu(:, k), b.varphi(k), parameters);
        end
        x_test = X{n_tasks};
        y_test = y{n_tasks};
        f = model{2};
        [error_deter(n_tasks), error_random(n_tasks)] = prediction_x(x_test, y_test, f.mu(:, n_tasks), f.varphi(n_tasks), parameters);

    else
        x_test = X{1};
        y_test = y{1};
        f = model{2};
        [error_deter, error_random] = prediction_x(x_test, y_test, f.mu(:, n_tasks), f.varphi(n_tasks), parameters);
    end
elseif strcmp(parameters.paradigm,'MTL')
    b = model;
    for k = 1:n_tasks
        x_test = X{k};
        y_test = y{k};
        [error_deter(k), error_random(k)] = prediction_x(x_test, y_test, b.mu(:, k), b.varphi(k), parameters);
    end
elseif strcmp(parameters.paradigm,'SCD')
    f = model{2};
    %%% Forward learning
    [error_deter, error_random] = prediction_x(X, y, f.mu(:, end), f.varphi(end), parameters);

elseif strcmp(parameters.paradigm,'MDA')
    mu = model.mu;
    varphi = model.varphi;
    x_test = X{n_tasks};
    y_test = y{n_tasks};
    [error_deter, error_random] = prediction_x(x_test, y_test, mu(:, 1), varphi, parameters);

end
end

function [error_deter, error_random] = prediction_x(x, y, mu, varphi, parameters)
%{
   Prediction

   This function computes the classification error and predicts the labels

   Input
   -----

   x: instance

   y: labels

   mu: classifier parameter

   parameters: model parameters

   Output
   ------
   
   error: classification error

%}
n_classes = parameters.n_classes;
mistakes_deter = [];
mistakes_random = [];
for j = 1:length(x(:, 1))
    [hat_y_deter, hat_y_random] = predict_label(x(j, :), mu, varphi, n_classes, parameters);
    if hat_y_deter ~= y(j) % Classification error
        mistakes_deter(j) = 1;
    else
        mistakes_deter(j) = 0;
    end
    if hat_y_random ~= y(j) % Classification error
        mistakes_random(j) = 1;
    else
        mistakes_random(j) = 0;
    end
end
error_deter = mean(mistakes_deter);
error_random = mean(mistakes_random);
end

function [y_deter, y_random] = predict_label(x, mu, varphi, n_classes, parameters)
%{

   Predict

   This function assigns labels to instances

   Input
   -----

   x: instance

   mu: classifier parameter

   n_classes: number of classes

   Output
   ------

   y_pred: predicted label

%}
for j=1:n_classes
    M(j,:)=feature_vector(x', j-1, n_classes, parameters.feature_mapping, parameters.feature_parameters)';
    c(j) = max([(M(j, :)*mu-varphi), 0]);
end
cx = sum(c);
for j=1:n_classes
    if cx == 0
        h(j)=1/n_classes;
    else
        h(j)=c(j)/cx;
    end
end
[~, y] = max(c);
y_deter = y-1;
y_random=find(mnrnd(1,h)==1)-1;
end