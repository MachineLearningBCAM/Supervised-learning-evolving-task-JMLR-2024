function [prmt] = optimization(type, k, parameters, c, opt)
%{
    Optimization

    This function solves the optimization using an algorithm based on
    Nesterov's method

   Input
   -----

    k: step

    parameters: model paramaters

    opt: parameters required to solve the optimization

    type:
        's' single task learning
        'f' forward learning
        'b' forward and backward learning

   Output
   ------
    
    prmt: classifier parameter and parameters of the optimization
%}
paradigm = parameters.paradigm;
lambda0 = parameters.lambda0;
n_classes = parameters.n_classes;
max_iter = parameters.max_iter;
m = parameters.m;
theta = 1;
theta0 = 1;
R_Ut = 0;
iter = 0;
tau(:, 1) = c.tau(:, k);
lambda(:, 1) = lambda0.*c.lambda(:, k);

if strcmp(paradigm, 'MTL') || strcmp(paradigm, 'SCD')
    f = opt{1};
    x = opt{2};
    F = [];
    h = [];
    if k == 1
        mu = zeros(m, 1);
        w = zeros(m, 1);
        w0(:, 1) = zeros(m, 1);
    else
        mu = c.mu(:, k-1);
        w = c.w(:, k-1);
        w0 = c.w0(:, k-1);
    end
    for i = 1:length(x(:, 1))
        M = [];
        for j = 1:n_classes
            M(end+1,:)=feature_vector(x(i, :)',j-1,n_classes, parameters.feature_mapping, parameters.feature_parameters)';
        end
        for j=1:n_classes
            aux=nchoosek(1:n_classes,j);
            for k=1:length(aux(:,1))
                idx=zeros(1,n_classes);
                idx(aux(k,:))=1;
                F(end+1, :) = (idx*M)./j;
                h(end+1, 1) = - 1/j;
            end
        end
    end
elseif strcmp(paradigm, 'MDA')
    F = [];
    h = [];
    x = opt{2};
    mu = zeros(m, 1);
    w = zeros(m, 1);
    w0(:, 1) = zeros(m, 1);
    for i = 1:length(x(:, 1))
        M = [];
        for j = 1:n_classes
            M(end+1,:)=feature_vector(x(i, :)',j-1,n_classes, parameters.feature_mapping, parameters.feature_parameters)';
        end
        for j=1:n_classes
            aux=nchoosek(1:n_classes,j);
            for k=1:length(aux(:,1))
                idx=zeros(1,n_classes);
                idx(aux(k,:))=1;
                F(end+1, :) = (idx*M)./j;
                h(end+1, 1) = - 1/j;
            end
        end
    end
elseif strcmp(paradigm, 'CL')
    if type == 'f'
        f = opt{1};
        x = opt{2};
        F = [];
        h = [];
        if k == 1
            mu = zeros(m, 1);
            w = zeros(m, 1);
            w0(:, 1) = zeros(m, 1);
        else
            mu = f.mu(:, k-1);
            w = f.w(:, k-1);
            w0 = f.w0(:, k-1);
        end
        for i = 1:length(x(:, 1))
            M = [];
            for j = 1:n_classes
                M(end+1,:)=feature_vector(x(i, :)',j-1,n_classes, parameters.feature_mapping, parameters.feature_parameters)';
            end
            for j=1:n_classes
                aux=nchoosek(1:n_classes,j);
                for k=1:length(aux(:,1))
                    idx=zeros(1,n_classes);
                    idx(aux(k,:))=1;
                    F(end+1, :) = (idx*M)./j;
                    h(end+1, 1) = - 1/j;
                end
            end
        end
    elseif type == 'b'
        f = opt{1};
        mu = f.mu(:, k);
        w = f.w(:, k);
        w0 = f.w0(:, k);
        F = f.F{k};
        h = f.h{k};
    elseif type == 's'
        x = opt{2};
        F = [];
        h = [];
        mu = zeros(m, 1);
        w = zeros(m, 1);
        w0(:, 1) = zeros(m, 1);
        for i = 1:length(x(:, 1))
            M = [];
            for j = 1:n_classes
                M(end+1,:)=feature_vector(x(i, :)',j-1,n_classes, parameters.feature_mapping, parameters.feature_parameters)';
            end
            for j=1:n_classes
                aux=nchoosek(1:n_classes,j);
                for k=1:length(aux(:,1))
                    idx=zeros(1,n_classes);
                    idx(aux(k,:))=1;
                    F(end+1, :) = (idx*M)./j;
                    h(end+1, 1) = - 1/j;
                end
            end
        end
    else
        disp('Error')
    end
end
muaux = mu;
v = F*muaux + h;
[varphi, ~] = max(v');
regularization = 0;
for i = 1:m
    regularization = regularization + (lambda(i)*abs(muaux(i)));
end
R_Ut_best_value = 1 - tau'*muaux + varphi + regularization;
while iter < max_iter
    iter = iter+1;
    muaux = w + theta*((1/theta0) - 1)*(w-w0);
    v = F*muaux + h;
    [varphi, idx_mv] = max(v');
    fi = F(idx_mv, :);
    regularization = 0;
    for i = 1:m
        subgradient_regularization(i) = lambda(i)*sign(muaux(i));
        regularization = regularization + (lambda(i)*abs(muaux(i)));
    end
    g = - tau + fi' + subgradient_regularization';
    theta0 = theta;
    theta = 2/(iter+1);
    alpha = 1/((iter+1)^(3/2));
    w0 = w;
    w = muaux - alpha*g;
    R_Ut = 1 - tau'*muaux + varphi + regularization;
    if R_Ut < R_Ut_best_value
        R_Ut_best_value = R_Ut;
        mu = muaux;
    end
end
v = F*w + h;
[varphi, ~] = max(v');
regularization = 0;
for i = 1:length(lambda)
    regularization = regularization + (lambda(i)*abs(w(i)));
end
R_Ut = 1 - tau'*w + varphi + regularization;
if R_Ut < R_Ut_best_value
    R_Ut_best_value = R_Ut;
    mu = w;
end
prmt.R_Ut = R_Ut;
prmt.mu = mu;
prmt.varphi = varphi;
if type == 'f'
    prmt.F = F;
    prmt.h = h;
    prmt.w = w;
    prmt.w0 = w0;
end

if strcmp(paradigm, 'MTL') || strcmp(paradigm, 'SCD')
    prmt.w = w;
    prmt.w0 = w0;
end
end
