function f = forward(k, stl, f, parameters)
%{
    Forward learning

    This function obtains classification rules performing forward learning

   Input
   -----

    k: step

    stl: mean and confindence vectors single task learning

    f: mean, confidence vectors, and classifier parameters forward learning

    parameters: model paramaters

   Output
   ------

    f: mean, confidence vectors, and classifier parameters forward learning
%}
paradigm = parameters.paradigm;
w = parameters.w;
m = parameters.m;
if strcmp(paradigm, 'CL') || strcmp(paradigm, 'MTL')
    if k == 1
        f.tau(:, 1) = stl.tau(:, k);
        f.s(:, 1) = stl.s(:, k);
        f.lambda(:, 1) = stl.lambda(:, k);
        f.d(:, 1) = expected_change_forward(m, k, w, stl);
    else
        f.d(:, k) = expected_change_forward(m, k, w, stl);
        f = tracking(f, k, parameters.m, stl, parameters);
    end
elseif strcmp(paradigm, 'SCD')
    f.d(:, k) = expected_change_forward(m, k, w, stl);
    if k == 1
        f.tau(:, 1) = zeros(m, 1);
        f.s(:, 1) = 999*ones(m, 1);
        f.lambda(:, 1) = 999*ones(m, 1);
        f.d(:, 1) = expected_change_forward(m, k, w, stl);
    else
        f = tracking(f, k, parameters.m, stl, parameters);
    end
elseif strcmp(paradigm, 'MDA')
    if k == 1
        f.tau(:, 1) = stl.tau(:, k);
        f.s(:, 1) = stl.s(:, k);
        f.lambda(:, 1) = stl.lambda(:, k);
        f.d(:, 1) = expected_change_forward(m, k, w, stl);
    else
        f.d(:, k) = expected_change_forward(m, k, w, stl);
        f = tracking(f, k, parameters.m, stl, parameters);
    end
end
opt{1} = f;
if strcmp(paradigm, 'CL') || strcmp(paradigm, 'SCD')
    opt{2} = stl.x;
    prmt = optimization('f', k, parameters, f, opt);
    f.mu(:, k) = prmt.mu;
    f.varphi(k) = prmt.varphi;
    f.R_Ut(k) = prmt.R_Ut;
    f.w(:, k) = prmt.w;
    f.w0(:, k) = prmt.w0;
    if strcmp(paradigm, 'CL')
        f.F{k} = prmt.F;
        f.h{k} = prmt.h;
    end
elseif strcmp(paradigm, 'MTL') || strcmp(paradigm, 'MDA')
    opt{2} = stl.x{k};
end
end

function f = tracking(f, k, m, stl, parameters)
%{
﻿   Tracking

     This function obtains mean vector estimates and confidence vectors

     Input
     -----
    k: step

    stl: mean and confindence vectors single task learning

    f: mean, confidence vectors, and classifier parameters forward learning

    m: length feature vector

     Output
     ------

    f: mean and confidence vectors forward learning
%}
paradigm = parameters.paradigm;
if strcmp(paradigm, 'CL') || strcmp(paradigm, 'MTL') || strcmp(paradigm, 'MDA')
    Phi(:, 1) = stl.tau(:, k);
    R = stl.s(:, k);
    for i = 1:m
        if R(i) == 0
            f.tau(i, k) = Phi(i, 1);
            f.s(i, k) = 0;
        else
            f.tau(i, k) = Phi(i, 1) + (R(i)/(f.s(i, k-1) + f.d(i, k) + R(i)))*(f.tau(i, k-1)-Phi(i, 1));
            f.s(i, k) = (R(i)^(-1) + (f.s(i, k-1)+f.d(i, k))^(-1))^(-1);
        end
        f.lambda(i, k) = sqrt(f.s(i, k));
    end
elseif strcmp(paradigm, 'SCD')
    Phi(:, 1) = stl.tau(:, k-1);
    R = stl.s(:, k-1);
    for i = 1:m
        if R(i) == 0
            f.tau(i, k) = Phi(i, 1);
            f.s(i, k) = 0;
        else
            f.tau(i, k) = f.tau(i, k-1) + (f.s(i, k-1)/(f.s(i, k-1) + R(i)))*(Phi(i, 1)-f.tau(i, k-1));
            f.s(i, k) = (R(i)/(f.s(i, k-1) + R(i)))*f.s(i, k-1)+f.d(i, k);
        end
        f.lambda(i, k) = sqrt(f.s(i, k));
    end
end
end

function d_forward = expected_change_forward(m, k, w, stl)
if k == 1
    d_forward(:, k) = 0.01.*ones(m, 1);
else
    j = max([k-w, 1]);
    for i = j+1:k
        vector_d(:, i-j) = (stl.tau(:, i)-stl.tau(:, i-1)).^2;
    end
    if length(vector_d(1, :))>1
        d_forward(:, 1) = mean(vector_d');
    else
        d_forward(:, 1) = vector_d;
    end
end
end
