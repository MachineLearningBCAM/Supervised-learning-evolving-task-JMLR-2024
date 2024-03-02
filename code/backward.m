function b = backward(k, stl, f, b, parameters)
%{
    Backward

    This function obtains classification rules performing forward and backward
    learning

   Input
   -----

    k: step

    stl: mean and confindence vectors single task learning

    f: mean, confidence vectors, and classifier parameters forward learning

    b: mean, confidence vectors, and classifier parameters forward and
    backward learning 

    parameters: model paramaters
    
   Output
   ------
    b: mean, confidence vectors, and classifier parameters forward and
    backward learning 
%}

m = parameters.m;
for i = 1:k
    b.d(:, i) = expected_change(k, i, parameters.w, stl.tau);
end
if strcmp(parameters.paradigm, 'CL')
    for i = k-1:-1:max([k-parameters.b_steps, 1])
        b = tracking_backward(k, i, parameters.m, stl, f, b, parameters);
        opt{1} = f;
        opt{2} = [];
        [prmt] = optimization('b', i, parameters, b, opt);
        b.mu(:, i) = prmt.mu;
        b.R_Ut(i) = prmt.R_Ut;
        b.varphi(i) = prmt.varphi;
    end
elseif strcmp(parameters.paradigm, 'MTL')
    b = tracking_backward(k, 1, parameters.m, stl, f, b, parameters);
    for i = 1:k
        opt{1} = f;
        opt{2} = stl.x{i};
        [prmt] = optimization('f', i, parameters, b, opt);
        b.mu(:, i) = prmt.mu;
        b.w(:, i) = prmt.w;
        b.w0(:, i) = prmt.w0;
        b.R_Ut(i) = prmt.R_Ut;
        b.varphi(i) = prmt.varphi;
    end
end
end

function b = tracking_backward(k, j, m, stl, f, b, parameters)
%{
﻿    Tracking_backward

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     k: step

     j: task index

     stl: mean and confidence vectors obtained at single task learning

     f: mean and confidence vectors obtained at forward learning

     b: mean and confidence vectors obtained at forward and backward learning
     
     m: length feature vector

     Output
     ------

    b: mean and confidence vectors obtained at forward and backward learning
%}
tau_backward = zeros(m , k);
s_backward = zeros(m , k);
tau_backward = zeros(m , k);
s_backward = zeros(m , k);

if strcmp(parameters.paradigm, 'CL')
    for i = k:-1:j+1
        if i == k
            tau_backward(:, i) = stl.tau(:, k);
            s_backward(:, i) = stl.s(:, k);
        else
            for c = 1:m
                if stl.s(c, i)  == 0
                    tau_backward(c, i) = stl.tau(c, i);
                    s_backward(c, i) = 0;
                else
                    tau_backward(c, i) = stl.tau(c, i) + (stl.s(c, i)/(s_backward(c, i+1) + b.d(c, i)+stl.s(c, i)))*(tau_backward(c, i+1)-stl.tau(c, i));
                    s_backward(c, i) = (stl.s(c, i)^(-1) + (s_backward(c, i+1) + b.d(c, i))^(-1))^(-1);
                end
            end
        end
    end
    for c = 1:m
        if f.s(c, j)== 0
            b.s(c, j) = f.s(c, j);
            b.tau(c, j) = f.tau(c, j);
        else
            b.tau(c, j) = f.tau(c, j) + (f.s(c, j)/(f.s(c, j) + s_backward(c, j+1)+b.d(c, j)))*(tau_backward(c, j+1) - f.tau(c, j));
            b.s(c, j) = (f.s(c, j)^(-1) + (s_backward(c, j+1)+b.d(c, j))^(-1))^(-1);
        end
    end
    b.lambda(:, j) = sqrt(b.s(:, j));
elseif strcmp(parameters.paradigm, 'MTL')
    for i = k:-1:1
        if i == k
            b.tau(:, i) = f.tau(:, k);
            b.s(:, i) = f.s(:, k);
        else
            for c = 1:m
                if f.s(c, i)  == 0
                    b.tau(c, i) = f.tau(c, i);
                    b.s(c, i) = 0;
                else
                    eta = f.s(c, i)/(f.s(c, i) + b.d(c, i));
                    b.tau(c, i) = f.tau(c, i) + eta*(b.tau(c, i+1)-f.tau(c, i));
                    b.s(c, i) = f.s(c, i) + (eta^2)*(b.s(c, i+1) - f.s(c, i) - b.d(c, i));
                end
            end
        end
        b.lambda(:, i) = sqrt(b.s(:, i));
    end
end
end

function d_backward = expected_change(t, i, w, tau_s)
w2 = floor((w)/2);
i1 = max([1, i-w2+2]);
i2 = min([t, i+w2]);
if i == t
    d_step = (tau_s(:, i)).^2;
end
for r = i1:i2
    if r == 1
        d_step(:, r-i1+1) = (tau_s(:, r)).^2;
    else
        d_step(:, r-i1+1) = (tau_s(:, r) - tau_s(:, r-1)).^2;
    end
end
if length(d_step(1, :))>1
    d_backward = mean(d_step');
else
    d_backward = d_step;
end
end