function [stl, f, b] = initialize()
%{

    Initialize

    This function initializes model variables

    Output
    ------

    stl: single task learning variables

    f: forward learning variables

    b: backward learning variables

%}

stl.tau = []; f.tau = []; b.tau = [];
stl.s = []; f.s = []; b.s = [];
stl.lambda = []; f.lambda = []; b.lambda = [];
f.d = [];
f.mu = []; f.R_Ut = [];
b.mu = []; b.R_Ut = [];
f.F = []; f.h = [];
end