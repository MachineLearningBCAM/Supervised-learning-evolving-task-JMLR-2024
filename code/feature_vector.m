function phi=feature_vector(x,y, n_classes, feature_mapping, feature_parameters)
%{
   Feature_vector

   This function obtains feature vectors

   Input
   -----

   x: new instance

   y: new label

   n_classes: number of classes

   Output
   ------

   phi: feature vector

%}
if strcmp(feature_mapping,'linear')
    x_phi = [1; x];
elseif strcmp(feature_mapping,'RFF')
    D_kernel = feature_parameters{1};
    u = feature_parameters{2};
    x_phi = [cos(u'*x); sin(u'*x)];
end
e = zeros(n_classes, 1);
e(y+1) = 1;
phi = kron(e, double(x_phi));
end