function [out_sensitivity] = relu_backprop(in_sensitivity, in)
%The backpropagation process of relu
%   input paramter:
%       in_sensitivity  : the sensitivity from the upper layer, shape: 
%                       : [number of images, number of outputs in feedforward]
%       in              : the input in feedforward process, shape: same as in_sensitivity
%   
%   output paramter:
%       out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity

% TODO
relu_grad = in;
relu_grad(relu_grad < 0) = 0;
relu_grad(relu_grad > 0) = 1;
out_sensitivity = relu_grad .* in_sensitivity;
end

