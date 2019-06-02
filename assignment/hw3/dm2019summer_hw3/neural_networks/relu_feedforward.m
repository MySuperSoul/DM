function [ out ] = relu_feedforward( in )
%The feedward process of relu
%   inputs:
%           in	: the input, shape: any shape of matrix
%   
%   outputs:
%           out : the output, shape: same as in

% TODO
tmp = in;
tmp(tmp < 0) = 0;
out = tmp;
end
