function s = Sigmoid(w, X)
    s = 1.0./(1.0 + exp(-w' * X));
end