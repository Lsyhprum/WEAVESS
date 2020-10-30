function [model, B,elapse] = LSH_learn(A, maxbits)
tmp_T = tic;
[Nsamples,Nfeatures] = size(A);
k = maxbits;
U = normrnd(0, 1, Nfeatures, k);
Z = A * U;
tmp = zeros(size(Z, 1), size(Z, 2));
B = (Z > tmp);
model.U = U;

elapse = toc(tmp_T);
end
