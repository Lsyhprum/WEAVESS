function [B,elapse] = LSH_compress(A, maxbits, model)
tmp_T = tic;

Ym = A * model.U;
tmp = zeros(size(Ym, 1), size(Ym, 2));
B = (Ym > tmp);

elapse = toc(tmp_T);
end
