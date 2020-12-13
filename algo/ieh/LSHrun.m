function LSHrun(dataset, base_path, query_path, ground_path, LSHtable_path, LSHfunc_path, n_threads)

trainset = fvecs_read (base_path);
testset = fvecs_read (query_path);
gnd = ivecs_read (ground_path);

num_threads = str2num(n_threads);
disp(dataset)
%disp(trainset)
trainset = trainset';
testset = testset';
gnd = gnd';
%disp(trainset(1:3,:));

tic
[model, trainB ,train_elapse] = LSH_learn(trainset, 16);
[testB,test_elapse] = LSH_compress(testset, 16, model);
%disp(train_elapse);
%disp(test_elapse);
%{
hamming_radius = 1;
idx = zeros(size(testB,1),1);
parfor (i = 1 : size(testB,1), num_threads)
min_dis = 10000000;
  for j = 1 : size(trainB,1)
    hamming_dis = sum(xor(testB(i,:),trainB(j,:)));
    if hamming_dis < 2
      L2dis = sum((testset(i,:) - trainset(j,:)).^2);
      if L2dis < min_dis
        min_dis = L2dis;
        idx(i) = j-1;
      end
    end
  end

end
error = idx - gnd(:,1);
errornum = length(find(error~=0));
disp(errornum);
%}
toc

fid = fopen(LSHtable_path,'wt');
for i = 1 : size(trainB,1);
  fprintf(fid,'%g ',trainB(i,:));
  fprintf(fid,'\n');
end
fclose(fid);

fid = fopen(LSHfunc_path,'wt');
for i = 1 : size(model.U,1);
  fprintf(fid,'%f ',model.U(i,:));
  fprintf(fid,'\n');
end
fclose(fid);
