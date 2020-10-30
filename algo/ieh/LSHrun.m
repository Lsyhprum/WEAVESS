trainset = fvecs_read ('F:/ANNS/DATASET/sift1M/sift_base.fvecs');
testset = fvecs_read ('F:/ANNS/DATASET/sift1M/sift_query.fvecs');
gnd = ivecs_read ('F:/ANNS/DATASET/sift1M/sift_groundtruth.ivecs');
disp(trainset)
trainset = trainset';
testset = testset';
gnd = gnd';
disp(trainset(1:3,:));
[model, trainB ,train_elapse] = LSH_learn(trainset, 16);
[testB,test_elapse] = LSH_compress(testset, 16, model);
disp(train_elapse);
disp(test_elapse);
%{
hamming_radius = 1;
idx = zeros(size(testB,1),1);
for i = 1 : size(testB,1)
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
fid = fopen('LSHtableSift.txt','wt');
for i = 1 : size(trainB,1);
  fprintf(fid,'%g ',trainB(i,:));
  fprintf(fid,'\n');
end
fclose(fid);

fid = fopen('LSHfuncSift.txt','wt');
for i = 1 : size(model.U,1);
  fprintf(fid,'%f ',model.U(i,:));
  fprintf(fid,'\n');
end
fclose(fid);
