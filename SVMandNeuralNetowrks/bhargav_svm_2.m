Labels=zeros(10000,1);
Labels([1:500,1001:1500,2001:2500,3001:3500,4001:4500,5001:5500,6001:6500,7001:7500,8001:8500,9001:9500],1)=1;
Data=load('data.tsv');
k=10;
crvf = crossvalind('Kfold',Labels, k);  
cp = classperf(Labels);                      

for i = 1:k                                  
    testInd = (crvf == i);               
    trainInd = ~testInd;                     
  
    svmModel = svmtrain( Data(trainInd,:), Labels(trainInd), 'Autoscale',true, 'Showplot',false, 'Method','QP', 'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',0.2);

    pred = svmclassify(svmModel, Data(testInd,:), 'Showplot',false);

    cp = classperf(cp, pred, testInd);
end

%# accuracy
cp.CorrectRate

%# confusion matrix
cp.CountingMatrix