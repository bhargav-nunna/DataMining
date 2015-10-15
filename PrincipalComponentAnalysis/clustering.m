%Principal Component Analysis to determine the clusters%

data=load('data.tsv'); %Change the file location to the location where currenct input data file is located i%
for x=1:1000
    data(:,x)=zNrom(data(:,x));
end
data2=(data'*data);
[V,D]=eig(data2);
plot(data*V(:,999),data*V(:,1000),'.');
