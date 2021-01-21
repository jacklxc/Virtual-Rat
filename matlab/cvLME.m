function LL=cvLME(TABLE,formula, fold)
% Perform k-fold cross-validation on RNN's LME model fiitted on TABLE using 
% formula provided

[col, ~] = size(TABLE);
single = floor(col / fold);
LL = 0;
for nx=1:fold
    test = TABLE(((nx-1)*single+1):nx*single,:);
    train = [TABLE(1:single*(nx-1),:);TABLE(single*nx+1:end,:)];
    model = fitlme(train, formula);
    fitted = predict(model,test);
    LL = LL + sum(log(fitted(test.hit_rate>0.5))) + sum(log(1-fitted(test.hit_rate<=0.5)));
end


