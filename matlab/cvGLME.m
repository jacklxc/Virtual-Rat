function LL=cvGLME(TABLE,formula, fold)
% Perform k-fold cross-validation on GLME model fiitted on TABLE using 
% formula provided

[col, ~] = size(TABLE);
single = floor(col / fold);
LL = 0;
for nx=1:fold
    test = TABLE(((nx-1)*single+1):nx*single,:);
    train = [TABLE(1:single*(nx-1),:);TABLE(single*nx+1:end,:)];
    model = fitglme(train, formula,'distribution','binomial');
    fitted = predict(model,test);
    LL = LL + sum(log(fitted(test.hit==1))) + sum(log(1-fitted(test.hit==0)));
end


