% This script fits a Linear Mixed Effect model to show the
% correlation between hit rate of RNN and rule encoding score predicted by linear
% regression classifier.
%% Load data table
clear;
load('../mats/SGD_table_target_RNN.mat')
SGD_table = array2table(SGD_table,...
    'VariableNames',{'RNNid','pro','right','switches','hit_rate',...
    'score0','score1','score2','score3','score4',...
    'encoding0','encoding1','encoding2','encoding3','encoding4'});
SGD_table.RNNid = num2str(SGD_table.RNNid);

%% Define formulas
formula0 = 'hit_rate ~ right * switches * encoding0 + (1|RNNid)';
formula1 = 'hit_rate ~ right * switches * encoding1 + (1|RNNid)';
formula2 = 'hit_rate ~ right * switches * encoding2 + (1|RNNid)';
formula3 = 'hit_rate ~ right * switches * encoding3 + (1|RNNid)';
formula4 = 'hit_rate ~ right * switches * encoding4 + (1|RNNid)';
%% fit models
%model = fitglme(SGD_table, formula,...
%    'distribution','binomial','DummyVarCoding','effect');
model0 = fitlme(SGD_table, formula0);
model1 = fitlme(SGD_table, formula1);
model2 = fitlme(SGD_table, formula2);
model3 = fitlme(SGD_table, formula3);
model4 = fitlme(SGD_table, formula4);
%%
disp(model0)
disp(model1)
disp(model2)
disp(model3)
disp(model4)
%%
SGD_table.fit0 = fitted(model0);
SGD_table.fit1 = fitted(model1);
SGD_table.fit2 = fitted(model2);
SGD_table.fit3 = fitted(model3);
SGD_table.fit4 = fitted(model4);
%% Convert TABLE back to Python readable format
SGD_table.RNNid = str2num(SGD_table.RNNid);
SGD_matrix = table2array(SGD_table);
save(['../mats/SGD_table_target_RNN_fitted.mat'],'SGD_matrix')

