clear;
model_type = 'middle'; % 'long','simple','simplest','middle','pro_simplest'

load('../mats/SGD_table_RNN.mat')
SGD_table = array2table(SGD_table,...
    'VariableNames',{'RNNid','pro','right','switches','hit_rate',...
    'score0','score1','score2','score3','score4',...
    'encoding0','encoding1','encoding2','encoding3','encoding4'});
SGD_table.RNNid = num2str(SGD_table.RNNid);

%% Define formulas
formulas = {};
LLs0 = [];
LLs1 = [];
LLs2 = [];
LLs3 = [];
LLs4 = [];
model_types={'long','simple','simplest','middle','pro*simplest','pro+simplest'};
for idx = 1:numel(model_types)
model_type = model_types{idx};
disp(model_type)
if strcmp(model_type,'long')
    formula0 = 'hit_rate ~ pro * switches * encoding0 + (1|RNNid)';
    formula1 = 'hit_rate ~ pro * switches * encoding1 + (1|RNNid)';
    formula2 = 'hit_rate ~ pro * switches * encoding2 + (1|RNNid)';
    formula3 = 'hit_rate ~ pro * switches * encoding3 + (1|RNNid)';
    formula4 = 'hit_rate ~ pro * switches * encoding4 + (1|RNNid)';
elseif strcmp(model_type,'simplest')
    formula0 = 'hit_rate ~ encoding0 + (1|RNNid)';
    formula1 = 'hit_rate ~ encoding1 + (1|RNNid)';
    formula2 = 'hit_rate ~ encoding2 + (1|RNNid)';
    formula3 = 'hit_rate ~ encoding3 + (1|RNNid)';
    formula4 = 'hit_rate ~ encoding4 + (1|RNNid)';
elseif strcmp(model_type,'pro*simplest')
    formula0 = 'hit_rate ~ pro * encoding0 + (1|RNNid)';
    formula1 = 'hit_rate ~ pro * encoding1 + (1|RNNid)';
    formula2 = 'hit_rate ~ pro * encoding2 + (1|RNNid)';
    formula3 = 'hit_rate ~ pro * encoding3 + (1|RNNid)';
    formula4 = 'hit_rate ~ pro * encoding4 + (1|RNNid)';
elseif strcmp(model_type,'pro+simplest')
    formula0 = 'hit_rate ~ pro + encoding0 + (1|RNNid)';
    formula1 = 'hit_rate ~ pro + encoding1 + (1|RNNid)';
    formula2 = 'hit_rate ~ pro + encoding2 + (1|RNNid)';
    formula3 = 'hit_rate ~ pro + encoding3 + (1|RNNid)';
    formula4 = 'hit_rate ~ pro + encoding4 + (1|RNNid)';
elseif strcmp(model_type,'simple')
    formula0 = 'hit_rate ~ encoding0 + (encoding0|RNNid)';
    formula1 = 'hit_rate ~ encoding1 + (encoding1|RNNid)';
    formula2 = 'hit_rate ~ encoding2 + (encoding2|RNNid)';
    formula3 = 'hit_rate ~ encoding3 + (encoding3|RNNid)';
    formula4 = 'hit_rate ~ encoding4 + (encoding4|RNNid)';
elseif strcmp(model_type,'middle')
    formula0 = 'hit_rate ~ encoding0 + encoding0:pro + encoding0:switches + encoding0:pro:switches + (1|RNNid)';
    formula1 = 'hit_rate ~ encoding1 + encoding1:pro + encoding1:switches + encoding1:pro:switches + (1|RNNid)';
    formula2 = 'hit_rate ~ encoding2 + encoding2:pro + encoding2:switches + encoding2:pro:switches + (1|RNNid)';
    formula3 = 'hit_rate ~ encoding3 + encoding3:pro + encoding3:switches + encoding3:pro:switches + (1|RNNid)';
    formula4 = 'hit_rate ~ encoding4 + encoding4:pro + encoding4:switches + encoding4:pro:switches + (1|RNNid)';
end

% Perform cross validation
fold = 10;
LL0 = cvLME(SGD_table,formula0,fold);
LL1 = cvLME(SGD_table,formula1,fold);
LL2 = cvLME(SGD_table,formula2,fold);
LL3 = cvLME(SGD_table,formula3,fold);
LL4 = cvLME(SGD_table,formula4,fold);

formulas = [formulas;formula0];
LLs0 = [LLs0;LL0];
LLs1 = [LLs1;LL1];
LLs2 = [LLs2;LL2];
LLs3 = [LLs3;LL3];
LLs4 = [LLs4;LL4];
end
result = table(formulas,LLs0,LLs1,LLs2,LLs3,LLs4);
disp(result)
