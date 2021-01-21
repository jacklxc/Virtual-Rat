% This script fits a Generalized Linear Mixed Effect model to show the
% correlation between hit rate and rule encoding score predicted by linear
% regression classifier.
clear
%% Define variables
experimentor = ''; % 'Duan','Erlich'
brain_area = 'all'; % 'mPFC','SC','FOF'
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    varnames = {'sessid','pro','right','switches','hit','ratindex',...
    'score0','score1','score2','score3','score4',...
    'accuracy0','accuracy1','accuracy2','accuracy3','accuracy4',...
    'encoding0','encoding1','encoding2','encoding3','encoding4'};
elseif strcmp(experimentor,'Erlich')
    varnames = {'sessid','pro','right','switches','hit','ratindex',...
    'score0','score1','score3','score4','empty_score'...
    'accuracy0','accuracy1','accuracy3','accuracy4','empty_accuracy'...
    'encoding0','encoding1','encoding3','encoding4','empty_encoding'};
end
%% Load data table
SGD_table_file_name = ['../mats/',experimentor,'SGD_table_',brain_area];
load(SGD_table_file_name);
SGD_table = array2table(SGD_table,...
    'VariableNames',varnames);
SGD_table.sessid = num2str(SGD_table.sessid);
%%
threshold = 0.6;
TABLE = SGD_table;
TABLE.good0 = SGD_table.accuracy0>=threshold;
TABLE.good1 = SGD_table.accuracy1>=threshold;
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    TABLE.good2 = SGD_table.accuracy2>=threshold;
end
TABLE.good3 = SGD_table.accuracy3>=threshold;
TABLE.good4 = SGD_table.accuracy4>=threshold;
%% Separately create table for different time steps
table0 = TABLE(TABLE.good0,:);
table1 = TABLE(TABLE.good1,:);
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    table2 = TABLE(TABLE.good2,:);
end
table3 = TABLE(TABLE.good3,:);
table4 = TABLE(TABLE.good4,:);
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
switch model_type
    case 'old'
        formula = 'hit ~ pro * switches + encoding0:good0 + encoding1:good1 + encoding2:good2 + encoding3:good3 + encoding4:good4 + (1|sessid)';
        formula0 = 'hit ~ pro * switches + encoding0:good0 + (1|sessid)';
            formula1 = 'hit ~ pro * switches + encoding1:good1 + (1|sessid)';
        if strcmp(experimentor,'Duan')|| strcmp(experimentor,'')
            formula2 = 'hit ~ pro * switches + encoding2:good2 + (1|sessid)';
        end 
        formula3 = 'hit ~ pro * switches + encoding3:good3 + (1|sessid)';
        formula4 = 'hit ~ pro * switches + encoding4:good4 + (1|sessid)';
    case 'long'
        formula0 = 'hit ~ pro * switches * encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro * switches * encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ pro * switches * encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro * switches * encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro * switches * encoding4 + (1|sessid)';
    case 'simplest'
        formula0 = 'hit ~ encoding0 + (1|sessid)';
        formula1 = 'hit ~ encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ encoding3 + (1|sessid)';
        formula4 = 'hit ~ encoding4 + (1|sessid)';
    case 'pro*simplest'
        formula0 = 'hit ~ pro * encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro * encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ pro * encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro * encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro * encoding4 + (1|sessid)';
    case 'pro+simplest'
        formula0 = 'hit ~ pro + encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro + encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ pro + encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro + encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro + encoding4 + (1|sessid)';
    case 'simple'
        formula0 = 'hit ~ encoding0 + (encoding0|sessid)';
        formula1 = 'hit ~ encoding1 + (encoding1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ encoding2 + (encoding2|sessid)';
        end
        formula3 = 'hit ~ encoding3 + (encoding3|sessid)';
        formula4 = 'hit ~ encoding4 + (encoding4|sessid)';
    case 'middle'
        formula0 = 'hit ~ encoding0 + encoding0:pro + encoding0:switches + encoding0:pro:switches + (1|sessid)';
        formula1 = 'hit ~ encoding1 + encoding1:pro + encoding1:switches + encoding1:pro:switches + (1|sessid)';
        if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
            formula2 = 'hit ~ encoding2 + encoding2:pro + encoding2:switches + encoding2:pro:switches + (1|sessid)';
        end
        formula3 = 'hit ~ encoding3 + encoding3:pro + encoding3:switches + encoding3:pro:switches + (1|sessid)';
        formula4 = 'hit ~ encoding4 + encoding4:pro + encoding4:switches + encoding4:pro:switches + (1|sessid)';
end
% Perform cross validation
fold = 10;
LL0 = cvGLME(table0,formula0,fold);
LL1 = cvGLME(table1,formula1,fold);
LL2 = cvGLME(table2,formula2,fold);
LL3 = cvGLME(table3,formula3,fold);
LL4 = cvGLME(table4,formula4,fold);

formulas = [formulas;formula0];
LLs0 = [LLs0;LL0];
LLs1 = [LLs1;LL1];
LLs2 = [LLs2;LL2];
LLs3 = [LLs3;LL3];
LLs4 = [LLs4;LL4];
end
result = table(formulas,LLs0,LLs1,LLs2,LLs3,LLs4);
disp(result)
