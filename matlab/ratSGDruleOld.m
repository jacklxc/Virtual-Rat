% This script fits a Generalized Linear Mixed Effect model to show the
% correlation between hit rate and rule encoding score predicted by linear
% regression classifier.
clear
%% Define variables
experimentor = 'Duan'; % 'Duan','Erlich'
brain_area = 'all'; % 'mPFC','SC','FOF'
model_type = 'middle'; % 'long','simple','simplest','middle','pro_simplest'
if strcmp(experimentor,'Duan')
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
if strcmp(experimentor,'Duan')
    TABLE.good2 = SGD_table.accuracy2>=threshold;
end
TABLE.good3 = SGD_table.accuracy3>=threshold;
TABLE.good4 = SGD_table.accuracy4>=threshold;
%% Separately create table for different time steps
table0 = TABLE(TABLE.good0,:);
table1 = TABLE(TABLE.good1,:);
if strcmp(experimentor,'Duan')
    table2 = TABLE(TABLE.good2,:);
end
table3 = TABLE(TABLE.good3,:);
table4 = TABLE(TABLE.good4,:);
%% Define formulas
switch model_type
    case 'old'
        formula = 'hit ~ pro * switches + encoding0:good0 + encoding1:good1 + encoding2:good2 + encoding3:good3 + encoding4:good4 + (1|sessid)';
        formula0 = 'hit ~ pro * switches + encoding0:good0 + (1|sessid)';
        formula1 = 'hit ~ pro * switches + encoding1:good1 + (1|sessid)';
    if strcmp(experimentor,'Duan')
        formula2 = 'hit ~ pro * switches + encoding2:good2 + (1|sessid)';
    end 
        formula3 = 'hit ~ pro * switches + encoding3:good3 + (1|sessid)';
        formula4 = 'hit ~ pro * switches + encoding4:good4 + (1|sessid)';
    case 'long'
        formula0 = 'hit ~ pro * switches * encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro * switches * encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ pro * switches * encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro * switches * encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro * switches * encoding4 + (1|sessid)';
    case 'simplest'
        formula0 = 'hit ~ encoding0 + (1|sessid)';
        formula1 = 'hit ~ encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ encoding3 + (1|sessid)';
        formula4 = 'hit ~ encoding4 + (1|sessid)';
    case 'pro*simplest'
        formula0 = 'hit ~ pro * encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro * encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ pro * encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro * encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro * encoding4 + (1|sessid)';
    case 'pro+simplest'
        formula0 = 'hit ~ pro + encoding0 + (1|sessid)';
        formula1 = 'hit ~ pro + encoding1 + (1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ pro + encoding2 + (1|sessid)';
        end
        formula3 = 'hit ~ pro + encoding3 + (1|sessid)';
        formula4 = 'hit ~ pro + encoding4 + (1|sessid)';
    case 'simple'
        formula0 = 'hit ~ encoding0 + (encoding0|sessid)';
        formula1 = 'hit ~ encoding1 + (encoding1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ encoding2 + (encoding2|sessid)';
        end
        formula3 = 'hit ~ encoding3 + (encoding3|sessid)';
        formula4 = 'hit ~ encoding4 + (encoding4|sessid)';
    case 'middle'
        formula0 = 'hit ~ encoding0 + encoding0:pro + encoding0:switches + encoding0:pro:switches + (1|sessid)';
        formula1 = 'hit ~ encoding1 + encoding1:pro + encoding1:switches + encoding1:pro:switches + (1|sessid)';
        if strcmp(experimentor,'Duan')
            formula2 = 'hit ~ encoding2 + encoding2:pro + encoding2:switches + encoding2:pro:switches + (1|sessid)';
        end
        formula3 = 'hit ~ encoding3 + encoding3:pro + encoding3:switches + encoding3:pro:switches + (1|sessid)';
        formula4 = 'hit ~ encoding4 + encoding4:pro + encoding4:switches + encoding4:pro:switches + (1|sessid)';
end
%% fit models
%model = fitglme(TABLE, formula,...
%    'distribution','binomial','DummyVarCoding','effect');
model0 = fitglme(table0, formula0,'distribution','binomial');
model1 = fitglme(table1, formula1,'distribution','binomial');
if strcmp(experimentor,'Duan')
    model2 = fitglme(table2, formula2,'distribution','binomial');
end
model3 = fitglme(table3, formula3,'distribution','binomial');
model4 = fitglme(table4, formula4,'distribution','binomial');
%%
disp(model0)
disp(model1)
if strcmp(experimentor,'Duan')
    disp(model2)
end
disp(model3)
disp(model4)
%%
table0.fit = fitted(model0);
table1.fit = fitted(model1);
if strcmp(experimentor,'Duan')
    table2.fit = fitted(model2);
end
table3.fit = fitted(model3);
table4.fit = fitted(model4);

%% Convert TABLE back to Python readable format
table0.sessid = str2num(table0.sessid);
SGD_matrix0 = table2array(table0);
save(['../mats/',experimentor,'SGD_table_0_',brain_area,'_',model_type,'.mat'],'SGD_matrix0')

table1.sessid = str2num(table1.sessid);
SGD_matrix1 = table2array(table1);
save(['../mats/',experimentor,'SGD_table_1_',brain_area,'_',model_type,'.mat'],'SGD_matrix1')

if strcmp(experimentor,'Duan')
    table2.sessid = str2num(table2.sessid);
    SGD_matrix2 = table2array(table2);
    save(['../mats/',experimentor,'SGD_table_2_',brain_area,'_',model_type,'.mat'],'SGD_matrix2')
end
    
table3.sessid = str2num(table3.sessid);
SGD_matrix3 = table2array(table3);
save(['../mats/',experimentor,'SGD_table_3_',brain_area,'_',model_type,'.mat'],'SGD_matrix3')

table4.sessid = str2num(table4.sessid);
SGD_matrix4 = table2array(table4);
save(['../mats/',experimentor,'SGD_table_4_',brain_area,'_',model_type,'.mat'],'SGD_matrix4')