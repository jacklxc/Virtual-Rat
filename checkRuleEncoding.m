% This script is just a sanity check of rule encoding scores (same_score)
% are different across different trial types (pro/anti, switch/block).
%% Load data table
load('mats/SGD_table_overfit_all_normalized.mat')
SGD_table = array2table(SGD_table,...
    'VariableNames',{'sessid','time_step','score','pro','right','switches',...
    'hit','accuracy','same_score','ratindex'});
SGD_table.sessid = num2str(SGD_table.sessid);
SGD_table.time_step = num2str(SGD_table.time_step);
%%
time_step = num2str(4);
threshold = 0.6;
specific =  SGD_table.accuracy>=threshold;% & SGD_table.time_step==time_step;
specific_table = SGD_table(specific,:);

%%
TABLE = specific_table;

model = fitlme(TABLE, 'same_score ~ time_step * switches * pro + (switches * pro|time_step) + (switches * pro|sessid)',...
    'DummyVarCoding','effect');
%model = fitlme(TABLE, 'same_score ~ switches * pro  + (switches * pro|sessid)',...
%    'DummyVarCoding','effect');

TABLE.fit = fitted(model);
%%
clf
hold on
step = num2str(4);
pro_switch = TABLE.switches==1 & TABLE.pro==1 & TABLE.time_step==step;
scatter(ones(size(TABLE.fit(pro_switch))),TABLE.fit(pro_switch))

anti_switch = TABLE.switches==1 & TABLE.pro==0 & TABLE.time_step==step;
scatter(2*ones(size(TABLE.fit(anti_switch))),TABLE.fit(anti_switch))

pro_block = TABLE.switches==0 & TABLE.pro==1 & TABLE.time_step==step;
scatter(3*ones(size(TABLE.fit(pro_block))),TABLE.fit(pro_block))

anti_block = TABLE.switches==0 & TABLE.pro==0 & TABLE.time_step==step;
scatter(4*ones(size(TABLE.fit(anti_block))),TABLE.fit(anti_block))

xlim([0,5])
ylabel('Fitted rule encoding (same score)')
set(gca, {'XTick', 'XTickLabel'}, {1:4, {'pro switch', 'anti switch',...
    'pro block', 'anti block'}})
title('Without normalization')
%%
clf
hold on
pro_switch = TABLE.switches==1 & TABLE.pro==1;
scatter(0.9*ones(size(TABLE.fit(pro_switch))),TABLE.fit(pro_switch),...
    'MarkerEdgeColor',[0,0.5,1])

anti_switch = TABLE.switches==1 & TABLE.pro==0;
scatter(1.9*ones(size(TABLE.fit(anti_switch))),TABLE.fit(anti_switch),...
    'MarkerEdgeColor',[1,0.5,0])

pro_block = TABLE.switches==0 & TABLE.pro==1;
scatter(2.9*ones(size(TABLE.fit(pro_block))),TABLE.fit(pro_block),...
    'MarkerEdgeColor',[0,0.5,1])

anti_block = TABLE.switches==0 & TABLE.pro==0;
scatter(3.9*ones(size(TABLE.fit(anti_block))),TABLE.fit(anti_block),...
    'MarkerEdgeColor',[1,0.5,0])

pro_switch = TABLE.switches==1 & TABLE.pro==1;
scatter(1.1*ones(size(TABLE.same_score(pro_switch))),TABLE.same_score(pro_switch),...
    'MarkerEdgeColor',[0,0,1])

anti_switch = TABLE.switches==1 & TABLE.pro==0;
scatter(2.1*ones(size(TABLE.same_score(anti_switch))),TABLE.same_score(anti_switch),...
    'MarkerEdgeColor',[1,0,0])

pro_block = TABLE.switches==0 & TABLE.pro==1;
scatter(3.1*ones(size(TABLE.same_score(pro_block))),TABLE.same_score(pro_block),...
    'MarkerEdgeColor',[0,0,1])

anti_block = TABLE.switches==0 & TABLE.pro==0;
scatter(4.1*ones(size(TABLE.same_score(anti_block))),TABLE.same_score(anti_block),...
'MarkerEdgeColor',[1,0,0])

xlim([0,5])
ylabel('Rule encoding (same score)')
set(gca, {'XTick', 'XTickLabel'}, {1:4, {'pro switch', 'anti switch',...
    'pro block', 'anti block'}})
title('LME fit')