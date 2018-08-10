% This script fits a Linear Mixed Effect model to show the
% correlation between hit rate and target encoding score of RNN predicted by linear
% regression classifier.
%% Load data table
clear;
load('mats/SGD_table_target_RNN.mat')
SGD_table = array2table(SGD_table,...
    'VariableNames',{'RNNid','time_step','score','pro','right','switches',...
    'hit_rate','same_score'});
% Normalize
SGD_table.normalized_score = zeros(numel(SGD_table.score),1);
SGD_table.normalized_hit_rate = zeros(numel(SGD_table.hit_rate),1);
RNNids = unique(SGD_table.RNNid);
for rx = 1:numel(RNNids)
    for tx = 0:4
        these_trials = SGD_table.RNNid==RNNids(rx) & SGD_table.time_step==tx;
        SGD_table.normalized_score(these_trials) = zscore(SGD_table.score(these_trials));
        SGD_table.normalized_hit_rate(these_trials) = zscore(SGD_table.hit_rate(these_trials));
    end
end
SGD_table.normalized_same_score = SGD_table.right.*SGD_table.normalized_score +...
    (1 - SGD_table.right).*-SGD_table.normalized_score;
SGD_table.time_step = num2str(SGD_table.time_step);
SGD_table.RNNid = num2str(SGD_table.RNNid);
%%
TABLE = SGD_table;
model = fitlme(TABLE, 'normalized_hit_rate ~ normalized_same_score * right * switches * time_step + (normalized_same_score * right * switches|RNNid) + (normalized_same_score * right * switches|time_step)');
TABLE.fit = fitted(model);

%% Convert TABLE back to Python readable format
TABLE.RNNid = str2num(TABLE.RNNid);
TABLE.time_step = str2num(TABLE.time_step);
SGD_matrix = table2array(TABLE);
save('mats/SGD_table_RNN_target_fitted.mat','SGD_matrix')

%% Compare y and y_hat vs x without left/right normalized
this_condition = TABLE.time_step~='0';
[binc,mu,se,n]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.normalized_hit_rate(this_condition),'n_bins',20);
[binc_fit,mu_fit,se_fit,n_fit]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.fit(this_condition),'n_bins',40);

figure(1)
clf(1)
hold on
errorbar(binc,mu,se,'o','color',[0,0.5,1])
errorbar(binc_fit,mu_fit,se_fit,'o','color',[0,0,1])

xlabel('Rule encoding')

%% Compare y and y_hat vs x considering left/right normalized
this_condition = TABLE.time_step~='0' & TABLE.right==1;
[binc,mu,se,n]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.normalized_hit_rate(this_condition),'n_bins',35);
[binc_fit,mu_fit,se_fit,n_fit]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.fit(this_condition),'n_bins',100);

figure(1)
clf(1)
hold on
errorbar(binc,mu,se,'o','color',[0,0.5,1])
plot(binc_fit,mu_fit,'o','color',[0,0,1])

this_condition = TABLE.time_step~='0' & TABLE.right==0;
[binc,mu,se,n]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.normalized_hit_rate(this_condition),'n_bins',35);
[binc_fit,mu_fit,se_fit,n_fit]=binned(TABLE.normalized_same_score(this_condition),...
    TABLE.fit(this_condition),'n_bins',100);

errorbar(binc,mu,se,'o','color',[1,0.5,0])
plot(binc_fit,mu_fit,'o','color',[1,0,0])

xlabel('Rule encoding')
ylabel('Hit rate')
%title('choice')
%% Check individual RNN's fits
clf;
to_plot = TABLE.RNNid==11;
to_plot_right = to_plot & TABLE.right==1 & TABLE.time_step~='0';
to_plot_left = to_plot & TABLE.right==0 & TABLE.time_step~='0';
hold on
scatter(TABLE.normalized_same_score(to_plot_right),TABLE.fit(to_plot_right),'b')
scatter(TABLE.normalized_same_score(to_plot_left),TABLE.fit(to_plot_left),'r')