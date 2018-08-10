% This script fits a Generalized Linear Mixed Effect model to show the
% correlation between hit rate and rule encoding score predicted by linear
% regression classifier.
%% Load data table
load('mats/SGD_table_overfit_all_normalized.mat')
SGD_table = array2table(SGD_table,...
    'VariableNames',{'sessid','time_step','score','pro','right','switches',...
    'hit','accuracy','same_score','ratindex'});
SGD_table.sessid = num2str(SGD_table.sessid);
SGD_table.time_step = num2str(SGD_table.time_step);

%%
time_step = 4;
threshold = 0.6;
%specific = SGD_table.time_step==time_step & SGD_table.accuracy>=threshold;
specific = SGD_table.accuracy>=threshold;
specific_table = SGD_table(specific,:);

%%
TABLE = specific_table;
model = fitglme(TABLE, 'hit ~ same_score * pro * time_step + (same_score * pro|sessid) + (same_score * pro|time_step)',...
    'distribution','binomial','DummyVarCoding','effect');
%model = fitglme(TABLE, 'hit ~ same_score * pro * time_step + (same_score * pro * time_step|sessid)',...
%    'distribution','binomial','DummyVarCoding','effect');
%model = fitglme(TABLE, 'hit ~ same_score * pro',...
%    'distribution','binomial','DummyVarCoding','effect');
TABLE.fit = fitted(model);

%% Compare y and y_hat vs x

[binc,mu,se,n]=binned(TABLE.same_score(TABLE.pro==1),...
    TABLE.hit(TABLE.pro==1),'n_bins',10);
[binc_fit,mu_fit,se_fit,n_fit]=binned(TABLE.same_score(TABLE.pro==1),...
    TABLE.fit(TABLE.pro==1),'n_bins',20);

sig4params_pro = nlinfit(binc_fit,mu_fit,@sig4,[0.7,0.1,-0.3,1]);
x_plot = -10:10:0.01;

figure(1)
clf(1)
hold on
errorbar(binc,mu,se,'o','color',[0,0.5,1])
plot(binc_fit,mu_fit,'o','color',[0,0,1])
plot(x_plot,sig4(sig4params_pro,x_plot),'color','b')

[binc,mu,se,n]=binned(TABLE.same_score(TABLE.pro==0),...
    TABLE.hit(TABLE.pro==0),'n_bins',10);
[binc_fit,mu_fit,se_fit,n_fit]=binned(TABLE.same_score(TABLE.pro==0),...
    TABLE.fit(TABLE.pro==0),'n_bins',20);
sig4params_anti = nlinfit(binc_fit,mu_fit,@sig4,[0.7,0.1,-0.3,1]);

errorbar(binc,mu,se,'o','color',[1,0.5,0])
plot(binc_fit,mu_fit,'o','color',[1,0,0])
plot(x_plot,sig4(sig4params_anti,x_plot),'color','r')

xlabel('Rule encoding')
ylabel('Hit rate')
%title('choice')
%% Convert TABLE back to Python readable format
TABLE.sessid = str2num(TABLE.sessid);
TABLE.time_step = str2num(TABLE.time_step);
SGD_matrix = table2array(TABLE);
save('mats/SGD_table_overfit_all_normalized_fitted.mat','SGD_matrix')
%% Check individual rat's fit
clf;
to_plot = TABLE.ratindex==15;
to_plot_pro = to_plot & TABLE.pro==1 & TABLE.time_step~='0';
to_plot_anti = to_plot & TABLE.pro==0 & TABLE.time_step~='0';
hold on
scatter(TABLE.same_score(to_plot_pro),TABLE.fit(to_plot_pro),'b')
scatter(TABLE.same_score(to_plot_anti),TABLE.fit(to_plot_anti),'r')
%% Psychometric curve
figure(2)
hold on
[binc_y,mu_y,se_y,n_y]=binned(TABLE.fit, TABLE.hit,'n_bins',20);
errorbar(binc_y,mu_y,se_y,'o','color','b')
xlabel('Fitted hit rate')
ylabel('Hit rate')
title('Psychometric curve')
