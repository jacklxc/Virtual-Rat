% This script fits a Generalized Linear Mixed Effect model to show the
% correlation between hit rate and rule encoding score predicted by linear
% regression classifier.
clear
%% Define variables
experimentor = ''; % 'Duan','Erlich', '' for all
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
formula0 = 'hit ~ pro * switches * encoding0 + (1|sessid)';
formula1 = 'hit ~ pro * switches * encoding1 + (1|sessid)';
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    formula2 = 'hit ~ pro * switches * encoding2 + (1|sessid)';
end
formula3 = 'hit ~ pro * switches * encoding3 + (1|sessid)';
formula4 = 'hit ~ pro * switches * encoding4 + (1|sessid)';
%% fit models
model0 = fitglme(table0, formula0,'distribution','binomial');
model1 = fitglme(table1, formula1,'distribution','binomial');
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    model2 = fitglme(table2, formula2,'distribution','binomial');
end
model3 = fitglme(table3, formula3,'distribution','binomial');
model4 = fitglme(table4, formula4,'distribution','binomial');
%%
disp(model0)
disp(model1)
if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    disp(model2)
end
disp(model3)
disp(model4)

%% Produce more fitted data for smoother visualization
start = -2;
finish = 2;
repeat = 1;

new_table0 = repmat(table0, repeat,1);
new_table0.encoding = new_table0.encoding0;
new_table0.encoding0 = unifrnd(start, finish, size(new_table0.encoding0));
[new_table0.fit,new_table0.CI] = predict(model0, new_table0);

new_table1 = repmat(table1, repeat,1);
new_table1.encoding = new_table1.encoding1;
new_table1.encoding1 = unifrnd(start, finish, size(new_table1.encoding1));
[new_table1.fit,new_table1.CI] = predict(model1, new_table1);

if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    new_table2 = repmat(table2, repeat,1);
    new_table2.encoding = new_table2.encoding2;
    new_table2.encoding2 = unifrnd(start, finish, size(new_table2.encoding2));
    [new_table2.fit,new_table2.CI] = predict(model2, new_table2);
end
new_table3 = repmat(table3, repeat,1);
new_table3.encoding = new_table3.encoding3;
new_table3.encoding3 = unifrnd(start, finish, size(new_table3.encoding3));
[new_table3.fit,new_table3.CI] = predict(model3, new_table3);

new_table4 = repmat(table4, repeat,1);
new_table4.encoding = new_table4.encoding4;
new_table4.encoding4 = unifrnd(start, finish, size(new_table4.encoding4));
[new_table4.fit,new_table4.CI] = predict(model4, new_table4);
%% Plot
tabs = {table0, table1, table2, table3, table4};
fittab = {new_table0, new_table1, new_table2, new_table3, new_table4};
labs = {'ITI','Rule','Delay','Target','Choice'};
pro_color = [55 125 34]/255;
anti_color = [236 100 43]/255;

    try

        figure(fig); clf
    catch
        fig = figure;
    end


x_init = 0.1;
y_init = 0.1;
x_gap = 0.03;
width = 0.15;
height = 0.2;


for xx = 1:5
    ax{xx} = draw.jaxes;

     this_data = tabs{xx};
    this_fit = fittab{xx};
    ecol = sprintf('encoding%d',xx-1);
    gcol = this_data.(sprintf('good%d',xx-1));
    pro = this_data.pro==1;
    [px,py,pe] = stats.binned(this_data.(ecol)(pro & gcol), this_data.hit(pro & gcol), 'n_bins',10);
    draw.errorplot(ax{xx},px,py,pe, 'Color', pro_color); 
    [anx,any,ane] = stats.binned(this_data.(ecol)(~pro & gcol), this_data.hit(~pro & gcol), 'n_bins',10);
    draw.errorplot(ax{xx},anx,any,ane, 'Color', anti_color); 


    gfcol = this_fit.(sprintf('good%d',xx-1));
    fpro = this_fit.pro==1;

    [pfx,pfy] = stats.binned(this_fit.(ecol)(gfcol & fpro), this_fit.fit(gfcol & fpro), 'n_bins',50);
    ph = plot(ax{xx},pfx,pfy, 'Color', pro_color); 

    [afx,afy] = stats.binned(this_fit.(ecol)(gfcol & ~fpro), this_fit.fit(gfcol & ~fpro), 'n_bins',50);
    ah = plot(ax{xx},afx,afy, 'Color', anti_color); 
    set([ph ah], 'LineWidth',2)


    ax{xx}.Position = [x_init + (xx-1)*x_gap + (xx-1)*width y_init width, height];
    ax{xx}.YLim = [0.5 1];
    ax{xx}.XLim = [-1.8 1.8];
    xlabel(ax{xx}, labs{xx});
    if (xx>1)
        ax{xx}.YTickLabel = [];
    end
    if xx == 1
        ylabel(ax{xx}, 'P(Correct)');
    end

    if xx == 3
        th = title(ax{xx},'P(Correct) vs. Rule Encoding in each time window.');

    end

end
%% Convert TABLE back to Python readable format
new_table0.sessid = str2num(new_table0.sessid);
SGD_matrix0 = table2array(new_table0);
save(['../mats/',experimentor,'SGD_table_0_',brain_area,'.mat'],'SGD_matrix0')

new_table1.sessid = str2num(new_table1.sessid);
SGD_matrix1 = table2array(new_table1);
save(['../mats/',experimentor,'SGD_table_1_',brain_area,'.mat'],'SGD_matrix1')

if strcmp(experimentor,'Duan') || strcmp(experimentor,'')
    new_table2.sessid = str2num(new_table2.sessid);
    SGD_matrix2 = table2array(new_table2);
    save(['../mats/',experimentor,'SGD_table_2_',brain_area,'.mat'],'SGD_matrix2')
end
    
new_table3.sessid = str2num(new_table3.sessid);
SGD_matrix3 = table2array(new_table3);
save(['../mats/',experimentor,'SGD_table_3_',brain_area,'.mat'],'SGD_matrix3')

new_table4.sessid = str2num(new_table4.sessid);
SGD_matrix4 = table2array(new_table4);
save(['../mats/',experimentor,'SGD_table_4_',brain_area,'.mat'],'SGD_matrix4')