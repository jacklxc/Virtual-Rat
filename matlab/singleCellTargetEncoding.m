% This script loads spike counts and check each cell's block trials' rule 
% encoding by using ROC.
clear
SD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_sesstab.mat');
CD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_celltab.mat');
SD = SD.sesstab;
CD = CD.celltab;
load('../mats/SessionInfo.mat')
load('../mats/CellIndexPerSession.mat')
%% Data analysis Pro/Anti for all
load('../mats/SpikeCountsPerSession.mat')
load('../mats/Erlich_sessions.mat')
time_steps = {'iti','rule','delay','target','choice'};
total_cells = 0;
total_sessions = numel(SpikeCountsPerSession);
for sx=1:total_sessions
    total_cells = total_cells+length(CellIndexPerSession{sx});
end
cell_table = cell(total_cells,10);
cell_idx = 0;
for sx=1:total_sessions
    display(['Extracting session index ',num2str(sx)])
    counts = SpikeCountsPerSession{sx};
    right = SessionInfo{sx}(:,2);
    switches = SessionInfo{sx}(:,3);
    for cx=1:size(counts,2) %for each cell
        cell_idx = cell_idx + 1;
        column_idx = 1;
        for tx=1:numel(time_steps) % for each time steps
            this_count = squeeze(counts(tx,cx,:));
            if tx==3 && ismember(sx,Erlich_sessions)
                cell_table(cell_idx,column_idx:(column_idx+1)) = {nan,nan};
            else
                [auc, p]=bootroc(this_count(right&~switches),this_count(~right&~switches));
                cell_table(cell_idx,column_idx:(column_idx+1)) = {auc,p};
            end
            column_idx = column_idx + 2;
        end
    end
end

cell_table = cell2table(cell_table,'VariableNames',{...
    'ITI_auc','ITI_p','rule_auc','rule_p','delay_auc','delay_p',...
    'target_auc','target_p','choice_auc','choice_p',});

% Save cell_table into python-readable array
single_AUC_p = table2array(cell_table);
save('../mats/single_target_AUC_p.mat','single_AUC_p')
%% Data analysis Pro/Anti for Duan
load('../mats/SpikeCountsPerSessionDuan.mat')
load('../mats/Duan_sessions.mat')
time_steps = {'iti','rule','delay','target','choice'};
total_cells = 0;
for sx=Duan_sessions
    total_cells = total_cells+length(CellIndexPerSession{sx});
end
cell_table = cell(total_cells,10);
cell_idx = 0;
for sx=Duan_sessions
    display(['Extracting session index ',num2str(sx)])
    counts = SpikeCountsPerSession{sx};
    right = SessionInfo{sx}(:,2);
    switches = SessionInfo{sx}(:,3);
    for cx=1:size(counts,2) %for each cell
        cell_idx = cell_idx + 1;
        column_idx = 1;
        for tx=1:numel(time_steps) % for each time steps
            this_count = squeeze(counts(tx,cx,:));
            [auc, p]=bootroc(this_count(right&~switches),this_count(~right&~switches));
            cell_table(cell_idx,column_idx:(column_idx+1)) = {auc,p};
            column_idx = column_idx + 2;
        end
    end
end
cell_table = cell2table(cell_table,'VariableNames',{...
    'ITI_auc','ITI_p','rule_auc','rule_p','delay_auc','delay_p',...
    'target_auc','target_p','choice_auc','choice_p',});

% Save cell_table into python-readable array
single_AUC_p = table2array(cell_table);
save('../mats/Duan_single_target_AUC_p.mat','single_AUC_p')

%% Data analysis Pro/Anti for Erlich
load('../mats/Erlich_sessions.mat')
load('../mats/SpikeCountsPerSessionErlich.mat')
time_steps = {'iti','rule','target','choice'};
total_cells = 0;
for sx=Erlich_sessions
    total_cells = total_cells+length(CellIndexPerSession{sx});
end
cell_table = cell(total_cells,8);
cell_idx = 0;
for sx=Erlich_sessions
    display(['Extracting session index ',num2str(sx)])
    counts = SpikeCountsPerSession{sx};
    right = SessionInfo{sx}(:,2);
    switches = SessionInfo{sx}(:,3);
    for cx=1:size(counts,2) %for each cell
        cell_idx = cell_idx + 1;
        column_idx = 1;
        for tx=1:numel(time_steps) % for each time steps
            this_count = squeeze(counts(tx,cx,:));
            [auc, p]=bootroc(this_count(right&~switches),this_count(~right&~switches));
            cell_table(cell_idx,column_idx:(column_idx+1)) = {auc,p};
            column_idx = column_idx + 2;
        end
    end
end

cell_table = cell2table(cell_table,'VariableNames',{...
    'ITI_auc','ITI_p','rule_auc','rule_p',...
    'target_auc','target_p','choice_auc','choice_p',});

% Save cell_table into python-readable array
single_AUC_p = table2array(cell_table);
save('../mats/Erlich_single_target_AUC_p.mat','single_AUC_p')