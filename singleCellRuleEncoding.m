% This script loads spike counts and check each cell's block trials' rule 
% encoding by using ROC.
%% Load data
SD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_sesstab.mat');
CD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_celltab.mat');
SD = SD.sesstab;
CD = CD.celltab;
load('SpikeCountsPerSession.mat')
%% Extract session data
time_steps = {'iti','rule','delay','target','choice'};

total_sessions = numel(SD.sessid);
%total_sessions = 1;
SwitchPerSession = cell(1,total_sessions);
ProPerSession = cell(1,total_sessions);
RightPerSession = cell(1,total_sessions);
CellIDperSession = cell(1,total_sessions);
RegionPerSession = cell(1,total_sessions);
for sx=1:total_sessions
    display(['Computing session index ',num2str(sx)])
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    CellIDperSession{sx} = CellData.cellid;
    RegionPerSession{sx} = CellData.region;
    
    pd = mloads(SessionData.protocol_data{1}); 
    peh = mloads(SessionData.parsed_events{1});
    
    pro = pd.side_lights==1;
    ProPerSession{sx} = pro;
    switches = pro(2:end) ~= pro(1:end-1);
    switches = [false;switches];
    SwitchPerSession{sx} = switches;
    
    ground_truth = pd.sides;
    truth_right = (ground_truth=='r')';
    target_right = ~xor(pro,truth_right);
    RightPerSession{sx} = target_right;
end
%% Data analysis Pro/Anti
cell_table = cell(total_cells,10);
cell_idx = 0;
for sx=1:total_sessions % for each session
    display(['Extracting session index ',num2str(sx)])
    counts = SpikeCountsPerSession{sx};
    pro = ProPerSession{sx};
    switches = SwitchPerSession{sx};
    cells = CellIDperSession{sx};
    regions = RegionPerSession{sx};
    for cx=1:size(counts,2) %for each cell
        cell_idx = cell_idx + 1;
        cell_table(cell_idx,1:2) = {cells(cx),regions{cx}};
        column_idx = 1;
        for tx=1:numel(time_steps) % for each time steps
            column_idx = column_idx + 2;
            this_count = squeeze(counts(tx,cx,:));
            [auc, p]=bootroc(this_count(pro&~switches),this_count(~pro&~switches));
            cell_table(cell_idx,column_idx:(column_idx+1)) = {auc,p};
        end
    end
end
cell_table = cell2table(cell_table,'VariableNames',{'cellid','region',...
    'ITI_auc','ITI_p','rule_auc','rule_p','delay_auc','delay_p',...
    'target_auc','target_p','choice_auc','choice_p',});

%% Save cell_table
save('mats/single_cell_block_auc_p.mat','cell_table')

%% Save cell_table into python-readable array
single_AUC_p = table2array(cell_table(:,3:end));
save('mats/single_AUC_p.mat','single_AUC_p')