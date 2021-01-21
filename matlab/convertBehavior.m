% This script converts behavioral data to python-readable .mat files.
%% Load data
SD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_sesstab.mat');
CD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_celltab.mat');
SD = SD.sesstab;
CD = CD.celltab;

%% Collect pro anti and other data from original spike data
total_sessions = numel(SD.sessid);
%total_sessions = 1;
SessionInfo = cell(1,total_sessions);

for sx=1:total_sessions
    SessionData = SD(sx,:);
    CellData = CD(CD.sessid==SessionData.sessid,:);
    pd = mloads(SessionData.protocol_data{1}); 
    pro = pd.side_lights==1;
    switches = pro(2:end) ~= pro(1:end-1);
    switches = [false;switches];
    ground_truth = pd.sides;
    truth_right = (ground_truth=='r')';
    target_right = ~xor(pro,truth_right);
    
    info = nan(numel(pro),4);
    info(:,1) = pro;
    info(:,2) = target_right;
    info(:,3) = switches;
    info(:,4) = pd.hits;
    SessionInfo{sx} = info;
end
save('../mats/SessionInfo.mat','SessionInfo')

%% Collect brain area data
total_sessions = numel(SD.sessid);
%total_sessions = 1;
BrainArea = cell(1,total_sessions);

for sx=1:total_sessions
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    areas = nan(1,numel(CellData.region));
    for cx = 1:numel(CellData.region)
        area_index = -1;
        if isequal(CellData.region{cx},'left mPFC')
            area_index = 0;
        elseif isequal(CellData.region{cx},'Left mPFC')
            area_index = 0;
        elseif isequal(CellData.region{cx},'right mPFC')
            area_index = 1;
        elseif isequal(CellData.region{cx},'Right mPFC')
            area_index = 1;
        elseif isequal(CellData.region{cx},'left SC')
            area_index = 2;
        elseif isequal(CellData.region{cx},'Left SC')
            area_index = 2;
        elseif isequal(CellData.region{cx},'right SC')
            area_index = 3;
        elseif isequal(CellData.region{cx},'Right SC')
            area_index = 3;
        elseif isequal(CellData.region{cx},'left FOF')
            area_index = 4;
        elseif isequal(CellData.region{cx},'Left FOF')
            area_index = 4;
        elseif isequal(CellData.region{cx},'right FOF')
            area_index = 5;
        elseif isequal(CellData.region{cx},'Right FOF')
            area_index = 5;
        else
            area_index = -1;
            display(CellData.region{cx})
        end
        areas(cx) = area_index;
    end
    BrainArea{sx} = areas; 
end

save('../mats/BrainArea.mat','BrainArea')

%% Collect rat name data
total_sessions = numel(SD.sessid);
RatIndexPerSession = zeros(total_sessions,1);
ratnames = unique(SD.ratname);
display(ratnames)
for rx = 1:numel(ratnames)
    RatIndexPerSession(strcmp(SD.ratname,ratnames{rx})) = rx;
end
save('../mats/RatIndexPerSession.mat','RatIndexPerSession')

%% Collect cellid per session
total_sessions = numel(SD.sessid);
%total_sessions = 5;
CellIndexPerSession = cell(1,total_sessions);

for sx=1:total_sessions
    SessionData = SD(sx,:);
    cellidx = find(CD.sessid==SessionData.sessid);
    CellIndexPerSession{sx} = cellidx';
end
save('../mats/CellIndexPerSession.mat','CellIndexPerSession')

%% Separate Duan and Erlich rats
load('../mats/RatIndexPerSession.mat')
Erlich_rats = [1,8,9,10,11,12];
Duan_rats = [2,3,4,5,6,7,13,14,15];
all_sessions = 1:numel(SD.sessid);
Erlich_sessions = all_sessions(ismember(RatIndexPerSession,Erlich_rats));
Duan_sessions = all_sessions(ismember(RatIndexPerSession,Duan_rats));
save('../mats/Erlich_sessions.mat','Erlich_sessions')
save('../mats/Duan_sessions.mat','Duan_sessions')