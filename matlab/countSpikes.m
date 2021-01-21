% This script produces spike counts per session from original session and
% cell data.
%% Load data
clear
SD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_sesstab.mat');
CD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_celltab.mat');
SD = SD.sesstab;
CD = CD.celltab;
total_sessions = size(SD,1);

%% Count spikes from all sessions
time_steps = {'iti','rule','delay','target','choice'};
SpikeCountsPerSession = cell(1,total_sessions);
total_cells = 0;
load('../mats/Erlich_sessions.mat')
for sx=1:total_sessions
    display(['Computing session index ',num2str(sx)])
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    
    pd = mloads(SessionData.protocol_data{1}); 
    peh = mloads(SessionData.parsed_events{1});
    
    cin = extract_state(peh,'cpoke1');
    target = extract_state(peh,'wait_for_spoke');
    
    start = {cin-1, cin, cin+1, target, target+.3}; 
    stop =  {cin cin+1 target target+0.3 target+1.25}; 
    % Count spikes
    % counts: (5 steps, num_cells recorded in this session, num_trials)
    counts = nan(numel(time_steps), numel(CellData.cellid), numel(peh));
    for cx = 1:numel(CellData.cellid)
        total_cells = total_cells+1;
        ts = mloads(CellData.spiketimes{cx});
        for tx = 1:numel(time_steps)
            counts(tx,cx,:)=qcount(ts,start{tx}, stop{tx});          
        end
    end
    SpikeCountsPerSession{sx} = counts;
end

% Fill invalid delay step from Erlich sessions by zeros
for sx=Erlich_sessions
    display(['Wiping session index ',num2str(sx)])
    counts = SpikeCountsPerSession{sx};
    counts(3,:,:) = 0;      
    SpikeCountsPerSession{sx} = counts;
end
save('../mats/SpikeCountsPerSession.mat','SpikeCountsPerSession')
%% Count spikes from Duan's sessions
time_steps = {'iti','rule','delay','target','choice'};
SpikeCountsPerSession = cell(1,total_sessions);
total_cells = 0;
load('../mats/Duan_sessions.mat')
for sx=Duan_sessions
    display(['Computing session index ',num2str(sx)])
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    
    pd = mloads(SessionData.protocol_data{1}); 
    peh = mloads(SessionData.parsed_events{1});
    
    cin = extract_state(peh,'cpoke1');
    target = extract_state(peh,'wait_for_spoke');
    
    start = {cin-1, cin, cin+1, target, target+.3}; 
    stop =  {cin cin+1 target target+0.3 target+1.25}; 
    % Count spikes
    % counts: (5 steps, num_cells recorded in this session, num_trials)
    counts = nan(numel(time_steps), numel(CellData.cellid), numel(peh));
    for cx = 1:numel(CellData.cellid)
        total_cells = total_cells+1;
        ts = mloads(CellData.spiketimes{cx});
        for tx = 1:numel(time_steps)
            counts(tx,cx,:)=qcount(ts,start{tx}, stop{tx});          
        end
    end
    SpikeCountsPerSession{sx} = counts;
end
save('../mats/SpikeCountsPerSessionDuan.mat','SpikeCountsPerSession')

%% Count spikes from Erlich's sessions
time_steps = {'iti','rule','target','choice'};
SpikeCountsPerSession = cell(1,total_sessions);
total_cells = 0;
load('../mats/Erlich_sessions.mat')
for sx=Erlich_sessions
    display(['Computing session index ',num2str(sx)])
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    
    pd = mloads(SessionData.protocol_data{1}); 
    peh = mloads(SessionData.parsed_events{1});
    
    cin = extract_state(peh,'cpoke1');
    target = extract_state(peh,'wait_for_spoke');
    
    %start = {cin-1 cin target-0.25 target target+.3};
    start = {cin-1, cin, target, target+.3};
    stop =  {cin cin+1 target+0.3 target+1.25};
    % Count spikes
    % counts: (4 steps, num_cells recorded in this session, num_trials)
    counts = nan(numel(time_steps), numel(CellData.cellid), numel(peh));
    for cx = 1:numel(CellData.cellid)
        total_cells = total_cells+1;
        ts = mloads(CellData.spiketimes{cx});
        for tx = 1:numel(time_steps)
            counts(tx,cx,:)=qcount(ts,start{tx}, stop{tx});          
        end
    end
    SpikeCountsPerSession{sx} = counts;
end
save('../mats/SpikeCountsPerSessionErlich.mat','SpikeCountsPerSession')