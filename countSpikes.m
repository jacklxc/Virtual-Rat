% This script produces spike counts per session from original session and
% cell data.
%% Load data
SD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_sesstab.mat');
CD = load('/Users/lixiangci/Documents/MATLAB/ProAntiData/pa_celltab.mat');
SD = SD.sesstab;
CD = CD.celltab;

%% Count spikes
time_steps = {'iti','rule','delay','target','choice'};

%total_sessions = numel(SD.sessid);
total_sessions = 1;
SpikeCountsPerSession = cell(1,total_sessions);
SwitchPerSession = cell(1,total_sessions);
ProPerSession = cell(1,total_sessions);
CellIDperSession = cell(1,total_sessions);
RegionPerSession = cell(1,total_sessions);
total_cells = 0;
for sx=1:total_sessions
    display(['Computing session index ',num2str(sx)])
    SessionData = SD(sx,:);
    CellData = CD(find(CD.sessid==SessionData.sessid),:);
    CellIDperSession{sx} = CellData.cellid;
    RegionPerSession{sx} = CellData.region;
    
    pd = mloads(SessionData.protocol_data{1}); 
    peh = mloads(SessionData.parsed_events{1});
    
    cin = extract_state(peh,'cpoke1');
    target = extract_state(peh,'wait_for_spoke');
    pro = pd.side_lights==1;
    ProPerSession{sx} = pro;
    switches = pro(2:end) ~= pro(1:end-1);
    switches = [false;switches];
    SwitchPerSession{sx} = switches;
    
    start = {cin-1 cin target-0.25 target target+.3};
    stop =  {cin cin+1.5 target target+0.3 target+1.25};
    
    % Count spikes
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

%% Save mat file
save('mats/SpikeCountsPerSession.mat','SpikeCountsPerSession')