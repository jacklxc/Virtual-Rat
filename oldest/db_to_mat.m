function db_to_mat
% db_to_mat
% Gets files from the database (the brodylab database) and saves each rat to a MAT file for offline analysis.
% Does not include data from learning, just the data that is in `alldata`

    system('mkdir -p ~/rat_data')

    rats = bdata('select distinct ratname from pa.alldata');

    for rx = 1:numel(rats)
        ratname = rats{rx};
        sessid = bdata('select distinct(sessid) from pa.alldata where ratname = "{S}"', ratname);
        SD = get_sessdata(sessid);
        save(sprintf('~/rat_data/%s.mat',ratname),'SD')
    end
