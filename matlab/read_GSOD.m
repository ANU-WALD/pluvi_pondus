

% GSOD
datadir='/g/data/xc0/original/meteo/global/gauge/GSOD/';
prodir='/g/data/xc0/user/vandijk/github/precip/';
tempdir='/g/data/xc0/user/vandijk/temp/'
outdir='/g/data/xc0/project/GlobalPrecip/'

%% station metadata
statfn=[datadir 'stations/isd-history.csv'];
data=importdata(statfn);
header=data{1};
for ri=2:length(data)
    txtdata=data{ri};
    txtdata = strrep(txtdata,'""',num2str(0));
    txtdata = strrep(txtdata,' ','_');
    txtdata(regexp(txtdata,'["+]'))=[];
    fields=textscan(txtdata,'%s %s %s %s %s %s %f %f %f %d %d','Delimiter',',');
    si=ri-1;
    GSOD.USAF(si)=string(fields{1});
    GSOD.WBAN(si)=string(fields{2});
    GSOD.STATION_NAME(si)=string(fields{3});
    GSOD.CTRY(si)=string(fields{4});
    GSOD.STATE(si)=string(fields{5});
    GSOD.ICAO(si)=string(fields{6});
    GSOD.LATLON(si,:)=[fields{7} fields{8}];
    GSOD.ELEV(si)=fields{9};
    GSOD.BEGIN_END(si,:)=[fields{10} fields{11}];
end
save([outdir 'GSODmeta.mat'],'GSOD')

%%
tic
yyyy=2015;
fprintf('\n Reformatting station data')
% Add station data
flist=dir([datadir num2str(yyyy) '/*.gz']);
unzipdir=[tempdir 'GSOD/'];
Nf=length(flist);
dates=[datenum([yyyy 1 1]):datenum([yyyy 12 31])]';
Nd=numel(dates);
% initialise
allP=single(NaN*zeros(Nf,Nd));
IDs=string(NaN*zeros(Nf,2));
h = waitbar(0,'Please wait...');
for fi=1:Nf
    fn=flist(fi).name;
    gfn=[datadir num2str(yyyy) '/' fn];
    gunzip(gfn,unzipdir)
    fn=erase(fn,'.gz');
    fullfn=[unzipdir fn];
    data = textread(fullfn, '%s','delimiter', '\n');
    delete(fullfn)
    % read data
    Ndf=length(data)-1;
    PRCP=NaN.*zeros(Nd,1);
    for ri=1:Ndf
        tstring=cell2mat(data(ri+1));
        yy=str2double(tstring(15:18));
        mm=str2double(tstring(19:20));
        dd=str2double(tstring(21:22));
        PRCP(dates==datenum([yy mm dd]))=str2double(tstring(119:123));
    end
    PRCP(PRCP==99.99)=NaN;
    allP(fi,:)=(PRCP*2.54)'; % 0.1 inch to mm
    % station IDs
    IDs(fi,1)=string(fn(1:6));
    IDs(fi,2)=string(fn(8:12));  
    waitbar(fi/Nf,h)
end
% save it all
GSOD.PRECIP=allP;
GSOD.ID=IDs;
outfn=[outdir 'GSOD_' num2str(yyyy) '.mat'];
fprintf('\n Saving %s',outfn)
save(outfn,'GSOD')
fprintf('\n Done! \n')

%%
[yy mm dd]=datevec(now);
timestamp=sprintf('%.0d%.2d%.2d',yy,mm,dd);










% %%
% yyyy=2018;
% mkdir([outdir num2str(yyyy)])
% % Initialise files
% %[yy mm dd]=datevec(now);
% %timestamp=sprintf('v%.0d%.2d%.2d',yy,mm,dd)
% fprintf('\n Initialising files')
% textHeader='LATITUDE,LONGITUDE,PRECIPMM,STATION_CODE,UPDATE';
% dates=datenum([yyyy 1 1]):datenum([yyyy 12 31]);
% for di=1:numel(dates)
%     [yy mm dd]=datevec(dates(di));
%     obsdatestr=sprintf('%.0d%.2d%.2d',yy,mm,dd);
%     %    fn=[outdir sprintf('%d/StationPrecip_%s_%s.csv',yyyy,obsdatestr,timestamp)];
%     fn=[outdir sprintf('%d/StationPrecip_%s.csv',yyyy,obsdatestr)];
%     fid = fopen(fn,'w');
%     fprintf(fid,'%s\n',textHeader);
%     fclose(fid);
% end
% fprintf('\n Reformatting station data')
% % Add station data
% flist=dir([datadir num2str(yyyy) '/*.gz']);
% unzipdir=[tempdir 'GSOD/'];
% [yy mm dd]=datevec(now);
% timestamp=sprintf('%.0d%.2d%.2d',yy,mm,dd);
% Nf=length(flist);
% h = waitbar(0,'Please wait...');
% for fi=1:Nf
%     fn=flist(fi).name;
%     gfn=[datadir num2str(yyyy) '/' fn];
%     gunzip(gfn,unzipdir)
%     fn=erase(fn,'.gz');
%     fullfn=[unzipdir fn];
%     data = textread(fullfn, '%s','delimiter', '\n');
%     delete(fullfn)
%     Nd=length(data)-1;
%     stat.yyyy=NaN.*zeros(Nd,1);
%     stat.mm=NaN.*zeros(Nd,1);
%     stat.dd=NaN.*zeros(Nd,1);
%     stat.PRCP=NaN.*zeros(Nd,1);
%     for ri=1:Nd
%         tstring=cell2mat(data(ri+1));
%         stat.yyyy(ri)=str2num(tstring(15:18));
%         stat.mm(ri)=str2num(tstring(19:20));
%         stat.dd(ri)=str2num(tstring(21:22));
%         stat.PRCP(ri)=str2num(tstring(119:123));
%     end
%     stat.PRCP(stat.PRCP==99.99)=NaN;
%     stat.PMM=stat.PRCP*2.54; % 0.1 inch to mm
%     stID=string(fn(1:6));
%     j=find(strcmp(GSOD.USAF,stID));
%     for di=1:numel(stat.dd)
%         PMMd=stat.PMM(di);
%         if isnan(PMMd)
%             % skip
%         else
%             obsdatestr=sprintf('%.0d%.2d%.2d',stat.yyyy(di),stat.mm(di),stat.dd(di));
%             fn=[outdir sprintf('%d/StationPrecip_%s.csv',yyyy,obsdatestr)];
%             stdata=sprintf('%0.3f,%0.3f,%0.2f,%s,%s\n',GSOD.LATLON(j,1),GSOD.LATLON(j,2),PMMd,stID,timestamp);
%             % write data to end of file
%             fid = fopen(fn,'a');
%             fprintf(fid,stdata);
%             fclose(fid);
%         end
%     end
%     waitbar(fi/Nf,h)
% end
% fprintf('\n Done! \n')
% 
% %1-6       Int.   Station number (WMO/DATSAV3 number)
% %YEAR    15-18     Int.   The year.
% %MODA    19-22     Int.
% % PRCP  119-123
% 
% 
% 
% 
% 
