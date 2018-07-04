

% space and time range
daterange.start=[2015 1 1];
daterange.end=[2015 12 31];
%daterange.end=[2016 12 14];
sdates=[datenum(daterange.start):datenum(daterange.end)]'; % create list of serial dates

RoI.tilesize=5; % degrees
LatLon=[51.419541, -0.948314]; %ECMWF
LatLon=[-23.811593, 123.600807]; %Tanami
LatLon=[-3.151657, 20.902991]; % Congo
RoI.ulclat=LatLon(1)-RoI.tilesize./2;
RoI.ulclon=LatLon(2)+RoI.tilesize./2;

% File specs
% GPM
fnpattern.GPM='/g/data/fj4/SatellitePrecip/GPM/global/global_0.1deg/GPM_10km_daily_precipitation_{yyyymmdd}.nc';
varname.GPM='precipitation';
% MSWEP v2.2
fnpattern.MSWEP='/g/data/xc0/original/meteo/global/MSWEP_V2.2/global_daily_010deg/{yyyymm}.nc';
varname.MSWEP='precipitation';
% ECMWF forecast
fnpattern.ECMWF='/g/data/xc0/original/TimeSeries/Climate/Forecasts/netcdf/{yyyy}/precip-ecmwfc-{yyyymmdd}.nc';
varname.ECMWF='precip';
% ERA Interim
fnpattern.ERAI='/g/data/xc0/original/TimeSeries/Climate/ERA-Interim/netcdf/{yyyy}/precip-eraint-{yyyymmdd}.nc';
varname.ERAI='precip';


% Configure queries
[yy,mm,dd]=datevec(sdates(1));
dstr.yyyymmdd=sprintf('%.0d%.2d%.2d',yy,mm,dd);
dstr.yyyymm=sprintf('%.0d%.2d',yy,mm);
dstr.yyyy=sprintf('%.0d',yy);
% GPM query
fn=strrep(fnpattern.GPM,'{yyyymmdd}',dstr.yyyymmdd);
lat=ncread(fn,'lat');
jlat=find(lat>=RoI.ulclat-RoI.tilesize & lat<=RoI.ulclat);
query.GPM.latstart=min(jlat);
query.GPM.latstride=numel(jlat);
lon=ncread(fn,'lon');
jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
query.GPM.lonstart=min(jlon);
query.GPM.lonstride=numel(jlon);
% MSWEP v2.2 query
dstr.yyyymm=sprintf('%.0d%.2d',yy,mm);
fn=strrep(fnpattern.MSWEP,'{yyyymm}',dstr.yyyymm);
lat=ncread(fn,'lat');
jlat=find(lat>=RoI.ulclat-RoI.tilesize & lat<=RoI.ulclat);
query.MSWEP.latstart=min(jlat);
query.MSWEP.latstride=numel(jlat);
lon=ncread(fn,'lon');
jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
query.MSWEP.lonstart=min(jlon);
query.MSWEP.lonstride=numel(jlon);
% ECMWF query
fn=strrep(strrep(fnpattern.ECMWF,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
lat=ncread(fn,'latitude');
jlat=find(lat>=RoI.ulclat-RoI.tilesize & lat<=RoI.ulclat);
query.ECMWF.latstart=min(jlat);
query.ECMWF.latstride=numel(jlat);
lon=ncread(fn,'longitude');
lon=[lon(121:240) ; lon(1:120)]; % FIX BECAUSE LON DATA APPEAR WRONG
jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
query.ECMWF.lonstart=min(jlon);
query.ECMWF.lonstride=numel(jlon);
% ERA-Int
fn=strrep(strrep(fnpattern.ERAI,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
lat=ncread(fn,'latitude');
jlat=find(lat>=RoI.ulclat-RoI.tilesize & lat<=RoI.ulclat);
query.ERAI.latstart=min(jlat);
query.ERAI.latstride=numel(jlat);
lon=ncread(fn,'longitude');
lon=[lon(121:240) ; lon(1:120)]; % FIX BECAUSE LON DATA APPEAR WRONG
jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
query.ERAI.lonstart=min(jlon);
query.ERAI.lonstride=numel(jlon);



sources={'GPM';'MSWEP';'ECMWF'; 'ERAI'};

% initialise variables
Nlat=query.MSWEP.latstride;
Nlon=query.MSWEP.lonstride;
Nd=numel(sdates);
for si=1:numel(sources)
    eval(['cube_' sources{si} '=single(NaN*zeros(Nlat,Nlon,Nd));'])
end

for di=1:Nd
    [yy,mm,dd]=datevec(sdates(di));
    dstr.yyyymmdd=sprintf('%.0d%.2d%.2d',yy,mm,dd);
    dstr.yyyymm=sprintf('%.0d%.2d',yy,mm);
    dstr.yyyy=sprintf('%.0d',yy);
    % load GPM
    fn=strrep(fnpattern.GPM,'{yyyymmdd}',dstr.yyyymmdd);
    datagrid=NaN.*zeros(query.GPM.lonstride,query.GPM.latstride);
    try
        datagrid=transpose(ncread(fn,varname.GPM,[query.GPM.lonstart query.GPM.latstart ],[query.GPM.lonstride query.GPM.latstride ]));
    end
    cube_GPM(:,:,di)=datagrid;
    % load MSWEP
    fn=strrep(fnpattern.MSWEP,'{yyyymm}',dstr.yyyymm);
    datagrid=NaN.*zeros(query.MSWEP.lonstride,query.MSWEP.latstride);
    try
        datagrid=transpose(ncread(fn,varname.MSWEP,[query.MSWEP.lonstart query.MSWEP.latstart dd],[query.MSWEP.lonstride query.MSWEP.latstride 1]));
    end
    cube_MSWEP(:,:,di)=datagrid;
    % load ECMWF
    fn=strrep(strrep(fnpattern.ECMWF,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
    datagrid=NaN.*zeros(query.ECMWF.lonstride,query.ECMWF.latstride);
    try
        datagrid=transpose(ncread(fn,varname.ECMWF,[query.ECMWF.lonstart query.ECMWF.latstart 1],[query.ECMWF.lonstride query.ECMWF.latstride 1]));
    end
    cube_ECMWF(:,:,di)=imresize(datagrid,[query.MSWEP.lonstride query.MSWEP.latstride],'cubic');
    % load EA Interim
    fn=strrep(strrep(fnpattern.ERAI,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
    datagrid=NaN.*zeros(query.ERAI.lonstride,query.ERAI.latstride);
    try
        datagrid=ncread(fn,varname.ERAI,[query.ERAI.lonstart query.ERAI.latstart 1],[query.ERAI.lonstride query.ERAI.latstride 1])';
    end
    cube_ERAI(:,:,di)=imresize(datagrid,[query.MSWEP.lonstride query.MSWEP.latstride],'cubic');
    %
    fprintf('.')
end
fprintf('Done\n')

%% Compare
x=query.MSWEP.lonstride/2;
y=query.MSWEP.latstride/2;
x_GPM=squeeze(cube_GPM(x,y,:));
x_ECMWF=squeeze(cube_ECMWF(x,y,:));
x_ERAI=squeeze(cube_ERAI(x,y,:));
y=squeeze(cube_MSWEP(x,y,:));
plot(y,'r') % MSWEP
hold
plot(x_GPM,'g')
plot(x_ECMWF,'b')
plot(x_ERAI,'k')
hold

%%
X=x_ECMWF; Y=y;
X(X<0)=0; Y(Y<0)=0;
X(X==0)=0.1.*min(X(X>0));
Y(Y==0)=0.1.*min(Y(Y>0));
X(isnan(X.*Y))=NaN;
Y(isnan(X.*Y))=NaN;
figure(1)
Xtr=log10(sinh(X));
Ytr=log10(sinh(Y));
plot(Xtr,Ytr,'o');
title('log-sinh'); xlabel('source'); ylabel('MSWEP')
%plot(log10(sinh(sort(X+1e-4))),log10(sinh(sort(Y+1e-4))),'o')
% figure(2)
% [temp,Yr]  = ismember(Y,unique(Y));
% [temp,Xr]  = ismember(X,unique(X));
% plot(Xr,Yr,'ok');
% title('rank'); xlabel('source'); ylabel('MSWEP')

% calculate regression parameters
jok=~isnan(Xtr.*Ytr);
Yj=Ytr(jok);
X2 = [ones(length(Xtr(jok)),1) Xtr(jok)];
pars=X2\Yj;
ypred = X2*pars;
R2 = 1 - sum((Yj - ypred).^2)/sum((Yj - mean(Yj)).^2);




mdl=fitlm(Xtr(jok),Ytr(jok));

slope(1)=mdl.Coefficients(2,1);


%%
cube=cube_MSWEP;
imagesc(sum(cube,3)); colorbar; axis equal; axis off

