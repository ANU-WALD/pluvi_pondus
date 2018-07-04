
function [greg]=member_stats(RoI,period)

% calculates grid regression stats for specified sources


%% parameters
% space and time range
%period.start=[2015 12 31];
%period.end=[2016 12 31];
%LatLon=[45, -0.948314];
%LatLon=[51.419541, -0.948314]; %ECMWF
%LatLon=[-23.811593, 123.600807]; %Tanami
%LatLon=[-3.151657, 20.902991]; % Congo
% RoI.tilesize=30; % degrees
% RoI.ulclat=LatLon(1)+RoI.tilesize./2;
% RoI.ulclon=LatLon(2)-RoI.tilesize./2;

%
%period.end=[2016 12 14];
sdates=[datenum(period.start):datenum(period.end)]'; % create list of serial dates
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
% ECMWF tmin
fnpattern.TMIN='/g/data/xc0/original/TimeSeries/Climate/Forecasts/netcdf/{yyyy}/tmin-ecmwfc-{yyyymmdd}.nc';
varname.TMIN='tmin';


% Configure queries
testdate=[2015 1 1];
[yy,mm,dd]=datevec(datenum(testdate));
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
gridres=360/length(lon);
lon=[lon(121:240) ; lon(1:120)]+gridres/2; % FIX BECAUSE LON DATA APPEAR WRONG
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
gridres=360/length(lon);
lon=[lon(121:240) ; lon(1:120)]+gridres/2; % FIX BECAUSE LON DATA APPEAR WRONG
jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
query.ERAI.lonstart=min(jlon);
query.ERAI.lonstride=numel(jlon);

% load data
%sources={'MSWEP'; 'GPM';'ECMWF'; 'ERAI'};
sources={'MSWEP'; 'GPM';'ECMWF'};
% initialise variables
Nlat=query.MSWEP.latstride;
Nlon=query.MSWEP.lonstride;
Nd=numel(sdates);
for si=1:numel(sources)
    eval(['cube_' sources{si} '=single(NaN*zeros(Nlat,Nlon,Nd));'])
end
cube_tmin=single(NaN*zeros(Nlat,Nlon,Nd));

%% Load data
fprintf('\n Loading data')
for di=1:Nd
    [yy,mm,dd]=datevec(sdates(di));
    dstr.yyyymmdd=sprintf('%.0d%.2d%.2d',yy,mm,dd);
    dstr.yyyymm=sprintf('%.0d%.2d',yy,mm);
    dstr.yyyy=sprintf('%.0d',yy);
    % load GPM
    if query.GPM.latstride==0
        % skip
    else
        fn=strrep(fnpattern.GPM,'{yyyymmdd}',dstr.yyyymmdd);
        datagrid=NaN.*zeros(query.GPM.lonstride,query.GPM.latstride);
        try
            datagrid=transpose(ncread(fn,varname.GPM,[query.GPM.lonstart query.GPM.latstart ],[query.GPM.lonstride query.GPM.latstride ]));
        end
        cube_GPM(:,:,di)=datagrid;
    end
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
    cube_ECMWF(:,:,di)=imresize(datagrid,[query.MSWEP.lonstride query.MSWEP.latstride],'bilinear');
%     % load ERA Interim
%     fn=strrep(strrep(fnpattern.ERAI,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
%     datagrid=NaN.*zeros(query.ERAI.lonstride,query.ERAI.latstride);
%     try
%         datagrid=ncread(fn,varname.ERAI,[query.ERAI.lonstart query.ERAI.latstart 1],[query.ERAI.lonstride query.ERAI.latstride 1])';
%     end
%     cube_ERAI(:,:,di)=imresize(datagrid,[query.MSWEP.lonstride query.MSWEP.latstride],'bilinear');

    % load ECMWF tmin
    fn=strrep(strrep(fnpattern.TMIN,'{yyyymmdd}',dstr.yyyymmdd),'{yyyy}',dstr.yyyy);
    datagrid=NaN.*zeros(query.ECMWF.lonstride,query.ECMWF.latstride);
    try
        datagrid=transpose(ncread(fn,varname.TMIN,[query.ECMWF.lonstart query.ECMWF.latstart 1],[query.ECMWF.lonstride query.ECMWF.latstride 1]));
    end
    cube_tmin(:,:,di)=imresize(datagrid,[query.MSWEP.lonstride query.MSWEP.latstride],'bilinear');
    %
    %fprintf('.')
end
fprintf('\n Data loaded')

%% Transform & calculate parameters
%tic
fprintf('\n Calculating statistics')
Nmin=6;
for si=2:numel(sources)
    eval(['Xcube=cube_' sources{si} ';']);  
    Xcube(cube_tmin<3)=NaN;
    R2=NaN.*zeros(Nlat,Nlon);
    Intercept=NaN.*zeros(Nlat,Nlon);
    Slope=NaN.*zeros(Nlat,Nlon);
    N=NaN.*zeros(Nlat,Nlon);
%     fpratio=NaN.*zeros(Nlat,Nlon);
%     mpratio=NaN.*zeros(Nlat,Nlon);
    for i=1:Nlat
        for j=1:Nlon
            X=squeeze(Xcube(i,j,:));            
            Y=squeeze(cube_MSWEP(i,j,:));            
            if sum((Y>0))<Nmin || sum((X>0))<Nmin
                % skip
            else
                X(X<0)=NaN; Y(Y<0)=NaN;
                jok=~isnan(X.*Y);
                Xp=sum(X(jok)>0)./numel(jok);
                Yp=sum(Y(jok)>0)./numel(jok);
%                 fpratio(i,j)=Yp./Xp;
%                 mpratio(i,j)=mean(Y(jok))./mean(X(jok));
                Xr=X; Yr=Y;
                Xr(Xr==0)=0.1.*min(X(X>0));
                Yr(Yr==0)=0.1.*min(Y(Y>0));
                Xtr=log10(sinh(double(Xr(jok))));
                Ytr=log10(sinh(double(Yr(jok))));
                % calculate regression parameters
                X2 = [ones(length(Xtr),1) Xtr];
                pars=X2\Ytr;
                Ypred = X2*pars;
                %p=polyfit(Xtr,Ytr,1);
                Intercept(i,j)=pars(1);
                Slope(i,j)=pars(2);
                R2(i,j) = 1 - sum((Ytr - Ypred).^2)/sum((Ytr - mean(Ytr)).^2);
                N(i,j)=sum(jok);
            end
        end
    end
    eval(['greg.' sources{si} '.N=N;'])
    eval(['greg.' sources{si} '.Slope=Slope;'])
    eval(['greg.' sources{si} '.Intercept=Intercept;'])
    eval(['greg.' sources{si} '.R2=R2;'])
%     eval(['greg.' sources{si} '.fpratio=fpratio;'])
%     eval(['greg.' sources{si} '.mpratio=mpratio;'])
end
fprintf('\n Statistics calculated')
%toc

% %% Show maps
% figure(2)
% %sumR2=(greg.ECMWF.R2+greg.GPM.R2+greg.ERAI.R2);
% % subplot(3,3,1); imagesc(greg.ECMWF.R2./sumR2,[0 2/3]); colorbar; title('w ECMWF'); axis off; axis equal;
% % subplot(3,3,2); imagesc(greg.ERAI.R2./sumR2,[0 2/3]); colorbar; title('w ERAI');axis off; axis equal;
% % subplot(3,3,3); imagesc(greg.GPM.R2./sumR2,[0 2/3]); colorbar; title('w GPM');axis off; axis equal;
% subplot(3,3,1); imagesc(greg.ECMWF.R2,[0 1]); colorbar; title('R2 ECMWF'); axis off; axis equal;
% %subplot(3,3,2); imagesc(greg.ERAI.R2,[0 1]); colorbar; title('R2 ERAI');axis off; axis equal;
% subplot(3,3,3); imagesc(greg.GPM.R2,[0 1]); colorbar; title('R2 GPM');axis off; axis equal;
% 
% subplot(3,3,4); imagesc(greg.ECMWF.Slope); colorbar; title('slope ECMWF'); axis off; axis equal;
% %subplot(3,3,5); imagesc(greg.ERAI.Slope); colorbar; title('slope ERAI');axis off; axis equal;
% subplot(3,3,6); imagesc(greg.GPM.Slope); colorbar; title('slope GPM');axis off; axis equal;
% 
% subplot(3,3,7); imagesc(greg.ECMWF.N); colorbar; title('int ECMWF'); axis off; axis equal;
% %subplot(3,3,8); imagesc(greg.ERAI.N); colorbar; title('int ERAI');axis off; axis equal;
% subplot(3,3,9); imagesc(greg.GPM.N); colorbar; title('int GPM');axis off; axis equal;
% 
% 
% % %% Show maps
% % figure(2)
% % subplot(3,3,1); imagesc(greg.ECMWF.fpratio,[0 2]); colorbar; title('fpratio ECMWF'); axis off; axis equal;
% % subplot(3,3,2); imagesc(greg.ERAI.fpratio,[0 2]); colorbar; title('fpratio ERAI');axis off; axis equal;
% % subplot(3,3,3); imagesc(greg.GPM.fpratio,[0 2]); colorbar; title('fpratio GPM');axis off; axis equal;
% %  
% % subplot(3,3,4); imagesc(greg.ECMWF.mpratio,[0 2]); colorbar; title('mpratio ECMWF'); axis off; axis equal;
% % subplot(3,3,5); imagesc(greg.ERAI.mpratio,[0 2]); colorbar; title('mpratio ERAI');axis off; axis equal;
% % subplot(3,3,6); imagesc(greg.GPM.mpratio,[0 2]); colorbar; title('mpratio GPM');axis off; axis equal;
% 
% 
% %%
% 
% figure(3)
% wn=max(greg.ECMWF.R2,0)./(max(greg.ECMWF.R2,0)+max(greg.GPM.R2,0));
% wn(isnan(wn))=0.5;
% ws=1-wn;
% %ws=medfilt2(ws,[5 5])
% subplot(1,2,1); imagesc(wn,[0 1]); colorbar; title('w ECMWF forecast'); axis off; axis equal;
% subplot(1,2,2); imagesc(ws, [0 1]); colorbar; title('w GPM');axis off; axis equal;
% colormap('jet')
% 



