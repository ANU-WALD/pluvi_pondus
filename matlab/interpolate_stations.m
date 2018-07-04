

addpath('/g/data/xc0/user/vandijk/matlab/')
indir='/g/data/xc0/project/GlobalPrecip/'

yyyy=2015;
fn=[indir 'GSOD_' num2str(yyyy) '.mat'];
load(fn);

%%
Ns=length(GSOD.ID);
slat=single(NaN.*zeros(Ns,1));
slon=single(NaN.*zeros(Ns,1));
for si=1:Ns
    j=find((GSOD.USAF'==GSOD.ID(si,1)).*(GSOD.WBAN'==GSOD.ID(si,2)));
    if isempty(j)
    else
        LATLON=GSOD.LATLON(j,:);
        if prod(LATLON)==0
        else
            slat(si)=LATLON(1);
            slon(si)=LATLON(2);
        end
    end
end
%%
dates=[datenum([yyyy 1 1]):datenum([yyyy 12 31])]';
di=11;
P=GSOD.PRECIP(:,di);
sok=~isnan(P.*slat.*slon);
xyz=[slon(sok) slat(sok) P(sok)];

gridres=0.1;
lat=[90-gridres/2:-gridres:-90+gridres/2];
lon=[-180+gridres/2:gridres:180-gridres/2];

% to generate accurate invDist functions
eqCircum=40070;merCircum=39931;
Cflatt=(eqCircum*(gridres/360))./(merCircum*(gridres/360)); % flattening factor (1.0035)
latL=(merCircum*(gridres/360)); % latitude dist
D0=25 ; % range of influence
MaskRadius=100; % means 50*gridres

dummy=single(zeros(numel(lat)+2.*MaskRadius,numel(lon)+2.*MaskRadius));
W=dummy;
P=dummy;
for sj=1:length(xyz)
    lonc=xyz(sj,1);
    latc=xyz(sj,2);
    zc=xyz(sj,3);
    % generate inverse distance mask 
    Rlonlat = cos(pi().*abs(latc)./180).*Cflatt;
    Xl=[-MaskRadius:1:MaskRadius].*latL;
    Yl=[-MaskRadius:1:MaskRadius].*latL.*Rlonlat;
    [X, Y]=meshgrid(Yl,Xl);
    D=(X.^2+Y.^2).^0.5;
    w=(exp(-D./D0)).^0.5;
    w(w<0.01)=0;
    % find central grid cell
    loni=find(abs(lon-lonc)==min(abs(lon-lonc)));
    lati=find(abs(lat-latc)==min(abs(lat-latc)));
    % add weights
    W(lati:lati+2*MaskRadius,loni:loni+2*MaskRadius)=W(lati:lati+2*MaskRadius,loni:loni+2*MaskRadius)+w;
    P(lati:lati+2*MaskRadius,loni:loni+2*MaskRadius)=P(lati:lati+2*MaskRadius,loni:loni+2*MaskRadius)+w.*zc;
end
W=W(MaskRadius+1:end-MaskRadius,MaskRadius+1:end-MaskRadius);
P=P(MaskRadius+1:end-MaskRadius,MaskRadius+1:end-MaskRadius);


%Ps = imgaussfilt(P,2.5);
%Ws = imgaussfilt(W,2.5);
%imagesc(log10(Ps./Ws))

%%

grid=W;
outdir='/g/data/xc0/project/GlobalPrecip/';
outfn=[outdir 'Wgauge_' datestr(dates(di),'yyyymmdd') '.nc'];
mappar.lats=lat;
mappar.lons=lon;
mappar.shortname='wgauge';
mappar.longname='gauge weight';
mappar.unit='-';
writeNCmap(outfn, grid, mappar);

grid=P./W;
grid(isnan(grid))=0;
outfn=[outdir 'Pgauge_' datestr(dates(di),'yyyymmdd') '.nc'];
mappar.lats=lat;
mappar.lons=lon;
mappar.shortname='P';
mappar.longname='precipitation';
mappar.unit='mm/d';
writeNCmap(outfn, grid, mappar);


%%

 
