
%
addpath('/g/data/xc0/user/vandijk/github/precip/')
addpath('/g/data/xc0/user/vandijk/matlab/')

% time range
period.start=[2014 1 1];
period.end=[2017 12 31];
tilesize= 30;

% derive query parameters
ulclons=[-180:tilesize:180-tilesize];
ulclats=[90:-tilesize:-90+tilesize];
NlonR=numel(ulclons);
NlatR=numel(ulclats);

% Initialise results
gridres=0.1;
lon=[-180+gridres/2:gridres:180-gridres/2];
lat=[90-gridres/2:-gridres:-90+gridres/2];
Nlat=numel(lat); Nlon=numel(lon);
%sources={'GPM';'ECMWF'; 'ERAI'};
sources={'GPM';'ECMWF'};
vars={'R2';'Intercept';'Slope';'N'};
for si=1:numel(sources)
    for vi=1:numel(vars)
        eval([sources{si} '.' vars{vi} '=single(NaN.*zeros(Nlat,Nlon));'])
    end
end

%% calculate
for loni=1:NlonR
    for lati=1:NlatR
        RoI.tilesize=tilesize; % degrees
        RoI.ulclon=ulclons(loni);
        RoI.ulclat=ulclats(lati);
        fprintf('\n Doing: lat %d , lon %d',RoI.ulclat,RoI.ulclon)
        [greg]=member_stats(RoI,period);
        % assign
        jlat=find(lat>=RoI.ulclat-RoI.tilesize & lat<=RoI.ulclat);
        jlon=find(lon>=RoI.ulclon & lon<=RoI.ulclon+RoI.tilesize);
        minlat=min(jlat); maxlat=max(jlat); minlon=min(jlon); maxlon=max(jlon);
        for si=1:numel(sources)
            for vi=1:numel(vars)
                eval([sources{si} '.' vars{vi} '(minlat:maxlat,minlon:maxlon)=greg.' sources{si} '.' vars{vi} ';'])
                fprintf('.')
            end
        end
        fprintf('\n')
    end    
end


%% Show maps
figure(1)
sumR2=(ECMWF.R2+GPM.R2+ERAI.R2);
% subplot(3,3,1); imagesc(ECMWF.R2./sumR2,[0 2/3]); colorbar; title('w ECMWF'); axis off; axis equal;
% subplot(3,3,2); imagesc(ERAI.R2./sumR2,[0 2/3]); colorbar; title('w ERAI');axis off; axis equal;
% subplot(3,3,3); imagesc(GPM.R2./sumR2,[0 2/3]); colorbar; title('w GPM');axis off; axis equal;
subplot(3,3,1); imagesc(ECMWF.R2,[0 1]); colorbar; title('R2 ECMWF'); axis off; axis equal;
subplot(3,3,2); imagesc(ERAI.R2,[0 1]); colorbar; title('R2 ERAI');axis off; axis equal;
subplot(3,3,3); imagesc(GPM.R2,[0 1]); colorbar; title('R2 GPM');axis off; axis equal;

subplot(3,3,4); imagesc(ECMWF.Slope,[0 2]); colorbar; title('slope ECMWF'); axis off; axis equal;
subplot(3,3,5); imagesc(ERAI.Slope,[0 2]); colorbar; title('slope ERAI');axis off; axis equal;
subplot(3,3,6); imagesc(GPM.Slope,[0 2]); colorbar; title('slope GPM');axis off; axis equal;

subplot(3,3,7); imagesc(ECMWF.Intercept,[-5 5]); colorbar; title('int ECMWF'); axis off; axis equal;
subplot(3,3,8); imagesc(ERAI.Intercept,[-5 5]); colorbar; title('int ERAI');axis off; axis equal;
subplot(3,3,9); imagesc(GPM.Intercept,[-5 5]); colorbar; title('int GPM');axis off; axis equal;

colormap('jet')
linkaxes


%%
outdir='/g/data/xc0/project/GlobalPrecip/';
mappar.lats=lat;
mappar.lons=lon;
mappar.unit='-';
for si=1:numel(sources)
    for vi=1:numel(vars)
        outfn=[outdir sources{si} '.' vars{vi} '.nc'];
        eval(['grid=' sources{si} '.' vars{vi} ';'])
        grid(isnan(grid))=0;
        mappar.shortname= vars{vi} ;    
        writeNCmap(outfn, grid, mappar);
    end
end

%%
wgauge=ncread('/g/data/xc0/project/GlobalPrecip/Wgauge_20150111.nc','wgauge');
wecmwf=max(ECMWF.R2,0.001);
wgpm=max(GPM.R2,0.001);
rw_gauge=wgauge./(wgauge+wecmwf+wgpm);
rw_ecmwf=wecmwf./(wgauge+wecmwf+wgpm);
rw_gpm=wgpm./(wgauge+wecmwf+wgpm);
vars={'rw_gauge','rw_ecmwf','rw_gpm'};
for vi=1:numel(vars)
    outfn=[outdir vars{vi} '.nc'];
    eval(['grid=' vars{vi} ';'])
    grid(isnan(grid))=0;
    mappar.shortname= vars{vi} ;
    writeNCmap(outfn, grid, mappar);
end




