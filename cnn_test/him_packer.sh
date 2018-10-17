module load gdal
module load nco
module load cdo

DATE=$(date -d "2017-01-01")
END=$(date -d "2017-02-01")
while [ "$DATE" != "$END" ]; do
	for BAND in {1..16}; do
		FILE="/g/data/rr5/satellite/obs/himawari8/FLDK/"$(date +%Y -d "$DATE")"/"$(date +%m -d "$DATE")"/"$(date +%d -d "$DATE")"/"$(date +%H%M -d "$DATE")"/"$(date +%Y%m%d%H%M -d "$DATE")"00-P1S-ABOM_OBS_B"$(printf %02d $BAND)"-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc"
		if [ -f $FILE ]; then
			gdalwarp -of netCDF -r cubic -t_srs '+proj=longlat +datum=WGS84 +no_defs' -te 135.0 -39.0 154.0 -29.0 -tr 0.02 -0.02 $FILE B$BAND.nc
			ncrename -v Band1,B$BAND B$BAND.nc
		fi
	done
	cdo merge B1.nc B2.nc B3.nc B4.nc B5.nc B6.nc B7.nc B8.nc B9.nc B10.nc B11.nc B12.nc B13.nc B14.nc B15.nc B16.nc HIM8_$(date +%Y%m%d%H%M -d "$DATE")_SW_AU.nc
	rm B*.nc
	DATE=$(date -d "$DATE + 30 min")
done
