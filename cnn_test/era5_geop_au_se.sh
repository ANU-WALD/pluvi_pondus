module load gdal/2.0.0
module load cdo

DATE=$(date -ud "2015-01-01")
END=$(date -ud "2018-07-01")
while [ "$DATE" != "$END" ]; do
	echo $DATE
	FILE="/g/data/fj4/ECMWF/ERA5/press_levels_"$(date +%Y%m -ud "$DATE")".nc"
	echo $FILE
	if [ ! -f $FILE ]; then
		echo "PROBLEM"
		exit 1
	fi

	cdo selname,z $FILE /g/data/fj4/scratch/tmpz1.nc
        cdo sellonlatbox,130.0,150.0,-30.0,-40.0 /g/data/fj4/scratch/tmpz1.nc /g/data/fj4/scratch/eu_tmpz_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmpz?.nc
	
        cdo selname,t $FILE /g/data/fj4/scratch/tmpt1.nc
	cdo sellonlatbox,90.0,180.0,0.0,-60.0 /g/data/fj4/scratch/tmpt1.nc /g/data/fj4/scratch/eu_tmpt_$(date +%Y%m%d%H%M -d "$DATE").nc
        cdo sellonlatbox,130.0,150.0,-30.0,-40.0 /g/data/fj4/scratch/tmpt1.nc /g/data/fj4/scratch/eu_tmpt_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmpt?.nc
        
	cdo selname,r $FILE /g/data/fj4/scratch/tmpr1.nc
	cdo sellonlatbox,90.0,180.0,0.0,-60.0 /g/data/fj4/scratch/tmpr1.nc /g/data/fj4/scratch/eu_tmpr_$(date +%Y%m%d%H%M -d "$DATE").nc
        cdo sellonlatbox,130.0,150.0,-30.0,-40.0 /g/data/fj4/scratch/tmpr1.nc /g/data/fj4/scratch/eu_tmpr_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmpr?.nc

	DATE=$(date -ud "$DATE + 1 month")
done

cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmpz_*.nc /g/data/fj4/scratch/AU_SE_NATIVE_Z_ERA5.nc
cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmpt_*.nc /g/data/fj4/scratch/AU_SE_NATIVE_T_ERA5.nc
cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmpr_*.nc /g/data/fj4/scratch/AU_SE_NATIVE_RH_ERA5.nc
rm /g/data/fj4/scratch/eu_tmpz_*.nc
rm /g/data/fj4/scratch/eu_tmpt_*.nc
rm /g/data/fj4/scratch/eu_tmpr_*.nc
