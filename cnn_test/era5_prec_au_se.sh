module load gdal/2.0.0
module load cdo

DATE=$(date -ud "2015-01-01" -u)
END=$(date -ud "2018-07-01" -u)
while [ "$DATE" != "$END" ]; do
	FILE="/g/data/fj4/ECMWF/ERA5/precip_"$(date +%Y%m -ud "$DATE")".nc"
	echo $FILE
	if [ ! -f $FILE ]; then
		echo "PROBLEM"
		exit 1
	fi

	cdo selname,tp $FILE /g/data/fj4/scratch/tmp1.nc
	cdo sellonlatbox,130.0,150.0,-30.0,-40.0 /g/data/fj4/scratch/tmp1.nc /g/data/fj4/scratch/eu_tmp_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmp?.nc
	DATE=$(date -d "$DATE + 1 month" -u)
done

cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmp_*.nc /g/data/fj4/scratch/AU_SE_NATIVE_TP_ERA5.nc
rm /g/data/fj4/scratch/eu_tmp_*.nc
