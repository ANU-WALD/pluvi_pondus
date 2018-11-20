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
	#cdo selhour,0,6,12,18 $FILE /g/data/fj4/scratch/tmp1.nc
	#cdo selname,tp /g/data/fj4/scratch/tmp1.nc /g/data/fj4/scratch/tmp2.nc 	
	cdo selname,tp $FILE /g/data/fj4/scratch/tmp2.nc 	
	#cdo sellonlatbox,110.0,158.0,-9.0,-45.0 /g/data/fj4/scratch/tmp.nc /g/data/fj4/scratch/tmp_$(date +%Y%m%d%H%M -d "$DATE").nc
	#cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/fj4/scratch/tmp2.nc /g/data/fj4/scratch/infilep.nc
	cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/fj4/scratch/tmp2.nc /g/data/fj4/scratch/eu_tmp_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmp?.nc
	#cdo remapcon,outgrid.txt /g/data/fj4/scratch/infilep.nc /g/data/fj4/scratch/eu_tmp_$(date +%Y%m%d%H%M -d "$DATE").nc
	#rm /g/data/fj4/scratch/infilep.nc
	DATE=$(date -d "$DATE + 1 month" -u)
done

#cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmp_*.nc /g/data/fj4/scratch/EU_TP_ERA5.nc
cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmp_*.nc /g/data/fj4/scratch/EU_TP_FULL_ERA5.nc
rm /g/data/fj4/scratch/eu_tmp_*.nc
