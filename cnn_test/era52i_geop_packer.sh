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
	cdo selhour,0,6,12,18 /g/data/fj4/scratch/tmpz1.nc /g/data/fj4/scratch/tmpz2.nc
	cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/fj4/scratch/tmpz2.nc /g/data/fj4/scratch/infile.nc
	rm /g/data/fj4/scratch/tmpz?.nc
	cdo remapbic,outgrid.txt /g/data/fj4/scratch/infile.nc /g/data/fj4/scratch/eu_tmpz_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/infile.nc
	DATE=$(date -ud "$DATE + 1 month")
done

cdo -b f32 mergetime /g/data/fj4/scratch/eu_tmpz_*.nc /g/data/fj4/scratch/EU_Z_ERA5.nc
rm /g/data/fj4/scratch/eu_tmpz_*.nc
