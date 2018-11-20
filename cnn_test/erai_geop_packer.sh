module load cdo

DATE=$(date -ud "2015-01-01")
END=$(date -ud "2018-07-01")
while [ "$DATE" != "$END" ]; do
	echo $DATE
	cdo sellevel,100000,85000,50000 /g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/z/z_6hrs_ERAI_historical_an-pl_$(date +%Y%m%d -d "$DATE")_*.nc /g/data/fj4/scratch/tmpz.nc
	cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/fj4/scratch/tmpz.nc /g/data/fj4/scratch/tmpz_eu_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/tmpz.nc
	DATE=$(date -ud "$DATE + 1 month")
done

cdo -b F32 mergetime /g/data/fj4/scratch/tmpz_eu_*.nc /g/data/fj4/scratch/EU_Z_ERAI.nc
rm /g/data/fj4/scratch/tmpz_eu_*.nc
