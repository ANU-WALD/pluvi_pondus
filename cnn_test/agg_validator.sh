module load cdo

DATE=$(date -ud "2015-01-01" -u)
FILE="/g/data/fj4/ECMWF/ERA5/precip_"$(date +%Y%m -ud "$DATE")".nc"
echo $FILE

cdo selname,tp $FILE /g/data/fj4/scratch/tmp.nc
cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/fj4/scratch/tmp.nc /g/data/fj4/scratch/raw_t.nc
cdo -b 32 timselsum,6 /g/data/fj4/scratch/raw_t.nc /g/data/fj4/scratch/agg_t.nc
