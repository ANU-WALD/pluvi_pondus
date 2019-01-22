module load cdo

DATE=$(date -ud "2015-01-01")
END=$(date -ud "2018-07-01")
while [ "$DATE" != "$END" ]; do
	echo $DATE

	#1.- Chop Geographical area
	cdo sellonlatbox,-50.0,40.0,75.0,15.0 /g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/tp/tp_3hrs_ERAI_historical_fc-sfc_$(date +%Y%m%d -d "$DATE")_*.nc /g/data/fj4/scratch/eu_tmp.nc

	#2.- Select mid accumulated ranges
	cdo selhour,6,18 /g/data/fj4/scratch/eu_tmp.nc /g/data/fj4/scratch/eu_tp_mid.nc

	#3.- Shift fordwards 6h mid ranges
	cdo shifttime,6hours /g/data/fj4/scratch/eu_tp_mid.nc /g/data/fj4/scratch/eu_tp_mid_shifted.nc

	#4.- Change name of end tp -> tpe
	cdo setname,tpi /g/data/fj4/scratch/eu_tp_mid_shifted.nc /g/data/fj4/scratch/eu_tp_mid_shifted_renamed.nc

	#5.- Select end accumulated ranges
	cdo selhour,0,12 /g/data/fj4/scratch/eu_tmp.nc /g/data/fj4/scratch/eu_tp_end.nc

	#6.- Merge into one file
	cdo merge /g/data/fj4/scratch/eu_tp_end.nc /g/data/fj4/scratch/eu_tp_mid_shifted_renamed.nc /g/data/fj4/scratch/eu_tp_merged.nc

	#6.- Expr
	cdo expr,'tp=(tp-tpi)' /g/data/fj4/scratch/eu_tp_merged.nc /g/data/fj4/scratch/eu_tp_end_corr.nc

	#6.- Merge both back into one file
	cdo mergetime /g/data/fj4/scratch/eu_tp_end_corr.nc /g/data/fj4/scratch/eu_tp_mid.nc /g/data/fj4/scratch/int_$(date +%Y%m%d%H%M -d "$DATE").nc
	rm /g/data/fj4/scratch/eu_*.nc

	DATE=$(date -ud "$DATE + 1 month")
done

cdo -b F32 mergetime /g/data/fj4/scratch/int_*.nc /g/data/fj4/scratch/EU_TP_ERAI.nc
rm /g/data/fj4/scratch/int_*.nc
