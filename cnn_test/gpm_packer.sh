module load gdal/2.0.0
module load cdo

DATE=wq$(date -d "2017-01-01")
END=$(date -d "2017-02-01")
while [ "$DATE" != "$END" ]; do
	ENDDATE=$(date -d "$DATE + 29 min + 59 sec")
	FILE="/g/data/fj4/SatellitePrecip/GPM/global/late/"$(date +%Y%m -d "$DATE")"/3B-HHR-L.MS.MRG.3IMERG."$(date +%Y%m%d -d "$DATE")"-S"$(date +%H%M%S -d "$DATE")"-E"$(date +%H%M%S -d "$ENDDATE")"."$(printf "%04d" $(( 10#$(date '+%H * 60 + %M' -d "$DATE") )))".V05B.RT-H5"
	if [ ! -f $FILE ]; then
		echo "PROBLEM"
		exit 1
	fi
	echo $FILE
	/g/data1/xc0/software/conda-envs/rs3/bin/python h52nc4.py -i $FILE -o out.nc
	cdo sellonlatbox,135.0,154.0,-29.0,-39.0 out.nc GPM_$(date +%Y%m%d%H%M -d "$DATE")_SW_AU.nc
        rm out.nc
	DATE=$(date -d "$DATE + 30 min")
done
