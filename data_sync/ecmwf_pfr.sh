#!/bin/bash

module load grib_api
module load netcdf
RUNS=("00" "12")

DOWNLOAD_PFR() {

	FTP_PATH='ftp://dissemination.ecmwf.int/DATA/PFR/'$1
	OUTDIR='/g/data/xc0/user/pablo'

	if [ ! -d "$OUTDIR" ]; then
 		echo "Creating dir"$OUTDIR
		#mkdir $OUTDIR
	fi

	cd $OUTDIR
	curl -l $FTP_PATH'/*' --user pfr_pull:$PFR_PSWD | while read NAME; do 
		DATE=$1
		for RUN in "${RUNS[@]}"; do
			for LEAD in {3..12..3}; do
				FNAME='E1E'$(date +%m%d -d "$DATE")$RUN'00'$(date +%m%d%H%M -d "$DATE + $LEAD hour")'1'
				if [[ $NAME == $FNAME ]] && [ ! -f $outdir'/'$NAME'.nc' ]; then
					echo $NAME' does not exist!'
					curl -O $FTP_PATH'/'$NAME --user pfr_pull:$PFR_PSWD
					grib_to_netcdf -o $NAME"_INT.nc" $NAME
					nccopy -k 4 -d 3 -c 'longitude/200,latitude/200' $NAME'_INT.nc' $NAME'.nc'
					rm $NAME'_INT.nc'
					rm $NAME
				fi
			done
			DATE=$(date -d "$DATE + 12 hour")
		done
	done
}


if [ -z "$1" ]; then
	# If no argument supplied then we update the collection"
	
	#D-2
	DATE=`date -d "2 days ago 12:00" +%Y%m%d`
	DOWNLOAD_PFR "$DATE"

	#D-1
	DATE=`date -d "yesterday 12:00" +%Y%m%d`
	DOWNLOAD_PFR "$DATE"

	#TODAY
	DATE=`date +%Y%m%d`
	DOWNLOAD_PFR "$DATE"
else
	# Else we download the specified date YYYYMMDD"
	DOWNLOAD_PFR "$1"
fi
