#!/bin/bash

module load grib_api
module load netcdf
RUNS=("00" "12")
OUTDIR='/g/data/ub8/global/Precipitation/NWP'
FTP_PATH='ftp://dissemination.ecmwf.int/'

echo $PFR_PSWD

DOWNLOAD_PFR() {
	if [ ! -d "$OUTDIR" ]; then
 		echo "Creating dir"$OUTDIR
		#mkdir $OUTDIR
	fi

	for RUN in {00,12}; do
		DATE=$(date -d "$1 + $RUN hour")
		for LEAD in {3..12..3}; do
			DATE=$(date -d "$DATE + 3 hour")
			FMT_DATE=$(date -d "$1" +%m%d)
			FMT_LEAD=$(date -d "$DATE" +%m%d%H%M)
			FNAME="E1D"$FMT_DATE$RUN"00"$FMT_LEAD"1"
                        if [ -f $OUTDIR"/"$FNAME".nc" ]; then
				echo "File "$FNAME".nc is already available"
				continue
			fi
			curl --user pfr_pull:$PFR_PSWD --head $FTP_PATH$1"/"$FNAME
			if [[ ! $? -eq 0 ]]; then
				echo "File "$FNAME" does not exists"
				continue
			fi
			curl --user pfr_pull:$PFR_PSWD -O $FTP_PATH$1"/"$FNAME
			grib_to_netcdf -o $OUTDIR"/"$FNAME".nc" $FNAME
			rm $FNAME
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
