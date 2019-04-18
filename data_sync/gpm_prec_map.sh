#!/bin/bash

OUTDIR="/g/data/fj4/SatellitePrecip/GPM/global/early/"

DOWNLOAD_GPM_EARLY() {
	DATE=$(date -d "$1" +%Y%m%d)
	echo $DATE

	FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/early/'${1:0:6}
	echo $FTP_PATH

	for SNAP in {0..47}; do
		MINS=$(($SNAP * 30))
		START=$(date -d "$DATE + $MINS minute" +%H%M%S)
		END=$(date -d "$DATE + $(($MINS + 30)) minute - 1 second" +%H%M%S)
		END_NAME=$(date -d "$DATE + $(($MINS + 30)) minute" +%H%M)
		IM_NAME="GPM"$DATE$END_NAME".png"
		echo $IM_NAME

		FNAME="3B-HHR-E.MS.MRG.3IMERG."$DATE"-S"$START"-E"$END"."`printf "%04d" $MINS`".V05B.RT-H5"
		if [ ! -f $OUTDIR$FNAME ]; then
			curl --user  $NRT_ACCOUNT:$NRT_ACCOUNT --head $FTP_PATH"/"$FNAME
			if [[ ! $? -eq 0 ]]; then
				echo "File "$FNAME" does not exists"
				continue
			fi
			curl -O $FTP_PATH'/'$FNAME --user $NRT_ACCOUNT:$NRT_ACCOUNT
			mv $FNAME $OUTDIR
		fi

        	if [ -f $IM_NAME ]; then
			echo "File "$IM_NAME" is already available"
			continue
		fi

		/g/data/xc0/software/python/miniconda3/bin/python gpm_prec_map.py $OUTDIR$FNAME $IM_NAME
		gsutil cp $IM_NAME gs://pluvi_pondus/
	done
	
        /g/data/xc0/software/python/miniconda3/bin/python gpm_3hprec_map.py $DATE
	gsutil cp GPM3H$DATE"*" gs://pluvi_pondus/
}

if [ -z "$1" ]; then
	# If no argument supplied the we update the collection"

	#TWO DAYS AGO
	DATE=`date -d "2 days ago 12:00" +%Y%m%d`
	DOWNLOAD_GPM_EARLY "$DATE"

	#YESTERDAY
	DATE=`date -d "yesterday 12:00" +%Y%m%d`
	DOWNLOAD_GPM_EARLY "$DATE"

	#TODAY
	DATE=`date +%Y%m%d`
	DOWNLOAD_GPM_EARLY "$DATE"
else
	# Else we download the specified date YYYYMMDD"
	DOWNLOAD_GPM_EARLY "$1"
fi

