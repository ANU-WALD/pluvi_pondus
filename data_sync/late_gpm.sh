#!/bin/bash

DOWNLOAD_GPM_LATE() {
	FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/late/'${1:0:6}
	OUTDIR='/g/data/fj4/SatellitePrecip/GPM/global/late/'${1:0:6}

	if [ ! -d "$OUTDIR" ]; then
 		echo "Creating dir"$OUTDIR
		mkdir $OUTDIR
	fi

	cd $OUTDIR
	curl -l $FTP_PATH'/*' --user $NRT_ACCOUNT:$NRT_ACCOUNT | while read NAME; do 
    		echo $NAME
		if [[ $NAME == 3B-HHR-L.MS.MRG.3IMERG.$1* ]] && [ ! -f $OUTDIR'/'$NAME ]; then
    			echo $NAME" Does not exist!"
			curl -O $FTP_PATH'/'$NAME --user $NRT_ACCOUNT:$NRT_ACCOUNT
		fi
	done
}

if [ -z "$1" ]; then
	# If no argument supplied the we update the collection"

	#YESTERDAY
	DATE=`date -d "yesterday 12:00" +%Y%m%d`
	DOWNLOAD_GPM_LATE "$DATE"

	#TODAY
	DATE=`date +%Y%m%d`
	DOWNLOAD_GPM_LATE "$DATE"
else
	# Else we download the specified date YYYYMMDD"
	DOWNLOAD_GPM_LATE "$1"
fi

