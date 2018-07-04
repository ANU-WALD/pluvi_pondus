#!/bin/bash

DOWNLOAD_GPM_EARLY() {
	FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/early/'${1:0:6}
	OUTDIR='/g/data/fj4/SatellitePrecip/GPM/global/early/'${1:0:6}

	if [ ! -d "$OUTDIR" ]; then
 		echo "Creating dir"$OUTDIR
		mkdir $OUTDIR
	fi

	cd $OUTDIR
	curl -l $FTP_PATH'/*' --user $NRT_ACCOUNT:$NRT_ACCOUNT | while read NAME; do 
		if [[ $NAME == 3B-HHR-E.MS.MRG.3IMERG.$1* ]] && [ ! -f $OUTDIR'/'$NAME ]; then
    			echo $NAME" Does not exist!"
			curl -O $FTP_PATH'/'$NAME --user $NRT_ACCOUNT:$NRT_ACCOUNT
		fi
	done
}

if [ -z "$1" ]; then
	# If no argument supplied the we update the collection"

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

