#!/bin/bash

DOI=$1

DATE=`date +%Y%m`

FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/early/'
OUTDIR='/g/data/fj4/SatellitePrecip/GPM/global/early/'$DOI

if [ -d "$OUTDIR" ]; then
	echo $OUTDIR$DATE
	# mkdir $OUTDIR
fi

if [ ! -d "$OUTDIR" ]; then
	echo $OUTDIR$DATE
fi
