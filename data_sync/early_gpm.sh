#!/bin/bash

#YESTERDAY
DATE=`date -d "yesterday 12:00" +%Y%m%d`

FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/early/'${DATE:0:6}
OUTDIR='/g/data/fj4/SatellitePrecip/GPM/global/early/'${DATE:0:6}

if [ ! -d "$OUTDIR" ]; then
 	echo "Creating dir"$OUTDIR
	mkdir $OUTDIR
fi

cd $OUTDIR
curl -l $FTP_PATH'/*' --user $NRT_ACCOUNT:$NRT_ACCOUNT | while read NAME; do 
        if [[ $NAME == 3B-HHR-E.MS.MRG.3IMERG.$DATE* ]] && [ ! -f $OUTDIR'/'$NAME ]; then
    		echo $NAME" Does not exist!"
		curl -O $FTP_PATH'/'$NAME --user $NRT_ACCOUNT:$NRT_ACCOUNT
	fi
done


#TODAY
DATE=`date +%Y%m%d`

FTP_PATH='ftp://jsimpson.pps.eosdis.nasa.gov/NRTPUB/imerg/early/'${DATE:0:6}
OUTDIR='/g/data/fj4/SatellitePrecip/GPM/global/early/'${DATE:0:6}

if [ ! -d "$OUTDIR" ]; then
 	echo "Creating dir"$OUTDIR
	mkdir $OUTDIR
fi

cd $OUTDIR
curl -l $FTP_PATH'/*' --user $NRT_ACCOUNT:$NRT_ACCOUNT | while read NAME; do 
        if [[ $NAME == 3B-HHR-E.MS.MRG.3IMERG.$DATE* ]] && [ ! -f $OUTDIR'/'$NAME ]; then
    		echo $NAME" Does not exist!"
		curl -O $FTP_PATH'/'$NAME --user $NRT_ACCOUNT:$NRT_ACCOUNT
	fi
done

