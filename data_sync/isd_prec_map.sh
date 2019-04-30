#!/bin/bash
COMPUTE_ISD() {
        python isd_6hprec_map.py $1
}
wget -r --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 5 ftp://ftp.ncdc.
noaa.gov/pub/data/noaa/isd-lite/2019/
if [ -z "$1" ]; then
        # If no argument supplied then we update the collection"
        #D-2
        DATE=`date -d "2 days ago 12:00" +%Y%m%d`
        COMPUTE_ISD "$DATE"
        #D-1
        DATE=`date -d "yesterday 12:00" +%Y%m%d`
        COMPUTE_ISD "$DATE"
else
        # Else we download the specified date YYYYMMDD"
        COMPUTE_ISD "$1"
fi
rm -rf ftp.ncdc.noaa.gov
gsutil cp *.png gs://pluvi_pondus/
rm *.png
