#!/bin/bash

in=2018-04-01
while [ "$in" != 2018-07-01 ]; do
  in=$(date -I -d "$in + 1 day")
  x=$(date -d "$in" +%Y%m%d)
  echo "./late_gpm.sh" $x
  ./late_gpm.sh $x
done
