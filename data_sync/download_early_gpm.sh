#!/bin/bash

in=2018-07-10
while [ "$in" != 2018-07-25 ]; do
  in=$(date -I -d "$in + 1 day")
  x=$(date -d "$in" +%Y%m%d)
  echo "./final_gpm.sh" $x
  ./final_gpm.sh $x
  echo "./early_gpm.sh" $x
  ./early_gpm.sh $x
  echo "./late_gpm.sh" $x
  ./late_gpm.sh $x
done
