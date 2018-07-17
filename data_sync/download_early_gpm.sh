#!/bin/bash

in=2018-07-01
while [ "$in" != 2018-07-20 ]; do
  in=$(date -I -d "$in + 1 day")
  x=$(date -d "$in" +%Y%m%d)
  ./final_gpm.sh $x
  ./early_gpm.sh $x
  ./final_gpm.sh $x
done
