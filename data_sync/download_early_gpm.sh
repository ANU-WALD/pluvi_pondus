#!/bin/bash

in=2015-01-01
while [ "$in" != 2018-07-10 ]; do
  in=$(date -I -d "$in + 1 day")
  x=$(date -d "$in" +%Y%m%d)
  ./final_gpm.sh $x
done
