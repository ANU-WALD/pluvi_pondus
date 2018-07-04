#!/bin/bash

in=2015-01-07
while [ "$in" != 2015-11-01 ]; do
  in=$(date -I -d "$in + 1 day")
  x=$(date -d "$in" +%Y%m%d)
  ./early_gpm.sh $x
done
