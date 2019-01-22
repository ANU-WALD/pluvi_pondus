#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
from datetime import timedelta, date

def monthly_range(start_date, end_date):
    start_date = start_date.replace(day=1)
    
    while start_date < end_date:
        last_day_month = start_date + timedelta(days=33)
        last_day_month = last_day_month.replace(day=1)
        last_day_month = last_day_month - timedelta(days=1)
        yield start_date, last_day_month
        start_date = last_day_month + timedelta(days=1)
        
server = ECMWFDataServer()

start_date = date(2015, 4, 1)
end_date = date(2018, 7, 1)
for (d0, d1) in monthly_range(start_date, end_date):
    server.retrieve({
        "type": "fc",
        "class": "ei",
        "dataset": "interim",
        "date": "{0}/to/{1}".format(d0.strftime('%Y-%m-%d'), d1.strftime('%Y-%m-%d')),
        "expver": "1",
        "levtype": "sfc",
        "area": "75/-50/15/40", # North, West, South, East. Default: global
        "grid": "0.75/0.75",
        "param": "228.128",
        "step": "6/12",
        "stream": "oper",
        "time": "00:00:00/12:00:00",
        "format": "netcdf",
        "target": "/g/data1/fj4/ECMWF/ERAInt/precip_{0}.nc".format(d0.strftime('%Y%m')),
   })
