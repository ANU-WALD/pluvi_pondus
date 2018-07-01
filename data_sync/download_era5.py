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

start_date = date(2018, 1, 1)
end_date = date(2018, 4, 1)
for (d0, d1) in monthly_range(start_date, end_date):
    server.retrieve({
        "type": "fc",
        "class": "ea",
        "dataset": "era5",
        #"date": "{0}/to/{1}".format(d0.strftime('%Y-%m-%d'), d1.strftime('%Y-%m-%d')),
        "date": "2018-01-01/to/2018-06-01",
        "expver": "1",
        "levtype": "sfc",
        "param": "228.128/260015",
        #"step": "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18",
        "step": "0/1/2/3/4/5/6/7/8/9/10/11",
        "stream": "oper",
        "time": "06:00:00/18:00:00",
        "grid": "0.25/0.25",
        "format": "netcdf",
        "target": "/g/data1/fj4/ERA5/precip_{0}.nc".format(d0.strftime('%Y%m')),
    })
