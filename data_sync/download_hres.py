import sys
import ecmwfapi
import re
import os
from datetime import timedelta, date

def monthly_range(start_date, end_date):
    start_date = start_date.replace(day=1)
    
    while start_date < end_date:
        last_day_month = start_date + timedelta(days=33)
        last_day_month = last_day_month.replace(day=1)
        last_day_month = last_day_month - timedelta(days=1)
        yield start_date, last_day_month
        start_date = last_day_month + timedelta(days=1)
        
start_date = date(2016, 1, 1)
end_date = date(2018, 1, 1)
for (d0, d1) in monthly_range(start_date, end_date):
    req = """retrieve,
    class   = od,
    date    = {0}/to/{1},
    stream  = oper,
    levtype = sfc,
    expver  = 1,
    time    = 00/12,
    step    = 1/2/3/4/5/6/7/8/9/10/11/12,
    type    = fc,
    param   = 228.128,
    grid    = 0.1/0.1,
    format  = netcdf,
    target  = \"/g/data/fj4/ECMWF/HRES/precip_{2}.nc\"""".format(d0.strftime('%Y-%m-%d'), 
                                                             d1.strftime('%Y-%m-%d'),
                                                             d0.strftime('%Y%m'))
    print(req)

    if "WEBMARS_TARGET" in os.environ:
        target = os.environ["WEBMARS_TARGET"]
    else:
        m = re.search('\\btar(g(e(t)?)?)?\s*=\s*([^\'",\s]+|"[^"]*"|\'[^\']*\')', req, re.I|re.M)
        if m is None:
            raise Exception("Cannot extract target")

        target=m.group(4)
        if target is None:
            raise Exception("Cannot extract target")

    if target[0] == target[-1]:
        if target[0] in ['"', "'"]:
            target = target[1:-1]


    c = ecmwfapi.ECMWFService('mars')
    c.execute(req, target)
