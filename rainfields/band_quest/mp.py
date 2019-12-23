import multiprocessing
import xarray as xr
import datetime
import os
import numpy as np
import time

def read(q, fname):
    dsg = xr.open_dataset(fname)
    times = dsg.time.data
    np.random.shuffle(times)
    for i, t in enumerate(times):
        d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
        if not os.path.isfile("/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(d.strftime("%Y%m%d_%H%M%S"))):
            continue

        rf_fp = "/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(d.strftime("%Y%m%d_%H%M%S"))
        dsp = xr.open_dataset(rf_fp)
        prec = dsp['precipitation'].data[2:, 402:]
        dsp.close()
        b = dsg['B8'].sel(time=t).data[2:, 402:]
           
        #print((d, np.nanmean(b[:, :, None]), np.nanmean(prec[:, :, None])))    
        q.put((i, d, np.nanmean(b[:, :, None]), np.nanmean(prec[:, :, None])))

    q.put('DONE')

    dsg.close()

def wait_q(q, s):
    time.sleep(s)
    return not q.empty()

def wait_q(q,s):
    time.sleep(s)
    return not q.empty()

if __name__ == '__main__':
    fnames = ["/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181101.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181102.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181103.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181104.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181105.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181106.nc",
              "/data/pluvi_pondus/HIM8_AU_2B/0809/HIM8_AU_20181107.nc"]

    q = multiprocessing.Queue()
    jobs = []
    for fname in fnames:
        p = multiprocessing.Process(target=read, args=(q, fname,))
        p.daemon = True
        jobs.append(p)
        p.start()

    """
    for p in jobs:
        p.join()
    """
    time.sleep(2)

    print("AAAAAAA", q.empty())

    a = True
    a = []
    while len(a)<len(jobs):
        o = q.get()
        if o == "DONE":
            a.append(o)
        print(o)

    print("AAAAAAA", q.empty())
    for p in jobs:
        p.join()
