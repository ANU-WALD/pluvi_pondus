import imageio
import numpy as np
import glob
import xarray as xr
import datetime



def convert_video(date):
    writer = imageio.get_writer('composite_mystery_{}.mp4'.format(date), fps=4)
        
    dsg = xr.open_dataset("/data/GPM_HIM8/GPM_{}.nc".format(date))
    dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_{}.nc".format(date))

    for i, t in enumerate(dsg.time[100:295]):
        d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
        if np.datetime64(d) not in dsh.time.values:
            continue

        prec = dsg['PrecCal'].sel(time=t).values
        prec = np.clip(prec, 0, 30)/30

        b8 = (dsh['B8'].sel(time=t).values[::4,::4])
        b8 = (b8 - 197.)/(257.-197.)

        prec = (prec*255).astype(np.uint8)
        b8 = (b8*255).astype(np.uint8)

        stack = np.hstack((b8, prec))
        writer.append_data(stack)
    writer.close()


    dsg.close()
    dsh.close()


dates = ["201811","201812","201901","201902","201903"]
dates = ["201812"]

for date in dates:
    convert_video(date)
