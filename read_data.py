import xarray as xr
import pygrib as pygrib

# ds = xr.open_dataset("data/1.grib", engine="cfgrib")

# print(xr)

grbs = pygrib.open("data/1.grib")
grb = grbs.read(1)[0]
data = grb.values

# grbs.seek(0), tell, read, readline, close for each "message"
for grb in grbs:
    print(grb)

grb = grbs.select(name="Lake bottom temperature")[0]
print(grb.values - 273.15)
print(grb.values.shape, grb.values.min(), grb.values.max())

lats, lons = grb.latlons()
print(lats.shape, lats.min(), lats.max(), lons.shape, lons.min(), lons.max())

grbs.close()
