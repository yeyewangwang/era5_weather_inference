import cdsapi

cds = cdsapi.Client()
cds.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "variable": "temperature",
        "pressure_level": "1000",
        "product_type": "reanalysis",
        "date": "2017-12-01/2017-12-31",
        "time": "12:00",
        "format": "grib",
    },
    "download.grib",
)
